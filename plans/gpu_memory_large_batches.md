# GPU memory scaling for large single-batch cell counts + `allow_missing_features`

Status: **fixes 1 and 2 implemented** (see "Implementation status" at the
bottom). Fixes 3-5 remain investigation-only, for if 1+2 aren't enough.
Triggered by: NBAtlas malignantCells stability sweep (`anglemania_analysis/tasks/nmf`)
hitting GPU OOM when trying to run `allow_missing_features=True` on the malignant
compartment, after it had already worked fine on the immune/stroma compartment
(see that repo's `TODO.md`, "Does `allow_missing_features` fix marker recovery?").

Related existing docs — this note is additive, not a duplicate:
- `plans/optimization.md` #5 already flags `StreamingZscoreStats`'s float64
  accumulators as *the* documented OOM risk at 20k+ genes, but stops at "needs
  validation, don't implement blindly." This note works out *why* the malignant
  compartment specifically hits that risk, and adds fixes beyond just the dtype.
- `CLAUDE.md` "Scaling to large gene panels" documents the `finalize()` buffer-count
  fix and benchmarks at 20k cells x 20k genes x 4 batches. This note is about a
  different regime: one batch alone can be far bigger than that benchmark's whole
  dataset, and `allow_missing_features` can push the gene universe well past 20k too.

## Real numbers behind the diagnosis

Checked directly against the actual malignantCells h5ad
(`NB_Atlas_morethan150cells_malignantCells.h5ad`, backed-mode read, no GPU needed):

- **42,413 genes** in the raw panel (far more than the 20k-gene benchmark in CLAUDE.md).
- **54 `Sample` batches**, cell counts highly skewed: mean 4,526, but max **22,434**
  (`Dong2020_T162`), with `Dong2020_T75` (17,564), `Dong2020_T174` (15,222),
  `Dong2020_T69` (13,840) close behind — a handful of single batches are
  themselves bigger than the entire 20k-cell CLAUDE.md benchmark.
- Strict-mode `intersect_genes` (current default, `allow_missing_features=False`)
  already measured at 7,993 genes for this dataset (per `anglemania_analysis`'s
  `TODO.md`) — small, unproblematic.
- `allow_missing_features=True` with `min_samples_frac=0.5` changes `intersect_genes`
  from "present in literally every one of 54 batches" to "present in >= 27 of 54"
  (`_batches.py::intersect_genes`, the `Counter`-based branch) — on the immune/stroma
  compartment this took the pool from 7,140 -> 16,969 (2.38x). Applied to malignant's
  much larger 42,413-gene panel, the same relaxation is very plausibly landing in the
  20-30k+ range, not yet measured directly (see "Next steps" below).

## Diagnosis: two compounding, mechanically distinct bottlenecks

`anglemania()` already loops per batch and streams the cross-batch reduction
(`StreamingZscoreStats`, `_stats.py`) — it never holds every batch's data at once.
So this is *not* the textbook "whole dataset doesn't fit on one GPU" problem that
`dask`/`rapids-singlecell`/`dask-cuda` solve by chunking the cell axis of one big
array. Two different things are actually blowing up:

**1. `O(genes^2)` buffers, worsened by `allow_missing_features` inflating `n_genes`.**
`StreamingZscoreStats` holds two *persistent*, whole-run `(n_genes, n_genes)` **float64**
accumulators (`_wz_sum`, `_wz2_sum`, `_stats.py:35-36` — already flagged in
`optimization.md` #5). On top of that, `factorise`/`extract_angles` (`_angles.py`)
allocate 4-5 more `(n_genes, n_genes)` **float32** temporaries *per batch*
(`cov`, `vlr`, `vlp`, `result`/ratio, plus `corr` and `perm_corr` both held at once
inside `factorise`) — these stay alive simultaneously because they're local
variables never explicitly freed/reused in place (contrast with `finalize()`,
which was already rewritten with in-place ops + `del` for exactly this reason).
None of this scales with cell count at all — it scales purely with `n_genes^2`,
so `allow_missing_features` growing the gene universe is what's newly exposing it
for malignant cells, independent of anything about batch sizes.

**2. Per-batch dense-matrix copies scale with that batch's own cell count, and
malignant's batches are extremely skewed.** `align_to_common_genes`,
`normalize_matrix`, and `permute_matrix` each materialize a full
`(cells_b x n_genes)` dense array (original, permuted, centered, normalized —
several full copies coexist per batch in `factorise`). For most of the 54
batches this is small (median ~3,150 cells), but for `Dong2020_T162`
(22,434 cells) at an inflated `allow_missing_features` gene count, several
GB-sized copies exist simultaneously for that one batch alone.

These compound: bottleneck 1 sets a high *persistent* floor for the whole run;
bottleneck 2 adds a large *transient* spike on top of it for the few biggest
batches — so the run can OOM specifically (and only) on those large batches,
which matches the earlier observation that the two biggest bootstrap draws
(185k/200k cells) were the only ones that failed even under strict mode
(`stability_pyanglemania_bootstrap.py`'s existing GPU-then-CPU-retry logic).

## Candidate fixes, roughly in order of effort/payoff

1. **Move `StreamingZscoreStats`'s accumulators to host (numpy) memory.**
   They're touched exactly once per batch (`update()`), so there's no repeated
   GPU<->host ping-pong — pull that batch's `zscores` to host with one
   `cupy.asnumpy()`, accumulate in plain numpy float64, discard the GPU copy.
   Removes the two biggest *persistent* GPU buffers (they're the float64 ones)
   for the cost of one `(genes^2)` D2H transfer per batch (low, relative to
   per-batch compute time). This is a variant of `optimization.md` #5's fix that
   sidesteps the "does float32 lose precision?" question entirely, since it
   doesn't change dtype — worth trying before/alongside float32 validation.

2. **Cell-chunk the per-batch dense-matrix construction and GEMM
   (fixes bottleneck 2).** `cov[j,k] = sum_i (X[i,j]-mean_j)(X[i,k]-mean_k)`
   decomposes additively over row-blocks of `X`: `cov = sum_chunks
   (X_chunk_centered.T @ X_chunk_centered)`. Two passes per batch — one to
   accumulate the per-gene mean (and the permuted matrix's own mean), one to
   accumulate `cov`/`cov_perm` chunk by chunk — bounds GPU memory to
   `O(chunk_size x genes + genes^2)` regardless of that batch's total cell
   count, with the same result as processing the whole batch at once (exact,
   not a subsample). The default `permute_row_or_column="column"` permutation
   is per-row-independent, so it's chunkable the same way. Doesn't touch
   bottleneck 1 at all — the accumulator itself is still `genes^2`.

3. **Gene-block-tile the angle/z-score computation (fixes bottleneck 1,
   the "real" fix, most invasive).** FlashAttention-style: tile *both* axes
   of the `(genes, genes)` output into blocks (e.g. 2,000 genes), compute
   `cov_block = X_centered[:, gi].T @ X_centered[:, gj]` per block pair
   (small, e.g. 2000x2000x4B = 16MB), fold each block's contribution directly
   into a host-resident accumulator slice, discard the GPU block. Peak GPU
   memory becomes `O(cells x genes + block^2)`, independent of total gene
   count — composes with fix 2 (chunk cells for the GEMM's contraction axis,
   chunk genes for its output axes). This is the one lever that actually caps
   memory regardless of how far `allow_missing_features`/`min_samples_frac`
   push `n_genes`.

4. **Hybrid GPU-compute/CPU-elementwise split, as a smaller step toward 3.**
   Only `cov = x_centered.T @ x_centered` is a genuine GEMM that benefits from
   the GPU; `vlr`, `vlp`, the ratio, and the null z-scoring are all
   memory-bandwidth-bound elementwise ops on `(genes, genes)` arrays. Doing
   the matmul on GPU, immediately `.get()`-ing `cov` to host, and finishing
   `vlr`/`vlp`/ratio/z-score in numpy trades some elementwise speed for using
   host RAM (typically 64GB-1TB on these nodes vs. 16-80GB VRAM) for the
   buffers that don't actually need GPU parallelism.

5. **Managed/unified memory oversubscription as a quick experiment, not a
   real fix.** cupy's managed-memory allocator (or RMM's managed pool) pages
   GPU allocations to host RAM automatically on oversubscription — zero code
   change, worth a 10-minute try, but degrades badly under sustained
   random-access working sets that are a large multiple of GPU capacity,
   which is close to what a `genes^2` elementwise chain looks like. Don't
   rely on this as the actual solution.

## What `rapids-singlecell`/`dask-cuda` actually solve (and don't)

Worth being precise about this since it's the natural comparison: their core
ops (normalization, HVG stats, randomized PCA, kNN) are all `O(cells x genes)`,
not quadratic in genes, so chunking the *cell* axis via `dask.array` +
`dask-cuda`'s automatic spill-to-host is sufficient for their workload. This
package's problem is different in kind — the *output itself*
(`(genes, genes)`) is quadratic in genes — so cell-chunking (fix 2) is
necessary but not sufficient; it needs to be paired with genuine gene-block
tiling (fix 3) to cap memory when the gene universe itself grows large. The
closest real precedent for fix 3 is FlashAttention's block-wise online
reduction over an `(N, N)` attention matrix, not anything in rapids-singlecell
itself (checked: no primitive there computes a full gene-by-gene or
cell-by-cell dense matrix the way this pipeline does).

## Implementation status

Fixes 1 and 2 are implemented (fix 3/4/5 are not — revisit only if 1+2 turn
out insufficient on the actual malignantCells run):

- **Fix 1** (`_stats.py::StreamingZscoreStats`): the two persistent
  `(genes x genes)` float64 accumulators (`_wz_sum`, `_wz2_sum`) are now
  always plain numpy/host arrays, regardless of what array module a batch's
  z-score matrix comes from. `update()` does one `to_numpy()` (a
  `cupy.asnumpy()` D2H transfer when fed a cupy batch) and accumulates on
  host from there; `finalize()` always returns numpy. Dtype is unchanged
  (still float64 — see `plans/optimization.md` #5's update note), so this
  doesn't touch the catastrophic-cancellation question at all, only where
  the buffers live. `StreamingZscoreStats.__init__` dropped its now-unused
  `xp` parameter.
- **Fix 2** (`_angles.py::factorise_chunked`, wired up in `_anglemania.py`
  via a new `cell_chunk_size` parameter on `pp.anglemania()`, default `None`
  = unchanged behavior): processes one batch in row-chunks of at most
  `cell_chunk_size` cells, taking each chunk from raw counts through
  alignment/permutation/normalization to an accumulated contribution to the
  uncentered Gram matrix (`sum_sq`) and column sums, then discarding the
  chunk. The centered Gram matrix (and from it, `cosine`/`phi_s`'s angle
  matrix) is recovered exactly from those accumulated moments after the
  loop via `sum_sq - n*outer(mean, mean)` — the same sum-of-squares identity
  `StreamingZscoreStats` already uses across batches, applied here within a
  batch across cell-chunks. Bounds peak memory to
  `O(cell_chunk_size x genes + genes^2)` regardless of that batch's actual
  cell count.

  Only supports the combination that's both chunkable without a
  global/second pass *and* is what the actual failing workload uses:
  `permute_row_or_column="column"` (row-independent permutation),
  `method` in `("cosine", "phi_s")` (both reduce to the centered Gram
  matrix; `"spearman"` needs a global rank, not implemented), and
  `normalization_method` in `("divide_by_total_counts", "pflog1ppf")` (both
  per-cell-only; `"find_residuals"` needs global stats before centering,
  not implemented). Other combinations raise `ValueError` rather than
  silently falling back to full materialization.

  Caveat documented in the docstring: the permuted null is not bit-identical
  across different `cell_chunk_size` values for the same `seed` (generating
  the permutation's random keys is itself chunked, so the RNG draw sequence
  differs), though it's still a valid independent random permutation either
  way. The *real* (unpermuted) angle matrix matches the unchunked path
  almost exactly (floating point only) regardless of chunk size — verified
  in `tests/test_angles.py` (`test_angles_from_moments_accumulation_is_chunk_size_independent`,
  `test_factorise_chunked_single_chunk_matches_factorise`) and end-to-end in
  `tests/test_anglemania.py` (`test_anglemania_cell_chunk_size_single_chunk_matches_unchunked`,
  using a `cell_chunk_size` larger than every batch so there's exactly one
  chunk per batch — same rng draw as the unchunked path, selected genes
  match exactly).

  `align_to_common_genes` (`_batches.py`) itself is unchanged — it's called
  once per chunk instead of once per batch, so a chunk's row-slice is
  aligned/densified at chunk scale, never at whole-batch scale.

**GPU-validated** (SLURM job 31490, Tesla P40, cupy 14.1.1): all 14
`tests/test_gpu.py` cases pass, including new chunking-specific parity checks
(`factorise_chunked` cupy vs numpy, `cell_chunk_size` end-to-end on a cupy
sparse-backed `AnnData`, single-chunk == unchunked). More importantly, a
synthetic single-batch reproduction of the actual failure mode confirmed the
fix works: a 60,000-cell x 15,000-gene batch (density 0.08, roughly
Dong2020_T162's scale with an inflated `allow_missing_features` gene count)
hit a genuine `cupy.cuda.memory.OutOfMemoryError` with `cell_chunk_size=None`
(wanted 7.2GB more after already holding ~20.9GB on the P40's 23GB) — the
exact bottleneck-2 failure this note describes — and completed successfully
with `cell_chunk_size=5000` (135s) and `cell_chunk_size=2000` (41s). Wall-clock
numbers from that same run session are confounded by cupy's one-time NVRTC
kernel-compilation cost on first use (the `cell_chunk_size=None` runs, being
first in sequence, paid a warm-up cost later chunked runs didn't) — not yet
re-measured with a proper warm-up pass, so no clean chunking-overhead number
exists yet, only the OOM-vs-succeeds result.

Not done, and why they're lower priority now: fix 3 (gene-block tiling) is
the only lever that caps the *persistent* `genes^2` floor itself (fix 1 just
moves it off GPU, doesn't shrink it) — worth revisiting if a single batch's
`(genes, genes)` buffers alone don't fit even after 1+2. Fix 4 (hybrid
GPU/CPU elementwise split) and fix 5 (managed memory) were never more than
fallback ideas.

### Next steps

- Run the actual NBAtlas malignantCells `allow_missing_features=True` sweep
  with `cell_chunk_size` set (e.g. matching the immune/stroma pilot's
  `--allow-missing-features --min-samples-frac 0.5` convention) and confirm
  it completes without OOM on the two batches (`Dong2020_T162` at 22,434
  cells, etc.) that motivated this note.
- Measure the actual post-`allow_missing_features` gene pool size for
  malignantCells directly (`intersect_genes` bookkeeping only, no GPU needed)
  to confirm whether bottleneck 1 (now mitigated by fix 1) would have been
  prohibitive on its own, independent of the large-batch bottleneck 2 fix.
- `stability_pyanglemania_bootstrap.py` in `anglemania_analysis` would need
  a `--cell-chunk-size` CLI flag (plumbed to `anglemania_kwargs`) to actually
  use this from that script's GPU run.
