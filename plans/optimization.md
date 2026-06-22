# Optimization notes

## Branch: experiment/per-gene-aggregate-selection

Implements alternative 3 from the "other approaches to gene ranking"
discussion (see the `## 6.` writeup below) as an opt-in
`anglemania(..., selection_method="per_gene")`, kept off `main` because it
is a genuinely different, non-R-faithful selection criterion rather than a
performance optimization of the existing one:
`score_genes_by_aggregate`/`select_top_genes` in `_select.py` score each
gene directly (sum of direction-adjusted `|mean_zscore| * sn_zscore` over
its surviving pairs from the existing on-device `prefilter_gene_pairs`)
and take the top `max_n_genes` by that score, instead of
`rank_gene_pairs`/`extract_unique_genes`'s rank-pairs-then-walk approach.
`xp`-dispatched (works on numpy and cupy), with a GPU-vs-CPU parity test
in `test_gpu.py`. Not benchmarked against the pairwise path's runtime or
validated against R/the existing algorithm's gene sets -- it exists to
explore the idea, not as a recommended default.

## Implementation status

Items #1, #3, #4 were implemented per the `--> implement` markers (#2 and #5
were left alone per `--> don't implement`). #6 was discussed in depth
afterward (see #6 below for the full writeup); its sub-items (c) (defer
`geneA`/`geneB` string materialization until after ranking) and (d) (bypass
pandas entirely on the GPU path, via verified `cupy.sort`+`searchsorted`)
were both then implemented by request. All 64 existing tests (56 CPU + 8
GPU, including a new cupy-vs-pandas ranking parity test added alongside (d))
pass and `ruff check` is clean after every change in this file.

- **#1** (`_select.py::prefilter_gene_pairs`): now resolves `xp` from
  `get_array_module(mean_zscore)` and does the `triu_indices`/gather/
  threshold-mask entirely on-device, calling `to_numpy()` only on the
  post-filter arrays. Verified with a fair (apples-to-apples, including the
  pandas DataFrame construction both versions need) before/after benchmark
  at 8000 genes, ~38% pairs passing: **6.9s/iter new vs 11.9s/iter old
  (~1.7x), identical output DataFrame**. (An earlier *unfair* comparison
  that skipped the DataFrame-building step in the "old" reference made the
  new version look slower -- worth remembering if re-benchmarking this:
  always compare the full function, not a partial reimplementation of it.)
- **#3** (sparse column-indexing): `_anglemania.py` now converts each
  batch to CSC once, right after row-selection; `genes_passing_min_cells`'s
  sparse branch uses `xp.diff(X.tocsc().indptr)` instead of bincount.
  Verified `.tocsc()` is a no-op (`is`-identical) on an already-CSC matrix
  in both scipy and cupyx, so this costs nothing when already converted.
  Verified CSC boolean column-masking is ~33x faster than CSR's on this GPU
  (1.09ms vs 36.7ms at 5000x10000, 5% density), identical results.
- **#4** (RNG): `factorise` now builds an `xp.random.default_rng(seed)`
  and threads it through `permute_matrix`/`_shuffle_full`/`_shuffle_nonzero`
  (signatures gained an `rng` parameter) instead of mutating global RNG
  state via `xp.random.seed()`; the random sort-key arrays are generated as
  `float32` directly rather than `rand()`'s hardcoded `float64`. Updated the
  four `test_angles.py` unit tests that called these internals directly
  with the old signature. `factorise`'s own public contract (`seed: int`,
  reproducibility) is unchanged and its existing tests needed no edits.

## Context: where the time goes today

Benchmark from a prior run, 20k cells x 20k genes x 4 batches, `max_n_genes=2000`,
default `prefilter_threshold=0.5` (see also CLAUDE.md "Scaling to large gene panels"):

| stage                                                | GPU    | CPU                  |
| ----------------------------------------------------- | ------ | -------------------- |
| gene filtering                                        | 0.1s   | 6.2s                 |
| per-batch `factorise` (all 4 batches)                  | ~6s    | ~229s                |
| `finalize`                                             | 0.5s   | 15.8s                |
| `prefilter_gene_pairs` (~54.5M of 200M pairs passed)   | 59.5s  | 49.2s                |
| `rank_gene_pairs`                                      | 124.5s | 129.2s                |
| `extract_unique_genes`                                 | 0.01s  | 0.01s                 |
| **total**                                              | **191s** | **430s** (2.25x slower) |

The GPU's 30-38x advantage is confined to `factorise`/`finalize` (dense numeric
work). `prefilter_gene_pairs` and `rank_gene_pairs` run on CPU/numpy/pandas
regardless of backend and dominate the GPU-backed total (96% of the 191s).
**`prefilter_gene_pairs` is actually slower on the GPU pipeline than the CPU
one** (59.5s vs 49.2s) — that's the smoking gun for caveat #1 below.

Everything below was checked against the actual installed `cupy==14.1.1`,
`scipy`, and the `ref_packages/rapids-singlecell` / `ref_packages/scanpy`
sources in this repo, and several claims are backed by microbenchmarks run on
this machine (not just docs) — numbers are reproducible but specific to this
GPU/CPU; re-benchmark before relying on the exact multipliers elsewhere.

## 1. `_select.py::prefilter_gene_pairs` transfers full (genes x genes) matrices to host before filtering — this is the "do we need cupy->numpy?" caveat --> implement

`prefilter_gene_pairs` (`_select.py:17-63`) does, unconditionally:

```python
i_idx, j_idx = np.triu_indices(n, k=1)              # _select.py:37, CPU, full upper triangle
mean_flat = to_numpy(mean_zscore)[i_idx, j_idx]      # _select.py:38
sd_flat   = to_numpy(sds_zscore)[i_idx, j_idx]       # _select.py:39
sn_flat   = to_numpy(sn_zscore)[i_idx, j_idx]        # _select.py:40
```

`to_numpy()` D2H-transfers the *entire* (genes x genes) matrix even though only
the upper triangle is ever read, and the threshold filter (`keep = ...`,
`_select.py:43`) is applied only *after* all three full transfers. At 20k
genes that's 3 x 3.2GB = 9.6GB moved over PCIe, plus two 200M-element int64
index arrays built on the CPU side purely to gather — before a single
"does this pair pass?" check has happened. This matches the benchmark above
almost exactly: it's why this step is *slower* on the GPU pipeline than on
CPU (no transfer needed there).

**Fix**: do the triu-extraction and threshold filtering on-device. `cupy`
supports `cp.triu_indices`, boolean masking, and `cp.abs` identically to
numpy, so `prefilter_gene_pairs` can take `xp` and run the
`(sn_flat >= sn_thr) & (abs(mean_flat) >= mean_thr)` mask (including the
relax-by-0.1 retry loop) entirely on-device, then call `to_numpy()` only on
the already-filtered (much smaller) 1-D arrays before building the
`pandas.DataFrame`.

**Impact**: in the benchmark above, 54.5M of ~200M pairs passed — filtering
first cuts the host transfer from "the whole matrix" to "54.5M x 5
small arrays" (~1.7GB), roughly 5-6x less data moved, and skips materializing
the two 200M-length CPU-side index arrays. Should flip this step back into
GPU-faster-than-CPU territory.

**Effort/risk**: low. Same output, same R-matching semantics, change is local
to one function; only needs `xp`/`get_array_module(mean_zscore)` threaded in
instead of assuming numpy.

## 2. `extract_angles`'s gene-gene matmul: GEMM vs symmetric rank-k update (verified, GPU-only win) --> don't implement

`extract_angles` (`_angles.py:97-100`) computes the covariance as a plain
matmul:

```python
x_centered = X - X.mean(axis=0, keepdims=True)
cov = x_centered.T @ x_centered     # _angles.py:98 — symmetric output, computed as a full GEMM
```

Since `cov` is symmetric, this is a textbook BLAS `syrk` (symmetric rank-k
update), which only needs to compute half the output. Both backends expose
it: `scipy.linalg.blas.get_blas_funcs(['syrk'], ...)` and `cupy.cublas.syrk`
(confirmed present in the installed `cupy==14.1.1`; signature
`syrk(trans, a, out=None, alpha=1.0, beta=0.0, lower=False)`).

**Benchmarked on this GPU** (8000x8000, float32):
- `x_centered.T @ x_centered` (gemm): **104ms**
- `cupy.cublas.syrk('T', x_centered, out=out, lower=False)`: **48ms** (2.15x), bit-exact with gemm on the populated triangle.
- BUT `syrk` only fills one triangle, and the rest of the pipeline
  (`get_dstat`'s per-column `nanmean`/`nanvar`, the z-scoring, `StreamingZscoreStats`)
  needs the *full* symmetric matrix. A naive mirror (`out + out.T - diag(diag(out))`)
  costs **54ms** — enough to erase the whole syrk saving (48+54 ≈ 104, a wash).
  A leaner in-place fancy-index mirror (`out[tril_idx] = out.T[tril_idx]`, no
  full transpose/add temporaries) costs only **19ms**, verified bit-exact —
  net **67ms vs 104ms, ~1.55x**.
- **On CPU this does NOT help**: `scipy`'s `ssyrk` wrapper benchmarked *slower*
  than numpy's `@` at the same size (2.98s vs 2.79s for 8000x8000), likely
  because numpy's `@` already dispatches to a multithreaded BLAS gemm while the
  raw `scipy.linalg.blas` wrapper call doesn't get the same threading — so
  don't bother with this on the CPU path.

This is exactly the technique `rapids-singlecell` uses for its sparse PCA
covariance (`preprocessing/_sparse_pca/_helper.py`: computes a Gram matrix
then calls a **custom CUDA kernel** `_spca.copy_upper_to_lower` to mirror it
in one pass) — confirming the pattern is sound, but also why they needed a
real kernel rather than a cupy-level mirror: ours is the achievable
in-scope approximation, not the optimal one.

**Recommendation**: low priority. ~1.55x on a sub-step of `factorise`, which
is already only ~6s for 4 batches on GPU (the CPU side, where `factorise`
costs 229s, gets no benefit from this). Also requires a `cupy`-specific
branch inside `_angles.py` (`if xp is cupy: ... else: X.T @ X`), which cuts
against `_utils.py`'s stated design of never importing cupy outside
`get_array_module`. Worth it only if profiling at much larger gene panels
(100k+ genes) shows this matmul, not transfer/pandas overhead, dominating.

## 3. Redundant/sub-optimal sparse column-indexing in batch prep --> implement

Per batch, the pipeline does two separate column-subset operations on a CSR
matrix:

```python
batch_X[label] = X_b[:, mask]                          # _anglemania.py:155
...
dense = to_dense(X[:, np.asarray(src_cols)], xp)         # _batches.py:124, inside align_to_common_genes
```

Column indexing is the *minor* axis for CSR, which cupyx implements via a
dedicated histogram/argsort kernel (`cupyx/scipy/sparse/_csr.py:
_minor_index_fancy`, confirmed by reading the installed source — not a naive
CSC round-trip, so it's not catastrophically slow, but it's inherently more
work than CSC's native major-axis slice for the same operation, and it's done
twice per batch).

Also, `genes_passing_min_cells`'s sparse branch (`_batches.py:69`) gets
nnz-per-gene via `xp.bincount(X.tocsr().indices, minlength=...)`, an O(nnz)
scan over every stored entry. `rapids-singlecell`'s own
`preprocessing/_hvg/_pearson_residuals.py:67-70` computes the identical
per-gene nnz count via `X_batch.tocsc()` then `cp.diff(X_batch.indptr)` — O(n_genes),
because CSC's `indptr` already encodes per-column nnz directly.

**Fix**: convert each batch to CSC once, right after row-selection
(`X_b = X_full[idx].tocsc()`), and reuse it for: (a) `cp.diff(indptr)` instead
of bincount, (b) both column subsets (`mask`, then `src_cols`), which become
native major-axis slices on CSC instead of two kernel-backed minor-axis ops
on CSR.

**Impact**: gene filtering is already cheap in absolute terms (0.1s GPU /
6.2s CPU per the table above), so this is a minor win in isolation — but it's
free (no algorithm change, two fewer kernel-backed indexing ops per batch)
and low-risk.

## 4. RNG: legacy global API vs `default_rng`, float64 sort keys (minor, free) --> implement

`factorise`/`permute_matrix` use the legacy global RNG:

```python
xp.random.seed(seed)                          # _angles.py:141
order = xp.argsort(xp.random.rand(*X.shape))  # _angles.py:42, rand() is hardcoded float64
```

Checked: `cupy.random.Generator` has **no `.permuted(axis=...)`**
(confirmed via introspection on the installed cupy — `AttributeError`), so the
O(n) per-axis Fisher-Yates-style shuffle numpy's modern `Generator` offers
isn't available on GPU; the existing argsort-based shuffle is already close
to the best vectorized primitive cupy has for "independently permute each row."

Benchmarked (8000x8000): `default_rng().random(shape, dtype=float32)` is
2.5-4x faster than legacy `rand()` for the random-draw step alone (1.6ms vs
6.5ms), but since the subsequent `argsort` dominates total cost, the
end-to-end shuffle is only ~5% faster (283ms vs 299ms).

**Recommendation**: low-effort, no-risk cleanup — switch to
`xp.random.default_rng(seed)` and request `dtype=X.dtype` instead of
hardcoded float64. Worth doing for hygiene (legacy `xp.random.seed()` mutates
*global*, process-wide RNG state, which `default_rng(seed)` avoids) more than
for raw speed.

## 5. `StreamingZscoreStats` accumulators are always float64 — needs validation, not a free win --> don't implement

`_stats.py:35-36`:

```python
self._wz_sum = xp.zeros(shape, dtype=xp.float64)
self._wz2_sum = xp.zeros(shape, dtype=xp.float64)
```

These are allocated float64 unconditionally, even though the z-score
matrices fed into `update()` are float32 (`align_to_common_genes` forces
`xp.float32`). These two buffers are exactly what CLAUDE.md already flags as
the OOM risk at 20k+ genes — 3.2GB each at 20k genes; float32 would halve
that to 1.6GB each, i.e. the single biggest lever for raising the gene-panel
ceiling on a fixed GPU memory budget.

**Why this isn't a trivial fix**: the streaming-variance identity
(`sum(w*z^2) - sum(w*z)^2/sum(w)`) is a textbook case for catastrophic
cancellation, and float64 accumulation is the standard mitigation regardless
of the input dtype — this was likely a deliberate choice, not an oversight.

**Recommendation**: don't change blindly. Benchmark float32 accumulation
against the current float64 path's output on a real multi-batch dataset
(e.g. the existing 20k x 20k benchmark fixture) and check whether
`mean_zscore`/`sds_zscore`/`sn_zscore` stay within an acceptable tolerance
before touching this.

## 6. `rank_gene_pairs`'s pandas `.rank()` cost — the other CLAUDE.md-flagged bottleneck -->

At 124.5s GPU / 129.2s CPU (the table above), this is the single largest line
item in the whole pipeline, and never touches the GPU.

Checked scanpy's and rapids-singlecell's own top-N gene-ranking code for a
faster pattern: both explicitly tried `np.argpartition` and rejected it —
*"interestingly, np.argpartition is slightly slower"*
(`scanpy/preprocessing/_highly_variable_genes.py:515`,
`rapids_singlecell/preprocessing/_hvg/_seurat_cellranger.py:176`, both use a
full sort instead). So pyanglemania's pandas `.rank()` + `.sort_values()`
approach (`_select.py:88-92`) isn't naively wrong — it's the same family of
solution the reference packages converged on — it's just running on a
CPU-only, potentially 10s-of-millions-of-rows DataFrame regardless of backend.

**Two levers, in order of effort**:
- (a) Fixing #1 feeds a smaller table into this step, but at permissive
  `prefilter_threshold` values the post-filter table can still be huge (54.5M
  rows in the benchmark above). Raising `prefilter_threshold` is already the
  documented user-facing lever (CLAUDE.md) — free, no code change, do this
  first when profiling after #1.
- (b) Move the prefilter -> rank -> select chain onto `cudf`, which is
  API-compatible (`.rank(method=...)`, `.sort_values()`) and could be built
  directly from the cupy arrays surviving the on-device filter in #1, with no
  host round-trip at all until `extract_unique_genes`. `cudf` is a natural
  sibling dependency in the ecosystem this package targets — confirmed
  `ref_packages/rapids-singlecell/pyproject.toml:35-36` bundles
  `cudf-cu12>=25.10`/`cudf-cu13>=25.10` alongside `cupy-cuda12x` in its own
  `rapids-cu12`/`rapids-cu13` extras. But it's a new, heavyweight,
  CUDA-version-pinned optional dependency pyanglemania doesn't currently
  have, and notably `rapids-singlecell` itself never uses `cudf` in its own
  preprocessing/HVG code — only in `tools/_clustering.py` for cugraph
  interop — so adopting it here would be a real precedent shift for this
  package, not "just doing what rapids-singlecell does." Only worth it if
  profiling after (a) still shows this step dominating.
- (c) **Implemented**: stop materializing `geneA`/`geneB` as name strings
  before ranking/sorting. R's own `select_genes_cpp` works with integer
  gene indices throughout and only maps to names at the very end
  (`select_genes.R:83-90`) — the Python port did the opposite (strings
  attached in `prefilter_gene_pairs`, before `rank_gene_pairs`'s two
  `.rank()` calls and `.sort_values()` ever run, so they drag two string
  columns through an O(M log M) sort for nothing). `prefilter_gene_pairs`
  now returns `geneA_idx`/`geneB_idx` (ints) instead of `geneA`/`geneB`
  (strings); `extract_unique_genes` takes a `gene_names` array and maps
  indices to names only for the handful of genes actually selected;
  `_anglemania.py` does the one remaining full-table string lookup itself,
  once, on the already-ranked table, before storing
  `uns["anglemania"]["prefiltered_df"]` (same columns/order/content as
  before — verified via the existing `test_anglemania_end_to_end_basic`
  column-order assertion). Benchmarked end-to-end (real
  `prefilter_gene_pairs`→`rank_gene_pairs`→`extract_unique_genes`, 8000
  genes, 12.18M surviving pairs): **21.5s vs 33.3s (~1.5x), identical
  selected-gene output**. This is backend-agnostic (helps numpy and cupy
  callers equally, since it doesn't touch the rank/sort *implementation*,
  just what flows through it) and required no new `xp`-branching.
- (d) **Implemented**: bypass pandas entirely on the GPU path.
  `rank_gene_pairs` now branches on `xp.__name__ == "cupy"`: cupy input
  ranks via `_min_rank` (`cupy.sort` + `cupy.searchsorted`, verified
  bit-exact against `pandas.Series.rank(method="min")`) and transfers to
  host only once, after sorting; numpy input still uses plain pandas
  `.rank()`, which benchmarked *faster* than the same sort/searchsorted
  approach on CPU (31.1s vs 17.7s pandas-on-ints at 10M rows -- pandas'
  Cython rank is already well-tuned there), so this is a real
  `if xp is cupy:` branch, not a single "better" algorithm for both.
  `prefilter_gene_pairs` correspondingly stopped calling `to_numpy()`
  internally -- it now returns a dict of arrays in their native backend,
  letting `rank_gene_pairs` do the one host transfer at the very end
  instead.

  **Correctness pitfall caught by a parity test, not by inspection**: the
  combined `rank` column ties constantly (it's a weighted sum of two
  *integer* ranks at rational weights -- e.g. `rank_mean=1, rank_sd=5`
  and `rank_mean=4, rank_sd=3` both give `3.4` at the default `(0.4,
  0.6)` weights), and pandas' default `sort_values` (quicksort) vs cupy's
  default `argsort` broke those ties differently -- 7.2% of rows
  disagreed in a 60-gene test case (`tests/test_gpu.py::
  test_rank_gene_pairs_cupy_native_path_matches_pandas`, added
  specifically because this is exactly the kind of thing "it ran without
  crashing" doesn't catch). Fixed by sorting with `kind="stable"` on
  *both* branches, which breaks every tie by the pairs' original
  (pre-rank, `triu_indices`-determined) order -- identical input on both
  backends, verified to make the two paths agree exactly afterward. This
  is a deliberate, documented behavior change from the pre-existing
  (default-quicksort) tie-breaking, justified because it makes selection
  reproducible across backends, which is a correctness property worth
  more than matching the old implementation's incidental tie order.

  Benchmarked end-to-end with the real functions at 20k genes (76.1M
  surviving pairs, a larger and thus harder case than the 54.5M-pair
  CLAUDE.md benchmark): **14.2s, vs 184s for the combined
  `prefilter_gene_pairs` + `rank_gene_pairs` cost in that benchmark
  (prefilter 59.5s + rank 124.5s) — ~13x**.

## Checked and ruled out (don't re-investigate)

- **CPU-side `syrk`** (#2): benchmarked slower than `@`, not just "no
  better" — skip it on the CPU path entirely.
- **`cupy.random.Generator.permuted`**: doesn't exist in cupy (confirmed on
  the installed 14.1.1); the argsort-based shuffle in `_angles.py` is already
  close to cupy's best available vectorized option for per-row permutation.
- **`np.argpartition` for top-N selection** (#6): both scanpy and
  rapids-singlecell benchmarked this against a full sort and found it
  slightly slower; not a lever for `_select.py`.
- **cuML**: no primitive found there applicable to pairwise gene-gene
  correlation, batch permutation, or weighted streaming reduction — its use
  in `rapids-singlecell` is for regression/clustering algorithms unrelated to
  this pipeline's shape of work.
