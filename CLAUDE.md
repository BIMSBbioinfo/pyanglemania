# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The core port is implemented and tested on CPU (numpy/scipy). What exists:

- `src/pyanglemania/` — the package (see Architecture below).
- `tests/` — pytest suite covering each module plus the full `pp.anglemania()` pipeline (dense and sparse input, CPU and GPU).
- `notebooks/tutorial.ipynb` — the vignette-equivalent walkthrough (simulate batched data → unintegrated UMAP → `pp.anglemania` → compare against `highly_variable_genes` → Harmony integration of both gene sets). Re-execute after changing the public API: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=pyanglemania --output tutorial.ipynb notebooks/tutorial.ipynb` (kernel registered via `python -m ipykernel install --user --name pyanglemania`).
- `envs/pyanglemania.yml` — the conda env (`mamba env update -f envs/pyanglemania.yml` to sync after editing).
- `plans/implementation_plan.md` — the original one-paragraph brief this package derives from.
- `ref_packages/` — full git checkouts kept **only as reference material**, untracked by this repo (not gitignored, just not yet added): `anglemania` (the R package being ported), `scanpy`, `scvi-tools`, `rapids-singlecell` (the Python target ecosystem), `anndata`. Treat these as read-only library sources to read and crib patterns from, not code to modify.

### GPU status

GPU support (array-API dispatch — the same code runs on numpy or cupy depending on what `adata.X`
already holds) has been execution-validated end to end on this sandbox's Tesla P40s (dense *and*
cupyx-sparse `adata.X`, every `method`/`permutation_function`/`normalization_method` combination,
and numerical parity vs. the numpy path for every deterministic step — see `tests/test_gpu.py`,
skipped automatically when no CUDA device is reachable). Two environment-specific things had to be
fixed/learned along the way, not package bugs but worth knowing:

1. **cupy's JIT compiler needs CUDA headers new enough for its bundled CCCL.** This sandbox's
   system CUDA (`/usr/local/cuda-12.3`) doesn't define `__nv_fp8_e8m0`, which cupy 14's bundled
   `cupy/_core/include/cupy/_cccl` headers reference — any kernel that pulls in that template
   (NVRTC-compiled elementwise/reduction ops) fails to compile until `CUDA_PATH`/`CUDA_HOME` point
   at headers that do. Fixed here by adding `cuda-cudart-dev=12.9` (conda-forge) to
   `envs/pyanglemania.yml` and exporting `CUDA_PATH=CUDA_HOME=$CONDA_PREFIX` before running anything
   that imports cupy.
2. **`cupyx.scipy.sparse` has no `.getnnz(axis=...)`** (raises `ValueError`, unlike scipy's, which
   supports it), and a sparse `X != 0` compiles a much heavier kernel that's more likely to hit (1).
   `_batches.py::genes_passing_min_cells` works around both by counting stored entries via
   `xp.bincount(X.tocsr().indices, ...)` instead, which is exactly what `getnnz(axis=0)` computes
   and works identically on numpy/scipy and cupy/cupyx.

If GPU tests start failing on a fresh box, check both of those before assuming a real regression.

### Scaling to large gene panels (tens of thousands of genes)

Benchmarked end-to-end at 20k cells x 20k genes (4 batches) on GPU. Two real findings, both fixed:

1. **`StreamingZscoreStats.finalize()` used to OOM** at this scale: the naive chained expression for
   the weighted variance (`(wz2_sum - wz_sum*wz_sum/w_sum) / denom`, then `sqrt(clip(...))`, then a
   doubly-nested `xp.where(...)` for the SNR) allocates 8-10 full `(genes x genes)` float64 buffers
   at once (3.2 GB each at 20k genes) on top of the two accumulators already held. Fixed by rewriting
   it with in-place ops (`-=`, `/=`, `out=`) and dropping the accumulators once they're no longer
   needed, which gets this down to ~1-2 extra buffers at any moment — see the comments in `_stats.py`.
2. **`_select.py::extract_unique_genes` used to dominate total runtime** (8+ of ~14 minutes in the
   20k x 20k benchmark): it deduplicated the *entire* prefiltered gene-pair table before truncating
   to `max_n_genes`, via `numpy.unique` on a 100M+-element string array once a permissive
   `prefilter_threshold` let tens of millions of pairs through (54.5M pairs in that benchmark, out of
   ~200M possible at 20k genes — unsurprising, since 0.5 isn't a strict z-score cutoff). R's own
   `extract_rows_for_unique_genes` has the identical cost, so this isn't a Python regression, but
   it's avoidable: since the table is already rank-sorted, the first `max_n_genes` unique genes are
   fully determined by some prefix of it. Fixed by growing that prefix exponentially until it has
   enough, which is a behavior-preserving optimization (see the docstring/tests for why the result is
   provably identical). This dropped the same 20k x 20k benchmark from 839s to 191s total.

After both fixes, CPU vs GPU at 20k cells x 20k genes x 4 batches (same data, `max_n_genes=2000`,
default `prefilter_threshold=0.5`; numbers from one run each, not averaged):

| stage | GPU | CPU |
| --- | --- | --- |
| gene filtering (`genes_passing_min_cells`, all batches) | 0.1s | 6.2s |
| per-batch `factorise` (all 4 batches, align+compute+update) | ~6s | ~229s |
| `finalize` | 0.5s | 15.8s |
| `prefilter_gene_pairs` (~54.5M of ~200M pairs passed) | 59.5s | 49.2s |
| `rank_gene_pairs` | 124.5s | 129.2s |
| `extract_unique_genes` | 0.01s | 0.01s |
| **total** | **191s** | **430s** (2.25x slower) |

The GPU's advantage is concentrated entirely in the numeric, matmul/elementwise-heavy steps
(`factorise`, `finalize`: 30-38x faster on this sandbox's Tesla P40s) — `prefilter_gene_pairs` and
`rank_gene_pairs` run on CPU/numpy/pandas regardless of which backend `adata.X` uses (`_select.py`
calls `to_numpy()` immediately), so they cost the same either way and are why the *overall* speedup
is a more modest ~2.25x rather than 30x+. Porting the prefilter/rank step to stay on-GPU (e.g. via
cudf, or filtering before transferring off-device) would be the next lever for end-to-end speed at
large gene-panel scale, but hasn't been done. Raising `prefilter_threshold` is the user-facing lever
to cut the prefilter/rank cost down on very large gene panels in the meantime.

## Development commands

```bash
# after editing envs/pyanglemania.yml
mamba env update -f envs/pyanglemania.yml

# install the package (editable) into the active env
pip install -e . --no-build-isolation

# tests (CPU-only tests/test_gpu.py skips itself if no CUDA device is reachable)
pytest tests/ -q
pytest tests/test_stats.py -q -k streaming   # single file / -k filter

# tests including GPU, on a box with CUDA headers new enough for cupy's bundled CCCL
# (see "GPU status" above if this errors instead of skipping)
CUDA_PATH=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX pytest tests/test_gpu.py -q

# lint
ruff check src/ tests/
```

## The task

Port the R/Bioconductor package **anglemania** (`ref_packages/anglemania`) to Python with GPU support, structured to integrate into the **scanpy** / **rapids-singlecell** architecture (`AnnData` in/out, scanpy-style `pp.anglemania(adata, batch_key=...)`) rather than being a from-scratch reimplementation of the R package's internal structure.

Two explicit deviations from the R implementation, per the brief:
1. Don't recreate the R package's internals 1:1 — transplant the *algorithm*, fitted to scanpy/rapids-singlecell idioms (AnnData in/out, GPU arrays via cupy where rapids-singlecell would use them).
2. Change the computation strategy: the R version computes a per-batch correlation/z-score matrix and **persists each one to disk** (via `bigstatsr::FBM`, file-backed matrices) before reducing them to mean/SD/SNR across batches at the end. The Python version computes the running mean/SD **on the fly** instead — see `StreamingZscoreStats` in `src/pyanglemania/preprocessing/_stats.py`, which never holds more than one batch's matrix plus the accumulators at once.

## The anglemania algorithm (ground truth: `ref_packages/anglemania/R/`)

Read `anglemania.R`, `compute_angles.R`, `stats.R`, `select_genes.R`, `prepare_anglemania.R` in that package before changing the corresponding Python step — the R source is the spec. Pipeline (entry point `anglemania()` in `R/anglemania.R`, ported to `src/pyanglemania/preprocessing/_anglemania.py::anglemania`):

1. **Batch setup** (`prepare_anglemania.R` → `_batches.py`): combined `anglemania_batch` key from `batch_key`/`dataset_key` (`add_unique_batch_key`), split by batch (`split_obs_indices_by_batch`), per-dataset `weight`s so each *dataset* contributes equally regardless of how many batches it's split into (`compute_dataset_weights`).
2. **Gene filtering** (`prepare_anglemania.R` → `_batches.py`): drop genes below `min_cells_per_gene` per batch (`genes_passing_min_cells`), reduce to the gene intersection across batches, or to genes present in at least `min_samples_per_gene` batches if `allow_missing_features=True` (`intersect_genes`), then densify/reorder/zero-pad each batch to that common gene set (`align_to_common_genes`).
3. **Per-batch angle computation** (`compute_angles.R::factorise` → `_angles.py::factorise`), for each batch's `(cells x genes)` matrix:
   - Permute to build a null distribution (`permute_matrix`: `"sample"` shuffles every value, `"permute_nonzero"` shuffles only nonzero entries, leaving zeros in place; `permute_row_or_column` keeps R's parameter values but they map to the *opposite* numpy axis here since this package stores cells x genes where R stores genes x cells — see the docstring in `factorise`).
   - Normalize both the real and permuted matrices (`normalize_matrix`: default `"divide_by_total_counts"` = CP10K + log1p; alternate `"find_residuals"` regresses out log total counts per gene. Note R's docs also mention a third choice, `"scale_by_total_counts"`, but R's own `normalize_matrix` never implements it — only these two real choices are ported).
   - Gene-gene relationship matrix for both (`extract_angles`: Pearson correlation across cells, i.e. the "angle" between mean-centered gene vectors; `"spearman"` ranks first — ties broken by original order rather than R's tie-averaging, to keep this vectorized on both numpy and cupy). Diagonal is NaN.
   - Per-gene (per-column) `mean`/`sd` of the **permuted** matrix (`get_dstat`), then z-score the **real** matrix against that null. This makes the z-score matrix asymmetric (entry `(i, j)` is standardized against gene `j`'s own null, not gene `i`'s) — intentional, matches R, and only the upper triangle is read downstream anyway.
4. **Cross-batch reduction** (`stats.R::get_list_stats` → `_stats.py::StreamingZscoreStats`): weighted `mean_zscore`, weighted `sds_zscore`, and `sn_zscore = |mean| / sd` per gene pair, accumulated batch-by-batch instead of from a list of all batches' matrices (the core streaming deviation; see the module docstring for the single-pass identity this relies on).
5. **Prefilter + select** (`select_genes.R` → `_select.py`): keep gene pairs whose `|mean_zscore|` and `sn_zscore` both clear a threshold, auto-relaxed by -0.1 if nothing passes (`prefilter_gene_pairs`); rank by a weighted combination of rank(|mean z-score|) and rank(sd z-score) (`score_weights`, default `(0.4, 0.6)` favors sd; `rank_gene_pairs`); take unique genes from the top-ranked pairs up to `max_n_genes` (`extract_unique_genes`). Results land in `adata.var["anglemania_genes"]` / `adata.uns["anglemania"]`.

All public parameters from R's `anglemania()`/`check_params` are preserved with the same names/values: `batch_key`, `dataset_key`, `max_n_genes`, `min_cells_per_gene`, `min_samples_per_gene`, `allow_missing_features`, `method`, `permute_row_or_column`, `permutation_function`, `prefilter_threshold`, `do_normalize`, `normalization_method`, `score_weights`, `direction`. (`layer` is new, since AnnData has no exact equivalent of R's fixed `counts()` accessor.)

## Architecture

```
src/pyanglemania/
  _utils.py                 # numpy/cupy + scipy/cupyx.sparse dispatch (get_array_module, to_dense, to_numpy)
  datasets.py                # example_adata() synthetic dataset (port of R's sce_example())
  preprocessing/
    __init__.py               # exposes anglemania()
    _anglemania.py             # orchestrator + parameter validation (the public pp.anglemania entry point)
    _batches.py                  # batch/dataset weighting, gene filtering/intersection, zero-padding
    _angles.py                    # normalize_matrix, permute_matrix, extract_angles, factorise
    _stats.py                      # StreamingZscoreStats (the streaming cross-batch reduction)
    _select.py                      # prefilter_gene_pairs, rank_gene_pairs, extract_unique_genes
```

Nothing in this package moves data to the GPU itself — like `rapids-singlecell`, it dispatches to numpy or cupy based on whatever array `adata.X` (or `layer`) already holds (e.g. after `rapids_singlecell.get.anndata_to_GPU(adata)`). `get_array_module` in `_utils.py` is the single place that decides this; every other function takes an explicit `xp` parameter rather than importing numpy/cupy itself.

`scanpy` (`ref_packages/scanpy`) is the reference for AnnData API conventions (`adata.uns`/`adata.var` write-back, `key_added`-style patterns) used in `_anglemania.py`. `rapids-singlecell` (`ref_packages/rapids-singlecell`) is the reference for the GPU dispatch pattern and for `preprocessing/_hvg/`, the existing scanpy/rapids-singlecell feature-selection function `anglemania` is meant to compete with/replace. `scvi-tools` (`ref_packages/scvi-tools`) is reference for the downstream integration models (e.g. `SCVI`) that anglemania-selected genes feed into, not for the algorithm itself. `anndata` (`ref_packages/anndata`) is the reference for `AnnData` semantics (views vs. copies, backed/sparse storage) underlying all of the above.
