# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**The selection is gene-level, not pair-level (since 0.1.1).** There is no gene-*pair* ranking,
prefiltering or weighted pair scoring anywhere in the package any more: R's
`prefilter_gene_pairs`/`rank_gene_pairs`/`extract_unique_genes` (the old `_select.py`) and the
`prefilter_threshold`/`score_weights`/`direction` parameters were removed. Every gene gets its own
row-sum score over all its partners, corrected for expression, and genes are ranked on that directly
(steps 4–5 below). Anything in `plans/` that talks about pair prefilter/rank/extract stages, SNR
(`sn_zscore`) ranking or `score_weights` describes the pre-0.1.1 pipeline and is kept only as history.
The design and the experiments behind the change live outside this repo, in
`~/projects/CRC1588/dataset_integration/anglemania_analysis/tasks/nmf/docs/tasks/`
(`pergene_score_into_pyanglemania.md`, `pergene_signal_noise_R_axes.md`,
`permutation_null_necessity.md`); the package reproduces that task's `gene_scores.py` +
`add_binned_zscores` (`R_z_binned`, and `score_binned`, called `WSN` here) to float32 precision.

### Pipeline at a glance

```
AnnData (raw counts in X or `layer`)
  │
  ├─ 1. batch setup        anglemania_batch key, per-dataset weights w_b          (_batches.py)
  ├─ 2. gene filtering     min_cells_per_gene per batch → intersection, or ≥ min_samples_per_gene
  │                        batches if allow_missing_features (missing_mode="mask" | "zero")
  │
  ├─ 3. per batch b        normalize (CP10K+log1p | residuals | pflog1ppf)          (_angles.py)
  │                        angles: gene×gene correlation (cosine | spearman | phi_s)
  │                        null:   same on a permuted matrix → per-gene mean/sd
  │                        z_b = (angles − null mean_j) / null sd_j,  symmetrised (z + zᵀ)/2
  │                        └─ folded into running sums, then dropped                (_stats.py)
  │
  ├─ 4. cross-batch        M_ij = weighted mean of z_b,  S_ij = weighted sd        (StreamingZscoreStats)
  │     → per gene         signal_i = Σⱼ M_ij²,  noise_i = Σⱼ S_ij²
  │                        R_i = signal_i / (signal_i + noise_i)
  │                        (+ debiased signal_db_i = Σⱼ (M_ij² − κ_ij S_ij²), ICC_db_i)
  │
  └─ 5. selection          z-score log10 signal, R, ICC_db within n_bins expression bins
                           WSN = w₀·signal_z_binned + w₁·R_z_binned                (_gene_level.py)
                           top max_n_genes by score: "R" (R_z_binned, default) | "WSN" | "ICC"
                           → adata.var["anglemania_*"], adata.uns["anglemania"]
```

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

At 20k cells x 20k genes x 4 batches the GPU path was benchmarked before 0.1.1 (191s GPU vs 430s
CPU end to end), but most of that time was the pair-level prefilter/rank stages that no longer
exist, so those numbers are obsolete; the per-batch part still holds (per-batch `factorise`, all 4
batches: ~6s GPU vs ~229s CPU on this sandbox's Tesla P40s). The lesson that carried over: the
cross-batch reduction must never build several full `(genes x genes)` float64 temporaries at once
(3.2 GB each at 20k genes). The old `finalize()` did and OOM'd. `StreamingZscoreStats.gene_scores()`
now works in row chunks (`chunk_size`) and never materialises `M`/`S` as full matrices. No current
end-to-end benchmark of the gene-level pipeline exists yet.

The per-batch step is bounded by whole-batch materialization (`align_to_common_genes` +
`factorise` each hold the full `(cells x genes)` matrix, several copies at once, for one batch at a
time). That becomes its own OOM risk independent of total gene-panel size when a *single batch* is
very large (tens of thousands of cells) — see `plans/gpu_memory_large_batches.md` (triggered by a
54-batch dataset with per-batch cell counts up to 22k). Two fixes from that investigation are
implemented: `StreamingZscoreStats`'s cross-batch accumulators now live in host memory rather than
GPU memory unconditionally (`_stats.py`), and `pp.anglemania(..., cell_chunk_size=...)` processes an
oversized batch in row-chunks instead of all at once (`_angles.py::factorise_chunked`), for the
`method in ("cosine", "phi_s")` / `normalization_method in ("divide_by_total_counts", "pflog1ppf")`
/ `permute_row_or_column="column"` combination (raises `ValueError` for anything else rather than
silently ignoring the chunk size). Both default off/unused unless a batch is actually too large to
fit, so the default path is unaffected.

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
   - Normalize both the real and permuted matrices (`normalize_matrix`: default `"divide_by_total_counts"` = CP10K + log1p; alternate `"find_residuals"` regresses out log total counts per gene. Note R's docs also mention a third choice, `"scale_by_total_counts"`, but R's own `normalize_matrix` never implements it — only these two are ported from R as-is. A third, non-R choice, `"pflog1ppf"`, was added later — see "Extensions beyond the R port" below).
   - Gene-gene relationship matrix for both (`extract_angles`: Pearson correlation across cells, i.e. the "angle" between mean-centered gene vectors; `"spearman"` ranks first — ties broken by original order rather than R's tie-averaging, to keep this vectorized on both numpy and cupy). Diagonal is NaN. A third, non-R `method`, `"phi_s"`, was added later — see below.
   - Per-gene (per-column) `mean`/`sd` of the **permuted** matrix (`get_dstat`), then z-score the **real** matrix against that null. This makes the z-score matrix asymmetric (entry `(i, j)` is standardized against gene `j`'s own null, not gene `i`'s) — intentional, matches R, and only the upper triangle is read downstream anyway.
4. **Cross-batch reduction** (`stats.R::get_list_stats` → `_stats.py::StreamingZscoreStats`). The
   weighted cross-batch mean `M` and sd `S` per gene pair are accumulated batch-by-batch instead of
   from a list of all batches' matrices (the core streaming deviation; see the module docstring for
   the single-pass identity this relies on). Unlike R, these pair-level `M`/`S` are only an
   intermediate and are never ranked or returned (R's `sn_zscore = |mean|/sd` pair SNR is gone). Each
   per-batch z matrix is symmetrised (`(z + zᵀ)/2`) on the way in, then reduced to one row per gene (`gene_scores()`, row-chunked, never materialising `M`/`S`): `signal = Σⱼ M_ij²`,
   `noise = Σⱼ S_ij²`, `R = signal / (signal + noise)`. With `allow_missing_features=True` a pair
   only uses the batches where both genes are present (per-pair weights `W1 = Pᵀ diag(w) P` from the
   per-batch presence vectors, not accumulated); row sums are rescaled by `(p−1)/n_valid`. With
   `missing_mode="mask"` (default) each batch's z (angles *and* permutation null) is computed on its
   present genes only and handed to `update(z, w, index)` as a `(q, q)` block; `missing_mode="zero"`
   is R's zero-padding + full-weight averaging (`plans/missing_features_nan_masking.md`). Masked,
   κ = ΣW²/(ΣW)² differs per pair and `R`'s null floor is ≈κ, so `R` is inflated for genes in fewer
   batches; `signal_db = Σⱼ (M² − κ S²)` / `ICC_db` are the debiased versions (`ICC_db_z_binned`,
   selectable via `score="ICC"`). `dataset_presence=True` keeps only genes present in ≥ 1 batch of every
   `dataset_key` group.
5. **Expression-binned selection** (no R counterpart → `_gene_level.py`): `log10 signal` and `R` are
   z-scored within `n_bins` equal-frequency bins of mean CP10K+log1p expression (accumulated per
   batch during the loop, `_batches.py::lognorm_column_sums`), `WSN = signal_R_weights[0] ·
   signal_z_binned + signal_R_weights[1] · R_z_binned`, and the top `max_n_genes` by `score`
   (`score="R"` → `R_z_binned`, the default; `"WSN"`; `"ICC"` → `ICC_db_z_binned`) are selected. Per-gene columns land in
   `adata.var["anglemania_*"]`; `adata.uns["anglemania"]` holds `params`, `intersect_genes`,
   `anglemania_genes`.

R's per-batch parameters from `anglemania()`/`check_params` are preserved with the same names/values: `batch_key`, `dataset_key`, `max_n_genes`, `min_cells_per_gene`, `min_samples_per_gene`, `allow_missing_features`, `method`, `permute_row_or_column`, `permutation_function`, `do_normalize`, `normalization_method`. R's pair-level `prefilter_threshold`/`score_weights`/`direction` are gone (the gene-level sums are sums of squares, hence sign-blind; `direction` has no analogue). New: `layer` (AnnData has no exact equivalent of R's fixed `counts()` accessor), `score`, `n_bins`, `signal_R_weights` (deliberately not reusing the name `score_weights`), `missing_mode`, `dataset_presence`, `cell_chunk_size`.

## Extensions beyond the R port

Two `normalize_matrix`/`extract_angles` choices have no R counterpart, added to evaluate
proportionality (CoDA) as an alternative to correlation for the angle/z-score pipeline. Source
papers are kept in `papers/` (not the algorithm's R spec — these are reference material for these
two additions only):

- `normalize_matrix(..., "pflog1ppf")`: the shifted-centered-log-ratio transform ("PFlog1pPF (CLR)")
  from `papers/2022.05.06.490859v3.full.pdf` (Booeshaghi, Hallgrímsdóttir, Gálvez-Merchán & Pachter,
  "Depth normalization for single-cell genomics count data"). Per cell: divide by the cell's total
  count (`u = x / sum(x)`, a proportional-fitting/PF step), `log1p`, then subtract the cell's own
  mean log-proportion (a second PF step, done as centering since it follows a log) — equivalent to
  `sc.pp.normalize_total(adata, target_sum=1); sc.pp.log1p(adata); adata.X -= adata.X.mean(axis=1)`
  (confirmed against `papers/PFlogPF_convo_zulip.md`, a Zulip thread where the paper's authors'
  collaborators worked out this exact scanpy-equivalent formula). Implemented with plain `xp` ops
  rather than literal scanpy calls, to keep numpy/cupy dispatch (`normalize_matrix` never imports
  scanpy/AnnData; it's a pure array function called from inside `factorise`).
- `extract_angles(..., "phi_s")`: the symmetric proportionality metric φs from
  `papers/s41598-017-16520-0-2.pdf` (Quinn, Richardson, Lovell & Crowley, "propr: An R-package for
  Identifying Proportionally Abundant Features Using Compositional Data Analysis") —
  `VLR(i,j) / VLP(i,j)`, the variance of the log-ratio `A_i - A_j` over the variance of the
  log-product `A_i + A_j`, for whatever log-ratio matrix `A` is passed in. Low φs means proportional
  (the *opposite* sense from a correlation, where high means related) and it's unbounded above
  rather than capped at 1.
  **Deliberately does not use propr's own CLR** to build `A` (propr replaces zeros in raw counts
  with 1, then centers per-sample log-counts — depth-dependent in exactly the way the PFlogPF paper
  argues against). Use `normalization_method="pflog1ppf"` to build `A` instead: pass
  `method="phi_s", normalization_method="pflog1ppf"` together to `factorise`/`anglemania()` so the
  matrix φs operates on is the shifted-CLR transform, not raw counts.

Both compose with the rest of the pipeline (permutation, per-batch z-scoring, cross-batch streaming
reduction, gene-level scoring/selection) unmodified, since they preserve the existing function contracts:
`normalize_matrix` still returns a `(cells x genes)` array, `extract_angles` still returns a
symmetric `(genes x genes)` array with NaN diagonal. No changes were needed downstream
of `factorise`. Tested for CPU/GPU parity the same way as the R-ported choices (`tests/test_angles.py`,
`tests/test_gpu.py`).

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
    _stats.py                      # StreamingZscoreStats (streaming cross-batch reduction -> per-gene signal/noise/R)
    _gene_level.py                  # binned_zscores (expression-binned R_z_binned/WSN/ICC_db_z_binned), top_genes
```

Nothing in this package moves data to the GPU itself — like `rapids-singlecell`, it dispatches to numpy or cupy based on whatever array `adata.X` (or `layer`) already holds (e.g. after `rapids_singlecell.get.anndata_to_GPU(adata)`). `get_array_module` in `_utils.py` is the single place that decides this; every other function takes an explicit `xp` parameter rather than importing numpy/cupy itself.

`scanpy` (`ref_packages/scanpy`) is the reference for AnnData API conventions (`adata.uns`/`adata.var` write-back, `key_added`-style patterns) used in `_anglemania.py`. `rapids-singlecell` (`ref_packages/rapids-singlecell`) is the reference for the GPU dispatch pattern and for `preprocessing/_hvg/`, the existing scanpy/rapids-singlecell feature-selection function `anglemania` is meant to compete with/replace. `scvi-tools` (`ref_packages/scvi-tools`) is reference for the downstream integration models (e.g. `SCVI`) that anglemania-selected genes feed into, not for the algorithm itself. `anndata` (`ref_packages/anndata`) is the reference for `AnnData` semantics (views vs. copies, backed/sparse storage) underlying all of the above.
