"""Top-level ``anglemania`` entry point.

Ported from anglemania's R ``anglemania()`` (``R/anglemania.R``), restructured
around ``AnnData`` and streaming cross-batch statistics instead of
file-backed per-batch matrices -- see ``_stats.py`` for that part. The
public surface (parameter names/values, the two-pass prefilter-then-rank
selection) is kept faithful to the R function so results are comparable.
"""

from __future__ import annotations

import numpy as np

from .._utils import get_array_module, vmessage
from ._angles import factorise, factorise_chunked
from ._batches import (
    add_unique_batch_key,
    align_to_common_genes,
    compute_dataset_weights,
    genes_passing_min_cells,
    intersect_genes,
    split_obs_indices_by_batch,
)
from ._select import extract_unique_genes, prefilter_gene_pairs, rank_gene_pairs
from ._stats import StreamingZscoreStats


def _check_params(
    adata,
    batch_key,
    dataset_key,
    max_n_genes,
    method,
    min_cells_per_gene,
    min_samples_per_gene,
    permute_row_or_column,
    permutation_function,
    prefilter_threshold,
    normalization_method,
    score_weights,
    direction,
    cell_chunk_size,
):
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key {batch_key!r} must be a column in adata.obs")
    if dataset_key is not None and dataset_key not in adata.obs.columns:
        raise ValueError(f"dataset_key {dataset_key!r} must be a column in adata.obs")
    if max_n_genes is not None and (not isinstance(max_n_genes, int) or max_n_genes < 1):
        raise ValueError("max_n_genes must be a positive integer or None")
    if method not in ("cosine", "spearman", "phi_s"):
        raise ValueError(f"method must be 'cosine', 'spearman' or 'phi_s', got {method!r}")
    if min_cells_per_gene < 1:
        raise ValueError("min_cells_per_gene must be >= 1")
    if min_samples_per_gene < 1:
        raise ValueError("min_samples_per_gene must be >= 1")
    if permute_row_or_column not in ("row", "column"):
        raise ValueError(
            f"permute_row_or_column must be 'row' or 'column', got {permute_row_or_column!r}"
        )
    if permutation_function not in ("sample", "permute_nonzero"):
        raise ValueError(
            "permutation_function must be 'sample' or 'permute_nonzero', "
            f"got {permutation_function!r}"
        )
    if prefilter_threshold <= 0:
        raise ValueError("prefilter_threshold must be positive")
    if normalization_method not in ("divide_by_total_counts", "find_residuals", "pflog1ppf"):
        raise ValueError(
            "normalization_method must be 'divide_by_total_counts', "
            f"'find_residuals' or 'pflog1ppf', got {normalization_method!r}"
        )
    if score_weights is not None and (
        len(score_weights) != 2 or not all(0 <= w <= 1 for w in score_weights)
    ):
        raise ValueError("score_weights must be a length-2 sequence of values in [0, 1]")
    if direction not in ("both", "anticor", "cor"):
        raise ValueError(f"direction must be 'both', 'anticor' or 'cor', got {direction!r}")
    if cell_chunk_size is not None and (
        not isinstance(cell_chunk_size, int) or cell_chunk_size < 1
    ):
        raise ValueError("cell_chunk_size must be a positive integer or None")


def anglemania(
    adata,
    batch_key: str,
    dataset_key: str | None = None,
    *,
    layer: str | None = None,
    max_n_genes: int | None = 2000,
    min_cells_per_gene: int = 1,
    min_samples_per_gene: int = 2,
    allow_missing_features: bool = False,
    method: str = "cosine",
    permute_row_or_column: str = "column",
    permutation_function: str = "sample",
    prefilter_threshold: float = 0.5,
    do_normalize: bool = True,
    normalization_method: str = "divide_by_total_counts",
    score_weights: tuple[float, float] = (0.4, 0.6),
    direction: str = "both",
    cell_chunk_size: int | None = None,
    verbose: bool = True,
):
    """Select genes with batch-invariant, biologically informative gene-gene angles.

    For each batch (``batch_key``, optionally nested under ``dataset_key``),
    computes the gene-gene angle (correlation) matrix on ``adata.X`` (or
    ``layer``, expected to hold raw counts) and z-scores it against a
    permuted null built from that same batch. Those per-batch z-score
    matrices are then reduced, batch by batch, into a weighted mean/sd/SNR
    across batches (kept as a running accumulator rather than ever holding
    every batch's matrix at once -- see :class:`._stats.StreamingZscoreStats`),
    and genes are selected from the gene pairs with the most consistently
    extreme angles across batches.

    Parameters mirror anglemania's R function of the same name; see
    ``ref_packages/anglemania/R/anglemania.R`` for the original. Two
    ``method``/``normalization_method`` choices are not from R:
    ``method="phi_s"`` (a proportionality metric in place of correlation)
    and ``normalization_method="pflog1ppf"`` (a shifted-CLR transform,
    intended to be used together) -- see ``_angles.py``'s
    ``extract_angles``/``normalize_matrix`` docstrings.

    ``cell_chunk_size`` (not from R): if given, each batch's angle
    computation is done in row-chunks of at most this many cells instead of
    materializing the whole ``(cells x genes)`` batch at once -- bounds peak
    memory to roughly ``O(cell_chunk_size x genes + genes^2)`` regardless of
    that batch's actual cell count, for batches too large to fit otherwise
    (see ``plans/gpu_memory_large_batches.md`` and
    ``_angles.py::factorise_chunked``). Only supported for the default
    ``permute_row_or_column="column"`` together with
    ``method in ("cosine", "phi_s")`` and ``normalization_method in
    ("divide_by_total_counts", "pflog1ppf")`` -- other combinations raise
    ``ValueError`` rather than silently ignoring the chunk size. ``None``
    (default) keeps the original whole-batch behavior, unchanged.

    Modifies ``adata`` in place:

    - ``adata.var["anglemania_genes"]``: boolean mask of selected genes.
    - ``adata.uns["anglemania"]``: dict with ``params``, ``intersect_genes``,
      ``prefiltered_df`` (ranked gene-pair statistics), and
      ``anglemania_genes``.
    - **GPU only, sparse input only**: once every batch's cells have been
      partitioned out of ``adata.X``/``adata.layers[layer]`` (the input is
      no longer read after that point), that layer is replaced with an
      empty same-shape placeholder to free its GPU memory -- redundant
      otherwise, since the per-batch partitions already hold every cell.
      Doesn't happen for CPU (numpy) input, where host RAM is rarely the
      binding constraint, or for dense GPU input (a same-shape placeholder
      wouldn't save anything). If you need the raw layer to survive the
      call (e.g. for downstream steps in the same script), pass a copy or
      re-read it afterward.

    Returns ``adata``.
    """
    _check_params(
        adata,
        batch_key,
        dataset_key,
        max_n_genes,
        method,
        min_cells_per_gene,
        min_samples_per_gene,
        permute_row_or_column,
        permutation_function,
        prefilter_threshold,
        normalization_method,
        score_weights,
        direction,
        cell_chunk_size,
    )

    vmessage(verbose, "Preparing input...")
    add_unique_batch_key(adata, batch_key, dataset_key)
    weights = compute_dataset_weights(adata.obs, batch_key, dataset_key)
    batch_indices = split_obs_indices_by_batch(adata)

    X_full = adata.X if layer is None else adata.layers[layer]
    xp, sp_mod = get_array_module(X_full)
    var_names = np.asarray(adata.var_names)

    vmessage(verbose, f"Filtering each batch to at least {min_cells_per_gene} cells per gene...")
    batch_X: dict[str, object] = {}
    batch_genes: dict[str, list[str]] = {}
    for label, idx in batch_indices.items():
        X_b = X_full[idx]
        if sp_mod.issparse(X_b):
            # CSC once, up front: both the nnz-per-gene count below and the
            # column subsetting it (and align_to_common_genes) do are then
            # native major-axis ops instead of CSR's costlier minor-axis
            # ones (see plans/optimization.md #3).
            X_b = X_b.tocsc()
        mask = genes_passing_min_cells(X_b, min_cells_per_gene, xp, sp_mod)
        batch_genes[label] = list(var_names[mask])
        batch_X[label] = X_b[:, mask]

    # Every batch's data needed downstream now lives in batch_X (already
    # reduced to that batch's own passing genes) -- X_full itself is never
    # read again after this point. On GPU this matters: a caller following
    # this package's own convention (move the *whole* layer to GPU before
    # calling anglemania() so xp resolves to cupy) leaves X_full's full
    # (cells x genes) sparse matrix resident on GPU for the rest of the
    # call otherwise, on top of batch_X's per-batch copies -- redundant,
    # since batch_X already holds every cell, just batch-partitioned. This
    # was the actual cause of a "bottleneck 1"-shaped OOM
    # (plans/gpu_memory_large_batches.md) that showed up even after fixes 1
    # and 2 there: not the per-batch angle-matrix buffers (already bounded),
    # but this structural double-residency, at 18,417 genes with a batch
    # skew large enough to need cell_chunk_size in the first place. Freeing
    # it here (GPU only -- host RAM is rarely the constraint CPU callers
    # hit) drops the caller's own reference too, since ``adata.X``/
    # ``adata.layers[layer]`` was the only other place holding it alive.
    if xp.__name__ == "cupy" and sp_mod.issparse(X_full):
        # AnnData's X/layers setters validate the replacement's shape
        # against adata.shape (rejecting None outright), so free by
        # replacing with an empty same-shape sparse placeholder rather than
        # nulling the slot -- an empty CSR matrix costs next to nothing,
        # unlike a same-shape dense zeros array (which is why this only
        # applies to sparse input; a GPU-resident dense adata.X can't be
        # shrunk this way and is left alone).
        full_shape, full_dtype = X_full.shape, X_full.dtype
        del X_full
        placeholder = sp_mod.csr_matrix(full_shape, dtype=full_dtype)
        if layer is None:
            adata.X = placeholder
        else:
            adata.layers[layer] = placeholder

    common_genes = intersect_genes(
        list(batch_genes.values()), allow_missing_features, min_samples_per_gene, verbose
    )
    if max_n_genes is not None and max_n_genes > len(common_genes):
        vmessage(
            verbose,
            f"{max_n_genes} is larger than the number of intersected genes. "
            f"Setting max_n_genes to {len(common_genes)}",
        )
        max_n_genes = len(common_genes)

    vmessage(verbose, "Computing angles and transforming to z-scores...")
    stats = StreamingZscoreStats(len(common_genes))
    for label, idx in batch_indices.items():
        if cell_chunk_size is not None:
            zscores = factorise_chunked(
                batch_X[label],
                batch_genes[label],
                common_genes,
                xp,
                cell_chunk_size,
                method=method,
                permute_row_or_column=permute_row_or_column,
                permutation_function=permutation_function,
                normalization_method=normalization_method,
                do_normalize=do_normalize,
            )
        else:
            X_dense = align_to_common_genes(batch_X[label], batch_genes[label], common_genes, xp)
            zscores = factorise(
                X_dense,
                xp,
                method=method,
                permute_row_or_column=permute_row_or_column,
                permutation_function=permutation_function,
                normalization_method=normalization_method,
                do_normalize=do_normalize,
            )
            del X_dense
        stats.update(zscores, float(weights[label]))
        del zscores

    # batch_X (every batch's own reduced partition) is never read again past
    # this point either -- same redundancy as X_full above, just discovered
    # later: on GPU it was staying resident through the entire prefilter/
    # rank/select stage below, on top of forcing that stage itself onto CPU
    # (mean_zscore/sds_zscore/sn_zscore come back from the host-resident
    # StreamingZscoreStats as plain numpy now, so prefilter_gene_pairs/
    # rank_gene_pairs -- which infer their backend from *that* input --
    # always ran on CPU regardless of --gpu, silently losing the ~13x
    # on-device speedup plans/optimization.md #6d measured for this stage).
    # Freeing batch_X and moving the z-score matrices back to GPU here
    # restores that speedup; by this point X_full and batch_X together were
    # the dominant GPU consumers, so there's ample room for three more
    # (genes x genes) arrays.
    del batch_X

    vmessage(verbose, "Computing statistics...")
    mean_zscore, sds_zscore, sn_zscore = stats.finalize()
    if xp.__name__ == "cupy":
        mean_zscore = xp.asarray(mean_zscore)
        sds_zscore = xp.asarray(sds_zscore)
        sn_zscore = xp.asarray(sn_zscore)

    vmessage(verbose, "Pre-filtering features...")
    prefiltered = prefilter_gene_pairs(
        mean_zscore,
        sds_zscore,
        sn_zscore,
        zscore_mean_threshold=prefilter_threshold,
        zscore_sn_threshold=prefilter_threshold,
        verbose=verbose,
    )

    vmessage(verbose, "Extracting filtered features...")
    ranked = rank_gene_pairs(prefiltered, score_weights=score_weights, direction=direction)
    common_genes_arr = np.asarray(common_genes)
    selected_genes = extract_unique_genes(ranked, common_genes_arr, max_n_genes)

    # geneA/geneB strings are looked up once here, on the already-ranked
    # table, instead of before ranking/sorting -- see prefilter_gene_pairs's
    # docstring (plans/optimization.md #6).
    prefiltered_df = ranked.assign(
        geneA=common_genes_arr[ranked["geneA_idx"].to_numpy()],
        geneB=common_genes_arr[ranked["geneB_idx"].to_numpy()],
    )[["geneA", "geneB", "mean_zscore", "sd_zscore", "sn_zscore", "rank"]]

    adata.var["anglemania_genes"] = adata.var_names.isin(selected_genes)
    adata.uns["anglemania"] = {
        "params": {
            "batch_key": batch_key,
            "dataset_key": dataset_key,
            "max_n_genes": max_n_genes,
            "min_cells_per_gene": min_cells_per_gene,
            "min_samples_per_gene": min_samples_per_gene,
            "allow_missing_features": allow_missing_features,
            "method": method,
            "permute_row_or_column": permute_row_or_column,
            "permutation_function": permutation_function,
            "prefilter_threshold": prefilter_threshold,
            "do_normalize": do_normalize,
            "normalization_method": normalization_method,
            "score_weights": list(score_weights),
            "direction": direction,
            "cell_chunk_size": cell_chunk_size,
        },
        "intersect_genes": common_genes,
        "prefiltered_df": prefiltered_df.reset_index(drop=True),
        "anglemania_genes": selected_genes,
    }
    vmessage(verbose, f"Selected {len(selected_genes)} genes for integration.")
    return adata
