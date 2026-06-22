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
from ._angles import factorise
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
):
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key {batch_key!r} must be a column in adata.obs")
    if dataset_key is not None and dataset_key not in adata.obs.columns:
        raise ValueError(f"dataset_key {dataset_key!r} must be a column in adata.obs")
    if max_n_genes is not None and (not isinstance(max_n_genes, int) or max_n_genes < 1):
        raise ValueError("max_n_genes must be a positive integer or None")
    if method not in ("cosine", "spearman"):
        raise ValueError(f"method must be 'cosine' or 'spearman', got {method!r}")
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
    if normalization_method not in ("divide_by_total_counts", "find_residuals"):
        raise ValueError(
            "normalization_method must be 'divide_by_total_counts' or "
            f"'find_residuals', got {normalization_method!r}"
        )
    if score_weights is not None and (
        len(score_weights) != 2 or not all(0 <= w <= 1 for w in score_weights)
    ):
        raise ValueError("score_weights must be a length-2 sequence of values in [0, 1]")
    if direction not in ("both", "anticor", "cor"):
        raise ValueError(f"direction must be 'both', 'anticor' or 'cor', got {direction!r}")


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
    ``ref_packages/anglemania/R/anglemania.R`` for the original.

    Modifies ``adata`` in place:

    - ``adata.var["anglemania_genes"]``: boolean mask of selected genes.
    - ``adata.uns["anglemania"]``: dict with ``params``, ``intersect_genes``,
      ``prefiltered_df`` (ranked gene-pair statistics), and
      ``anglemania_genes``.

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
    stats = StreamingZscoreStats(len(common_genes), xp)
    for label, idx in batch_indices.items():
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
        stats.update(zscores, float(weights[label]))
        del X_dense, zscores

    vmessage(verbose, "Computing statistics...")
    mean_zscore, sds_zscore, sn_zscore = stats.finalize()

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
            "score_weights": score_weights,
            "direction": direction,
        },
        "intersect_genes": common_genes,
        "prefiltered_df": prefiltered_df.reset_index(drop=True),
        "anglemania_genes": selected_genes,
    }
    vmessage(verbose, f"Selected {len(selected_genes)} genes for integration.")
    return adata
