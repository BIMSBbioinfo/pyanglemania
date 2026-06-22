"""Batch/dataset bookkeeping, ported from anglemania's R ``prepare_anglemania.R``."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .._utils import to_dense, to_numpy, vmessage


def add_unique_batch_key(adata, batch_key: str, dataset_key: str | None = None) -> pd.Series:
    """Add a combined ``anglemania_batch`` column to ``adata.obs`` in place.

    Mirrors the R ``add_unique_batch_key``: labels are ``"{dataset}:{batch}"``
    when a ``dataset_key`` is given, otherwise just the batch value.
    """
    batch = adata.obs[batch_key].astype(str)
    if dataset_key is not None:
        dataset = adata.obs[dataset_key].astype(str)
        combined = dataset + ":" + batch
    else:
        combined = batch
    adata.obs["anglemania_batch"] = combined.astype("category")
    return adata.obs["anglemania_batch"]


def compute_dataset_weights(
    obs: pd.DataFrame, batch_key: str, dataset_key: str | None = None
) -> pd.Series:
    """Per-``anglemania_batch`` weight so each dataset contributes equally.

    Mirrors the R ``.set_weights``: with more than one dataset, a batch's
    weight is ``1 / n_batches_in_its_dataset / n_datasets``, renormalized to
    a mean of 1. Without a meaningful ``dataset_key`` every batch gets
    weight 1.
    """
    if dataset_key is not None and obs[dataset_key].nunique() > 1:
        info = obs[["anglemania_batch", dataset_key, batch_key]].drop_duplicates()
        n_samples = info.groupby(dataset_key)["anglemania_batch"].transform("count")
        n_groups = info[dataset_key].nunique()
        weight = 1.0 / n_samples / n_groups
        weight = weight / weight.mean()
    else:
        info = obs[["anglemania_batch"]].drop_duplicates()
        weight = pd.Series(1.0, index=info.index)
    return pd.Series(weight.to_numpy(), index=info["anglemania_batch"].to_numpy())


def split_obs_indices_by_batch(adata) -> dict[str, np.ndarray]:
    """Map each ``anglemania_batch`` label to its row indices in ``adata``."""
    labels = adata.obs["anglemania_batch"]
    return {
        str(label): np.flatnonzero((labels == label).to_numpy())
        for label in labels.cat.categories
    }


def genes_passing_min_cells(X, min_cells_per_gene: int, xp, sp_mod) -> np.ndarray:
    """Boolean mask of genes detected in at least ``min_cells_per_gene`` cells.

    For sparse ``X``, converts to CSC first: ``.getnnz(axis=...)`` isn't
    implemented for cupyx's sparse matrices (raises ValueError), but CSC's
    ``indptr`` already encodes per-column (per-gene) nnz directly, so
    ``diff(indptr)`` is an O(genes) read instead of an O(nnz) scan (e.g. via
    bincount on CSR's column-index array). ``.tocsc()`` is a no-op (returns
    ``X`` itself) when ``X`` is already CSC, which it is whenever the caller
    has pre-converted for column-indexing (see ``align_to_common_genes``).
    """
    if sp_mod.issparse(X):
        counts = xp.diff(X.tocsc().indptr)
    else:
        counts = (X != 0).sum(axis=0)
    return to_numpy(counts).ravel() >= min_cells_per_gene


def intersect_genes(
    gene_lists: list[list[str]],
    allow_missing_features: bool,
    min_samples_per_gene: int,
    verbose: bool = True,
) -> list[str]:
    """Reduce per-batch gene lists to the common set used downstream.

    Ports ``get_intersect_genes``: by default the strict intersection across
    every batch (ordered as in the first batch); with
    ``allow_missing_features=True``, any gene seen in at least
    ``min_samples_per_gene`` batches (sorted alphabetically, as in R).
    """
    if not allow_missing_features:
        vmessage(verbose, "Using the intersection of filtered genes from all batches...")
        common = set(gene_lists[0])
        for genes in gene_lists[1:]:
            common &= set(genes)
        result = [g for g in gene_lists[0] if g in common]
    else:
        vmessage(
            verbose,
            f"Using genes which are present in minimally {min_samples_per_gene} samples...",
        )
        counts = Counter(g for genes in gene_lists for g in genes)
        result = sorted(g for g, n in counts.items() if n >= min_samples_per_gene)
    vmessage(verbose, f"Number of genes in intersected set: {len(result)}")
    return result


def align_to_common_genes(X, batch_genes: list[str], common_genes: list[str], xp):
    """Densify ``X`` and reorder/pad its columns to match ``common_genes``.

    Genes in ``common_genes`` absent from ``batch_genes`` (only possible
    with ``allow_missing_features=True``) are filled with zeros, matching
    the zero-padding ``prepare_matrices`` does in R before treating every
    batch as having the same gene set.

    If ``X`` is sparse, callers should pass it in as CSC: column-indexing
    below is then a native major-axis slice rather than CSR's costlier
    minor-axis one (the caller in ``_anglemania.py`` already converts once,
    up front, and reuses it for ``genes_passing_min_cells`` too).
    """
    gene_pos = {g: i for i, g in enumerate(batch_genes)}
    src_cols = []
    dst_cols = []
    for dst, g in enumerate(common_genes):
        src = gene_pos.get(g)
        if src is not None:
            src_cols.append(src)
            dst_cols.append(dst)

    out = xp.zeros((X.shape[0], len(common_genes)), dtype=xp.float32)
    if src_cols:
        dense = to_dense(X[:, np.asarray(src_cols)], xp).astype(xp.float32)
        out[:, np.asarray(dst_cols)] = dense
    return out
