"""Top-level ``anglemania`` entry point.

Ported from anglemania's R ``anglemania()`` (``R/anglemania.R``), restructured
around ``AnnData`` and streaming cross-batch statistics instead of
file-backed per-batch matrices -- see ``_stats.py`` for that part. The
per-batch angle/z-score computation and its parameters are kept faithful to
the R function; the selection step is not. R ranks *gene pairs* and takes
unique genes from the top pairs, so a gene's effective score is the rank of
the single best pair it is in. Here every gene instead gets a row-sum score
over all its partners (``signal``/``noise``/``R``, see ``_stats.py``),
corrected for expression by within-bin z-scoring (``_gene_level.py``), and
genes are ranked on that directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils import get_array_module, vmessage
from ._angles import factorise, factorise_chunked
from ._batches import (
    add_unique_batch_key,
    align_to_common_genes,
    compute_dataset_weights,
    genes_in_every_dataset,
    genes_passing_min_cells,
    intersect_genes,
    lognorm_column_sums,
    split_obs_indices_by_batch,
)
from ._gene_level import binned_zscores, top_genes
from ._stats import StreamingZscoreStats

# `score` value -> the per-gene column (in ``binned_zscores``' output) genes are ranked on
_SCORES = {"R": "R_z_binned", "WSN": "WSN", "ICC": "ICC_db_z_binned"}
_MISSING_MODES = ("mask", "zero")


def _check_params(
    adata,
    batch_key,
    dataset_key,
    max_n_genes,
    method,
    min_cells_per_gene,
    min_samples_per_gene,
    missing_mode,
    dataset_presence,
    permute_row_or_column,
    permutation_function,
    normalization_method,
    score,
    n_bins,
    signal_R_weights,
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
    if missing_mode not in _MISSING_MODES:
        raise ValueError(f"missing_mode must be one of {_MISSING_MODES}, got {missing_mode!r}")
    if dataset_presence and dataset_key is None:
        raise ValueError("dataset_presence=True needs a dataset_key")
    if permute_row_or_column not in ("row", "column"):
        raise ValueError(
            f"permute_row_or_column must be 'row' or 'column', got {permute_row_or_column!r}"
        )
    if permutation_function not in ("sample", "permute_nonzero"):
        raise ValueError(
            "permutation_function must be 'sample' or 'permute_nonzero', "
            f"got {permutation_function!r}"
        )
    if normalization_method not in ("divide_by_total_counts", "find_residuals", "pflog1ppf"):
        raise ValueError(
            "normalization_method must be 'divide_by_total_counts', "
            f"'find_residuals' or 'pflog1ppf', got {normalization_method!r}"
        )
    if score not in _SCORES:
        raise ValueError(f"score must be one of {tuple(_SCORES)}, got {score!r}")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if len(signal_R_weights) != 2 or not all(w >= 0 for w in signal_R_weights):
        raise ValueError("signal_R_weights must be a length-2 sequence of non-negative values")
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
    missing_mode: str = "mask",
    dataset_presence: bool = False,
    method: str = "cosine",
    permute_row_or_column: str = "column",
    permutation_function: str = "sample",
    do_normalize: bool = True,
    normalization_method: str = "divide_by_total_counts",
    score: str = "R",
    n_bins: int = 20,
    signal_R_weights: tuple[float, float] = (0.5, 0.5),
    cell_chunk_size: int | None = None,
    verbose: bool = True,
):
    """Select genes with batch-invariant, biologically informative gene-gene angles.

    For each batch (``batch_key``, optionally nested under ``dataset_key``),
    computes the gene-gene angle (correlation) matrix on ``adata.X`` (or
    ``layer``, expected to hold raw counts) and z-scores it against a
    permuted null built from that same batch. Those per-batch z-score
    matrices are then reduced, batch by batch, into a weighted cross-batch
    mean ``M`` and sd ``S`` per gene pair (kept as a running accumulator
    rather than ever holding every batch's matrix at once -- see
    :class:`._stats.StreamingZscoreStats`), and from there into one score
    per gene:

    - ``signal = sum_j M_ij^2``: how strongly the gene's angles to all
      other genes are reproduced across batches;
    - ``noise = sum_j S_ij^2``: how much of them is batch-specific;
    - ``R = signal / (signal + noise)``: the reproducible fraction.

    ``log10(signal)`` and ``R`` are z-scored within equal-frequency bins of
    mean log-normalized expression (``n_bins``), giving ``signal_z_binned``
    and ``R_z_binned``, and the weighted signal/noise score ``WSN =
    signal_R_weights[0] * signal_z_binned + signal_R_weights[1] *
    R_z_binned``. The ``max_n_genes`` genes with the highest ``score`` are
    selected: ``"R"`` (default) ranks on ``R_z_binned``, ``"WSN"`` on
    ``WSN``, ``"ICC"`` on ``ICC_db_z_binned`` (see below).

    The per-batch parameters mirror anglemania's R function of the same
    name; see ``ref_packages/anglemania/R/anglemania.R`` for the original.
    R's pair-level selection parameters (``prefilter_threshold``,
    ``score_weights``, ``direction``) have no gene-level counterpart and
    are gone -- the row sums are sums of squares, hence sign-blind. Two
    ``method``/``normalization_method`` choices are not from R:
    ``method="phi_s"`` (a proportionality metric in place of correlation)
    and ``normalization_method="pflog1ppf"`` (a shifted-CLR transform,
    intended to be used together) -- see ``_angles.py``'s
    ``extract_angles``/``normalize_matrix`` docstrings.

    **Missing genes** (``allow_missing_features=True``: a gene is kept if it
    passes ``min_cells_per_gene`` in at least ``min_samples_per_gene``
    batches). With ``missing_mode="mask"`` (default, not from R), each
    batch's angles and permutation null are computed on the genes present
    in that batch only, and each gene pair is reduced over the batches in
    which both genes are present. ``missing_mode="zero"`` is R's behavior
    instead -- absent genes are zero-padded columns (which also enter the
    batch's permutation null) whose z-scores of 0 are averaged in with full
    weight, shrinking ``M`` and inflating ``S`` in proportion to how often a
    gene is missing (see ``plans/missing_features_nan_masking.md``).

    Masked, each pair also has its own ``kappa = sum w^2 / (sum w)^2`` over
    its shared batches, and ``R``'s pure-noise floor is ``~kappa`` -- so
    ``R`` (and ``R_z_binned``, ``WSN``) is inflated for genes present in
    fewer batches. ``score="ICC"`` ranks on the binned
    debiased ``ICC_db = sum_j (M^2 - kappa S^2) / (sum_j (M^2 - kappa S^2)
    + noise)`` instead, whose null is ~0 at every presence level (without
    missing genes it is a monotone transform of ``R``, so it ranks genes
    within each bin as ``R_z_binned`` does).
    ``dataset_presence=True`` additionally requires a gene to be present in
    at least one batch of every ``dataset_key`` group, which guards against
    annotation gaps (a gene with zero counts across a whole dataset).

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
    - ``adata.var["anglemania_{signal,noise,R,signal_db,ICC_db,mean_lognorm,
      n_batches_present,expr_bin,signal_z_binned,R_z_binned,WSN,
      ICC_db_z_binned}"]``: the per-gene scores and their inputs; NaN for
      genes outside ``intersect_genes``.
    - ``adata.uns["anglemania"]``: dict with ``params``, ``intersect_genes``,
      ``anglemania_genes`` (selected genes, best first), and ``presence``
      (``batches x intersect_genes`` boolean DataFrame; pair counts ``n_ij``
      are ``presence.T @ presence``).
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
        missing_mode,
        dataset_presence,
        permute_row_or_column,
        permutation_function,
        normalization_method,
        score,
        n_bins,
        signal_R_weights,
        cell_chunk_size,
    )

    vmessage(verbose, "Preparing input...")
    add_unique_batch_key(adata, batch_key, dataset_key)
    weights = compute_dataset_weights(adata.obs, batch_key, dataset_key)
    batch_indices = split_obs_indices_by_batch(adata)
    if len(batch_indices) < 2:
        raise ValueError("anglemania needs at least 2 batches")

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
    if dataset_presence and allow_missing_features:
        info = adata.obs[["anglemania_batch", dataset_key]].drop_duplicates()
        batch_dataset = dict(
            zip(info["anglemania_batch"].astype(str), info[dataset_key].astype(str))
        )
        common_genes = genes_in_every_dataset(common_genes, batch_genes, batch_dataset)
        vmessage(verbose, f"Number of genes present in every dataset: {len(common_genes)}")
    if max_n_genes is not None and max_n_genes > len(common_genes):
        vmessage(
            verbose,
            f"{max_n_genes} is larger than the number of intersected genes. "
            f"Setting max_n_genes to {len(common_genes)}",
        )
        max_n_genes = len(common_genes)

    vmessage(verbose, "Computing angles and transforming to z-scores...")
    gene_pos = {g: i for i, g in enumerate(common_genes)}
    stats = StreamingZscoreStats(len(common_genes))
    lognorm_sum = np.zeros(len(common_genes))
    n_present = np.zeros(len(common_genes), dtype=np.int64)
    mask = missing_mode == "mask"
    for label, idx in batch_indices.items():
        src = [i for i, g in enumerate(batch_genes[label]) if g in gene_pos]
        dst = np.asarray([gene_pos[batch_genes[label][i]] for i in src], dtype=np.int64)
        n_present[dst] += 1
        lognorm_sum[dst] += lognorm_column_sums(batch_X[label][:, np.asarray(src)], xp, sp_mod)

        # Masked: the batch's z-scores cover its present genes only (in
        # universe order), so no zero-padded column enters its null either.
        # Zero: the full universe, absent genes zero-padded, as in R.
        index = np.sort(dst) if mask else None
        z_genes = [common_genes[i] for i in index] if mask else common_genes
        if cell_chunk_size is not None:
            zscores = factorise_chunked(
                batch_X[label],
                batch_genes[label],
                z_genes,
                xp,
                cell_chunk_size,
                method=method,
                permute_row_or_column=permute_row_or_column,
                permutation_function=permutation_function,
                normalization_method=normalization_method,
                do_normalize=do_normalize,
            )
        else:
            X_dense = align_to_common_genes(batch_X[label], batch_genes[label], z_genes, xp)
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
        stats.update(zscores, float(weights[label]), index)
        del zscores
    del batch_X

    vmessage(verbose, "Computing gene-level scores...")
    gene_scores = stats.gene_scores()
    presence = pd.DataFrame(stats.presence, index=list(batch_indices), columns=common_genes)
    del stats
    mean_lognorm = lognorm_sum / sum(len(idx) for idx in batch_indices.values())
    binned = binned_zscores(
        mean_lognorm,
        gene_scores["signal"],
        gene_scores["R"],
        ICC_db=gene_scores["ICC_db"],
        n_bins=n_bins,
        signal_R_weights=signal_R_weights,
    )
    selected_genes = top_genes(binned[_SCORES[score]].to_numpy(), common_genes, max_n_genes)

    columns = {
        "signal": gene_scores["signal"],
        "noise": gene_scores["noise"],
        "R": gene_scores["R"],
        "signal_db": gene_scores["signal_db"],
        "ICC_db": gene_scores["ICC_db"],
        "mean_lognorm": mean_lognorm,
        "n_batches_present": n_present,
        **{c: binned[c].to_numpy() for c in binned.columns},
    }
    pos = adata.var_names.get_indexer(common_genes)
    for name, values in columns.items():
        col = np.full(adata.n_vars, np.nan)
        col[pos] = values
        adata.var[f"anglemania_{name}"] = col
    adata.var["anglemania_genes"] = adata.var_names.isin(selected_genes)
    adata.uns["anglemania"] = {
        "params": {
            "batch_key": batch_key,
            "dataset_key": dataset_key,
            "max_n_genes": max_n_genes,
            "min_cells_per_gene": min_cells_per_gene,
            "min_samples_per_gene": min_samples_per_gene,
            "allow_missing_features": allow_missing_features,
            "missing_mode": missing_mode,
            "dataset_presence": dataset_presence,
            "method": method,
            "permute_row_or_column": permute_row_or_column,
            "permutation_function": permutation_function,
            "do_normalize": do_normalize,
            "normalization_method": normalization_method,
            "score": score,
            "n_bins": n_bins,
            "signal_R_weights": list(signal_R_weights),
            "cell_chunk_size": cell_chunk_size,
        },
        "intersect_genes": common_genes,
        "anglemania_genes": selected_genes,
        "presence": presence,
    }
    vmessage(verbose, f"Selected {len(selected_genes)} genes for integration.")
    return adata
