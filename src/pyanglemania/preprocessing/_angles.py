"""Per-batch angle (gene-gene correlation) computation.

Ported from anglemania's R ``compute_angles.R``. Everything here takes a
dense ``(cells x genes)`` matrix in whatever array module the caller passes
as ``xp`` (``numpy`` or ``cupy``), so it runs unmodified on CPU or GPU.
"""

from __future__ import annotations

from ._batches import align_to_common_genes


def normalize_matrix(X, xp, method: str = "divide_by_total_counts"):
    """Normalize a dense ``(cells x genes)`` matrix.

    ``"divide_by_total_counts"`` (default): CP10K + log1p, i.e. each cell's
    counts are divided by that cell's total count, scaled by 1e4, then
    log1p'd. ``"find_residuals"``: log1p the counts, then for every gene
    regress out the cell's log1p total count and keep the residual.

    These two are the choices the upstream R ``normalize_matrix`` actually
    implements -- its own docs advertise a third, ``"scale_by_total_counts"``,
    but that branch doesn't exist in the R source, so it isn't ported here
    either.

    ``"pflog1ppf"`` is not from R: it's the shifted-centered-log-ratio
    transform from Booeshaghi, Hallgrímsdóttir, Gálvez-Merchán & Pachter,
    "Depth normalization for single-cell genomics count data" (PFlog1pPF,
    a.k.a. "shifted CLR"), equivalent to
    ``sc.pp.normalize_total(adata, target_sum=1); sc.pp.log1p(adata);
    adata.X -= adata.X.mean(axis=1)``: a proportional-fitting step (each
    cell's counts divided by that cell's total, i.e. ``u = x / sum(x)``), a
    pseudocount-1 log (``log1p(u)``), then a second proportional-fitting
    step done as centering in log-space (subtracting each cell's own mean
    log-proportion) rather than dividing in count-space, since a log has
    already been applied.
    """
    if method == "divide_by_total_counts":
        total = X.sum(axis=1, keepdims=True)
        return xp.log1p(X / total * 1e4)
    if method == "find_residuals":
        total = xp.log1p(X.sum(axis=1))
        x_log = xp.log1p(X)
        x_centered = x_log - x_log.mean(axis=0, keepdims=True)
        total_centered = total - total.mean()
        slopes = (total_centered @ x_centered) / xp.sum(total_centered * total_centered)
        return x_centered - xp.outer(total_centered, slopes)
    if method == "pflog1ppf":
        total = X.sum(axis=1, keepdims=True)
        log_u = xp.log1p(X / total)
        return log_u - log_u.mean(axis=1, keepdims=True)
    raise ValueError(
        "normalization_method must be 'divide_by_total_counts', "
        f"'find_residuals' or 'pflog1ppf', got {method!r}"
    )


def _shuffle_full(X, axis: int, xp, rng):
    """Independently permute every 1-D slice of ``X`` along ``axis``."""
    order = xp.argsort(rng.random(X.shape, dtype=xp.float32), axis=axis)
    return xp.take_along_axis(X, order, axis=axis)


def _shuffle_nonzero(X, axis: int, xp, rng):
    """Independently permute only the nonzero entries of each slice along ``axis``.

    Zero entries stay exactly where they were. The values that land on the
    nonzero positions are a uniform random permutation of that slice's
    original nonzero values: one argsort (keyed on a random value, with
    zeros pushed past it via +inf) shuffles just the nonzero values, a
    second argsort (keyed on the zero/nonzero mask) recovers the original
    nonzero positions, and a single scatter pairs them up. Which of the
    nonzero positions gets which shuffled value doesn't depend on the
    pairing order, so no tie-breaking/stability between the two orders is
    needed for this to be a correct uniform shuffle.
    """
    mask = X != 0
    random_key = xp.where(mask, rng.random(X.shape, dtype=xp.float32), xp.inf)
    value_order = xp.argsort(random_key, axis=axis)
    shuffled_values = xp.take_along_axis(X, value_order, axis=axis)

    position_order = xp.argsort((~mask).astype(xp.int8), axis=axis)
    result = xp.empty_like(X)
    xp.put_along_axis(result, position_order, shuffled_values, axis=axis)
    return result


def permute_matrix(X, axis: int, function: str, xp, rng):
    if function == "sample":
        return _shuffle_full(X, axis, xp, rng)
    if function == "permute_nonzero":
        return _shuffle_nonzero(X, axis, xp, rng)
    raise ValueError(
        f"permutation_function must be 'sample' or 'permute_nonzero', got {function!r}"
    )


def extract_angles(X, method: str, xp):
    """Gene-gene relationship matrix for a dense ``(cells x genes)`` matrix.

    ``"cosine"``: Pearson correlation of genes across cells -- the angle
    between mean-centered gene vectors (same as anglemania's R
    ``extract_angles``, which despite the name computes a centered
    correlation via ``big_cor``). ``"spearman"``: the same, computed on
    per-gene ranks across cells (ties broken by original order rather than
    R's tie-averaging, so this stays vectorized on both numpy and cupy).

    ``"phi_s"`` is not from R: it's the symmetric proportionality metric
    phi_s from Quinn, Richardson, Lovell & Crowley, "propr: An R-package for
    Identifying Proportionally Abundant Features Using Compositional Data
    Analysis" -- ``VLR(i, j) / VLP(i, j)``, the variance of the log-ratio
    ``X_i - X_j`` over the variance of the log-product ``X_i + X_j``, low
    for proportional gene pairs and unbounded above otherwise (the inverse
    sense of a correlation). ``X`` is expected to already be a log-ratio
    matrix; in this package that's ``normalize_matrix(..., "pflog1ppf")``,
    deliberately *not* propr's own per-sample CLR (raw counts, log'd after
    replacing zeros with 1) -- see that function's docstring.

    Returns a symmetric ``(genes x genes)`` matrix with NaN on the diagonal.
    """
    if method == "spearman":
        X = xp.argsort(xp.argsort(X, axis=0), axis=0).astype(X.dtype)
    elif method not in ("cosine", "phi_s"):
        raise ValueError(f"method must be 'cosine', 'spearman' or 'phi_s', got {method!r}")

    x_centered = X - X.mean(axis=0, keepdims=True)
    cov = x_centered.T @ x_centered

    if method == "phi_s":
        # var/cov up to the shared factor 1/(n - 1), which cancels in the
        # ratio below, so skip it: VLR(i,j) = var(X_i - X_j), and X_i, X_j
        # already mean-zero makes that sum((x_i - x_j)^2) = cov_ii + cov_jj
        # - 2*cov_ij directly; VLP is the same with X_i + X_j.
        var = xp.diagonal(cov)
        vlr = var[:, None] + var[None, :] - 2 * cov
        vlp = var[:, None] + var[None, :] + 2 * cov
        result = vlr / vlp
    else:
        norm = xp.sqrt(xp.sum(x_centered * x_centered, axis=0))
        result = cov / xp.outer(norm, norm)

    n = result.shape[0]
    idx = xp.arange(n)
    result[idx, idx] = xp.nan
    return result


def get_dstat(perm_corr, xp):
    """Per-gene (per-column) mean/sd of a permuted null correlation matrix."""
    mean = xp.nanmean(perm_corr, axis=0)
    sd = xp.sqrt(xp.nanvar(perm_corr, axis=0, ddof=1))
    return mean, sd


def factorise(
    X,
    xp,
    method: str = "cosine",
    seed: int = 1,
    permute_row_or_column: str = "column",
    permutation_function: str = "sample",
    normalization_method: str = "divide_by_total_counts",
    do_normalize: bool = True,
):
    """Z-score one batch's gene-gene angles against a permuted null.

    Ports anglemania's R ``factorise``: build a null distribution by
    permuting ``X``, normalize both the real and permuted matrices, compute
    their gene-gene angle matrices, then z-score the real one against the
    per-gene (per-column) mean/sd of the null -- so, like in R, the z-score
    matrix is not symmetric: entry ``(i, j)`` is standardized against gene
    ``j``'s own null background, not gene ``i``'s.

    ``permute_row_or_column`` keeps the R parameter's values ("row"/
    "column"), which refer to R's genes x cells matrix orientation; this
    package stores cells x genes, so "column" (R's default: permute within
    each cell, across genes) maps to ``axis=1`` here, and "row" maps to
    ``axis=0``.
    """
    axis = 1 if permute_row_or_column == "column" else 0
    rng = xp.random.default_rng(seed)
    x_perm = permute_matrix(X, axis, permutation_function, xp, rng)

    if do_normalize:
        X = normalize_matrix(X, xp, normalization_method)
        x_perm = normalize_matrix(x_perm, xp, normalization_method)

    corr = extract_angles(X, method, xp)
    perm_corr = extract_angles(x_perm, method, xp)

    mean, sd = get_dstat(perm_corr, xp)
    zscores = (corr - mean[None, :]) / sd[None, :]
    # Matches R's `zscores[is.na(zscores)] <- 0`: degenerate all-zero-
    # correlation columns (mean == sd == 0) become 0/0 == NaN, here too.
    return xp.where(xp.isnan(zscores), 0.0, zscores)


_CHUNKABLE_METHODS = ("cosine", "phi_s")
_CHUNKABLE_NORMALIZATIONS = ("divide_by_total_counts", "pflog1ppf")


def _angles_from_moments(col_sum, sum_sq, n_cells: int, method: str, xp):
    """``extract_angles``'s result, given only accumulated moments.

    ``sum_sq`` is the *uncentered* second moment ``sum_i(x_i @ x_i.T)`` and
    ``col_sum`` the per-column sum, both summable additively over row-chunks
    of ``X`` -- unlike ``extract_angles``, which needs the whole ``(cells x
    genes)`` matrix at once to center it first. The centered Gram matrix
    ``cov`` (``x_centered.T @ x_centered``, no ``1/(n-1)`` factor -- see
    ``extract_angles``'s docstring for why that factor is never applied
    there either, since it cancels in both the cosine and phi_s ratios) is
    recovered from the uncentered moments via the standard identity
    ``sum_i((x_i - mean) @ (x_i - mean).T) == sum_sq - n * outer(mean, mean)``,
    the same one already used across batches in ``_stats.py``.
    """
    mean = col_sum / n_cells
    cov = sum_sq - n_cells * xp.outer(mean, mean)

    if method == "phi_s":
        var = xp.diagonal(cov)
        vlr = var[:, None] + var[None, :] - 2 * cov
        vlp = var[:, None] + var[None, :] + 2 * cov
        result = vlr / vlp
    else:
        norm = xp.sqrt(xp.diagonal(cov))
        result = cov / xp.outer(norm, norm)

    n = result.shape[0]
    idx = xp.arange(n)
    result[idx, idx] = xp.nan
    return result


def factorise_chunked(
    X,
    batch_genes: list[str],
    common_genes: list[str],
    xp,
    cell_chunk_size: int,
    method: str = "cosine",
    seed: int = 1,
    permute_row_or_column: str = "column",
    permutation_function: str = "sample",
    normalization_method: str = "divide_by_total_counts",
    do_normalize: bool = True,
):
    """Memory-bounded equivalent of ``factorise``, for one very large batch.

    ``factorise`` (and the ``align_to_common_genes`` call that precedes it
    in ``_anglemania.py``) each materialize a full ``(cells x genes)`` dense
    array -- several of them coexist at once (raw, permuted, normalized real,
    normalized permuted) -- which is the "bottleneck 2" documented in
    ``plans/gpu_memory_large_batches.md``: fine for most batches, but a
    problem for a batch with tens of thousands of cells at a large gene
    count (e.g. ``allow_missing_features=True`` pushing the gene universe up).

    This computes the exact same z-score matrix as ``factorise`` would (up
    to floating point summation order, and the caveat about permutation
    randomness below), but processes ``X`` in row-chunks of at most
    ``cell_chunk_size`` cells, taking each chunk from raw counts through
    alignment/permutation/normalization to an accumulated contribution to
    the ``(genes x genes)`` Gram matrix, then discarding it -- so peak
    memory is ``O(cell_chunk_size x genes + genes^2)`` regardless of how
    many cells this batch actually has (see ``_angles_from_moments``'s
    docstring for the identity this relies on to stay exact, not a
    subsample). ``X`` is passed in *unaligned*, i.e. still only restricted
    to ``batch_genes`` (this batch's own passing genes) as in
    ``_anglemania.py``'s ``batch_X`` -- alignment to ``common_genes`` also
    has to happen per chunk, or the whole point is lost.

    Only supports the combination of options where every step from
    permutation through normalization is row-independent (chunkable without
    a separate pass to compute global statistics first) -- which happens to
    be this package's defaults, and the combination used by both
    ``method``/``normalization_method`` configs in the real workload this
    was written for (NBAtlas malignant-compartment stability sweep, see
    ``plans/gpu_memory_large_batches.md``):

    - ``permute_row_or_column="column"`` (the default): permutes within each
      cell, independently per row, so any row-chunking permutes correctly.
      ``"row"`` permutes within each *gene*, across all cells -- inherently
      needs every cell of a column at once, so isn't chunkable this way.
    - ``method`` must be ``"cosine"`` or ``"phi_s"``: both only need the
      centered Gram matrix (see ``_angles_from_moments``). ``"spearman"``
      needs a global rank over every cell in the batch, which isn't an
      additive statistic.
    - ``normalization_method`` must be ``"divide_by_total_counts"`` or
      ``"pflog1ppf"``: both normalize each cell using only that cell's own
      total, so they're row-independent. ``"find_residuals"`` regresses out
      each gene's dependence on log total counts, which needs the global
      per-gene mean (and the global total-count centering) *before* any row
      can be normalized -- a genuine two-pass dependency this function
      doesn't implement.

    Raises ``ValueError`` for any other combination rather than silently
    falling back to the full-materialization path, so a caller relying on
    the memory bound isn't surprised by an OOM anyway.

    Caveat: unlike ``factorise``, results are only reproducible for a fixed
    ``seed`` *and* ``cell_chunk_size`` together. The real (unpermuted) data's
    angle matrix matches ``factorise`` almost exactly (floating point only);
    the permuted null does not, because generating the permutation's random
    keys is itself chunked -- an RNG draw of shape ``(chunk, genes)`` per
    chunk is not the same draw sequence as one ``(cells, genes)`` call. The
    null is still a valid independent random permutation either way, just
    not a bit-identical one across chunk sizes.
    """
    if permute_row_or_column != "column":
        raise ValueError(
            "cell_chunk_size only supports permute_row_or_column='column' "
            f"(row-independent permutation); got {permute_row_or_column!r}, which "
            "permutes across all cells in a column and needs the whole batch at once."
        )
    if method not in _CHUNKABLE_METHODS:
        raise ValueError(
            f"cell_chunk_size does not support method={method!r} (needs a global "
            f"rank over every cell); use one of {_CHUNKABLE_METHODS}, or omit "
            "cell_chunk_size."
        )
    if normalization_method not in _CHUNKABLE_NORMALIZATIONS:
        raise ValueError(
            f"cell_chunk_size does not support normalization_method={normalization_method!r} "
            f"(needs a global pass before centering); use one of "
            f"{_CHUNKABLE_NORMALIZATIONS}, or omit cell_chunk_size."
        )
    if cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be a positive integer")

    n_cells = X.shape[0]
    n_genes = len(common_genes)
    rng = xp.random.default_rng(seed)

    sum_real = xp.zeros(n_genes, dtype=xp.float64)
    sum_perm = xp.zeros(n_genes, dtype=xp.float64)
    ss_real = xp.zeros((n_genes, n_genes), dtype=xp.float64)
    ss_perm = xp.zeros((n_genes, n_genes), dtype=xp.float64)

    for start in range(0, n_cells, cell_chunk_size):
        chunk = X[start : start + cell_chunk_size]
        chunk_real = align_to_common_genes(chunk, batch_genes, common_genes, xp)
        chunk_perm = permute_matrix(chunk_real, 1, permutation_function, xp, rng)

        if do_normalize:
            chunk_real = normalize_matrix(chunk_real, xp, normalization_method)
            chunk_perm = normalize_matrix(chunk_perm, xp, normalization_method)

        sum_real += chunk_real.sum(axis=0, dtype=xp.float64)
        sum_perm += chunk_perm.sum(axis=0, dtype=xp.float64)
        ss_real += (chunk_real.T @ chunk_real).astype(xp.float64)
        ss_perm += (chunk_perm.T @ chunk_perm).astype(xp.float64)
        del chunk, chunk_real, chunk_perm

    corr = _angles_from_moments(sum_real, ss_real, n_cells, method, xp)
    perm_corr = _angles_from_moments(sum_perm, ss_perm, n_cells, method, xp)
    del sum_real, sum_perm, ss_real, ss_perm

    mean, sd = get_dstat(perm_corr, xp)
    zscores = (corr - mean[None, :]) / sd[None, :]
    return xp.where(xp.isnan(zscores), 0.0, zscores)
