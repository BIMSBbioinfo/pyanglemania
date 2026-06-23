"""Per-batch angle (gene-gene correlation) computation.

Ported from anglemania's R ``compute_angles.R``. Everything here takes a
dense ``(cells x genes)`` matrix in whatever array module the caller passes
as ``xp`` (``numpy`` or ``cupy``), so it runs unmodified on CPU or GPU.
"""

from __future__ import annotations


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
