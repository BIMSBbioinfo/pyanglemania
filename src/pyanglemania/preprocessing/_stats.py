"""Streaming cross-batch reduction of per-batch z-score matrices.

This is the deviation from anglemania's R ``get_list_stats`` /
``big_mat_list_mean`` called for in the implementation plan: instead of
keeping every batch's z-score matrix around and reducing them all at once
at the end, :class:`StreamingZscoreStats` accumulates the running sums
needed for the final mean/sd/SNR one batch at a time, so a batch's z-score
matrix can be dropped as soon as it has been folded in -- only the
accumulators (the same shape as a single batch's matrix) are ever held.

The weighted mean/variance R computes -- ``mean = sum_b(w_b*z_b) /
sum_b(w_b)``, and an unbiased weighted variance with denominator
``sum_b(w_b) - sum_b(w_b^2) / sum_b(w_b)`` -- only need two running sums
per gene pair (``sum(w*z)``, ``sum(w*z^2)``) plus two running scalars
(``sum(w)``, ``sum(w^2)``), thanks to the identity
``sum_b(w_b*(z_b - mean)^2) == sum_b(w_b*z_b^2) - sum_b(w_b*z_b)^2 / sum_b(w_b)``,
which avoids needing the final mean before seeing every batch. ``sum(w)``/
``sum(w^2)`` can be plain scalars rather than per-pair matrices because
every per-batch z-score matrix produced by :func:`._angles.factorise` has
already had its NaNs replaced with 0 -- there is no per-pair "missing in
this batch" case left to weight differently from the rest of the matrix.
"""

from __future__ import annotations


class StreamingZscoreStats:
    """Accumulates weighted mean/sd/SNR of z-score matrices across batches."""

    def __init__(self, n_genes: int, xp):
        self.xp = xp
        shape = (n_genes, n_genes)
        self._w_sum = 0.0
        self._w_sq_sum = 0.0
        self._wz_sum = xp.zeros(shape, dtype=xp.float64)
        self._wz2_sum = xp.zeros(shape, dtype=xp.float64)

    def update(self, zscores, weight: float) -> None:
        """Fold one batch's z-score matrix in; it can be discarded after this."""
        self._w_sum += weight
        self._w_sq_sum += weight * weight
        self._wz_sum += weight * zscores
        self._wz2_sum += weight * zscores * zscores

    def finalize(self):
        """Return ``(mean_zscore, sds_zscore, sn_zscore)``, each (genes x genes).

        Meant to be called once, after every batch has been folded in via
        :meth:`update` -- it consumes the accumulators (see below) rather
        than just reading them.
        """
        xp = self.xp
        w_sum = self._w_sum
        denom = w_sum - self._w_sq_sum / w_sum

        mean = self._wz_sum / w_sum

        # var = (wz2_sum - wz_sum^2/w_sum) / denom, reformulated as
        # (wz2_sum - wz_sum*mean) / denom and computed with in-place ops so
        # this allocates only one extra (genes x genes) buffer. The naive
        # chained expression (`wz2_sum - wz_sum*wz_sum/w_sum`, then `/
        # denom`) instead briefly holds 4-5 such buffers alive at once on
        # top of the accumulators -- harmless at a few thousand genes, but
        # at tens of thousands of genes (each buffer several GB) that's the
        # difference between fitting in GPU memory and OOMing (see
        # tests/test_gpu.py and CLAUDE.md's "GPU status" for the benchmark
        # that caught this).
        var = self._wz_sum * mean
        var -= self._wz2_sum
        var /= -denom
        # Never needed again; drop them so their memory can be reused below.
        del self._wz_sum, self._wz2_sum
        xp.clip(var, 0, None, out=var)
        sd = xp.sqrt(var, out=var)

        n = mean.shape[0]
        idx = xp.arange(n)
        sd[idx, idx] = xp.nan  # avoid dividing by zero in sn below, as in R

        # Matches R's `replace_with_na()` on the SNR matrix: a gene pair
        # whose z-score happens to be identical across every batch (or is
        # on the diagonal) has sd == 0 or NaN, so sn would be +inf/NaN and
        # an inf would trivially clear any threshold; treat both as missing
        # instead. `xp.where` (rather than dividing then cleaning up)
        # sidesteps an actual division by zero, which cupy can't mask the
        # way `numpy.errstate` does (and cupy's ufuncs don't support a
        # `where=` kwarg to skip it the other way either).
        positive_sd = sd > 0
        safe_sd = xp.where(positive_sd, sd, 1.0)
        sn = xp.abs(mean)
        sn /= safe_sd
        del safe_sd
        sn[~positive_sd] = xp.nan
        return mean, sd, sn
