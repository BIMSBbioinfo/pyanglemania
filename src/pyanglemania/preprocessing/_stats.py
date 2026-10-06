"""Streaming cross-batch reduction of per-batch z-score matrices to per-gene scores.

This is the deviation from anglemania's R ``get_list_stats`` /
``big_mat_list_mean`` called for in the implementation plan: instead of
keeping every batch's z-score matrix around and reducing them all at once
at the end, :class:`StreamingZscoreStats` accumulates the running sums
needed for the cross-batch mean/sd one batch at a time, so a batch's
z-score matrix can be dropped as soon as it has been folded in -- only the
accumulators (the same shape as a single batch's matrix) are ever held.

Each per-batch matrix is symmetrised (``(z + z^T) / 2``) on the way in,
since :func:`._angles.factorise` standardises entry ``(i, j)`` against gene
``j``'s null only. From the accumulated sums, per gene pair ``(i, j)`` and
over the batches ``b`` in which *both* genes are present:

    W1_ij = sum_b w_b,   W2_ij = sum_b w_b^2,   kappa_ij = W2_ij / W1_ij^2
    M_ij  = sum_b w_b z_bij / W1_ij
    S2_ij = [sum_b w_b (z_bij - M_ij)^2 / W1_ij] / (1 - kappa_ij)

i.e. the weighted mean and unbiased weighted variance R uses, with
``sum_b w_b (z_b - M)^2 == sum_b w_b z_b^2 - M sum_b w_b z_b`` so the mean
isn't needed before every batch is seen. ``W1``/``W2`` are not accumulated:
they are ``P^T diag(w) P`` and ``P^T diag(w^2) P`` of the ``(batches x
genes)`` presence matrix ``P``, recomputed per row-chunk at the end. Without
missing genes they are plain scalars and everything reduces to the
complete-data formulas.

Missing genes (``allow_missing_features=True``) are excluded rather than
counted as ``z = 0``: a batch's z-score matrix covers only the genes present
in it (``update(..., index=...)``), and every pair is reduced over the
batches in which both genes are present. Zero-filling instead -- R's
behavior, reproduced by feeding zero-padded full matrices with
``index=None`` -- averages the padded zeros in with full weight, which for a
pair with real z-scores ``mu +/- sigma`` in ``B - k`` of ``B`` batches gives

    M  ~ (1 - k/B) mu                         (shrunk toward 0)
    S2 ~ sigma^2 + mu^2 k(B-k) / (B(B-1))     (inflated by the 0-vs-mu gap)

i.e. scores set by how often a gene is missing rather than how reproducible
its co-expression is (``plans/missing_features_nan_masking.md``). Pairs
shared by fewer than ``min_pair_batches`` batches get no ``S`` and are
skipped.

:meth:`StreamingZscoreStats.gene_scores` then reduces ``M``/``S2`` to one
row per gene, never materialising either as a full matrix:

    signal_i = sum_j M_ij^2          # relational energy shared across batches
    noise_i  = sum_j S2_ij           # batch-specific relational energy
    R_i      = signal_i / (signal_i + noise_i)

and the debiased pair (since ``E[M^2] = mu^2 + kappa S2``, the pure-noise
floor of ``R`` is ``~kappa`` -- which, once missing genes give each pair its
own ``kappa``, is higher for genes present in fewer batches):

    signal_db_i = sum_j (M_ij^2 - kappa_ij S2_ij)
    ICC_db_i    = signal_db_i / (signal_db_i + noise_i)     # ~0 under the null

with the sums over valid partners ``j != i``, rescaled by
``(n_genes - 1) / n_valid`` so they equal the plain sums when nothing is
missing.

The two accumulators live in **host (numpy) memory, always**, regardless of
whether ``update()`` is fed numpy or cupy z-score matrices -- see
``plans/gpu_memory_large_batches.md`` fix 1. They are touched exactly once
per batch (inside ``update()``), so pulling a cupy batch to host there is a
single ``cupy.asnumpy()`` D2H transfer per batch.
"""

from __future__ import annotations

import numpy as np

from .._utils import to_numpy


class StreamingZscoreStats:
    """Accumulates per-batch z-score matrices into per-gene signal/noise/R.

    Accumulators are always host (numpy) arrays -- see module docstring --
    so this class doesn't need to know which array module a caller's
    batches come from; ``update()`` figures that out per call.
    """

    def __init__(self, n_genes: int):
        shape = (n_genes, n_genes)
        self._n_genes = n_genes
        self._weights: list[float] = []
        self._present: list[np.ndarray] = []
        self._wz_sum = np.zeros(shape, dtype=np.float64)
        self._wz2_sum = np.zeros(shape, dtype=np.float64)

    def update(self, zscores, weight: float, index=None) -> None:
        """Fold one batch's z-score matrix in; it can be discarded after this.

        ``zscores`` is ``(q x q)`` over the genes at positions ``index``
        (ascending ints into the ``n_genes`` universe) -- the genes present
        in this batch; every other gene counts as absent from it. ``None``
        means ``zscores`` covers the whole universe and every gene is
        present.
        """
        n = self._n_genes
        if index is None:
            index = np.arange(n)
        index = np.asarray(index, dtype=np.int64)
        q = len(index)
        if zscores.shape != (q, q):
            raise ValueError(f"zscores must be ({q} x {q}) to match index, got {zscores.shape}")
        if q > 1 and np.any(np.diff(index) <= 0):
            raise ValueError("index must be strictly increasing")

        z = zscores + zscores.T  # symmetrise on-device, before the D2H copy
        z *= 0.5
        z = np.asarray(to_numpy(z), dtype=np.float64)
        del zscores
        z[np.arange(q), np.arange(q)] = 0.0  # diagonal carries no relational information

        present = np.zeros(n, dtype=bool)
        present[index] = True
        self._weights.append(float(weight))
        self._present.append(present)
        block = slice(None) if q == n else np.ix_(index, index)
        self._wz_sum[block] += weight * z
        z *= z
        z *= weight
        self._wz2_sum[block] += z

    @property
    def presence(self) -> np.ndarray:
        """``(batches x genes)`` boolean presence matrix, in ``update()`` order."""
        return np.stack(self._present)

    def pair_counts(self, rows=slice(None)) -> np.ndarray:
        """``n_ij``: number of batches sharing genes ``i`` (in ``rows``) and ``j``.

        Not determined by per-gene presence counts alone -- it depends on
        *which* batches overlap -- hence ``P^T P`` of the presence matrix.
        """
        P = self.presence.astype(np.float64)
        return (P[:, rows].T @ P).astype(np.int64)

    def gene_scores(self, min_pair_batches: int = 2, chunk_size: int = 2048) -> dict:
        """Per-gene ``signal``/``noise``/``R``/``signal_db``/``ICC_db``/``n_partners``.

        Computed in row-chunks of ``chunk_size`` genes, so peak extra memory
        is ``O(chunk_size x n_genes)`` on top of the accumulators. Genes with
        no valid partner get NaN.
        """
        if len(self._weights) < 2:
            raise ValueError("need at least 2 batches to estimate cross-batch spread")
        if min_pair_batches < 2:
            raise ValueError("min_pair_batches must be >= 2")

        n = self._n_genes
        w = np.asarray(self._weights)
        P = self.presence.astype(np.float64)
        complete = bool(P.all())
        if complete:
            W1, W2 = w.sum(), (w * w).sum()
            n_pairs = len(w)

        signal = np.empty(n)
        noise = np.empty(n)
        signal_db = np.empty(n)
        n_valid = np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            if not complete:
                Pc = P[:, s:e].T
                W1 = (Pc * w) @ P
                W2 = (Pc * (w * w)) @ P
                n_pairs = Pc @ P
            S1 = self._wz_sum[s:e]
            with np.errstate(invalid="ignore", divide="ignore"):
                M = S1 / W1
                V = self._wz2_sum[s:e] - S1 * M
                np.clip(V, 0, None, out=V)
                V /= W1
                kappa = W2 / (W1 * W1)
                V /= 1.0 - kappa
            valid = np.isfinite(M) & np.isfinite(V) & (n_pairs >= min_pair_batches)
            valid[np.arange(e - s), np.arange(s, e)] = False
            M *= M
            signal[s:e] = np.where(valid, M, 0.0).sum(axis=1)
            noise[s:e] = np.where(valid, V, 0.0).sum(axis=1)
            V *= kappa
            signal_db[s:e] = signal[s:e] - np.where(valid, V, 0.0).sum(axis=1)
            n_valid[s:e] = valid.sum(axis=1)
            del M, V, valid

        with np.errstate(invalid="ignore", divide="ignore"):
            scale = (n - 1) / n_valid
            signal *= scale
            noise *= scale
            signal_db *= scale
            R = signal / (signal + noise)
            ICC_db = signal_db / (signal_db + noise)
        return {
            "signal": signal,
            "noise": noise,
            "R": R,
            "signal_db": signal_db,
            "ICC_db": ICC_db,
            "n_partners": n_valid,
        }
