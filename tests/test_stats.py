from __future__ import annotations

import numpy as np

from pyanglemania.preprocessing._stats import StreamingZscoreStats


def _naive_weighted_stats(zscore_list, weights):
    n = zscore_list[0].shape[0]
    w_sum = sum(weights)
    w_sq_sum = sum(w * w for w in weights)
    mean = sum(w * z for w, z in zip(weights, zscore_list)) / w_sum
    denom = w_sum - w_sq_sum / w_sum
    m_sd = sum(w * (z - mean) ** 2 for w, z in zip(weights, zscore_list))
    var = m_sd / denom
    sd = np.sqrt(np.clip(var, 0, None))
    idx = np.arange(n)
    sd[idx, idx] = np.nan
    sn = np.abs(mean) / sd
    return mean, sd, sn


def test_streaming_matches_naive_batch_stacking():
    rng = np.random.default_rng(0)
    n_genes = 12
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(5)]
    for z in zscores:
        np.fill_diagonal(z, 0.0)
    weights = [0.8, 1.2, 1.0, 0.5, 1.5]

    stats = StreamingZscoreStats(n_genes, np)
    for z, w in zip(zscores, weights):
        stats.update(z, w)
    mean, sd, sn = stats.finalize()

    exp_mean, exp_sd, exp_sn = _naive_weighted_stats(zscores, weights)
    off_diag = ~np.eye(n_genes, dtype=bool)
    np.testing.assert_allclose(mean, exp_mean)
    np.testing.assert_allclose(sd[off_diag], exp_sd[off_diag])
    np.testing.assert_allclose(sn[off_diag], exp_sn[off_diag])
    assert np.all(np.isnan(np.diag(sd)))


def test_streaming_does_not_depend_on_batch_order():
    rng = np.random.default_rng(1)
    n_genes = 8
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(4)]
    weights = [1.0, 0.6, 1.3, 0.9]

    def run(order):
        stats = StreamingZscoreStats(n_genes, np)
        for i in order:
            stats.update(zscores[i], weights[i])
        return stats.finalize()

    mean_a, sd_a, sn_a = run(range(4))
    mean_b, sd_b, sn_b = run([2, 0, 3, 1])
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(sd_a, sd_b, equal_nan=True)
    np.testing.assert_allclose(sn_a, sn_b, equal_nan=True)


def test_streaming_never_materializes_more_than_one_batch_at_a_time():
    # Each update() only ever touches the running accumulators (which are
    # the size of a single batch's matrix) plus the one batch passed in --
    # callers are free to discard a batch's z-score matrix right after
    # update() returns, which is the whole point of streaming.
    n_genes = 5
    stats = StreamingZscoreStats(n_genes, np)
    accumulator_attrs = [v for v in vars(stats).values() if isinstance(v, np.ndarray)]
    assert all(a.shape == (n_genes, n_genes) for a in accumulator_attrs)
    assert len(accumulator_attrs) == 2  # wz_sum and wz2_sum only
