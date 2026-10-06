from __future__ import annotations

import numpy as np
import pytest

from pyanglemania.preprocessing._stats import StreamingZscoreStats


def _naive_gene_scores(zscore_list, weights, present_list, zero_fill=False, min_pair_batches=2):
    """Stack every batch (full, padded matrices), NaN out absent genes, reduce in one go.

    ``zero_fill=True`` instead counts absent entries as z = 0 present in
    every batch (R's / ``missing_mode="zero"`` behavior).
    """
    n = zscore_list[0].shape[0]
    Z = []
    for z, present in zip(zscore_list, present_list):
        z = (z + z.T) / 2
        fill = 0.0 if zero_fill else np.nan
        z[~present, :] = fill
        z[:, ~present] = fill
        Z.append(z)
    Z = np.stack(Z)
    w = np.asarray(weights)[:, None, None]
    m = ~np.isnan(Z)
    W1 = (w * m).sum(0)
    W2 = (w * w * m).sum(0)
    M = (w * np.where(m, Z, 0)).sum(0) / W1
    kappa = W2 / W1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        V = (w * np.where(m, (Z - M) ** 2, 0)).sum(0) / W1 / (1 - kappa)
    valid = (m.sum(0) >= min_pair_batches) & ~np.eye(n, dtype=bool)
    f = (n - 1) / valid.sum(1)
    signal = np.where(valid, M**2, 0).sum(1) * f
    noise = np.where(valid, V, 0).sum(1) * f
    signal_db = np.where(valid, M**2 - kappa * V, 0).sum(1) * f
    return {
        "signal": signal,
        "noise": noise,
        "R": signal / (signal + noise),
        "signal_db": signal_db,
        "ICC_db": signal_db / (signal_db + noise),
        "n_partners": valid.sum(1),
        "n_pairs": m.sum(0),
    }


def _random_batches(seed, n_genes, n_batches, missing=False):
    rng = np.random.default_rng(seed)
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(n_batches)]
    weights = list(rng.uniform(0.5, 1.5, n_batches))
    if missing:
        present = [rng.random(n_genes) > 0.3 for _ in range(n_batches)]
    else:
        present = [np.ones(n_genes, dtype=bool)] * n_batches
    return zscores, weights, present


def _feed(zscores, weights, present, n_genes, order=None):
    """Masked feeding: each batch's z restricted to its present genes."""
    stats = StreamingZscoreStats(n_genes)
    for i in order if order is not None else range(len(zscores)):
        idx = np.flatnonzero(present[i])
        stats.update(zscores[i][np.ix_(idx, idx)], weights[i], idx)
    return stats


KEYS = ("signal", "noise", "R", "signal_db", "ICC_db")


@pytest.mark.parametrize("missing", [False, True])
def test_streaming_matches_naive_batch_stacking(missing):
    zscores, weights, present = _random_batches(0, 12, 6, missing)
    got = _feed(zscores, weights, present, 12).gene_scores()
    exp = _naive_gene_scores(zscores, weights, present)
    for k in KEYS:
        np.testing.assert_allclose(got[k], exp[k], err_msg=k)
    np.testing.assert_array_equal(got["n_partners"], exp["n_partners"])


def test_zero_mode_feeding_matches_naive_zero_fill():
    # missing_mode="zero": zero-padded full matrices with index=None.
    zscores, weights, present = _random_batches(5, 10, 5, missing=True)
    stats = StreamingZscoreStats(10)
    for z, w, p in zip(zscores, weights, present):
        z = z.copy()
        z[~p, :] = 0
        z[:, ~p] = 0
        stats.update(z, w)
    got = stats.gene_scores()
    exp = _naive_gene_scores(zscores, weights, present, zero_fill=True)
    for k in KEYS:
        np.testing.assert_allclose(got[k], exp[k], err_msg=k)


def test_complete_data_matches_plain_weighted_moments():
    # With nothing missing, signal/noise are plain row sums of the weighted
    # mean^2 and unbiased weighted variance (R's get_list_stats formulas),
    # and ICC_db is a monotone (Moebius) transform of R with one scalar kappa.
    zscores, weights, _ = _random_batches(1, 10, 5)
    stats = StreamingZscoreStats(10)
    for z, w in zip(zscores, weights):
        stats.update(z, w)
    got = stats.gene_scores()

    Z = np.stack([(z + z.T) / 2 for z in zscores])
    w = np.asarray(weights)
    mean = np.tensordot(w, Z, 1) / w.sum()
    var = np.tensordot(w, (Z - mean) ** 2, 1) / (w.sum() - (w * w).sum() / w.sum())
    off = ~np.eye(10, dtype=bool)
    np.testing.assert_allclose(got["signal"], np.where(off, mean**2, 0).sum(1))
    np.testing.assert_allclose(got["noise"], np.where(off, var, 0).sum(1))
    kappa = (w * w).sum() / w.sum() ** 2
    R = got["R"]
    np.testing.assert_allclose(got["ICC_db"], (R * (1 + kappa) - kappa) / (1 - kappa + kappa * R))


def test_pair_counts_follow_which_batches_overlap():
    # genes 0 and 1 are both in 2 of 4 batches, but never the same ones:
    # per-gene counts (2, 2) don't determine n_01 = 0.
    present = np.ones((4, 5), dtype=bool)
    present[[0, 1], 0] = False
    present[[2, 3], 1] = False
    present[[0], 2] = False
    stats = StreamingZscoreStats(5)
    for p in present:
        idx = np.flatnonzero(p)
        stats.update(np.zeros((len(idx), len(idx))), 1.0, idx)
    n = stats.pair_counts()
    np.testing.assert_array_equal(n, present.T.astype(int) @ present.astype(int))
    assert n[0, 1] == 0 and n[0, 2] == 2 and n[1, 2] == 1 and n[3, 4] == 4
    np.testing.assert_array_equal(stats.pair_counts(slice(1, 3)), n[1:3])
    # gene 0 shares >= 2 batches with genes 2, 3, 4; gene 1 only with 3, 4
    np.testing.assert_array_equal(stats.gene_scores()["n_partners"][:2], [3, 2])


def test_absent_genes_ignore_whatever_the_batch_holds_there():
    # Feeding full padded matrices restricted to the present block, or the
    # present block directly, is the same thing -- whatever sat in the
    # absent rows/columns never reaches the accumulators.
    zscores, weights, present = _random_batches(2, 9, 4, missing=True)
    a = _feed(zscores, weights, present, 9)
    rng = np.random.default_rng(3)
    junk = [z + np.where(p[:, None] & p[None, :], 0, rng.normal(size=z.shape) * 100)
            for z, p in zip(zscores, present)]
    b = _feed(junk, weights, present, 9)
    ra, rb = a.gene_scores(), b.gene_scores()
    for k in KEYS:
        np.testing.assert_allclose(ra[k], rb[k])


def test_does_not_depend_on_batch_order_or_chunk_size():
    zscores, weights, present = _random_batches(4, 11, 5, missing=True)
    ref = _feed(zscores, weights, present, 11).gene_scores()
    for order, chunk in [([3, 0, 4, 1, 2], 2048), (None, 3), (None, 1)]:
        got = _feed(zscores, weights, present, 11, order).gene_scores(chunk_size=chunk)
        for k in KEYS:
            np.testing.assert_allclose(got[k], ref[k])


# --- planted-module simulation, ported from the highRisk_NB prototype's tests
B, P, MOD = 10, 300, 20


def _planted():
    rng = np.random.default_rng(0)
    zscores = []
    for b in range(B):
        z = rng.normal(size=(P, P))
        z[:MOD, :MOD] += 5.0  # invariant module
        z[MOD : 2 * MOD, MOD : 2 * MOD] += 5.0 * (1 if b % 2 else -1)  # sign-flipping module
        zscores.append(z)
    return zscores, list(rng.uniform(0.5, 2.0, B))


def test_planted_modules():
    zscores, w = _planted()
    sc = _feed(zscores, w, [np.ones(P, dtype=bool)] * B, P).gene_scores()
    flip, bg = slice(MOD, 2 * MOD), slice(2 * MOD, None)
    kappa = np.sum(np.square(w)) / np.sum(w) ** 2
    assert np.median(sc["ICC_db"][:MOD]) > 0.25  # ~20 of 299 partners carry signal
    assert np.median(sc["noise"][flip]) > 3 * np.median(sc["noise"][bg])
    assert np.median(sc["signal_db"][flip]) < 0.2 * np.median(sc["noise"][flip])
    assert abs(np.median(sc["ICC_db"][bg])) < 0.05  # pure-noise genes: ICC_db ~ 0
    assert abs(np.median(sc["R"][bg]) - kappa) < 0.05  # raw R floor ~ kappa


def test_missing_masked_unbiased_zero_fill_biased():
    # Invariant-module genes 0..7 absent from 4 of 10 batches.
    zscores, w = _planted()
    gone = np.arange(8)
    present = [np.ones(P, dtype=bool) for _ in range(B)]
    for b in (0, 3, 5, 8):
        present[b][gone] = False

    full = _feed(zscores, w, [np.ones(P, dtype=bool)] * B, P).gene_scores()
    masked_stats = _feed(zscores, w, present, P)
    masked = masked_stats.gene_scores()
    zero = StreamingZscoreStats(P)
    for z, wb, p in zip(zscores, w, present):
        z = z.copy()
        z[~p, :] = 0
        z[:, ~p] = 0
        zero.update(z, wb)
    zf = zero.gene_scores()

    n = masked_stats.pair_counts()
    assert n[0, 1] == 6 and n[0, 50] == 6 and n[50, 60] == B
    med = {k: np.median(d["ICC_db"][gone]) for k, d in (("full", full), ("mask", masked),
                                                         ("zero", zf))}
    assert abs(med["mask"] - med["full"]) < 0.05
    assert med["zero"] < med["full"] - 0.1
    # (full row sums: diluted by ~280 background partners, hence < the
    # prototype's 3x on within-module top-k sums)
    assert np.median(zf["noise"][gone]) > 1.3 * np.median(masked["noise"][gone])
    assert np.median(zf["signal"][gone]) < 0.6 * np.median(masked["signal"][gone])


def _gene_label_null(zscores, w, present, seed):
    """ICC_db/R with gene labels permuted within each batch's present genes."""
    rng = np.random.default_rng(seed)
    stats = StreamingZscoreStats(P)
    for z, wb, p in zip(zscores, w, present):
        idx = np.flatnonzero(p)
        perm = rng.permutation(idx)
        stats.update(z[np.ix_(perm, perm)], wb, idx)
    return stats.gene_scores()


def test_null_calibration_with_and_without_missing():
    zscores, w = _planted()
    complete = [np.ones(P, dtype=bool)] * B
    present = [np.ones(P, dtype=bool) for _ in range(B)]
    for b in (0, 1, 2):
        present[b][:120] = False
    kappa = np.sum(np.square(w)) / np.sum(w) ** 2

    for pres in (complete, present):
        null = [_gene_label_null(zscores, w, pres, s) for s in range(3)]
        icc = np.concatenate([n["ICC_db"] for n in null])
        assert abs(np.nanmedian(icc)) < 0.03
    # complete: raw R's null sits at kappa
    null = [_gene_label_null(zscores, w, complete, s) for s in range(3)]
    assert abs(np.median(np.concatenate([n["R"] for n in null])) - kappa) < 0.05


def test_update_validates_index():
    stats = StreamingZscoreStats(4)
    with pytest.raises(ValueError):
        stats.update(np.zeros((3, 3)), 1.0, [0, 1])
    with pytest.raises(ValueError):
        stats.update(np.zeros((2, 2)), 1.0, [2, 1])


def test_needs_two_batches():
    stats = StreamingZscoreStats(4)
    stats.update(np.zeros((4, 4)), 1.0)
    with pytest.raises(ValueError):
        stats.gene_scores()


def test_streaming_never_materializes_more_than_one_batch_at_a_time():
    # Each update() only ever touches the running accumulators (which are
    # the size of a single batch's matrix) plus the one batch passed in --
    # callers are free to discard a batch's z-score matrix right after
    # update() returns, which is the whole point of streaming.
    n_genes = 5
    stats = StreamingZscoreStats(n_genes)
    accumulator_attrs = [v for v in vars(stats).values() if isinstance(v, np.ndarray)]
    assert all(a.shape == (n_genes, n_genes) for a in accumulator_attrs)
    assert len(accumulator_attrs) == 2  # wz_sum and wz2_sum only
