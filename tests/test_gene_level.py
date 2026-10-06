from __future__ import annotations

import numpy as np

from pyanglemania.preprocessing._gene_level import binned_zscores, top_genes


def test_binned_zscores_are_standardized_within_each_bin():
    rng = np.random.default_rng(0)
    n = 200
    mean_lognorm = rng.gamma(2, size=n)
    signal = 10 ** (mean_lognorm + rng.normal(size=n))  # expression-confounded
    R = rng.uniform(0.1, 0.5, n)
    out = binned_zscores(mean_lognorm, signal, R, n_bins=10, signal_R_weights=(0.3, 0.7))

    assert out["expr_bin"].nunique() == 10
    assert (out.groupby("expr_bin").size() == 20).all()
    for col in ("signal_z_binned", "R_z_binned"):
        g = out.groupby("expr_bin")[col]
        np.testing.assert_allclose(g.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(g.std(ddof=1), 1)
    np.testing.assert_allclose(
        out["WSN"], 0.3 * out["signal_z_binned"] + 0.7 * out["R_z_binned"]
    )
    # bins are expression-ordered
    bin_max = [mean_lognorm[out["expr_bin"] == b].max() for b in range(10)]
    assert bin_max == sorted(bin_max)


def test_binned_zscores_leave_unscorable_genes_out():
    mean_lognorm = np.arange(8, dtype=float)
    signal = np.array([1, 2, 0, 4, np.nan, 6, 7, 8], dtype=float)
    R = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7, 0.8])
    out = binned_zscores(mean_lognorm, signal, R, n_bins=2)
    bad = [2, 3, 4]
    assert out.loc[bad].isna().all().all()
    assert out.drop(index=bad).notna().all().all()


def test_top_genes_orders_best_first_and_skips_nan():
    genes = ["a", "b", "c", "d", "e"]
    scores = [0.5, np.nan, 2.0, 0.5, -1.0]
    assert top_genes(scores, genes, 3) == ["c", "a", "d"]
    assert top_genes(scores, genes, None) == ["c", "a", "d", "e"]
