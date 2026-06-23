from __future__ import annotations

import numpy as np
import pytest

from pyanglemania.preprocessing._angles import (
    _shuffle_full,
    _shuffle_nonzero,
    extract_angles,
    factorise,
    get_dstat,
    normalize_matrix,
    permute_matrix,
)


def test_normalize_divide_by_total_counts_is_cp10k_log1p():
    X = np.array([[1.0, 3.0, 0.0], [2.0, 0.0, 2.0]])
    out = normalize_matrix(X, np, method="divide_by_total_counts")
    expected = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(out, expected)


def test_normalize_find_residuals_matches_per_gene_ols():
    rng = np.random.default_rng(0)
    X = rng.poisson(5, size=(50, 4)).astype(float)
    out = normalize_matrix(X, np, method="find_residuals")

    total = np.log1p(X.sum(axis=1))
    x_log = np.log1p(X)
    expected = np.empty_like(x_log)
    for g in range(X.shape[1]):
        slope, intercept = np.polyfit(total, x_log[:, g], 1)
        expected[:, g] = x_log[:, g] - (intercept + slope * total)
    np.testing.assert_allclose(out, expected, atol=1e-8)


def test_normalize_pflog1ppf_matches_paper_formula():
    rng = np.random.default_rng(6)
    X = rng.poisson(5, size=(30, 7)).astype(float)
    out = normalize_matrix(X, np, method="pflog1ppf")

    u = X / X.sum(axis=1, keepdims=True)
    log_u = np.log(u + 1.0)
    expected = log_u - log_u.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, atol=1e-8)
    # CLR centering: every cell's transformed row sums to ~0.
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-8)


def test_normalize_pflog1ppf_matches_scanpy_pipeline():
    anndata = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")

    rng = np.random.default_rng(7)
    X = rng.poisson(5, size=(30, 7)).astype(float)
    out = normalize_matrix(X, np, method="pflog1ppf")

    adata = anndata.AnnData(X.copy())
    sc.pp.normalize_total(adata, target_sum=1)
    sc.pp.log1p(adata)
    expected = adata.X - adata.X.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_normalize_matrix_rejects_unknown_method():
    with pytest.raises(ValueError):
        normalize_matrix(np.ones((2, 2)), np, method="scale_by_total_counts")


def test_shuffle_full_preserves_per_row_multiset():
    rng = np.random.default_rng(1)
    X = rng.poisson(3, size=(20, 8)).astype(float)
    out = _shuffle_full(X, axis=1, xp=np, rng=np.random.default_rng(0))
    assert out.shape == X.shape
    for row, row_out in zip(X, out):
        np.testing.assert_array_equal(np.sort(row), np.sort(row_out))
    # at least one row actually changed (extremely unlikely not to, with seed fixed)
    assert not np.array_equal(X, out)


def test_shuffle_nonzero_keeps_zero_positions_fixed():
    X = np.array([[0.0, 1.0, 0.0, 2.0, 3.0]] * 30)
    out = _shuffle_nonzero(X, axis=1, xp=np, rng=np.random.default_rng(0))
    assert (out[:, [0, 2]] == 0).all()
    for row in out:
        np.testing.assert_array_equal(np.sort(row[row != 0]), [1.0, 2.0, 3.0])
    # with 30 iid-shuffled rows of 3 nonzeros, expect more than one distinct order
    assert len({tuple(row[row != 0]) for row in out}) > 1


def test_shuffle_nonzero_axis0():
    X = np.array([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0], [0.0, 0.0]] * 10)
    out = _shuffle_nonzero(X, axis=0, xp=np, rng=np.random.default_rng(0))
    for col in range(X.shape[1]):
        zero_rows = X[:, col] == 0
        assert (out[zero_rows, col] == 0).all()
        np.testing.assert_array_equal(
            np.sort(X[~zero_rows, col]), np.sort(out[~zero_rows, col])
        )


def test_permute_matrix_dispatches_and_validates():
    X = np.ones((3, 3))
    with pytest.raises(ValueError):
        permute_matrix(X, axis=1, function="bogus", xp=np, rng=np.random.default_rng(0))


def test_extract_angles_cosine_matches_corrcoef():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 6))
    corr = extract_angles(X, "cosine", np)
    expected = np.corrcoef(X, rowvar=False)
    n = expected.shape[0]
    np.testing.assert_allclose(
        corr[~np.eye(n, dtype=bool)], expected[~np.eye(n, dtype=bool)], atol=1e-8
    )
    assert np.all(np.isnan(np.diag(corr)))


def test_extract_angles_spearman_uses_ranks():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 5))
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    expected = extract_angles(ranks, "cosine", np)
    actual = extract_angles(X, "spearman", np)
    np.testing.assert_allclose(actual, expected)


def test_extract_angles_phi_s_matches_vlr_vlp_formula():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 5))
    phi_s = extract_angles(X, "phi_s", np)

    n = X.shape[1]
    expected = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            vlr = np.var(X[:, i] - X[:, j], ddof=1)
            vlp = np.var(X[:, i] + X[:, j], ddof=1)
            expected[i, j] = vlr / vlp
    np.fill_diagonal(expected, np.nan)

    np.testing.assert_allclose(phi_s[~np.eye(n, dtype=bool)], expected[~np.eye(n, dtype=bool)])
    assert np.all(np.isnan(np.diag(phi_s)))
    np.testing.assert_allclose(phi_s, phi_s.T, equal_nan=True)


def test_extract_angles_phi_s_near_zero_for_proportional_genes():
    rng = np.random.default_rng(9)
    base = rng.normal(size=100)
    # gene B is gene A plus a constant shift in log-ratio space, i.e. a
    # perfectly proportional pair: their log-ratio is constant (var ~ 0).
    noise = rng.normal(scale=1e-6, size=100)
    X = np.column_stack([base, base + 3.0 + noise, rng.normal(size=100)])

    phi_s = extract_angles(X, "phi_s", np)
    assert phi_s[0, 1] < 1e-6
    assert phi_s[0, 2] > phi_s[0, 1]


def test_extract_angles_rejects_unknown_method():
    with pytest.raises(ValueError):
        extract_angles(np.ones((4, 3)), "bogus", np)


def test_get_dstat_ignores_diagonal():
    corr = np.array(
        [
            [np.nan, 0.2, 0.4],
            [0.2, np.nan, 0.6],
            [0.4, 0.6, np.nan],
        ]
    )
    mean, sd = get_dstat(corr, np)
    np.testing.assert_allclose(mean, [0.3, 0.4, 0.5])
    assert np.all(sd > 0)


def test_factorise_returns_finite_zero_diagonal():
    rng = np.random.default_rng(4)
    X = rng.poisson(5, size=(80, 10)).astype(float)
    z = factorise(X, np, seed=1)
    assert z.shape == (10, 10)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(np.diag(z), 0.0)


def test_factorise_is_seed_reproducible():
    rng = np.random.default_rng(5)
    X = rng.poisson(5, size=(80, 10)).astype(float)
    z1 = factorise(X.copy(), np, seed=7)
    z2 = factorise(X.copy(), np, seed=7)
    np.testing.assert_allclose(z1, z2)


def test_factorise_phi_s_with_pflog1ppf_returns_finite_zero_diagonal():
    rng = np.random.default_rng(10)
    X = rng.poisson(5, size=(80, 10)).astype(float)
    z = factorise(X, np, seed=1, method="phi_s", normalization_method="pflog1ppf")
    assert z.shape == (10, 10)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(np.diag(z), 0.0)
