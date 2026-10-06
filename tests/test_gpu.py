"""GPU (cupy) parity checks.

Skipped unless cupy is importable *and* a CUDA device is actually reachable
-- which, in this dev sandbox, depends on env vars unrelated to the package
itself (see CLAUDE.md): cupy's JIT kernel compiler needs `CUDA_PATH` pointed
at headers new enough to define types its bundled CCCL headers reference
(e.g. `__nv_fp8_e8m0`), which the system's default CUDA 12.3 headers lack.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _has_gpu = False

pytestmark = pytest.mark.skipif(not _has_gpu, reason="no reachable CUDA device")

import cupyx.scipy.sparse as csp  # noqa: E402

import pyanglemania as pa  # noqa: E402
from pyanglemania.preprocessing._angles import (  # noqa: E402
    extract_angles,
    factorise_chunked,
    get_dstat,
    normalize_matrix,
)
from pyanglemania.preprocessing._batches import (  # noqa: E402
    genes_passing_min_cells,
    lognorm_column_sums,
)
from pyanglemania.preprocessing._stats import StreamingZscoreStats  # noqa: E402


def test_normalize_matrix_matches_numpy():
    rng = np.random.default_rng(0)
    X_np = rng.poisson(5, size=(100, 30)).astype(np.float32)
    X_cp = cp.asarray(X_np)
    for method in ("divide_by_total_counts", "find_residuals", "pflog1ppf"):
        out_np = normalize_matrix(X_np, np, method)
        out_cp = cp.asnumpy(normalize_matrix(X_cp, cp, method))
        np.testing.assert_allclose(out_np, out_cp, atol=1e-4)


def test_extract_angles_cosine_matches_numpy():
    rng = np.random.default_rng(1)
    X_np = rng.normal(size=(80, 12)).astype(np.float32)
    X_cp = cp.asarray(X_np)
    corr_np = extract_angles(X_np, "cosine", np)
    corr_cp = cp.asnumpy(extract_angles(X_cp, "cosine", cp))
    mask = ~np.isnan(corr_np)
    np.testing.assert_allclose(corr_np[mask], corr_cp[mask], atol=1e-4)
    assert np.array_equal(np.isnan(corr_np), np.isnan(corr_cp))


def test_extract_angles_phi_s_matches_numpy():
    rng = np.random.default_rng(11)
    X_np = rng.normal(size=(80, 12)).astype(np.float32)
    X_cp = cp.asarray(X_np)
    phi_s_np = extract_angles(X_np, "phi_s", np)
    phi_s_cp = cp.asnumpy(extract_angles(X_cp, "phi_s", cp))
    mask = ~np.isnan(phi_s_np)
    np.testing.assert_allclose(phi_s_np[mask], phi_s_cp[mask], atol=1e-4)
    assert np.array_equal(np.isnan(phi_s_np), np.isnan(phi_s_cp))


def test_get_dstat_matches_numpy():
    rng = np.random.default_rng(2)
    corr_np = rng.normal(size=(20, 20)).astype(np.float32)
    np.fill_diagonal(corr_np, np.nan)
    corr_cp = cp.asarray(corr_np)
    mean_np, sd_np = get_dstat(corr_np, np)
    mean_cp, sd_cp = get_dstat(corr_cp, cp)
    np.testing.assert_allclose(mean_np, cp.asnumpy(mean_cp), atol=1e-4)
    np.testing.assert_allclose(sd_np, cp.asnumpy(sd_cp), atol=1e-4)


def test_streaming_stats_accepts_cupy_batches_same_as_numpy():
    # StreamingZscoreStats accumulators are always host (numpy) (fix 1 of
    # plans/gpu_memory_large_batches.md) -- update() must transparently pull a
    # cupy batch to host, giving the identical result as feeding the same
    # values in as numpy from the start.
    rng = np.random.default_rng(3)
    n_genes = 15
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(4)]
    present = [rng.random(n_genes) > 0.2 for _ in range(4)]
    weights = [0.7, 1.1, 1.0, 1.4]

    st_np = StreamingZscoreStats(n_genes)
    st_cp = StreamingZscoreStats(n_genes)
    for z, w, p in zip(zscores, weights, present):
        idx = np.flatnonzero(p)
        st_np.update(z[np.ix_(idx, idx)], w, idx)
        st_cp.update(cp.asarray(z[np.ix_(idx, idx)]), w, idx)
    got_np, got_cp = st_np.gene_scores(), st_cp.gene_scores()
    for k in ("signal", "noise", "R", "signal_db", "ICC_db"):
        assert isinstance(got_cp[k], np.ndarray)
        np.testing.assert_allclose(got_np[k], got_cp[k])


def test_genes_passing_min_cells_sparse_gpu():
    # Regression check: cupyx sparse has no `.getnnz(axis=...)`, and a
    # sparse `X != 0` compiles a much heavier kernel than dense ops --
    # the bincount-based implementation must avoid both.
    X_np = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32)
    X_cp = csp.csr_matrix(cp.asarray(X_np))
    mask = genes_passing_min_cells(X_cp, 2, cp, csp)
    np.testing.assert_array_equal(mask, [True, False, True])


@pytest.mark.parametrize("sparse", [False, True])
def test_anglemania_runs_on_gpu_backed_adata(sparse):
    adata = pa.datasets.example_adata()
    adata.X = cp.asarray(adata.X)
    if sparse:
        adata.X = csp.csr_matrix(adata.X)

    pa.pp.anglemania(adata, batch_key="batch", dataset_key="dataset", max_n_genes=15, verbose=False)

    assert adata.var["anglemania_genes"].sum() == 15
    scored = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    assert np.isfinite(scored[["anglemania_signal", "anglemania_R",
                               "anglemania_WSN"]].to_numpy()).all()


def test_anglemania_runs_on_gpu_backed_adata_with_phi_s():
    adata = pa.datasets.example_adata()
    adata.X = cp.asarray(adata.X)

    pa.pp.anglemania(
        adata,
        batch_key="batch",
        dataset_key="dataset",
        max_n_genes=15,
        method="phi_s",
        normalization_method="pflog1ppf",
        verbose=False,
    )

    assert adata.var["anglemania_genes"].sum() == 15
    scored = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    assert np.isfinite(scored[["anglemania_signal", "anglemania_R",
                               "anglemania_WSN"]].to_numpy()).all()


@pytest.mark.parametrize(
    "method,normalization_method", [("cosine", "divide_by_total_counts"), ("phi_s", "pflog1ppf")]
)
def test_factorise_chunked_matches_numpy_on_gpu(method, normalization_method):
    rng = np.random.default_rng(40)
    X_np = rng.poisson(5, size=(200, 20)).astype(np.float32)
    X_cp = cp.asarray(X_np)
    common_genes = [f"g{i}" for i in range(20)]

    z_np = factorise_chunked(
        X_np, common_genes, common_genes, np, cell_chunk_size=37, seed=1,
        method=method, normalization_method=normalization_method,
    )
    z_cp = cp.asnumpy(
        factorise_chunked(
            X_cp, common_genes, common_genes, cp, cell_chunk_size=37, seed=1,
            method=method, normalization_method=normalization_method,
        )
    )
    assert z_np.shape == z_cp.shape == (20, 20)
    assert np.isfinite(z_np).all() and np.isfinite(z_cp).all()
    # Not a bit-exact match: chunked cupy/numpy RNGs draw permutation keys
    # independently (see factorise_chunked's docstring) -- just check both
    # are well-formed, finite, zero-diagonal outputs of the right shape.
    np.testing.assert_allclose(np.diag(z_np), 0.0)
    np.testing.assert_allclose(np.diag(z_cp), 0.0)


def test_anglemania_cell_chunk_size_runs_on_gpu_sparse():
    adata = pa.datasets.example_adata()
    adata.X = csp.csr_matrix(cp.asarray(adata.X))

    pa.pp.anglemania(
        adata,
        batch_key="batch",
        dataset_key="dataset",
        max_n_genes=15,
        method="phi_s",
        normalization_method="pflog1ppf",
        cell_chunk_size=97,
        verbose=False,
    )

    assert adata.var["anglemania_genes"].sum() == 15
    scored = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    assert np.isfinite(scored[["anglemania_signal", "anglemania_R",
                               "anglemania_WSN"]].to_numpy()).all()


def test_anglemania_cell_chunk_size_gpu_single_chunk_matches_unchunked():
    unchunked = pa.datasets.example_adata()
    unchunked.X = cp.asarray(unchunked.X)
    chunked = pa.datasets.example_adata()
    chunked.X = cp.asarray(chunked.X)

    pa.pp.anglemania(
        unchunked, batch_key="batch", dataset_key="dataset", max_n_genes=15, verbose=False
    )
    pa.pp.anglemania(
        chunked, batch_key="batch", dataset_key="dataset", max_n_genes=15,
        cell_chunk_size=10_000, verbose=False,
    )
    assert list(unchunked.uns["anglemania"]["anglemania_genes"]) == list(
        chunked.uns["anglemania"]["anglemania_genes"]
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_lognorm_column_sums_matches_numpy(sparse):
    # (End-to-end CPU/GPU scores aren't comparable: cupy's and numpy's
    # default_rng draw different permutations, hence different nulls.)
    from scipy import sparse as sp

    rng = np.random.default_rng(5)
    X_np = rng.poisson(1, size=(120, 25)).astype(np.float32)
    X_np[3] = 0  # an all-zero cell
    X_cp = cp.asarray(X_np)
    if sparse:
        X_np, X_cp = sp.csr_matrix(X_np), csp.csr_matrix(X_cp)
    got_np = lognorm_column_sums(X_np, np, sp)
    got_cp = lognorm_column_sums(X_cp, cp, csp)
    assert isinstance(got_cp, np.ndarray)
    np.testing.assert_allclose(got_np, got_cp, rtol=1e-6)


@pytest.mark.parametrize("mode", ["mask", "zero"])
def test_anglemania_missing_modes_run_on_gpu_sparse(mode):
    adata = pa.datasets.example_adata()
    adata.obs["batch"] = (
        adata.obs["batch"].astype(str) + "_" + (np.arange(adata.n_obs) % 2).astype(str)
    ).astype("category")
    X = adata.X.copy()
    rng = np.random.default_rng(0)
    for b in adata.obs["batch"].cat.categories:
        X[np.ix_((adata.obs["batch"] == b).to_numpy(), rng.choice(adata.n_vars, 60, False))] = 0
    adata.X = csp.csr_matrix(cp.asarray(X))
    pa.pp.anglemania(adata, batch_key="batch", allow_missing_features=True,
                     missing_mode=mode, cell_chunk_size=97, max_n_genes=15, verbose=False)
    assert adata.var["anglemania_genes"].sum() == 15
    scored = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    assert (scored["anglemania_n_batches_present"] < 4).any()
    assert np.isfinite(scored[["anglemania_ICC_db", "anglemania_WSN"]].to_numpy()).all()
