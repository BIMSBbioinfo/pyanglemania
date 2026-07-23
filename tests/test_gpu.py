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
from pyanglemania.datasets import example_adata  # noqa: E402
from pyanglemania.preprocessing._angles import (  # noqa: E402
    extract_angles,
    factorise_chunked,
    get_dstat,
    normalize_matrix,
)
from pyanglemania.preprocessing._batches import genes_passing_min_cells  # noqa: E402
from pyanglemania.preprocessing._select import prefilter_gene_pairs, rank_gene_pairs  # noqa: E402
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
    # StreamingZscoreStats accumulators are always host (numpy) now (fix 1 of
    # plans/gpu_memory_large_batches.md) -- update() must transparently pull a
    # cupy batch to host, giving the identical result as feeding the same
    # values in as numpy from the start.
    rng = np.random.default_rng(3)
    n_genes = 15
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(4)]
    for z in zscores:
        np.fill_diagonal(z, 0.0)
    weights = [0.7, 1.1, 1.0, 1.4]

    st_np = StreamingZscoreStats(n_genes)
    st_cp = StreamingZscoreStats(n_genes)
    for z, w in zip(zscores, weights):
        st_np.update(z, w)
        st_cp.update(cp.asarray(z), w)
    mean_np, sd_np, sn_np = st_np.finalize()
    mean_cp, sd_cp, sn_cp = st_cp.finalize()
    assert isinstance(mean_cp, np.ndarray)
    off_diag = ~np.eye(n_genes, dtype=bool)
    np.testing.assert_allclose(mean_np, mean_cp)
    np.testing.assert_allclose(sd_np[off_diag], sd_cp[off_diag])
    np.testing.assert_allclose(sn_np[off_diag], sn_cp[off_diag])


def test_rank_gene_pairs_cupy_native_path_matches_pandas():
    # Continuous random data makes exact ties between pairs essentially
    # impossible, so row order (not just rank *values*) should agree
    # exactly between the cupy-native (sort+searchsorted) and pandas
    # rank() implementations -- see _select.py::_min_rank and
    # plans/optimization.md #6d for why these are two different code
    # paths in the first place (pandas wins on CPU, cupy-native wins on
    # GPU at scale).
    n = 60
    rng = np.random.default_rng(7)
    mean_np = rng.normal(size=(n, n))
    sd_np = np.abs(rng.normal(size=(n, n))) + 0.1
    sn_np = np.abs(rng.normal(size=(n, n))) + 0.1
    mean_cp, sd_cp, sn_cp = cp.asarray(mean_np), cp.asarray(sd_np), cp.asarray(sn_np)

    for direction in ("both", "anticor", "cor"):
        pre_np = prefilter_gene_pairs(mean_np, sd_np, sn_np, zscore_mean_threshold=0.1,
                                       zscore_sn_threshold=0.1, verbose=False)
        ranked_np = rank_gene_pairs(pre_np, score_weights=(0.4, 0.6), direction=direction)

        pre_cp = prefilter_gene_pairs(mean_cp, sd_cp, sn_cp, zscore_mean_threshold=0.1,
                                       zscore_sn_threshold=0.1, verbose=False)
        ranked_cp = rank_gene_pairs(pre_cp, score_weights=(0.4, 0.6), direction=direction)

        np.testing.assert_array_equal(
            ranked_np["geneA_idx"].to_numpy(), ranked_cp["geneA_idx"].to_numpy()
        )
        np.testing.assert_array_equal(
            ranked_np["geneB_idx"].to_numpy(), ranked_cp["geneB_idx"].to_numpy()
        )
        np.testing.assert_allclose(
            ranked_np["rank"].to_numpy(), ranked_cp["rank"].to_numpy()
        )
        assert ranked_cp["rank"].is_monotonic_increasing


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
    df = adata.uns["anglemania"]["prefiltered_df"]
    assert np.isfinite(df[["mean_zscore", "sd_zscore", "sn_zscore"]].to_numpy()).all()


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
    df = adata.uns["anglemania"]["prefiltered_df"]
    assert np.isfinite(df[["mean_zscore", "sd_zscore", "sn_zscore"]].to_numpy()).all()


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
    df = adata.uns["anglemania"]["prefiltered_df"]
    assert np.isfinite(df[["mean_zscore", "sd_zscore", "sn_zscore"]].to_numpy()).all()


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
