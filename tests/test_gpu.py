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
from pyanglemania.preprocessing._angles import extract_angles, get_dstat, normalize_matrix  # noqa: E402
from pyanglemania.preprocessing._batches import genes_passing_min_cells  # noqa: E402
from pyanglemania.preprocessing._stats import StreamingZscoreStats  # noqa: E402


def test_normalize_matrix_matches_numpy():
    rng = np.random.default_rng(0)
    X_np = rng.poisson(5, size=(100, 30)).astype(np.float32)
    X_cp = cp.asarray(X_np)
    for method in ("divide_by_total_counts", "find_residuals"):
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


def test_get_dstat_matches_numpy():
    rng = np.random.default_rng(2)
    corr_np = rng.normal(size=(20, 20)).astype(np.float32)
    np.fill_diagonal(corr_np, np.nan)
    corr_cp = cp.asarray(corr_np)
    mean_np, sd_np = get_dstat(corr_np, np)
    mean_cp, sd_cp = get_dstat(corr_cp, cp)
    np.testing.assert_allclose(mean_np, cp.asnumpy(mean_cp), atol=1e-4)
    np.testing.assert_allclose(sd_np, cp.asnumpy(sd_cp), atol=1e-4)


def test_streaming_stats_match_numpy_exactly():
    rng = np.random.default_rng(3)
    n_genes = 15
    zscores = [rng.normal(size=(n_genes, n_genes)) for _ in range(4)]
    for z in zscores:
        np.fill_diagonal(z, 0.0)
    weights = [0.7, 1.1, 1.0, 1.4]

    st_np = StreamingZscoreStats(n_genes, np)
    st_cp = StreamingZscoreStats(n_genes, cp)
    for z, w in zip(zscores, weights):
        st_np.update(z, w)
        st_cp.update(cp.asarray(z), w)
    mean_np, sd_np, sn_np = st_np.finalize()
    mean_cp, sd_cp, sn_cp = (cp.asnumpy(a) for a in st_cp.finalize())
    off_diag = ~np.eye(n_genes, dtype=bool)
    np.testing.assert_allclose(mean_np, mean_cp)
    np.testing.assert_allclose(sd_np[off_diag], sd_cp[off_diag])
    np.testing.assert_allclose(sn_np[off_diag], sn_cp[off_diag])


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
