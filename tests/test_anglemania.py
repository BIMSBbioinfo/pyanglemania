from __future__ import annotations

import pytest
from scipy import sparse as sp

import pyanglemania as pa
from pyanglemania.datasets import example_adata


def test_anglemania_sparse_X_matches_dense():
    dense = example_adata()
    sparse = example_adata()
    sparse.X = sp.csr_matrix(sparse.X)

    pa.pp.anglemania(dense, batch_key="batch", dataset_key="dataset", max_n_genes=15, verbose=False)
    pa.pp.anglemania(sparse, batch_key="batch", dataset_key="dataset", max_n_genes=15, verbose=False)

    assert list(dense.uns["anglemania"]["anglemania_genes"]) == list(
        sparse.uns["anglemania"]["anglemania_genes"]
    )


def test_anglemania_end_to_end_basic():
    adata = example_adata()
    out = pa.pp.anglemania(
        adata, batch_key="batch", dataset_key="dataset", max_n_genes=20, verbose=False
    )
    assert out is adata
    assert "anglemania_genes" in adata.var
    assert adata.var["anglemania_genes"].sum() == 20

    res = adata.uns["anglemania"]
    assert len(res["anglemania_genes"]) == 20
    assert set(res["anglemania_genes"]) <= set(adata.var_names)
    assert res["params"]["batch_key"] == "batch"

    df = res["prefiltered_df"]
    assert list(df.columns) == [
        "geneA",
        "geneB",
        "mean_zscore",
        "sd_zscore",
        "sn_zscore",
        "rank",
    ]
    assert df["rank"].is_monotonic_increasing


def test_anglemania_without_dataset_key():
    adata = example_adata()
    pa.pp.anglemania(adata, batch_key="batch", max_n_genes=10, verbose=False)
    assert adata.var["anglemania_genes"].sum() == 10


def test_anglemania_clamps_max_n_genes_to_intersect_size():
    adata = example_adata()
    pa.pp.anglemania(adata, batch_key="batch", max_n_genes=10_000, verbose=False)
    n_selected = adata.var["anglemania_genes"].sum()
    assert 0 < n_selected <= adata.n_vars
    assert adata.uns["anglemania"]["params"]["max_n_genes"] <= adata.n_vars


def test_anglemania_allow_missing_features_runs():
    adata = example_adata()
    pa.pp.anglemania(
        adata,
        batch_key="batch",
        dataset_key="dataset",
        allow_missing_features=True,
        min_samples_per_gene=1,
        max_n_genes=15,
        verbose=False,
    )
    assert adata.var["anglemania_genes"].sum() == 15


def test_anglemania_spearman_and_permute_nonzero_run():
    adata = example_adata()
    pa.pp.anglemania(
        adata,
        batch_key="batch",
        method="spearman",
        permutation_function="permute_nonzero",
        permute_row_or_column="row",
        max_n_genes=10,
        verbose=False,
    )
    assert adata.var["anglemania_genes"].sum() == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_key": "nope"},
        {"batch_key": "batch", "dataset_key": "nope"},
        {"batch_key": "batch", "max_n_genes": 0},
        {"batch_key": "batch", "method": "bogus"},
        {"batch_key": "batch", "min_cells_per_gene": 0},
        {"batch_key": "batch", "min_samples_per_gene": 0},
        {"batch_key": "batch", "permute_row_or_column": "bogus"},
        {"batch_key": "batch", "permutation_function": "bogus"},
        {"batch_key": "batch", "prefilter_threshold": 0},
        {"batch_key": "batch", "normalization_method": "bogus"},
        {"batch_key": "batch", "direction": "bogus"},
        {"batch_key": "batch", "score_weights": (0.4, 0.4, 0.2)},
    ],
)
def test_anglemania_param_validation(kwargs):
    adata = example_adata()
    with pytest.raises(ValueError):
        pa.pp.anglemania(adata, verbose=False, **kwargs)
