from __future__ import annotations

import numpy as np
import pandas as pd
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

    assert "prefiltered_df" not in res

    scores = adata.var.loc[res["intersect_genes"], "anglemania_R_z_binned"]
    assert scores.notna().all()
    # default score="R": selected genes are the top of R_z_binned, best first
    assert res["anglemania_genes"] == list(scores.sort_values(ascending=False).index[:20])
    for col in ("signal", "noise", "R", "mean_lognorm", "n_batches_present",
                "expr_bin", "signal_z_binned", "WSN", "ICC_db_z_binned"):
        assert f"anglemania_{col}" in adata.var
    R = adata.var["anglemania_R"]
    assert ((R > 0) & (R < 1)).all()


def test_anglemania_score_WSN():
    adata = example_adata()
    pa.pp.anglemania(adata, batch_key="batch", max_n_genes=10, score="WSN", verbose=False)
    wsn = adata.var["anglemania_WSN"]
    assert adata.uns["anglemania"]["anglemania_genes"] == list(
        wsn.sort_values(ascending=False).index[:10]
    )
    np.testing.assert_allclose(
        wsn, 0.5 * adata.var["anglemania_signal_z_binned"] + 0.5 * adata.var["anglemania_R_z_binned"]
    )


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


def _with_missing_genes():
    # 4 batches; 60 random genes zeroed out (hence "absent") in each one
    adata = example_adata()
    adata.obs["batch"] = (
        adata.obs["batch"].astype(str) + "_" + (np.arange(adata.n_obs) % 2).astype(str)
    ).astype("category")
    X = adata.X.copy()
    rng = np.random.default_rng(0)
    for b in adata.obs["batch"].cat.categories:
        gone = rng.choice(adata.n_vars, 60, replace=False)
        X[np.ix_((adata.obs["batch"] == b).to_numpy(), gone)] = 0
    adata.X = X
    return adata


def test_anglemania_allow_missing_features_runs():
    adata = _with_missing_genes()
    pa.pp.anglemania(
        adata,
        batch_key="batch",
        allow_missing_features=True,
        max_n_genes=15,
        verbose=False,
    )
    assert adata.var["anglemania_genes"].sum() == 15
    scored = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    assert scored["anglemania_n_batches_present"].min() == 2
    assert scored["anglemania_n_batches_present"].max() == 4
    assert scored["anglemania_WSN"].notna().all()
    # genes seen in only one batch are outside the universe, hence unscored
    outside = ~adata.var_names.isin(scored.index)
    assert outside.any() and adata.var.loc[outside, "anglemania_WSN"].isna().all()


def _manual_gene_scores(adata, mode, cell_chunk_size=None):
    """Per-batch factorise + StreamingZscoreStats by hand, for one missing_mode."""
    from pyanglemania.preprocessing._angles import factorise, factorise_chunked
    from pyanglemania.preprocessing._batches import align_to_common_genes
    from pyanglemania.preprocessing._stats import StreamingZscoreStats

    common = adata.uns["anglemania"]["intersect_genes"]
    presence = adata.uns["anglemania"]["presence"]
    stats = StreamingZscoreStats(len(common))
    for label, row in presence.iterrows():
        X_b = adata.X[(adata.obs["anglemania_batch"] == label).to_numpy()]
        batch_genes = list(adata.var_names[(X_b != 0).sum(0) >= 1])
        index = np.flatnonzero(row.to_numpy()) if mode == "mask" else None
        genes = [common[i] for i in index] if mode == "mask" else common
        if cell_chunk_size is None:
            z = factorise(align_to_common_genes(X_b[:, adata.var_names.isin(batch_genes)],
                                                batch_genes, genes, np), np)
        else:
            z = factorise_chunked(X_b[:, adata.var_names.isin(batch_genes)], batch_genes,
                                  genes, np, cell_chunk_size)
        stats.update(z, 1.0, index)
    return stats.gene_scores()


@pytest.mark.parametrize("mode", ["mask", "zero"])
@pytest.mark.parametrize("cell_chunk_size", [None, 97])
def test_anglemania_missing_mode_matches_manual_reduction(mode, cell_chunk_size):
    adata = _with_missing_genes()
    pa.pp.anglemania(adata, batch_key="batch", allow_missing_features=True,
                     missing_mode=mode, cell_chunk_size=cell_chunk_size, verbose=False)
    exp = _manual_gene_scores(adata, mode, cell_chunk_size)
    var = adata.var.loc[adata.uns["anglemania"]["intersect_genes"]]
    for k in ("signal", "noise", "R", "signal_db", "ICC_db"):
        np.testing.assert_allclose(var[f"anglemania_{k}"], exp[k], rtol=1e-10, err_msg=k)


def test_anglemania_missing_modes_agree_without_missing_genes():
    a, b = example_adata(), example_adata()
    for adata, mode in ((a, "mask"), (b, "zero")):
        pa.pp.anglemania(adata, batch_key="batch", dataset_key="dataset",
                         allow_missing_features=True, missing_mode=mode, verbose=False)
    cols = [c for c in a.var.columns if c.startswith("anglemania_")]
    pd.testing.assert_frame_equal(a.var[cols], b.var[cols])
    assert a.uns["anglemania"]["presence"].to_numpy().all()


def test_anglemania_missing_modes_differ_with_missing_genes():
    a, b = _with_missing_genes(), _with_missing_genes()
    for adata, mode in ((a, "mask"), (b, "zero")):
        pa.pp.anglemania(adata, batch_key="batch", allow_missing_features=True,
                         missing_mode=mode, verbose=False)
    universe = a.uns["anglemania"]["intersect_genes"]
    n_present = a.var.loc[universe, "anglemania_n_batches_present"]
    partial = n_present.index[n_present < 4]
    # zero-filling shrinks M toward 0 for partially-missing genes
    assert (b.var.loc[partial, "anglemania_signal"] < a.var.loc[partial, "anglemania_signal"]
            ).mean() > 0.9


def test_anglemania_presence_and_ICC_db_score():
    adata = _with_missing_genes()
    pa.pp.anglemania(adata, batch_key="batch", allow_missing_features=True,
                     score="ICC", max_n_genes=12, verbose=False)
    res = adata.uns["anglemania"]
    presence = res["presence"]
    assert list(presence.columns) == res["intersect_genes"]
    assert list(presence.index) == list(adata.obs["anglemania_batch"].cat.categories)
    np.testing.assert_array_equal(
        presence.sum(0).to_numpy(),
        adata.var.loc[res["intersect_genes"], "anglemania_n_batches_present"].to_numpy(),
    )
    icc = adata.var["anglemania_ICC_db_z_binned"]
    assert res["anglemania_genes"] == list(icc.sort_values(ascending=False).index[:12])


def test_anglemania_dataset_presence():
    adata = example_adata()
    X = adata.X.copy()
    gone = [0, 1, 2]  # absent from every batch of dataset1
    X[np.ix_((adata.obs["dataset"] == "dataset1").to_numpy(), gone)] = 0
    adata.X = X
    kw = dict(batch_key="batch", dataset_key="dataset", allow_missing_features=True,
              verbose=False)
    pa.pp.anglemania(adata, **kw)
    assert set(adata.var_names[gone]) <= set(adata.uns["anglemania"]["intersect_genes"])
    pa.pp.anglemania(adata, dataset_presence=True, **kw)
    universe = adata.uns["anglemania"]["intersect_genes"]
    assert not set(adata.var_names[gone]) & set(universe)
    assert len(universe) == adata.n_vars - 3


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
        {"batch_key": "batch", "normalization_method": "bogus"},
        {"batch_key": "batch", "score": "bogus"},
        {"batch_key": "batch", "score": "R_z_binned"},  # column name, not a score name
        {"batch_key": "batch", "missing_mode": "nan"},
        {"batch_key": "batch", "dataset_presence": True},
        {"batch_key": "batch", "n_bins": 0},
        {"batch_key": "batch", "signal_R_weights": (0.4, 0.4, 0.2)},
        {"batch_key": "batch", "signal_R_weights": (-0.5, 1.0)},
        {"batch_key": "batch", "cell_chunk_size": 0},
    ],
)
def test_anglemania_param_validation(kwargs):

    adata = example_adata()
    with pytest.raises(ValueError):
        pa.pp.anglemania(adata, verbose=False, **kwargs)


def test_anglemania_cell_chunk_size_runs_and_records_param():
    adata = example_adata()
    pa.pp.anglemania(
        adata, batch_key="batch", dataset_key="dataset", max_n_genes=15,
        cell_chunk_size=97, verbose=False,
    )
    assert adata.var["anglemania_genes"].sum() == 15
    assert adata.uns["anglemania"]["params"]["cell_chunk_size"] == 97


def test_anglemania_cell_chunk_size_single_chunk_matches_unchunked():
    # cell_chunk_size larger than every batch's cell count reduces to one
    # chunk per batch, processed with the same rng draw as the unchunked
    # path -- selected genes should match exactly.
    unchunked = example_adata()
    chunked = example_adata()

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


def test_anglemania_cell_chunk_size_rejects_incompatible_method():
    adata = example_adata()
    with pytest.raises(ValueError):
        pa.pp.anglemania(
            adata, batch_key="batch", method="spearman", cell_chunk_size=50, verbose=False
        )


def test_anglemania_cell_chunk_size_sparse_X():
    adata = example_adata()
    adata.X = sp.csr_matrix(adata.X)
    pa.pp.anglemania(
        adata, batch_key="batch", dataset_key="dataset", max_n_genes=15,
        cell_chunk_size=97, verbose=False,
    )
    assert adata.var["anglemania_genes"].sum() == 15


@pytest.mark.parametrize("removed", ["prefilter_threshold", "score_weights", "direction"])
def test_anglemania_rejects_removed_pair_level_params(removed):
    with pytest.raises(TypeError):
        pa.pp.anglemania(example_adata(), batch_key="batch", verbose=False, **{removed: 0.5})


def test_anglemania_needs_two_batches():
    adata = example_adata()
    adata.obs["one"] = "x"
    with pytest.raises(ValueError, match="2 batches"):
        pa.pp.anglemania(adata, batch_key="one", verbose=False)
