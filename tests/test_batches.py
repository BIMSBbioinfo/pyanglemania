from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from pyanglemania.datasets import example_adata
from pyanglemania.preprocessing._batches import (
    add_unique_batch_key,
    align_to_common_genes,
    compute_dataset_weights,
    genes_passing_min_cells,
    intersect_genes,
    split_obs_indices_by_batch,
)


def test_add_unique_batch_key_combines_dataset_and_batch():
    adata = example_adata()
    add_unique_batch_key(adata, "batch", "dataset")
    labels = set(adata.obs["anglemania_batch"].astype(str).unique())
    assert labels == {
        "dataset1:batch1",
        "dataset2:batch1",
        "dataset1:batch2",
        "dataset2:batch2",
    }


def test_add_unique_batch_key_without_dataset_key():
    adata = example_adata()
    add_unique_batch_key(adata, "batch", None)
    assert set(adata.obs["anglemania_batch"].astype(str).unique()) == {"batch1", "batch2"}


def test_compute_dataset_weights_equal_when_balanced():
    adata = example_adata()
    add_unique_batch_key(adata, "batch", "dataset")
    weights = compute_dataset_weights(adata.obs, "batch", "dataset")
    np.testing.assert_allclose(weights.to_numpy(), 1.0)


def test_compute_dataset_weights_downweights_oversampled_dataset():
    obs = pd.DataFrame(
        {
            "anglemania_batch": ["b1", "b2", "b3", "b4"],
            "dataset": ["d1", "d1", "d1", "d2"],
            "batch": ["b1", "b2", "b3", "b4"],
        }
    )
    weights = compute_dataset_weights(obs, "batch", "dataset")
    assert weights["b1"] < weights["b4"]
    np.testing.assert_allclose(weights[["b1", "b2", "b3"]].to_numpy(), weights["b1"])


def test_compute_dataset_weights_all_one_without_multi_dataset():
    obs = pd.DataFrame(
        {"anglemania_batch": ["b1", "b2"], "dataset": ["d1", "d1"], "batch": ["b1", "b2"]}
    )
    weights = compute_dataset_weights(obs, "batch", "dataset")
    np.testing.assert_allclose(weights.to_numpy(), 1.0)


def test_split_obs_indices_by_batch():
    adata = example_adata()
    add_unique_batch_key(adata, "batch", None)
    indices = split_obs_indices_by_batch(adata)
    assert set(indices) == {"batch1", "batch2"}
    assert len(indices["batch1"]) == 300
    assert len(indices["batch2"]) == 300
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(indices.values()))), np.arange(600)
    )


@pytest.mark.parametrize("as_sparse", [False, True])
def test_genes_passing_min_cells(as_sparse):
    X = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    if as_sparse:
        X = sp.csr_matrix(X)
    mask = genes_passing_min_cells(X, 2, np, sp)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_intersect_genes_strict_preserves_first_batch_order():
    genes = intersect_genes([["a", "b", "c"], ["c", "b", "d"]], False, 1, verbose=False)
    assert genes == ["b", "c"]


def test_intersect_genes_allow_missing():
    genes = intersect_genes(
        [["a", "b"], ["b", "c"], ["b", "c", "d"]],
        True,
        min_samples_per_gene=2,
        verbose=False,
    )
    assert genes == ["b", "c"]


def test_align_to_common_genes_pads_missing_with_zero():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = align_to_common_genes(X, ["a", "b"], ["a", "x", "b"], np)
    expected = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
    np.testing.assert_array_equal(out, expected)


def test_align_to_common_genes_reorders():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = align_to_common_genes(X, ["a", "b"], ["b", "a"], np)
    expected = np.array([[2.0, 1.0], [4.0, 3.0]])
    np.testing.assert_array_equal(out, expected)
