"""Synthetic datasets for tests and docstring examples."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def example_adata(seed: int = 42) -> ad.AnnData:
    """A small synthetic multi-batch, multi-dataset AnnData.

    Ports anglemania's R ``sce_example``: 300 genes, 600 cells split into
    two batches with different Poisson rates (so the batches differ in
    library size, but not in their underlying gene-gene structure), each
    batch evenly split across two datasets.
    """
    rng = np.random.default_rng(seed)
    counts = np.concatenate(
        [
            rng.poisson(lam=5, size=(300, 300)),  # genes x batch1 cells
            rng.poisson(lam=3, size=(300, 300)),  # genes x batch2 cells
        ],
        axis=1,
    ).T.astype(np.float32)  # -> cells x genes

    obs = pd.DataFrame(
        {
            "batch": np.repeat(["batch1", "batch2"], 300),
            "dataset": np.tile(np.repeat(["dataset1", "dataset2"], 150), 2),
        },
        index=[f"cell{i + 1}" for i in range(600)],
    )
    var = pd.DataFrame(index=[f"gene{i + 1}" for i in range(300)])
    return ad.AnnData(X=counts, obs=obs, var=var)
