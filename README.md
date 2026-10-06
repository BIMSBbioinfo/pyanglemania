# pyanglemania
## Introduction

A GPU-ready Python/AnnData port of the R/Bioconductor [anglemania](https://github.com/BIMSBbioinfo/anglemania) package: selects genes whose correlations to other genes stay invariant across batches, for use as integration features (in place of, or alongside, highly-variable genes). Unlike the R package, which ranks gene *pairs*, every gene gets its own expression-corrected score.

## Installation

```bash
pip install pyanglemania
```

GPU support is optional and not bundled: `pyanglemania` dispatches to numpy or cupy based on whatever array `adata.X` already holds, exactly like [rapids-singlecell](https://github.com/scverse/rapids_singlecell). To run on a GPU, install cupy for your CUDA version (or get it via rapids-singlecell) and move the data to the device yourself:

```bash
pip install "pyanglemania[rapids]"   # pulls cupy-cuda12x
```

The development environment (including the GPU stack, the tutorial's Harmony/scVI dependencies, and the CUDA headers cupy's JIT needs) is pinned in [`envs/pyanglemania.yml`](envs/pyanglemania.yml):

```bash
mamba env update -f envs/pyanglemania.yml
pip install -e . --no-build-isolation
```

## Usage

```python
import scanpy as sc
import pyanglemania as pa

adata = sc.read_h5ad("data.h5ad")           # raw counts in adata.X (or in a layer)
pa.pp.anglemania(adata, batch_key="batch", max_n_genes=2000)

genes = adata.var_names[adata.var["anglemania_genes"]]
```

Selected genes land in `adata.var["anglemania_genes"]` (boolean mask) and `adata.uns["anglemania"]` (parameters, gene universe, and the selected genes, best first). The per-gene scores are in `adata.var["anglemania_*"]`: with `M`/`S` the cross-batch mean/sd of each gene pair's z-scored correlation, `signal = Σⱼ M²`, `noise = Σⱼ S²`, `R = signal / (signal + noise)`, plus their within-expression-bin z-scores `signal_z_binned`/`R_z_binned` and the weighted signal/noise score `WSN = 0.5·signal_z_binned + 0.5·R_z_binned`. Genes are ranked on `score="R"` (`R_z_binned`, the default), `"WSN"`, or `"ICC"` (`ICC_db_z_binned`, see below).

With `allow_missing_features=True`, genes absent from a batch are masked out of that batch (`missing_mode="mask"`, the default) rather than zero-filled as in R (`missing_mode="zero"`). With missing genes `R` is inflated for genes present in fewer batches, so `score="ICC"` (debiased) is the safer selector there. On a GPU, move the layer to the device first — e.g. `rapids_singlecell.get.anndata_to_GPU(adata)` — and the same call runs on cupy.

## Tutorial

See [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) for a full walkthrough: simulating multi-batch data, running `pp.anglemania`, comparing the selected genes against batch-aware `highly_variable_genes`, and integrating both gene sets with Harmony.

See `CLAUDE.md` for architecture details and how this maps onto the original R algorithm.

## Releasing

Releases are published to PyPI by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using PyPI Trusted Publishing, so no API token lives in this repo. Per release:

1. Bump `version` in `pyproject.toml` and commit.
2. `git tag v<version> && git push origin v<version>`.

The workflow builds the sdist and wheel, runs `twine check`, verifies the tag matches the package version, and uploads through the `pypi` GitHub environment. The trusted publisher (owner `BIMSBbioinfo`, repository `pyanglemania`, workflow `publish.yml`, environment `pypi`) has to be registered once on pypi.org before the first release.
