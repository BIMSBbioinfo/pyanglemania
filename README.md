# pyanglemania
## Introduction

A GPU-ready Python/AnnData port of the R/Bioconductor [anglemania](https://github.com/BIMSBbioinfo/anglemania) package: selects genes whose pairwise correlations stay invariant across batches, for use as integration features (in place of, or alongside, highly-variable genes).

## Tutorial

See [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) for a full walkthrough: simulating multi-batch data, running `pp.anglemania`, comparing the selected genes against `highly_variable_genes`, and integrating both gene sets with Harmony.

See `CLAUDE.md` for architecture details and how this maps onto the original R algorithm.