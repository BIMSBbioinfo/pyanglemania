"""Prefilter gene pairs and rank/select genes for integration.

Ported from anglemania's R ``select_genes.R`` (and the C++ pair-extraction
in ``select_genes_cpp``). The heavy ``(genes x genes)`` matrices are only
read here; the result is small (at most a few thousand gene pairs) and is
returned as a plain pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils import get_array_module, to_numpy, vmessage


def prefilter_gene_pairs(
    mean_zscore,
    sds_zscore,
    sn_zscore,
    zscore_mean_threshold: float = 0.5,
    zscore_sn_threshold: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Keep gene pairs whose |mean z-score| and SNR both clear a threshold.

    Ports ``prefilter_angl``/``select_genes_cpp``: only the upper triangle
    (``i < j``) of each matrix is considered, one row per unordered gene
    pair. Thresholds are relaxed by 0.1 (down to 0) until at least one pair
    passes, exactly as in R.

    The upper-triangle gather and threshold mask run on whichever array
    module backs ``mean_zscore`` (numpy or cupy); only the pairs that
    survive filtering are ever pulled to the host, instead of transferring
    the full (genes x genes) matrices before filtering (see
    plans/optimization.md #1). Genes stay as integer indices
    (``geneA_idx``/``geneB_idx``) rather than name strings here -- like R's
    own ``select_genes_cpp``, which maps indices to gene names only at the
    very end (``select_genes.R``) -- so ``rank_gene_pairs``'s sort never
    has to drag string columns along (plans/optimization.md #6); callers
    map back to names once, after ranking, for the handful of rows that
    matter.
    """
    if zscore_mean_threshold <= 0 or zscore_sn_threshold <= 0:
        raise ValueError("zscore_mean_threshold and zscore_sn_threshold need to be positive")

    xp, _ = get_array_module(mean_zscore)
    n = mean_zscore.shape[0]
    i_idx, j_idx = xp.triu_indices(n, k=1)
    mean_flat = mean_zscore[i_idx, j_idx]
    sd_flat = sds_zscore[i_idx, j_idx]
    sn_flat = sn_zscore[i_idx, j_idx]

    mean_thr, sn_thr = zscore_mean_threshold, zscore_sn_threshold
    keep = (sn_flat >= sn_thr) & (xp.abs(mean_flat) >= mean_thr)
    while not bool(keep.any()):
        mean_thr -= 0.1
        sn_thr -= 0.1
        if mean_thr <= 0 or sn_thr <= 0:
            raise ValueError(
                "zscore_mean_threshold and zscore_sn_threshold need to be positive"
            )
        vmessage(verbose, "No genes passed the cutoff. Decreasing thresholds by 0.1...")
        keep = (sn_flat >= sn_thr) & (xp.abs(mean_flat) >= mean_thr)

    return pd.DataFrame(
        {
            "geneA_idx": to_numpy(i_idx[keep]),
            "geneB_idx": to_numpy(j_idx[keep]),
            "mean_zscore": to_numpy(mean_flat[keep]),
            "sd_zscore": to_numpy(sd_flat[keep]),
            "sn_zscore": to_numpy(sn_flat[keep]),
        }
    )


def rank_gene_pairs(
    prefiltered: pd.DataFrame,
    score_weights: tuple[float, float] = (0.4, 0.6),
    direction: str = "both",
) -> pd.DataFrame:
    """Add a combined rank column, ported from R ``select_genes``.

    Pairs are ranked by a weighted sum of the rank of the (possibly
    signed, depending on ``direction``) mean z-score and the rank of the sd
    z-score -- lower sd (more consistent across batches) ranks better.
    """
    if direction not in ("both", "anticor", "cor"):
        raise ValueError(f"direction must be 'both', 'anticor' or 'cor', got {direction!r}")

    df = prefiltered.copy()
    if direction == "both":
        mean_rank_key = -df["mean_zscore"].abs()
    elif direction == "anticor":
        mean_rank_key = df["mean_zscore"]
    else:  # cor
        mean_rank_key = -df["mean_zscore"]

    rank_mean = mean_rank_key.rank(method="min")
    rank_sd = df["sd_zscore"].rank(method="min")
    combined = rank_mean * score_weights[0] + rank_sd * score_weights[1]
    df["rank"] = combined.rank(method="min")
    return df.sort_values("rank")


def extract_unique_genes(
    ranked: pd.DataFrame, gene_names: list[str], max_n_genes: int | None
) -> list[str]:
    """Unique genes from ranked gene pairs, best-ranked pairs first.

    Ports ``extract_rows_for_unique_genes``: interleaves ``geneA_idx``/
    ``geneB_idx`` from each row (in rank order) and keeps the first
    ``max_n_genes`` unique genes, mapping integer indices to names only for
    that small result -- everything up to here works on indices, not
    strings (see ``prefilter_gene_pairs``'s docstring for why).

    A permissive ``prefilter_threshold`` can let tens of millions of pairs
    through at gene-panel sizes in the tens of thousands, and R's own
    ``extract_rows_for_unique_genes`` (and an earlier version of this
    function) deduplicates that *entire* table before truncating -- a
    `numpy.unique` over 100M+ entries this way, which dominates runtime far
    more than anything else in the pipeline. Since ``ranked`` is already
    sorted by rank, the first ``max_n_genes`` unique genes are fully
    determined by some prefix of it; growing that prefix exponentially
    until it has enough avoids ever deduplicating rows beyond what's
    needed, while still producing the exact same result as deduplicating
    the whole table up front.
    """
    gene_names = np.asarray(gene_names)
    idx_a = ranked["geneA_idx"].to_numpy()
    idx_b = ranked["geneB_idx"].to_numpy()
    n_pairs = len(idx_a)
    target = n_pairs if max_n_genes is None else max_n_genes

    prefix = min(n_pairs, max(target, 1))
    while True:
        interleaved = np.ravel(np.column_stack((idx_a[:prefix], idx_b[:prefix])))
        _, first_seen = np.unique(interleaved, return_index=True)
        if len(first_seen) >= target or prefix >= n_pairs:
            unique_idx = interleaved[np.sort(first_seen)]
            if max_n_genes is not None:
                unique_idx = unique_idx[:max_n_genes]
            return list(gene_names[unique_idx])
        prefix = min(n_pairs, prefix * 4)
