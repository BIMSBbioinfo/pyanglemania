"""Prefilter gene pairs and rank/select genes for integration.

Ported from anglemania's R ``select_genes.R`` (and the C++ pair-extraction
in ``select_genes_cpp``). The heavy ``(genes x genes)`` matrices are only
read here; the result is small (at most a few thousand gene pairs) and is
returned as a plain pandas DataFrame.

``prefilter_gene_pairs`` -> ``rank_gene_pairs`` -> ``extract_unique_genes``
is R's own pairwise-ranking algorithm (``anglemania(..., selection_method=
"pairwise")``, the default). ``score_genes_by_aggregate`` ->
``select_top_genes`` is an experimental, non-R-faithful alternative
(``selection_method="per_gene"``) that scores genes directly instead of
ranking pairs -- see their docstrings and plans/optimization.md.
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
) -> dict:
    """Keep gene pairs whose |mean z-score| and SNR both clear a threshold.

    Ports ``prefilter_angl``/``select_genes_cpp``: only the upper triangle
    (``i < j``) of each matrix is considered, one row per unordered gene
    pair. Thresholds are relaxed by 0.1 (down to 0) until at least one pair
    passes, exactly as in R.

    The upper-triangle gather and threshold mask run on whichever array
    module backs ``mean_zscore`` (numpy or cupy); only the pairs that
    survive filtering are ever computed at all, instead of building the
    full (genes x genes) pairs table before filtering (see
    plans/optimization.md #1). Returns a dict of arrays in that *same*
    array module -- not yet transferred to host -- so ``rank_gene_pairs``
    can rank/sort on-device too when ``xp`` is cupy (plans/optimization.md
    #6d); it does the one host transfer, once, after sorting.

    Genes stay as integer indices (``geneA_idx``/``geneB_idx``) rather than
    name strings here -- like R's own ``select_genes_cpp``, which maps
    indices to gene names only at the very end (``select_genes.R``) -- so
    ranking never has to drag string columns along (plans/optimization.md
    #6c); callers map back to names once, after ranking, for the handful
    of rows that matter.
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

    return {
        "geneA_idx": i_idx[keep],
        "geneB_idx": j_idx[keep],
        "mean_zscore": mean_flat[keep],
        "sd_zscore": sd_flat[keep],
        "sn_zscore": sn_flat[keep],
    }


def _min_rank(values, xp):
    """``pandas.Series(values).rank(method="min")``, vectorized in ``xp``.

    A tied value's rank is the rank of its first occurrence in sorted
    order, which ``searchsorted(sorted(values), values, side="left") + 1``
    gives directly -- verified bit-exact against pandas' own
    ``rank(method="min")`` (plans/optimization.md #6d). Assumes no NaNs,
    which holds here: ``prefilter_gene_pairs`` only ever keeps pairs whose
    stats already cleared a real-valued threshold.
    """
    sorted_values = xp.sort(values)
    return xp.searchsorted(sorted_values, values, side="left") + 1


def rank_gene_pairs(
    prefiltered: dict,
    score_weights: tuple[float, float] = (0.4, 0.6),
    direction: str = "both",
) -> pd.DataFrame:
    """Add a combined rank column, ported from R ``select_genes``.

    Pairs are ranked by a weighted sum of the rank of the (possibly
    signed, depending on ``direction``) mean z-score and the rank of the sd
    z-score -- lower sd (more consistent across batches) ranks better.

    Always returns a host-side pandas DataFrame, already sorted by rank --
    but *how* it gets there is backend-conditional. For cupy input, the
    whole rank+sort runs on-device via :func:`_min_rank` (``cupy.sort`` +
    ``cupy.searchsorted``), transferring to host only once, at the end:
    ~13x faster than pandas at realistic scale (124.5s -> 9.3s in the
    plans/optimization.md #6d benchmark), because it avoids forcing a
    host round-trip just to call ``pandas.Series.rank``. For numpy input,
    plain pandas ``.rank()`` is used as before -- benchmarked *faster*
    than the same ``sort``/``searchsorted`` approach on CPU (pandas' rank
    is already well-tuned there), so this isn't a single "better"
    algorithm, it's backend-dependent which one wins.

    Both branches sort with ``kind="stable"`` rather than each backend's
    default (pandas: quicksort; cupy: introsort-like), which matters
    because the combined ``rank`` column ties often: it's a weighted sum
    of two *integer* ranks, so e.g. ``rank_mean=1, rank_sd=5`` and
    ``rank_mean=4, rank_sd=3`` both give ``3.4`` at the default weights.
    A non-stable sort breaks those ties however the algorithm happens to
    -- verified to disagree between pandas and cupy's default sorts on
    realistic data, which would make ``extract_unique_genes``'s selection
    depend on which backend ran it. Stable sort instead breaks every tie
    by the pairs' original (pre-rank) order, which is identical input on
    both backends and verified to make the two paths agree exactly.
    """
    if direction not in ("both", "anticor", "cor"):
        raise ValueError(f"direction must be 'both', 'anticor' or 'cor', got {direction!r}")

    xp, _ = get_array_module(prefiltered["mean_zscore"])
    mean_zscore = prefiltered["mean_zscore"]
    sd_zscore = prefiltered["sd_zscore"]

    if direction == "both":
        mean_rank_key = -xp.abs(mean_zscore)
    elif direction == "anticor":
        mean_rank_key = mean_zscore
    else:  # cor
        mean_rank_key = -mean_zscore

    if xp.__name__ == "cupy":
        rank_mean = _min_rank(mean_rank_key, xp).astype(xp.float64)
        rank_sd = _min_rank(sd_zscore, xp).astype(xp.float64)
        combined = rank_mean * score_weights[0] + rank_sd * score_weights[1]
        final_rank = _min_rank(combined, xp)

        order = xp.argsort(final_rank, kind="stable")
        return pd.DataFrame(
            {
                "geneA_idx": to_numpy(prefiltered["geneA_idx"][order]),
                "geneB_idx": to_numpy(prefiltered["geneB_idx"][order]),
                "mean_zscore": to_numpy(mean_zscore[order]),
                "sd_zscore": to_numpy(sd_zscore[order]),
                "sn_zscore": to_numpy(prefiltered["sn_zscore"][order]),
                "rank": to_numpy(final_rank[order]),
            }
        )

    df = pd.DataFrame(prefiltered)
    rank_mean = pd.Series(mean_rank_key).rank(method="min")
    rank_sd = df["sd_zscore"].rank(method="min")
    combined = rank_mean * score_weights[0] + rank_sd * score_weights[1]
    df["rank"] = combined.rank(method="min")
    return df.sort_values("rank", kind="stable")


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


def score_genes_by_aggregate(
    prefiltered: dict,
    gene_names: list[str],
    direction: str = "both",
) -> pd.DataFrame:
    """Per-gene score from surviving gene pairs.

    An alternative to ``rank_gene_pairs`` + ``extract_unique_genes``: those
    rank gene *pairs* and walk the ranked list to find unique genes; this
    instead reduces each gene's surviving pairs (from
    ``prefilter_gene_pairs``) directly to one score per gene -- summing,
    over every surviving pair touching a gene, a direction-adjusted
    ``|mean_zscore| * sn_zscore``. ``sn_zscore`` is already
    ``|mean_zscore| / sd_zscore`` (see ``_stats.py::finalize``), so this
    single product already rewards pairs that are both strong *and*
    consistent across batches, without needing rank_gene_pairs's separate
    weighted combination of two ranks.

    This is a genuinely different selection criterion from R's, not a
    drop-in replacement: a gene with many moderate surviving correlations
    can outscore a gene with one very strong correlation, which pairwise
    ranking would tend to surface first. It also only needs an O(n_genes)
    sort at the end (there are far fewer genes than gene pairs), instead
    of rank_gene_pairs's O(n_pairs log n_pairs) -- see
    plans/optimization.md for the context this alternative was explored
    in and why it isn't the default.

    Returns a DataFrame with one row per gene (``gene``, ``score``,
    ``degree`` -- the count of surviving pairs touching it), sorted by
    score descending; pass to :func:`select_top_genes` to pick a final
    gene list.
    """
    if direction not in ("both", "anticor", "cor"):
        raise ValueError(f"direction must be 'both', 'anticor' or 'cor', got {direction!r}")

    xp, _ = get_array_module(prefiltered["mean_zscore"])
    mean_zscore = prefiltered["mean_zscore"]
    sn_zscore = prefiltered["sn_zscore"]

    if direction == "both":
        signed = xp.abs(mean_zscore)
    elif direction == "anticor":
        signed = xp.clip(-mean_zscore, 0, None)
    else:  # cor
        signed = xp.clip(mean_zscore, 0, None)

    pair_score = signed * sn_zscore
    n_genes = len(gene_names)
    # Each surviving pair contributes its score to *both* genes it connects.
    idx = xp.concatenate([prefiltered["geneA_idx"], prefiltered["geneB_idx"]])
    weight = xp.concatenate([pair_score, pair_score])

    gene_score = xp.bincount(idx, weights=weight, minlength=n_genes)
    gene_degree = xp.bincount(idx, minlength=n_genes)

    df = pd.DataFrame(
        {
            "gene": np.asarray(gene_names),
            "score": to_numpy(gene_score),
            "degree": to_numpy(gene_degree),
        }
    )
    return df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def select_top_genes(scores: pd.DataFrame, max_n_genes: int | None) -> list[str]:
    """Top ``max_n_genes`` genes by :func:`score_genes_by_aggregate`'s score.

    Excludes genes with no surviving pairs (``degree == 0``) rather than
    padding the result with arbitrary zero-score genes if fewer than
    ``max_n_genes`` qualify -- mirroring ``extract_unique_genes``'s same
    behavior for the pairwise path.
    """
    qualifying = scores.loc[scores["degree"] > 0, "gene"]
    n = len(qualifying) if max_n_genes is None else min(max_n_genes, len(qualifying))
    return list(qualifying.iloc[:n])
