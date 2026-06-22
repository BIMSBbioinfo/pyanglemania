from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyanglemania.preprocessing._select import (
    extract_unique_genes,
    prefilter_gene_pairs,
    rank_gene_pairs,
    score_genes_by_aggregate,
    select_top_genes,
)


def _toy_matrices():
    genes = ["g1", "g2", "g3"]
    # upper triangle: (g1,g2) mean=3.0 sd=0.3 sn=10; (g1,g3) mean=0.2 sd=0.5 sn=0.4;
    # (g2,g3) mean=0.1 sd=0.5 sn=0.2 -- only (g1,g2) clears a threshold of 1.0.
    mean = np.array(
        [
            [0.0, 3.0, 0.2],
            [3.0, 0.0, 0.1],
            [0.2, 0.1, 0.0],
        ]
    )
    sd = np.array(
        [
            [np.nan, 0.3, 0.5],
            [0.3, np.nan, 0.5],
            [0.5, 0.5, np.nan],
        ]
    )
    sn = np.abs(mean) / sd
    return genes, mean, sd, sn


def test_prefilter_gene_pairs_basic_threshold():
    _, mean, sd, sn = _toy_matrices()
    df = prefilter_gene_pairs(
        mean, sd, sn, zscore_mean_threshold=1.0, zscore_sn_threshold=1.0, verbose=False
    )
    # g1=0, g2=1, g3=2
    assert list(zip(df["geneA_idx"], df["geneB_idx"])) == [(0, 1)]


def test_prefilter_gene_pairs_rejects_nonpositive_threshold():
    _, mean, sd, sn = _toy_matrices()
    with pytest.raises(ValueError):
        prefilter_gene_pairs(
            mean, sd, sn, zscore_mean_threshold=0, zscore_sn_threshold=1, verbose=False
        )


def test_prefilter_gene_pairs_relaxes_threshold_until_a_pair_passes():
    _, mean, sd, sn = _toy_matrices()
    # Nothing clears 3.05/3.05 (max mean is exactly 3.0); one relaxation
    # step down to 2.95/2.95 should let (g1, g2) through.
    df = prefilter_gene_pairs(
        mean,
        sd,
        sn,
        zscore_mean_threshold=3.05,
        zscore_sn_threshold=3.05,
        verbose=False,
    )
    assert list(zip(df["geneA_idx"], df["geneB_idx"])) == [(0, 1)]


def test_rank_gene_pairs_both_direction_favors_large_abs_mean_and_small_sd():
    df = pd.DataFrame(
        {
            "geneA": ["a", "c"],
            "geneB": ["b", "d"],
            "mean_zscore": [3.0, -3.0],
            "sd_zscore": [0.1, 0.5],
            "sn_zscore": [30.0, 6.0],
        }
    )
    ranked = rank_gene_pairs(df, score_weights=(0.4, 0.6), direction="both")
    assert ranked.iloc[0]["geneA"] == "a"


def test_rank_gene_pairs_cor_direction_prefers_positive_mean():
    df = pd.DataFrame(
        {
            "geneA": ["a", "c"],
            "geneB": ["b", "d"],
            "mean_zscore": [3.0, -3.0],
            "sd_zscore": [0.2, 0.2],
            "sn_zscore": [15.0, 15.0],
        }
    )
    ranked = rank_gene_pairs(df, direction="cor")
    assert ranked.iloc[0]["geneA"] == "a"


def test_rank_gene_pairs_anticor_direction_prefers_negative_mean():
    df = pd.DataFrame(
        {
            "geneA": ["a", "c"],
            "geneB": ["b", "d"],
            "mean_zscore": [3.0, -3.0],
            "sd_zscore": [0.2, 0.2],
            "sn_zscore": [15.0, 15.0],
        }
    )
    ranked = rank_gene_pairs(df, direction="anticor")
    assert ranked.iloc[0]["geneA"] == "c"


def test_rank_gene_pairs_direction_validation():
    df = pd.DataFrame(
        {"geneA": ["a"], "geneB": ["b"], "mean_zscore": [1.0], "sd_zscore": [0.1], "sn_zscore": [10.0]}
    )
    with pytest.raises(ValueError):
        rank_gene_pairs(df, direction="bogus")


def test_extract_unique_genes_preserves_rank_order_and_caps():
    gene_names = ["a", "b", "c", "d", "e", "f"]
    # rank order: (a,b), (c,a), (e,f) -- by index: (0,1), (2,0), (4,5)
    ranked = pd.DataFrame({"geneA_idx": [0, 2, 4], "geneB_idx": [1, 0, 5]})
    genes = extract_unique_genes(ranked, gene_names, max_n_genes=4)
    assert genes == ["a", "b", "c", "e"]


def test_extract_unique_genes_no_cap():
    gene_names = ["a", "b", "c"]
    ranked = pd.DataFrame({"geneA_idx": [0, 2], "geneB_idx": [1, 0]})
    genes = extract_unique_genes(ranked, gene_names, max_n_genes=None)
    assert genes == ["a", "b", "c"]


def test_extract_unique_genes_matches_full_dedup_when_growth_is_needed():
    # The first 50 rows are all the same pair (no new genes), so the
    # initial prefix guess (== max_n_genes == 3) won't have 3 unique genes
    # yet and the exponential-growth loop has to kick in to find them.
    gene_names = ["dup_a", "dup_b", "c", "d", "e", "f", "g", "h"]
    rows = [(0, 1)] * 50 + [(2, 3), (4, 5), (6, 7)]
    ranked = pd.DataFrame(rows, columns=["geneA_idx", "geneB_idx"])

    got = extract_unique_genes(ranked, gene_names, max_n_genes=3)

    # Reference: dedup the whole table up front (the pre-optimization
    # behavior), then truncate -- must match exactly.
    interleaved = np.ravel(ranked[["geneA_idx", "geneB_idx"]].to_numpy())
    _, first_seen = np.unique(interleaved, return_index=True)
    expected = list(np.asarray(gene_names)[interleaved[np.sort(first_seen)][:3]])

    assert got == expected == ["dup_a", "dup_b", "c"]


def test_extract_unique_genes_fewer_unique_than_requested():
    gene_names = ["a", "b"]
    ranked = pd.DataFrame({"geneA_idx": [0, 0], "geneB_idx": [1, 1]})
    genes = extract_unique_genes(ranked, gene_names, max_n_genes=10)
    assert genes == ["a", "b"]


def _toy_prefiltered():
    # pairs (0,1) mean=2.0 sn=5.0 -> score 10.0; (0,2) mean=-1.0 sn=2.0 ->
    # score 2.0; (1,3) mean=3.0 sn=1.0 -> score 3.0.
    return {
        "geneA_idx": np.array([0, 0, 1]),
        "geneB_idx": np.array([1, 2, 3]),
        "mean_zscore": np.array([2.0, -1.0, 3.0]),
        "sd_zscore": np.array([0.4, 0.5, 1.0]),
        "sn_zscore": np.array([5.0, 2.0, 1.0]),
    }


def test_score_genes_by_aggregate_both_direction_sums_pair_scores():
    gene_names = ["g0", "g1", "g2", "g3"]
    scores = score_genes_by_aggregate(_toy_prefiltered(), gene_names, direction="both")
    assert list(scores["gene"]) == ["g1", "g0", "g3", "g2"]
    np.testing.assert_allclose(scores["score"].to_numpy(), [13.0, 12.0, 3.0, 2.0])
    assert list(scores["degree"]) == [2, 2, 1, 1]


def test_score_genes_by_aggregate_anticor_zeroes_positive_pairs():
    gene_names = ["g0", "g1", "g2", "g3"]
    scores = score_genes_by_aggregate(_toy_prefiltered(), gene_names, direction="anticor")
    # Only (0,2)'s negative mean_zscore contributes; (0,1) and (1,3) are
    # positive and clipped to 0, but their *degree* still counts.
    by_gene = dict(zip(scores["gene"], scores["score"]))
    assert by_gene == {"g0": 2.0, "g1": 0.0, "g2": 2.0, "g3": 0.0}
    assert list(scores["gene"]) == ["g0", "g2", "g1", "g3"]


def test_score_genes_by_aggregate_validates_direction():
    with pytest.raises(ValueError):
        score_genes_by_aggregate(_toy_prefiltered(), ["g0", "g1", "g2", "g3"], direction="bogus")


def test_select_top_genes_excludes_zero_degree():
    scores = pd.DataFrame(
        {
            "gene": ["a", "b", "c", "d"],
            "score": [10.0, 8.0, 0.0, 0.0],
            "degree": [2, 0, 0, 1],
        }
    )
    assert select_top_genes(scores, max_n_genes=3) == ["a", "d"]


def test_select_top_genes_caps_to_max_n_genes():
    scores = pd.DataFrame(
        {"gene": ["a", "b", "c"], "score": [10.0, 8.0, 5.0], "degree": [2, 2, 1]}
    )
    assert select_top_genes(scores, max_n_genes=2) == ["a", "b"]


def test_select_top_genes_none_returns_all_qualifying():
    scores = pd.DataFrame(
        {"gene": ["a", "b", "c"], "score": [10.0, 8.0, 0.0], "degree": [2, 1, 0]}
    )
    assert select_top_genes(scores, max_n_genes=None) == ["a", "b"]
