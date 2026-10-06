"""Expression-binned per-gene scores and gene selection.

Highly expressed genes have less sampling noise in their correlations, hence
larger cross-batch mean z-scores and smaller cross-batch spread -- so both
``signal`` and ``R`` (see ``_stats.py``) are partly expression scores. As in
DUBStepR (and scanpy's ``seurat`` HVG flavor), genes are put into
equal-frequency bins of mean log-normalized expression and the scores are
z-scored *within* each bin:

    signal_z_binned = z_bin(log10 signal)
    R_z_binned      = z_bin(R)
    WSN             = w_signal * signal_z_binned + w_R * R_z_binned   # weighted signal/noise
    ICC_db_z_binned = z_bin(ICC_db)

Higher is better for all four. Without missing genes there is one scalar
``kappa`` and ``ICC_db = (R(1 + kappa) - kappa) / (1 - kappa + kappa R)`` is a
monotone transform of ``R``, so ``ICC_db_z_binned`` ranks genes within each
bin exactly as ``R_z_binned`` does; with missing genes it removes ``R``'s
inflation for genes present in fewer batches (see ``_stats.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def binned_zscores(
    mean_lognorm,
    signal,
    R,
    ICC_db=None,
    n_bins: int = 20,
    signal_R_weights: tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """``expr_bin``/``signal_z_binned``/``R_z_binned``/``WSN`` per gene.

    Plus ``ICC_db_z_binned`` if ``ICC_db`` is given (same bins, same genes).

    Bins are equal-frequency on ``mean_lognorm``, ranked first (ties broken
    by order) so that many tied near-zero means still split into ``n_bins``
    equal-sized bins. Genes with a non-finite ``signal``/``R`` or
    ``signal <= 0`` (no valid partner) are left out of the binning and get
    NaN throughout.
    """
    df = pd.DataFrame(
        {
            "mean_lognorm": np.asarray(mean_lognorm, dtype=float),
            "signal": np.asarray(signal, dtype=float),
            "R": np.asarray(R, dtype=float),
        }
    )
    ok = np.isfinite(df["signal"]) & np.isfinite(df["R"]) & (df["signal"] > 0)
    t = df[ok].copy()
    t["expr_bin"] = pd.qcut(
        t["mean_lognorm"].rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    t["log_signal"] = np.log10(t["signal"])
    t["signal_z_binned"] = t.groupby("expr_bin")["log_signal"].transform(_z)
    t["R_z_binned"] = t.groupby("expr_bin")["R"].transform(_z)
    t["WSN"] = (
        signal_R_weights[0] * t["signal_z_binned"] + signal_R_weights[1] * t["R_z_binned"]
    )
    cols = ["expr_bin", "signal_z_binned", "R_z_binned", "WSN"]
    if ICC_db is not None:
        t["ICC_db"] = np.asarray(ICC_db, dtype=float)[ok.to_numpy()]
        t["ICC_db_z_binned"] = t.groupby("expr_bin")["ICC_db"].transform(_z)
        cols.append("ICC_db_z_binned")
    return t[cols].reindex(df.index)


def top_genes(scores, gene_names, max_n_genes: int | None) -> list[str]:
    """The ``max_n_genes`` best-scoring genes, best first; NaN scores never selected.

    Ties keep the input gene order (stable sort).
    """
    scores = np.asarray(scores, dtype=float)
    gene_names = np.asarray(gene_names)
    finite = np.flatnonzero(np.isfinite(scores))
    order = finite[np.argsort(-scores[finite], kind="stable")]
    if max_n_genes is not None:
        order = order[:max_n_genes]
    return list(gene_names[order])
