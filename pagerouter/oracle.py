"""
Experiment 3 — Oracle and complementarity analysis.

Quantifies the theoretical upper bound of routing and pairwise model complementarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oracle_score(matrix: pd.DataFrame, k: int) -> float:
    """Mean NED achieved by the oracle that picks the best-k models per page.

    k=1: per-page max. k>1: per-page mean of top-k scores.
    """
    n_models = matrix.shape[1]
    k = min(k, n_models)
    per_page = matrix.apply(lambda row: row.nlargest(k).mean(), axis=1)
    return float(per_page.mean())


def oracle_gain(matrix: pd.DataFrame) -> float:
    """Absolute NED gain of oracle-1 over the best static single model."""
    best_single = float(matrix.mean(axis=0).max())
    return oracle_score(matrix, k=1) - best_single


def oracle_curve(matrix: pd.DataFrame) -> pd.Series:
    """Oracle NED for k = 1, 2, 3, 5, and all models."""
    n_models = matrix.shape[1]
    ks = sorted({1, 2, 3, 5, n_models})
    return pd.Series({k: oracle_score(matrix, k) for k in ks}, name="oracle_ned")


def complementarity_matrix(matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pairwise complementarity Φ(i, j): fraction of pages where i fails and j succeeds.

    Φ(i, j) = P(NED_i < threshold AND NED_j >= threshold)
    """
    arr = matrix.values
    models = matrix.columns.tolist()
    n = len(models)
    fails   = (arr < threshold)   # shape (pages, models)
    succeeds = (arr >= threshold)

    comp = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            comp[i, j] = float(np.mean(fails[:, i] & succeeds[:, j]))

    return pd.DataFrame(comp, index=models, columns=models)


def per_stratum_oracle_gap(df: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Oracle gain broken down by doc_type and layout_type strata."""
    rows = []
    page_attrs = df[["page_id", "doc_type", "layout_type"]].drop_duplicates("page_id").set_index("page_id")

    for stratum_col in ["doc_type", "layout_type"]:
        for stratum_value, pages in page_attrs.groupby(stratum_col).groups.items():
            pages_in_matrix = [p for p in pages if p in matrix.index]
            if not pages_in_matrix:
                continue
            sub = matrix.loc[pages_in_matrix]
            best_single = float(sub.mean(axis=0).max())
            oracle = float(sub.max(axis=1).mean())
            rows.append({
                "stratum_col":   stratum_col,
                "stratum_value": stratum_value,
                "best_single_ned": best_single,
                "oracle_ned":    oracle,
                "gap":           oracle - best_single,
                "n_pages":       len(pages_in_matrix),
            })

    return pd.DataFrame(rows).sort_values(["stratum_col", "gap"], ascending=[True, False])
