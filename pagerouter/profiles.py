"""
Experiment 1 — Page-level capability profiles.

Compute mean NED per model per stratum (doc_type and layout_type).
Outputs: model × stratum score matrices for heatmap visualization.
"""

from __future__ import annotations

import pandas as pd


def compute_stratum_profiles(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """Compute mean NED score for each (model, stratum) pair.

    Returns long-form DataFrame with columns [model, stratum, mean_ned, n_pages].
    """
    grouped = (
        df.groupby(["model", stratum_col])["ned_score"]
        .agg(mean_ned="mean", n_pages="count")
        .reset_index()
        .rename(columns={stratum_col: "stratum"})
    )
    return grouped


def compute_score_matrix(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """Build a model × stratum mean-NED matrix.

    Returns DataFrame of shape (n_models, n_strata).
    """
    profiles = compute_stratum_profiles(df, stratum_col)
    matrix = profiles.pivot(index="model", columns="stratum", values="mean_ned")
    matrix.columns.name = None
    matrix.index.name = "model"
    return matrix


def rank_models_per_stratum(matrix: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each stratum (1 = best NED).

    Returns same shape as matrix, values are integer ranks.
    """
    return matrix.rank(axis=0, ascending=False, method="min").astype(int)
