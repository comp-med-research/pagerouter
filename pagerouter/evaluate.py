"""Shared evaluation utilities used across experiments."""

from __future__ import annotations

import pandas as pd


def mean_ned(scores: pd.Series) -> float:
    return float(scores.mean())


def oracle_gap_recovered(
    router_ned: float,
    best_single_ned: float,
    oracle_ned: float,
) -> float:
    """Fraction of the oracle gap recovered by a router.

    recovered = (router_ned - best_single_ned) / (oracle_ned - best_single_ned)
    Returns 0.0 if there is no gap to recover.
    """
    gap = oracle_ned - best_single_ned
    if gap == 0.0:
        return 0.0
    return (router_ned - best_single_ned) / gap


def best_single_realized_ned(train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> tuple[float, str]:
    """Pick parser with highest *train* (omni) mean NED, report its mean NED on *test* (real5).

    Aligns with 'train on omni, evaluate on real5' baselines.
    """
    champion = train_matrix.mean(axis=0).idxmax()
    # column may be missing on test for some parsers — use float mean with NaNs skipped
    if champion not in test_matrix.columns:
        raise ValueError(f"Train champion {champion!r} not in test matrix columns.")
    ned = float(test_matrix[champion].mean(skipna=True))
    return ned, str(champion)


def per_page_ned(selections: pd.Series, matrix: pd.DataFrame) -> pd.Series:
    """Look up each page's NED score given the model selected for that page."""
    return pd.Series(
        {page_id: matrix.at[page_id, model] for page_id, model in selections.items()},
        name="ned_score",
    )


def routing_summary(
    selections: pd.Series,
    matrix: pd.DataFrame,
    label: str = "router",
    oracle_ned: float | None = None,
    best_single_ned: float | None = None,
) -> dict:
    """Compute a summary dict for a routing policy.

    Returns keys: label, mean_ned, oracle_gap_pct, n_pages.
    oracle_ned and best_single_ned are computed from matrix if not provided.
    """
    scores = per_page_ned(selections, matrix)
    router_ned = mean_ned(scores)
    if oracle_ned is None:
        oracle_ned = float(matrix.max(axis=1).mean())
    if best_single_ned is None:
        best_single_ned = float(matrix.mean(axis=0).max())
    return {
        "label": label,
        "mean_ned": router_ned,
        "oracle_gap_pct": oracle_gap_recovered(router_ned, best_single_ned, oracle_ned),
        "n_pages": len(scores),
    }
