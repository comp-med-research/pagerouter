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


def summarize_routing_fallback(
    page_diag: pd.DataFrame,
    router: str,
    *,
    n_train_strata: int | None = None,
) -> dict:
    """Aggregate per-page stratum diagnostic into one router-level row."""
    n = len(page_diag)
    n_fallback = int(page_diag["used_fallback"].sum())
    n_champion = n - n_fallback
    if n_train_strata is None:
        seen = page_diag.drop_duplicates("stratum_key")
        n_train_strata = int((seen["train_bucket_size"] > 0).sum())
    return {
        "router": router,
        "n_test_pages": n,
        "n_stratum_champion": n_champion,
        "n_fallback": n_fallback,
        "frac_stratum_champion": n_champion / n if n else 0.0,
        "frac_fallback": n_fallback / n if n else 0.0,
        "n_train_strata": n_train_strata,
    }


def summarize_routing_fallback_by_stratum(
    page_diag: pd.DataFrame,
    router: str,
    train_tbl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per stratum: train bucket size and test-page champion vs fallback counts."""
    test_agg = (
        page_diag.groupby("stratum_key", sort=True)
        .agg(
            n_test_pages=("page_id", "count"),
            n_test_fallback=("used_fallback", "sum"),
        )
        .reset_index()
    )
    test_agg["n_test_stratum_champion"] = test_agg["n_test_pages"] - test_agg["n_test_fallback"]
    test_agg["frac_test_stratum_champion"] = (
        test_agg["n_test_stratum_champion"] / test_agg["n_test_pages"]
    )

    if train_tbl is not None and not train_tbl.empty:
        out = train_tbl.merge(test_agg, on="stratum_key", how="outer")
        out["router"] = router
        out["train_bucket_size"] = out["train_bucket_size"].fillna(0).astype(int)
        out["n_test_pages"] = out["n_test_pages"].fillna(0).astype(int)
        out["n_test_fallback"] = out["n_test_fallback"].fillna(0).astype(int)
        out["n_test_stratum_champion"] = out["n_test_stratum_champion"].fillna(0).astype(int)
        out["frac_test_stratum_champion"] = out["frac_test_stratum_champion"].fillna(0.0)
        cols = [
            "router",
            "stratum_key",
            "train_bucket_size",
            "stratum_champion_model",
            "n_test_pages",
            "n_test_stratum_champion",
            "n_test_fallback",
            "frac_test_stratum_champion",
        ]
        return out[cols].sort_values(["train_bucket_size", "stratum_key"], ascending=[False, True])

    test_agg["router"] = router
    test_agg["train_bucket_size"] = page_diag.groupby("stratum_key")["train_bucket_size"].max().reindex(
        test_agg["stratum_key"]
    ).values
    test_agg["stratum_champion_model"] = page_diag.groupby("stratum_key")["stratum_champion_model"].first().reindex(
        test_agg["stratum_key"]
    ).values
    return test_agg[
        [
            "router",
            "stratum_key",
            "train_bucket_size",
            "stratum_champion_model",
            "n_test_pages",
            "n_test_stratum_champion",
            "n_test_fallback",
            "frac_test_stratum_champion",
        ]
    ]
