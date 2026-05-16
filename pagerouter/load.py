"""Data loading and validation for pagerouter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {"page_id", "model", "ned_score", "doc_type", "layout_type", "dataset"}
VALID_DATASETS = {"omni", "real5"}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OMNI  = DATA_DIR / "omni_predictions.csv"
DEFAULT_REAL5 = DATA_DIR / "real5_predictions.csv"


def load_predictions(
    omni_path: str | Path = DEFAULT_OMNI,
    real5_path: str | Path = DEFAULT_REAL5,
) -> pd.DataFrame:
    """Load and combine OmniDocBench and Real5 prediction CSVs.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns:
        [page_id, model, ned_score, doc_type, layout_type, dataset]
    """
    omni  = pd.read_csv(omni_path)
    real5 = pd.read_csv(real5_path)
    df = pd.concat([omni, real5], ignore_index=True)
    df["ned_score"] = pd.to_numeric(df["ned_score"], errors="coerce")
    validate_schema(df)
    return df


def get_matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Pivot to a page × model NED score matrix for one dataset.

    Parameters
    ----------
    df:
        Combined dataframe as returned by load_predictions().
    dataset:
        One of "omni" or "real5".

    Returns
    -------
    pd.DataFrame
        Shape (n_pages, 14). Index is page_id, columns are model names.
        Missing predictions are left as NaN.
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {dataset!r}")
    subset = df[df["dataset"] == dataset]
    return subset.pivot(index="page_id", columns="model", values="ned_score")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the dataframe does not conform to the expected schema.

    Checks
    ------
    - All expected columns are present.
    - ned_score is in [0, 1].
    - dataset values are in VALID_DATASETS.
    - No duplicate (page_id, model, dataset) rows.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    bad_scores = df["ned_score"].notna() & ((df["ned_score"] < 0) | (df["ned_score"] > 1))
    if bad_scores.any():
        raise ValueError(f"{bad_scores.sum()} rows have ned_score outside [0, 1]")

    bad_datasets = ~df["dataset"].isin(VALID_DATASETS)
    if bad_datasets.any():
        bad = df.loc[bad_datasets, "dataset"].unique().tolist()
        raise ValueError(f"Unexpected dataset values: {bad}")

    dupes = df.duplicated(subset=["page_id", "model", "dataset"])
    if dupes.any():
        raise ValueError(f"{dupes.sum()} duplicate (page_id, model, dataset) rows")
