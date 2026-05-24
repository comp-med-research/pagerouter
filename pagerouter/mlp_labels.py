"""Oracle and stratum labels for MLP router training (OmniDocBench omni split)."""

from __future__ import annotations

import pandas as pd

from pagerouter.routing import MODELS, StratumMeanChampionRouter, StratumMode


def _nan_argmax_row(row: pd.Series) -> str:
    """Argmax over parsers; skips NaNs; ties broken by first max."""
    if row.isna().all():
        raise ValueError("No valid NED scores for page (all NaN).")
    return str(row.idxmax())


def per_page_oracle_labels(matrix: pd.DataFrame) -> pd.Series:
    """Per-page label = parser with highest NED on that page (oracle on omni).

    Parameters
    ----------
    matrix:
        Training (omni) page × model NED matrix.
    """
    labels = matrix.apply(_nan_argmax_row, axis=1)
    labels.name = "label"
    return labels


def _mean_champion_labels(
    matrix: pd.DataFrame,
    df_omni: pd.DataFrame,
    group_cols: list[str],
) -> pd.Series:
    """Label each page with the parser that has highest mean NED in its train bucket."""
    attrs = (
        df_omni[["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    common = matrix.index.intersection(attrs.index)
    matrix = matrix.loc[common]
    attrs = attrs.loc[common]
    attrs.index.name = "page_id"

    bucket_to_model: dict[tuple[str, ...], str] = {}
    df_join = attrs.reset_index()
    for key_vals, g in df_join.groupby(group_cols, sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (str(key_vals),)
        else:
            key_vals = tuple(str(v) for v in key_vals)
        page_ids = g["page_id"].tolist()
        sub = matrix.loc[page_ids]
        mean_per_model = sub.mean(axis=0, skipna=True)
        if mean_per_model.isna().all():
            raise ValueError(f"Bucket {key_vals!r} has no valid scores.")
        bucket_to_model[key_vals] = str(mean_per_model.idxmax())

    labels = []
    for pid in matrix.index:
        key = tuple(str(attrs.at[pid, col]) for col in group_cols)
        labels.append(bucket_to_model[key])

    return pd.Series(labels, index=matrix.index, name="label")


def per_doc_type_labels(matrix: pd.DataFrame, df_omni: pd.DataFrame) -> pd.Series:
    """Label = parser with highest mean NED for that doc_type on omni train pages."""
    return _mean_champion_labels(matrix, df_omni, ["doc_type"])


def per_layout_labels(matrix: pd.DataFrame, df_omni: pd.DataFrame) -> pd.Series:
    """Label = parser with highest mean NED for that layout_type on omni train pages."""
    return _mean_champion_labels(matrix, df_omni, ["layout_type"])


def per_stratum_labels(matrix: pd.DataFrame, df_omni: pd.DataFrame) -> pd.Series:
    """Stratum = (doc_type, layout_type). Label = parser with highest *mean* NED in that stratum on omni.

    Every page in the same stratum receives the same label.
    """
    return _mean_champion_labels(matrix, df_omni, ["doc_type", "layout_type"])


def stratum_best_lookup(matrix_omni: pd.DataFrame, df_omni: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map (doc_type, layout_type) → parser with highest mean NED on omni in that stratum."""
    attrs = (
        df_omni[["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    common = matrix_omni.index.intersection(attrs.index)
    matrix_omni = matrix_omni.loc[common]
    attrs = attrs.loc[common]
    attrs.index.name = "page_id"

    out: dict[tuple[str, str], str] = {}
    df_join = attrs.reset_index()
    for (doc_t, lay_t), g in df_join.groupby(["doc_type", "layout_type"]):
        page_ids = g["page_id"].tolist()
        sub = matrix_omni.loc[page_ids]
        mean_per_model = sub.mean(axis=0, skipna=True)
        if not mean_per_model.isna().all():
            out[(doc_t, lay_t)] = str(mean_per_model.idxmax())
    return out


def stratum_table_baseline_selections(
    test_matrix: pd.DataFrame,
    train_matrix: pd.DataFrame,
    predictions_df: pd.DataFrame,
    mode: StratumMode,
) -> pd.Series:
    """Apply train mean-NED champions per metadata bucket to each test page."""
    router = StratumMeanChampionRouter(mode).fit(train_matrix, predictions_df)
    test_ids = test_matrix.index.astype(str)
    meta = (
        predictions_df[["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .assign(page_id=lambda d: d["page_id"].astype(str))
    )
    eval_df = meta[meta["page_id"].isin(test_ids)]
    preds = router.predict(eval_df)
    common = preds.index.astype(str).intersection(test_ids)
    return preds.loc[common].rename("model")


def per_stratum_baseline_selections(
    test_matrix: pd.DataFrame,
    train_matrix: pd.DataFrame,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.Series:
    """For each test page, select omni (doc_type, layout_type) champion; fallback to omni best-single."""
    _ = train_df  # kept for backward-compatible call sites
    return stratum_table_baseline_selections(test_matrix, train_matrix, df, "both")


def model_to_index(labels: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Encode string model names to 0..K-1 using fixed MODELS order."""
    name_to_idx = {m: i for i, m in enumerate(MODELS)}
    missing = set(labels.unique()) - set(MODELS)
    if missing:
        raise ValueError(f"Unknown model labels: {missing}")
    y = labels.map(name_to_idx)
    return y, name_to_idx
