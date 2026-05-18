"""Oracle and stratum labels for MLP router training (OmniDocBench omni split)."""

from __future__ import annotations

import pandas as pd

from pagerouter.routing import MODELS


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


def per_stratum_labels(matrix: pd.DataFrame, df_omni: pd.DataFrame) -> pd.Series:
    """Stratum = (doc_type, layout_type). Label = parser with highest *mean* NED in that stratum on omni.

    Every page in the same stratum receives the same label.
    """
    attrs = (
        df_omni[["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    common = matrix.index.intersection(attrs.index)
    matrix = matrix.loc[common]
    attrs = attrs.loc[common]
    attrs.index.name = "page_id"

    stratum_to_model: dict[tuple[str, str], str] = {}
    df_join = attrs.reset_index()
    for (doc_t, lay_t), g in df_join.groupby(["doc_type", "layout_type"]):
        page_ids = g["page_id"].tolist()
        sub = matrix.loc[page_ids]
        mean_per_model = sub.mean(axis=0, skipna=True)
        if mean_per_model.isna().all():
            raise ValueError(f"Stratum {(doc_t, lay_t)!r} has no valid scores.")
        stratum_to_model[(doc_t, lay_t)] = str(mean_per_model.idxmax())

    labels = []
    for pid in matrix.index:
        key = (attrs.at[pid, "doc_type"], attrs.at[pid, "layout_type"])
        labels.append(stratum_to_model[key])

    out = pd.Series(labels, index=matrix.index, name="label")
    return out


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


def per_stratum_baseline_selections(
    test_matrix: pd.DataFrame,
    train_matrix: pd.DataFrame,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.Series:
    """For each real5 page, select omni stratum-best parser; fallback to omni best-single."""
    lookup = stratum_best_lookup(train_matrix, train_df)
    fallback = str(train_matrix.mean(axis=0).idxmax())
    attrs = (
        df[df["dataset"] == "real5"][["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    preds: dict[str, str] = {}
    for pid in test_matrix.index.astype(str):
        if pid not in attrs.index:
            continue
        key = (str(attrs.at[pid, "doc_type"]), str(attrs.at[pid, "layout_type"]))
        preds[pid] = lookup.get(key, fallback)
    return pd.Series(preds, name="model")


def model_to_index(labels: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Encode string model names to 0..K-1 using fixed MODELS order."""
    name_to_idx = {m: i for i, m in enumerate(MODELS)}
    missing = set(labels.unique()) - set(MODELS)
    if missing:
        raise ValueError(f"Unknown model labels: {missing}")
    y = labels.map(name_to_idx)
    return y, name_to_idx
