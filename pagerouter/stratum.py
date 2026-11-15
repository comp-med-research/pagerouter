"""Stratum mean-champion baseline (labels for MLP / validation)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

StratumMode = Literal["doc_type", "layout_type", "both"]


def _page_attrs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["page_id", "doc_type", "layout_type"]].drop_duplicates("page_id").set_index("page_id")


class StratumMeanChampionRouter:
    """Highest mean train NED parser within a metadata stratum; global fallback if unseen."""

    def __init__(self, mode: StratumMode) -> None:
        self.mode = mode

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "StratumMeanChampionRouter":
        attrs = _page_attrs(df)
        common = matrix.index.intersection(attrs.index)
        mat = matrix.loc[common]
        attrs = attrs.loc[common]
        joined = mat.join(attrs, how="inner")
        model_cols = list(mat.columns)

        self.best_overall: str = joined[model_cols].mean(axis=0).idxmax()
        self.champions: dict[tuple[str, ...], str] = {}
        self.train_bucket_sizes: dict[tuple[str, ...], int] = {}

        if self.mode == "doc_type":
            for dt, g in joined.groupby("doc_type", sort=False):
                key = (str(dt),)
                self.champions[key] = g[model_cols].mean(axis=0).idxmax()
                self.train_bucket_sizes[key] = len(g)
        elif self.mode == "layout_type":
            for lt, g in joined.groupby("layout_type", sort=False):
                key = (str(lt),)
                self.champions[key] = g[model_cols].mean(axis=0).idxmax()
                self.train_bucket_sizes[key] = len(g)
        else:
            for (doc, lay), g in joined.groupby(["doc_type", "layout_type"], sort=False):
                key = (str(doc), str(lay))
                self.champions[key] = g[model_cols].mean(axis=0).idxmax()
                self.train_bucket_sizes[key] = len(g)

        return self

    def _key_for_row(self, row: pd.Series) -> tuple[str, ...]:
        if self.mode == "doc_type":
            return (str(row["doc_type"]),)
        if self.mode == "layout_type":
            return (str(row["layout_type"]),)
        return (str(row["doc_type"]), str(row["layout_type"]))

    @staticmethod
    def format_stratum_key(key: tuple[str, ...]) -> str:
        return "|".join(key)