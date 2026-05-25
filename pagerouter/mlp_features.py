"""Feature construction for multimodal MLP router (image, layout, metadata)."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import torch

from pagerouter.multimodal_fusion import FeatureFusion, l2_normalize_rows

FeatureMode = Literal["image", "layout", "image_layout", "image_metadata", "layout_metadata", "all"]

FEATURE_MODE_LABELS: dict[str, str] = {
    "image": "visual only",
    "layout": "layout only",
    "image_layout": "visual+layout",
    "image_metadata": "visual+metadata",
    "layout_metadata": "layout+metadata",
    "all": "visual+layout+metadata",
}


def _page_attrs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["page_id", "doc_type", "layout_type"]].drop_duplicates("page_id").set_index("page_id")


def fit_metadata_columns(train_df: pd.DataFrame) -> list[str]:
    """One-hot column names from omni train split (doc_type + layout_type)."""
    attrs = _page_attrs(train_df[train_df["dataset"] == "omni"])
    doc = pd.get_dummies(attrs["doc_type"].astype(str), prefix="doc", dtype=float)
    lay = pd.get_dummies(attrs["layout_type"].astype(str), prefix="layout", dtype=float)
    return doc.columns.tolist() + lay.columns.tolist()


def metadata_matrix(
    df: pd.DataFrame,
    page_ids: list[str] | pd.Index,
    *,
    columns: list[str],
) -> torch.Tensor:
    """One-hot metadata rows aligned to ``page_ids`` (unknown strata → zeros)."""
    attrs = _page_attrs(df)
    doc = pd.get_dummies(attrs["doc_type"].astype(str), prefix="doc", dtype=float)
    lay = pd.get_dummies(attrs["layout_type"].astype(str), prefix="layout", dtype=float)
    meta = pd.concat([doc, lay], axis=1)
    meta = meta.reindex(columns=columns, fill_value=0.0)
    rows = meta.reindex([str(p) for p in page_ids], fill_value=0.0)
    return torch.tensor(rows.values, dtype=torch.float32)


def build_feature_matrix(
    mode: FeatureMode,
    *,
    fusion: FeatureFusion = "concat",
    page_ids: list[str] | pd.Index,
    image: torch.Tensor | None = None,
    layout: torch.Tensor | None = None,
    df: pd.DataFrame | None = None,
    metadata_columns: list[str] | None = None,
) -> torch.Tensor:
    """Concatenate modality blocks according to ``mode`` and fusion rule.

    For ``image_layout`` / ``all`` with both tensors present:

    - ``concat`` — ``[img | lay | (meta)]`` (baseline).
    - ``norm_concat`` — L2-normalize img and lay rows, then concatenate.
    - ``weighted_avg`` — mean of normalized img and lay when dims match; otherwise raw
      ``img|lay|(meta)`` for ``WeightedAvgFusionRouter`` (learned projection).
    - ``gmu`` / ``bilinear`` — raw ``img|lay|(meta)``; fusion happens inside ``FusionMLPRouter``.
    """
    needs_image = mode in ("image", "image_layout", "image_metadata", "all")
    needs_layout = mode in ("layout", "image_layout", "layout_metadata", "all")
    needs_meta = mode in ("image_metadata", "layout_metadata", "all")
    needs_pair = needs_image and needs_layout

    di = dl = 0
    if needs_pair:
        assert image is not None and layout is not None
        di, dl = int(image.shape[1]), int(layout.shape[1])

    learned_pair_fusion = fusion in ("gmu", "bilinear") or (
        fusion == "weighted_avg" and needs_pair and di != dl
    )

    fused_block: torch.Tensor | None = None
    if needs_pair:
        assert image is not None and layout is not None
        if learned_pair_fusion:
            fused_block = None
        elif fusion == "weighted_avg":
            fused_block = 0.5 * (l2_normalize_rows(image) + l2_normalize_rows(layout))
        elif fusion == "norm_concat":
            fused_block = torch.cat([l2_normalize_rows(image), l2_normalize_rows(layout)], dim=1)
        elif fusion == "concat":
            fused_block = torch.cat([image, layout], dim=1)
        else:
            raise ValueError(f"Unknown fusion={fusion!r}")

    parts: list[torch.Tensor] = []

    if needs_pair:
        assert image is not None and layout is not None
        if learned_pair_fusion:
            parts.append(torch.cat([image, layout], dim=1))
        elif fused_block is not None:
            parts.append(fused_block)
    elif needs_image:
        if image is None:
            raise ValueError(f"feature_mode={mode!r} requires image embeddings")
        parts.append(l2_normalize_rows(image) if fusion == "norm_concat" else image)
    elif needs_layout:
        if layout is None:
            raise ValueError(f"feature_mode={mode!r} requires layout embeddings")
        parts.append(l2_normalize_rows(layout) if fusion == "norm_concat" else layout)

    if needs_meta:
        if df is None or metadata_columns is None:
            raise ValueError(f"feature_mode={mode!r} requires df and metadata_columns")
        parts.append(metadata_matrix(df, page_ids, columns=metadata_columns))

    if not parts:
        raise ValueError(f"No feature blocks for mode={mode!r}")
    return torch.cat(parts, dim=1)
