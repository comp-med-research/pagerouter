"""Load OmniDocBench module scores (CDM, TEDS, reading-order) from quick_match raw JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

MetricKind = Literal["cdm", "teds", "reading_order"]

MODEL_RENAME = {"glm_ocr": "glmocr"}


def gt_lookup(gt_path: Path) -> dict[str, dict[str, str]]:
    with open(gt_path, encoding="utf-8") as f:
        pages = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for page in pages:
        pi = page.get("page_info") or {}
        img = str(pi.get("image_path") or "")
        base = Path(img).name
        attr = pi.get("page_attribute") or {}
        out[base] = {
            "doc_type": str(attr.get("data_source", "") or ""),
            "layout_type": str(attr.get("layout", "") or ""),
        }
    return out


def list_models(raw_dir: Path, *, eval_suffix: str | None = None) -> list[str]:
    """Table JSON stems ``{model}[_{eval_suffix}]_quick_match`` used to discover models."""
    names: list[str] = []
    suf_tok = eval_suffix.strip("_") if eval_suffix else None
    for p in sorted(raw_dir.glob("*_quick_match_table_result.json")):
        stem = p.name.removesuffix("_quick_match_table_result.json")
        if suf_tok is not None:
            if not stem.endswith(f"_{suf_tok}"):
                continue
        names.append(stem)
    return names


def _stem_to_canonical_model(stem: str, *, eval_suffix: str | None = None) -> str:
    base = stem
    if eval_suffix:
        tok = "_" + eval_suffix.strip("_")
        if base.endswith(tok):
            base = base[: -len(tok)]
    return MODEL_RENAME.get(base, base)


def _reading_order_value(row: dict) -> float | None:
    ed = row.get("edit")
    if ed is None:
        m = row.get("metric") or {}
        ed = m.get("Edit_dist")
    if ed is None:
        return None
    return 1.0 - float(ed)


def _metric_value_factory(kind: MetricKind) -> Callable[[dict], float | None]:
    if kind == "cdm":

        def _fn(row: dict) -> float | None:
            v = (row.get("metric") or {}).get("CDM")
            return float(v) if v is not None else None

        return _fn
    if kind == "teds":

        def _fn(row: dict) -> float | None:
            v = (row.get("metric") or {}).get("TEDS")
            return float(v) if v is not None else None

        return _fn
    if kind == "reading_order":
        return _reading_order_value
    raise ValueError(f"unknown metric kind: {kind!r}")


def _json_suffix(kind: MetricKind) -> str:
    if kind == "cdm":
        return "display_formula_result.json"
    if kind == "teds":
        return "table_result.json"
    if kind == "reading_order":
        return "reading_order_result.json"
    raise ValueError(f"unknown metric kind: {kind!r}")


def aggregate_mean_per_page(rows: list[dict], value_fn: Callable[[dict], float | None]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        img = Path(str(row.get("image_name") or row.get("img_id") or "")).name
        if not img:
            continue
        v = value_fn(row)
        if v is None:
            continue
        buckets[img].append(float(v))
    return {page: sum(vals) / len(vals) for page, vals in buckets.items()}


def build_module_long_df(
    raw_dir: Path,
    gt_path: Path,
    kind: MetricKind,
    *,
    models: list[str] | None = None,
    eval_suffix: str | None = None,
) -> pd.DataFrame:
    """Rows: page_id × model with ned_score = module metric (for oracle tooling)."""
    lookup = gt_lookup(gt_path)
    value_fn = _metric_value_factory(kind)
    json_suffix = _json_suffix(kind)

    model_list = models if models is not None else list_models(raw_dir, eval_suffix=eval_suffix)
    rows: list[dict] = []
    for raw_name in model_list:
        model = _stem_to_canonical_model(raw_name, eval_suffix=eval_suffix)
        path = raw_dir / f"{raw_name}_quick_match_{json_suffix}"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        page_scores = aggregate_mean_per_page(data, value_fn)
        for page_id, score in page_scores.items():
            if page_id not in lookup:
                continue
            attr = lookup[page_id]
            rows.append(
                {
                    "page_id": page_id,
                    "model": model,
                    "ned_score": score,
                    "doc_type": attr["doc_type"],
                    "layout_type": attr["layout_type"],
                    "dataset": "omni",
                }
            )

    return pd.DataFrame(rows)


def module_score_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """page × model matrix; duplicates (page_id, model) averaged."""
    if long_df.empty:
        return pd.DataFrame()
    sub = long_df[long_df["dataset"] == "omni"]
    dedup = sub.groupby(["page_id", "model"], as_index=False)["ned_score"].mean()
    return dedup.pivot(index="page_id", columns="model", values="ned_score")
