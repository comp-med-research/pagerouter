"""
Build pagerouter CSVs from OmniDocBench quick_match scores on the v1.6 hard (296) subset.

Writes under data/hard296/:
  - omni_slice.csv          — NED rows with dataset=omni (oracle / profiles / clustering on hard pages)
  - empty_real5_stub.csv    — header-only stub so load_predictions() works
  - test_as_real5.csv       — same scores as omni_slice but dataset=real5 for Experiment 4:
                              train on full data/omni_predictions.csv, "test" on hard slice.

Scores column ``score`` becomes ``ned_score``. Renames glm_ocr → glmocr to match pagerouter.

Usage:
  python scripts/prepare_hard296_pagerouter.py
  python scripts/prepare_hard296_pagerouter.py --scores /path/to/scores.csv --gt /path/to/OmniDocBench_v16_hard296.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MODEL_RENAME = {"glm_ocr": "glmocr"}

DEFAULT_SCORES = ROOT.parent / "rankshift" / "results" / "omnidocbench_eval" / "omnidoc_e2e_quick_match_hard296" / "scores.csv"
DEFAULT_GT = ROOT.parent / "rankshift" / "data" / "omnidocbench" / "OmniDocBench_v16_hard296.json"


def _empty_stub_frame() -> pd.DataFrame:
    """Typed empty frame so concat with scored omni preserves float ned_score."""
    return pd.DataFrame(
        {
            "page_id": pd.Series(dtype="string"),
            "model": pd.Series(dtype="string"),
            "ned_score": pd.Series(dtype="float64"),
            "doc_type": pd.Series(dtype="string"),
            "layout_type": pd.Series(dtype="string"),
            "dataset": pd.Series(dtype="string"),
        }
    )


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES, help="rankshift scores.csv (long format)")
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT, help="296-page OmniDocBench GT JSON")
    ap.add_argument("--out-dir", type=Path, default=DATA / "hard296", help="Output directory")
    ap.add_argument("--alignment", default="quick_match", help="Keep only rows with this alignment value")
    args = ap.parse_args()

    if not args.scores.is_file():
        raise SystemExit(f"Scores file not found: {args.scores}")
    if not args.gt.is_file():
        raise SystemExit(f"GT JSON not found: {args.gt}")

    lookup = gt_lookup(args.gt)
    raw = pd.read_csv(args.scores)
    if "alignment" in raw.columns:
        raw = raw[raw["alignment"] == args.alignment]
    missing_gt = []
    rows = []
    for _, r in raw.iterrows():
        img = str(r["image"])
        if img not in lookup:
            missing_gt.append(img)
            continue
        m = MODEL_RENAME.get(str(r["model_name"]), str(r["model_name"]))
        attr = lookup[img]
        ned = float(r["score"])
        rows.append(
            {
                "page_id": img,
                "model": m,
                "ned_score": ned,
                "doc_type": attr["doc_type"],
                "layout_type": attr["layout_type"],
                "dataset": "omni",
            }
        )
    if missing_gt:
        raise SystemExit(f"{len(missing_gt)} score rows missing from GT (showing up to 5): {missing_gt[:5]}")

    omni_df = pd.DataFrame(rows)
    real5_df = omni_df.copy()
    real5_df["dataset"] = "real5"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    omni_path = args.out_dir / "omni_slice.csv"
    stub_path = args.out_dir / "empty_real5_stub.csv"
    test_path = args.out_dir / "test_as_real5.csv"

    omni_df.to_csv(omni_path, index=False)
    real5_df.to_csv(test_path, index=False)
    _empty_stub_frame().to_csv(stub_path, index=False)

    print(f"Wrote {omni_path} ({len(omni_df)} rows, {omni_df['page_id'].nunique()} pages)")
    print(f"Wrote {test_path} (dataset=real5, same rows)")
    print(f"Wrote {stub_path} (0 rows)")


if __name__ == "__main__":
    main()
