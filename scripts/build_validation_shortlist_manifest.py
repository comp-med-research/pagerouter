"""Build Real5 / hard296 validation shortlist manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

VISUAL = [
    ("jina-clip-v2", "per_doc_type"),
    ("dinov2-small", "per_doc_type"),
    ("colqwen2-v1.0", "per_doc_type"),
    ("dit-large", "per_stratum"),
    ("clip-l-14", "per_stratum"),
]

LAYOUT = [
    ("doclayout-yolo-stats", "per_layout"),
    ("docling-heron-101-stats", "per_layout"),
    ("layoutlmv3-detected", "per_stratum"),
    ("layoutlmv3-detected", "per_page"),
]

FUSIONS = ("norm_concat", "weighted_avg")
COMBO_MODES = ("image_layout", "all")


def _emb_dim(path: Path) -> int:
    obj = torch.load(path, weights_only=False)
    return int(obj["embeddings"].shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "data" / "embeddings" / "omni_hard296",
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    dims: dict[str, int] = {}
    for enc, _ in VISUAL + LAYOUT:
        if enc in dims:
            continue
        path = args.emb_dir / f"{enc}.pt"
        if not path.is_file():
            raise SystemExit(f"Missing embedding: {path}")
        dims[enc] = _emb_dim(path)

    rows: list[dict[str, str]] = []
    combo_keys: set[tuple[str, str, str, str, str]] = set()

    for visual, v_label in VISUAL:
        rows.append(
            {
                "section": "visual",
                "config_id": f"visual|{visual}|{v_label}",
                "feature_mode": "image",
                "feature_fusion": "concat",
                "visual_encoder": visual,
                "layout_encoder": "",
                "label_type": v_label,
                "skip_reason": "",
            }
        )

    for layout, l_label in LAYOUT:
        rows.append(
            {
                "section": "layout",
                "config_id": f"layout|{layout}|{l_label}",
                "feature_mode": "layout",
                "feature_fusion": "concat",
                "visual_encoder": "",
                "layout_encoder": layout,
                "label_type": l_label,
                "skip_reason": "",
            }
        )

    for visual, v_label in VISUAL:
        for layout, _l_label in LAYOUT:
            label_type = v_label
            for fusion in FUSIONS:
                for mode in COMBO_MODES:
                    key = (visual, layout, fusion, mode, label_type)
                    if key in combo_keys:
                        continue
                    combo_keys.add(key)
                    rows.append(
                        {
                            "section": "combo",
                            "config_id": (
                                f"combo|{visual}+{layout}|{fusion}|{mode}|{label_type}"
                            ),
                            "feature_mode": mode,
                            "feature_fusion": fusion,
                            "visual_encoder": visual,
                            "layout_encoder": layout,
                            "label_type": label_type,
                            "skip_reason": "",
                        }
                    )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    n_skip = int((out["skip_reason"] != "").sum())
    print(f"Wrote {args.out_csv} ({len(out)} rows, {n_skip} skipped)")
    print(json.dumps(dims, indent=2))


if __name__ == "__main__":
    main()
