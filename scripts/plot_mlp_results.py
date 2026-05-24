"""
Plot MLP ablation bar chart from ``mlp_ablation.csv``.

Usage::

  PYTHONPATH=. python scripts/plot_mlp_results.py \\
      --ablation-csv results/mlp/real5/metrics/mlp_ablation.csv \\
      --figures-dir results/mlp/real5/figures \\
      --test-set-label Real5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter import viz  # noqa: E402


def _is_fusion_method_ablation(df: pd.DataFrame) -> bool:
    if "feature_fusion" not in df.columns:
        return False
    mlp = df[df["kind"] == "mlp"] if "kind" in df.columns else df
    if mlp.empty:
        return False
    fusions = set(mlp["feature_fusion"].dropna().astype(str).str.strip()) - {"", "nan"}
    return len(fusions) > 1


def _is_multimodal_ablation(df: pd.DataFrame) -> bool:
    if "feature_mode" not in df.columns:
        return False
    mlp = df[df["kind"] == "mlp"] if "kind" in df.columns else df
    if mlp.empty:
        return False
    modes = set(mlp["feature_mode"].dropna().astype(str).str.strip()) - {"", "nan"}
    fusion_modes = {"image_layout", "image_metadata", "layout_metadata", "all"}
    if modes & fusion_modes:
        return True
    primary = modes & {"image", "layout"}
    return len(primary) > 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation-csv", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path, required=True)
    ap.add_argument("--test-set-label", default="Real5")
    ap.add_argument("--out-name", default="mlp_ablation.pdf")
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="Use feature_mode grouping (writes mlp_multimodal_ablation.pdf by default)",
    )
    ap.add_argument(
        "--fusion-method",
        action="store_true",
        help="Use feature_fusion grouping (image+layout fusion ablation plot)",
    )
    args = ap.parse_args()

    if not args.ablation_csv.is_file():
        raise SystemExit(f"Ablation CSV not found: {args.ablation_csv}")

    df = pd.read_csv(args.ablation_csv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name
    if args.multimodal and out_name == "mlp_ablation.pdf":
        out_name = "mlp_multimodal_ablation.pdf"
    if args.fusion_method and out_name == "mlp_ablation.pdf":
        out_name = "mlp_fusion_method_ablation.pdf"
    out_path = args.figures_dir / out_name
    if args.fusion_method or _is_fusion_method_ablation(df):
        viz.plot_mlp_fusion_method_ablation(df, out_path, test_set_label=args.test_set_label)
    elif args.multimodal or _is_multimodal_ablation(df):
        viz.plot_mlp_multimodal_ablation(df, out_path, test_set_label=args.test_set_label)
    elif "kind" in df.columns and (df["kind"] == "logistic").any():
        viz.plot_mlp_logistic_ablation(df, out_path, test_set_label=args.test_set_label)
    else:
        viz.plot_mlp_ablation(df, out_path, test_set_label=args.test_set_label)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
