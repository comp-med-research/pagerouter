"""
Experiment 1 — Page-level capability profiles.

Computes mean NED per model per stratum and writes model × stratum matrices.
Outputs figures/capability_heatmap_doctype.pdf and figures/capability_heatmap_layout.pdf.

Usage:
  python scripts/run_profiles.py
  python scripts/run_profiles.py --dataset omni
  python scripts/run_profiles.py --out-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pagerouter import load, profiles, viz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--dataset", choices=["omni", "real5", "both"], default="omni")
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="Figure output directory (default: project figures/)")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="CSV output directory (default: project results/)")
    args = ap.parse_args()

    figures_dir = args.figures_dir or args.out_dir
    results_dir = args.results_dir or RESULTS

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)

    for stratum_col in ["doc_type", "layout_type"]:
        matrix = profiles.compute_score_matrix(df, stratum_col)
        matrix.to_csv(results_dir / f"profiles_{stratum_col}.csv")
        viz.plot_capability_heatmap(
            matrix,
            stratum_col=stratum_col,
            out_path=figures_dir / f"capability_heatmap_{stratum_col}.pdf",
        )
        print(f"Wrote capability_heatmap_{stratum_col}.pdf")


if __name__ == "__main__":
    main()
