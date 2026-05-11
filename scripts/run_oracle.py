"""
Experiment 3 — Oracle and complementarity analysis.

Computes oracle NED curves, pairwise complementarity, and per-stratum oracle gaps.
Outputs headline oracle bar chart and complementarity heatmap.

Usage:
  python scripts/run_oracle.py
  python scripts/run_oracle.py --threshold 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pagerouter import evaluate, load, oracle, viz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--dataset", choices=["omni", "real5"], default="omni")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="NED threshold for complementarity (default: 0.8)")
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)
    matrix = load.get_matrix(df, dataset=args.dataset)

    curve = oracle.oracle_curve(matrix)
    curve.to_csv(RESULTS / f"oracle_curve_{args.dataset}.csv", header=True)

    best_single = evaluate.mean_ned(matrix.max(axis=1))
    print(f"Oracle gain: {oracle.oracle_gain(matrix):.4f}")
    print(f"Oracle curve:\n{curve}")

    viz.plot_oracle_barchart(
        curve,
        best_single_ned=best_single,
        out_path=args.out_dir / f"oracle_barchart_{args.dataset}.pdf",
    )

    comp = oracle.complementarity_matrix(matrix, threshold=args.threshold)
    comp.to_csv(RESULTS / f"complementarity_{args.dataset}.csv")
    viz.plot_complementarity_heatmap(
        comp,
        out_path=args.out_dir / f"complementarity_{args.dataset}.pdf",
    )

    stratum_gaps = oracle.per_stratum_oracle_gap(df, matrix)
    stratum_gaps.to_csv(RESULTS / f"stratum_oracle_gaps_{args.dataset}.csv", index=False)
    print(f"Wrote oracle figures and results for dataset={args.dataset}")

    viz.plot_coverage_curves(
        curve,
        out_path=args.out_dir / f"coverage_curves_{args.dataset}.pdf",
    )


if __name__ == "__main__":
    main()
