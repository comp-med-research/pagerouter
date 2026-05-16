"""
Experiment 3 — Oracle and complementarity analysis.

Computes oracle NED curves, pairwise complementarity, and per-stratum oracle gaps.
Outputs headline oracle bar chart, complementarity heatmap, global oracle-vs-models chart,
and **per-stratum** oracle-vs-models PDFs under
``figures/oracle_vs_models_by_{doc_type,layout_type}_{dataset}/``.

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
    ap.add_argument(
        "--min-pages-stratum",
        type=int,
        default=5,
        help="Minimum pages required to emit one oracle-vs-models PDF per stratum (default: 5)",
    )
    ap.add_argument(
        "--skip-stratum-oracle-plots",
        action="store_true",
        help="Do not write per-stratum oracle_vs_models PDFs",
    )
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
    matrix = load.get_matrix(df, dataset=args.dataset)

    curve = oracle.oracle_curve(matrix)
    curve.to_csv(results_dir / f"oracle_curve_{args.dataset}.csv", header=True)

    oracle_one = evaluate.mean_ned(matrix.max(axis=1))
    model_means = matrix.mean(axis=0).sort_values(ascending=False)
    model_means.rename("mean_ned").to_csv(results_dir / f"model_mean_ned_{args.dataset}.csv")
    best_fixed_mean = float(model_means.max())

    print(f"Oracle-1 (mean per-page max NED): {oracle_one:.4f}")
    print(f"Best fixed model mean NED:       {best_fixed_mean:.4f} ({model_means.idxmax()})")
    print(f"Oracle gain: {oracle.oracle_gain(matrix):.4f}")
    print(f"Oracle curve:\n{curve}")

    viz.plot_oracle_vs_models(
        matrix,
        out_path=figures_dir / f"oracle_vs_models_{args.dataset}.pdf",
        title=f"Oracle-1 vs fixed models ({args.dataset})",
        dataset_scope=args.dataset,
    )

    if not args.skip_stratum_oracle_plots:
        n_written = 0
        for col in ("doc_type", "layout_type"):
            paths = viz.plot_oracle_vs_models_per_stratum(
                df,
                matrix,
                stratum_col=col,
                out_dir=figures_dir,
                dataset=args.dataset,
                min_pages=args.min_pages_stratum,
            )
            n_written += len(paths)
            print(f"  Stratum oracle plots ({col}): {len(paths)} PDFs → {figures_dir / f'oracle_vs_models_by_{col}_{args.dataset}'}")
        print(f"  Total stratum oracle-vs-models PDFs: {n_written}")

    viz.plot_oracle_barchart(
        curve,
        best_single_ned=best_fixed_mean,
        out_path=figures_dir / f"oracle_barchart_{args.dataset}.pdf",
    )

    comp = oracle.complementarity_matrix(matrix, threshold=args.threshold)
    comp.to_csv(results_dir / f"complementarity_{args.dataset}.csv")
    viz.plot_complementarity_heatmap(
        comp,
        out_path=figures_dir / f"complementarity_{args.dataset}.pdf",
    )

    stratum_gaps = oracle.per_stratum_oracle_gap(df, matrix)
    stratum_gaps.to_csv(results_dir / f"stratum_oracle_gaps_{args.dataset}.csv", index=False)
    print(f"Wrote oracle figures and results for dataset={args.dataset}")

    viz.plot_coverage_curves(
        curve,
        out_path=figures_dir / f"coverage_curves_{args.dataset}.pdf",
    )


if __name__ == "__main__":
    main()
