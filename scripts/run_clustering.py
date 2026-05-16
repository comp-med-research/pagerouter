"""
Experiment 2 — Behavioral clustering.

Clusters models by cosine similarity of their 1,355-dim NED score vectors.
Outputs figures/clustering_dendrogram.pdf and results/pairwise_cosine.csv.

Usage:
  python scripts/run_clustering.py
  python scripts/run_clustering.py --dataset omni
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pagerouter import clustering, load, viz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--dataset", choices=["omni", "real5"], default="omni")
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

    vectors = clustering.get_score_vectors(matrix)
    sim = clustering.compute_pairwise_cosine(vectors)
    sim.to_csv(results_dir / f"pairwise_cosine_{args.dataset}.csv")

    linkage = clustering.hierarchical_cluster(sim)
    viz.plot_clustering_dendrogram(
        linkage,
        model_labels=matrix.columns.tolist(),
        out_path=figures_dir / f"clustering_dendrogram_{args.dataset}.pdf",
    )
    print(f"Wrote clustering_dendrogram_{args.dataset}.pdf")


if __name__ == "__main__":
    main()
