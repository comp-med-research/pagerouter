"""
Experiment 4 — Lightweight routing baselines.

Trains routers on OmniDocBench and evaluates on Real5 (cross-domain).
Reports mean NED and % of oracle gap recovered for each router.

Routers: BestSingleRouter, MetadataRouter, LogisticRouter, XGBoostRouter.

Usage:
  python scripts/run_routing.py
  python scripts/run_routing.py --routers best metadata logistic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pagerouter import evaluate, load, viz
from pagerouter.routing import (
    BestSingleRouter,
    LogisticRouter,
    MetadataRouter,
    XGBoostRouter,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

ROUTER_REGISTRY = {
    "best":     BestSingleRouter,
    "metadata": MetadataRouter,
    "logistic": LogisticRouter,
    "xgboost":  XGBoostRouter,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--routers", nargs="+", choices=list(ROUTER_REGISTRY),
                    default=list(ROUTER_REGISTRY))
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)

    train_df = df[df["dataset"] == "omni"]
    test_df  = df[df["dataset"] == "real5"]
    train_matrix = load.get_matrix(train_df, dataset="omni")
    test_matrix  = load.get_matrix(test_df, dataset="real5")

    oracle_ned = evaluate.mean_ned(test_matrix.max(axis=1))
    summaries = []

    for name in args.routers:
        router = ROUTER_REGISTRY[name]()
        router.fit(train_matrix, train_df)
        selections = router.predict(test_df)
        summary = evaluate.routing_summary(selections, test_matrix, label=name)
        summaries.append(summary)
        print(f"{name}: mean_ned={summary['mean_ned']:.4f}  oracle_gap_pct={summary['oracle_gap_pct']:.1%}")

    viz.plot_routing_results(
        summaries,
        oracle_ned=oracle_ned,
        out_path=args.out_dir / "routing_results.pdf",
    )
    print("Wrote routing_results.pdf")


if __name__ == "__main__":
    main()
