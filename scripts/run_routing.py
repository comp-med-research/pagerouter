"""
Experiment 4 — Lightweight routing baselines.

Trains routers on OmniDocBench and evaluates on Real5 (cross-domain).
Reports mean NED and % of oracle gap recovered for each router.

Routers: BestSingleRouter, StratumMeanChampionRouter (doc / layout / joint),
MetadataRouter, LogisticRouter, XGBoostRouter.

Usage:
  python scripts/run_routing.py
  python scripts/run_routing.py --routers best metadata logistic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pagerouter import evaluate, load, viz
from pagerouter.routing import (
    BestSingleRouter,
    LogisticRouter,
    MetadataRouter,
    StratumMeanChampionRouter,
    XGBoostRouter,
    train_stratum_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

ROUTER_REGISTRY = {
    "best":         BestSingleRouter,
    "stratum_doc":  lambda: StratumMeanChampionRouter("doc_type"),
    "stratum_layout": lambda: StratumMeanChampionRouter("layout_type"),
    "stratum_joint": lambda: StratumMeanChampionRouter("both"),
    "metadata":     MetadataRouter,
    "logistic":     LogisticRouter,
    "xgboost":      XGBoostRouter,
}

# Slugs in --routers vs plot / console labels
ROUTER_LABELS = {
    "best":           "Best OmniDoc model",
    "stratum_doc":    "Best Omni mean (doc-type stratum)",
    "stratum_layout": "Best Omni mean (layout stratum)",
    "stratum_joint":  "Best Omni mean (doc×layout stratum)",
    "metadata":       "metadata",
    "logistic":       "logistic",
    "xgboost":        "xgboost",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument(
        "--routers",
        nargs="+",
        choices=list(ROUTER_REGISTRY),
        default=list(ROUTER_REGISTRY),
    )
    ap.add_argument(
        "--test-set-label",
        default="Real5",
        help="Short name for target split in plot x-axis / fixed-model legend (default: Real5)",
    )
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="Figure output directory (default: project figures/)")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Reserved for future CSV logs (figures only today)")
    args = ap.parse_args()

    figures_dir = args.figures_dir or args.out_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir or (figures_dir.parent / "metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)

    train_df = df[df["dataset"] == "omni"]
    test_df  = df[df["dataset"] == "real5"]
    train_matrix = load.get_matrix(train_df, dataset="omni")
    test_matrix  = load.get_matrix(test_df, dataset="real5")

    oracle_ned = evaluate.mean_ned(test_matrix.max(axis=1))
    best_fixed_on_test = float(test_matrix.mean(axis=0).max())
    summaries = []
    fallback_rows: list[dict] = []
    stratum_detail_frames: list = []

    for name in args.routers:
        router = ROUTER_REGISTRY[name]()
        router.fit(train_matrix, train_df)
        selections = router.predict(test_df)
        label = ROUTER_LABELS[name]
        summary = evaluate.routing_summary(selections, test_matrix, label=label)
        summaries.append(summary)
        print(f"{label}: mean_ned={summary['mean_ned']:.4f}  oracle_gap_pct={summary['oracle_gap_pct']:.1%}")

        if hasattr(router, "diagnose"):
            page_diag = router.diagnose(test_df)
            page_diag.insert(0, "router", label)
            train_tbl = train_stratum_snapshot(router)
            n_train_strata = len(train_tbl) if train_tbl is not None else 1
            fallback_rows.append(
                evaluate.summarize_routing_fallback(
                    page_diag,
                    label,
                    n_train_strata=n_train_strata,
                )
            )
            stratum_detail_frames.append(
                evaluate.summarize_routing_fallback_by_stratum(page_diag, label, train_tbl)
            )
        else:
            fallback_rows.append(
                {
                    "router": label,
                    "n_test_pages": len(test_df["page_id"].unique()),
                    "n_stratum_champion": None,
                    "n_fallback": None,
                    "frac_stratum_champion": None,
                    "frac_fallback": None,
                    "n_train_strata": None,
                }
            )

    viz.plot_routing_results(
        summaries,
        oracle_ned=oracle_ned,
        out_path=figures_dir / "routing_results.pdf",
        best_fixed_ned=best_fixed_on_test,
        test_set_label=args.test_set_label,
    )
    print("Wrote routing_results.pdf")

    fallback_summary = pd.DataFrame(fallback_rows)
    fallback_path = results_dir / "routing_fallback_summary.csv"
    fallback_summary.to_csv(fallback_path, index=False)
    print(f"Wrote {fallback_path}")
    print(fallback_summary.to_string(index=False))

    if stratum_detail_frames:
        stratum_detail = pd.concat(stratum_detail_frames, ignore_index=True)
        detail_path = results_dir / "routing_stratum_detail.csv"
        stratum_detail.to_csv(detail_path, index=False)
        print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
