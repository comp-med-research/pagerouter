"""
Oracle-1 vs fixed models for OmniDocBench **module** scores (CDM, TEDS, reading-order).

Reads ``*_quick_match_{display_formula,table,reading_order}_result.json`` under a quick_match
raw directory, aggregates mean metric per page (mean CDM / mean TEDS / mean (1 − edit) for RO),
joins ``doc_type`` / ``layout_type`` from the OmniDocBench GT JSON, then reuses the Experiment 3
oracle-vs-models plots with metric-appropriate axis labels.

Outputs (per metric) under ``--figures-dir/<metric>/``::

  - oracle_vs_models_{omni|real5}.pdf
  - oracle_vs_models_by_{doc_type,layout_type}_{omni|real5}/*.pdf

Usage::

  PYTHONPATH=. python scripts/run_oracle_module_metrics.py
  PYTHONPATH=. python scripts/run_oracle_module_metrics.py \\
      --gt-json /path/to/OmniDocBench_v16_hard296.json \\
      --raw-dir /path/to/OmniDocBench/result \\
      --eval-suffix hard296 \\
      --figures-dir figures/module_metrics_hard296
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pagerouter import load, viz
from pagerouter.omnidoc_module_scores import MetricKind, build_module_long_df, module_score_matrix

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW = ROOT.parent / "rankshift" / "results" / "omnidocbench_eval" / "omnidoc_e2e_quick_match" / "raw"
DEFAULT_RAW_REAL5 = ROOT.parent / "rankshift" / "results" / "omnidocbench_eval" / "real5_e2e_quick_match" / "raw"
DEFAULT_GT = ROOT.parent / "rankshift" / "data" / "omnidocbench" / "OmniDocBench.json"

METRIC_META: dict[str, dict[str, str]] = {
    "cdm": {
        "display": "CDM",
    },
    "teds": {
        "display": "TEDS",
    },
    "reading_order": {
        "display": "reading-order accuracy",
    },
}


def run_one(
    kind: MetricKind,
    raw_dir: Path,
    gt_path: Path,
    figures_dir: Path,
    results_dir: Path,
    *,
    min_pages_stratum: int,
    skip_stratum: bool,
    eval_suffix: str | None,
    dataset: str,
) -> None:
    meta = METRIC_META[kind]
    long_df = build_module_long_df(
        raw_dir, gt_path, kind, eval_suffix=eval_suffix, dataset=dataset
    )
    if long_df.empty:
        raise SystemExit(f"No rows built for metric={kind!r} (check raw_dir and gt json).")

    long_df = long_df.copy()
    long_df["ned_score"] = pd.to_numeric(long_df["ned_score"], errors="coerce")
    # Table TEDS can be negative in raw OmniDocBench outputs; clip for oracle plots / schema check.
    long_df["ned_score"] = long_df["ned_score"].clip(lower=0.0, upper=1.0)
    load.validate_schema(long_df)

    matrix = module_score_matrix(long_df, dataset=dataset)
    oracle_one = float(matrix.max(axis=1).mean())
    model_means = matrix.mean(axis=0).sort_values(ascending=False)
    best_fixed_mean = float(model_means.max())
    best_name = model_means.idxmax()

    out_fig = figures_dir / kind
    out_res = results_dir / kind
    out_fig.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    model_means.rename("mean_score").to_csv(out_res / f"model_mean_{kind}_{dataset}.csv")
    print(f"[{kind}] Oracle-1 (mean per-page max): {oracle_one:.4f}")
    print(f"[{kind}] Best fixed model mean:       {best_fixed_mean:.4f} ({best_name})")

    scope = "OmniDocBench (digital)" if dataset == "omni" else "Real5 (scan)"
    disp = meta["display"]
    x_title = f"Oracle-1 vs fixed models ({scope}) — {disp}"

    viz.plot_oracle_vs_models(
        matrix,
        out_path=out_fig / f"oracle_vs_models_{dataset}.pdf",
        title=x_title,
        dataset_scope=dataset,
        metric_display_name=disp,
    )

    if not skip_stratum:
        n_written = 0
        for col in ("doc_type", "layout_type"):
            paths = viz.plot_oracle_vs_models_per_stratum(
                long_df,
                matrix,
                stratum_col=col,
                out_dir=out_fig,
                dataset=dataset,
                min_pages=min_pages_stratum,
                metric_display_name=disp,
            )
            n_written += len(paths)
            print(
                f"[{kind}] Stratum plots ({col}): {len(paths)} PDFs → "
                f"{out_fig / f'oracle_vs_models_by_{col}_{dataset}'}"
            )
        print(f"[{kind}] Total stratum PDFs: {n_written}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--eval-suffix",
        default=None,
        help="OmniDocBench save suffix embedded in filenames (e.g. hard296 for "
             "{model}_hard296_quick_match_*.json). Required when raw dir mixes full-bench and subset runs.",
    )
    ap.add_argument(
        "--dataset",
        choices=["omni", "real5"],
        default="omni",
        help="Which dataset label / eval regime (default: omni)",
    )
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--gt-json", type=Path, default=DEFAULT_GT)
    ap.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRIC_META.keys()),
        default=list(METRIC_META.keys()),
        help="Subset of module metrics (default: all)",
    )
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "module_metrics")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "module_metrics")
    ap.add_argument("--min-pages-stratum", type=int, default=5)
    ap.add_argument("--skip-stratum-oracle-plots", action="store_true")
    args = ap.parse_args()

    if not args.raw_dir:
        args.raw_dir = DEFAULT_RAW_REAL5 if args.dataset == "real5" else DEFAULT_RAW

    if not args.raw_dir.is_dir():
        raise SystemExit(f"Raw dir not found: {args.raw_dir}")
    if not args.gt_json.is_file():
        raise SystemExit(f"GT JSON not found: {args.gt_json}")

    for k in args.metrics:
        run_one(
            k,  # type: ignore[arg-type]
            args.raw_dir,
            args.gt_json,
            args.figures_dir,
            args.results_dir,
            min_pages_stratum=args.min_pages_stratum,
            skip_stratum=args.skip_stratum_oracle_plots,
            eval_suffix=args.eval_suffix,
            dataset=args.dataset,
        )


if __name__ == "__main__":
    main()
