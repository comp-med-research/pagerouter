"""
Run Experiments 1–3 (profiles, clustering, oracle) on Real5 NED predictions,
plus module-metric oracle plots (CDM, TEDS, reading order) from rankshift Real5 quick_match raw JSON.

Outputs::

  results/baselines_no_train/real5/metrics/   — CSVs
  results/baselines_no_train/real5/figures/   — PDFs (incl. module_metrics/<metric>/)

Module raw JSONs default to::

  ../rankshift/results/omnidocbench_eval/real5_e2e_quick_match/raw

Usage::

  PYTHONPATH=. python scripts/run_real5_baselines.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "baselines_no_train" / "real5"
METRICS = BASE / "metrics"
FIGURES = BASE / "figures"
DEFAULT_RAW5 = ROOT.parent / "rankshift" / "results" / "omnidocbench_eval" / "real5_e2e_quick_match" / "raw"
DEFAULT_GT = ROOT.parent / "rankshift" / "data" / "omnidocbench" / "OmniDocBench.json"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--module-raw-dir",
        type=Path,
        default=DEFAULT_RAW5,
        help="Directory with *_quick_match_*_result.json (default: rankshift real5_e2e_quick_match/raw)",
    )
    ap.add_argument("--gt-json", type=Path, default=DEFAULT_GT)
    ap.add_argument("--skip-module-metrics", action="store_true")
    args = ap.parse_args()

    omni_csv = ROOT / "data" / "omni_predictions.csv"
    real5_csv = ROOT / "data" / "real5_predictions.csv"
    for p in (omni_csv, real5_csv):
        if not p.is_file():
            raise SystemExit(f"Missing {p}")
    if not args.module_raw_dir.is_dir():
        raise SystemExit(f"Module raw dir not found: {args.module_raw_dir}")
    if not args.gt_json.is_file():
        raise SystemExit(f"GT JSON not found: {args.gt_json}")

    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [
        "--omni",
        str(omni_csv),
        "--real5",
        str(real5_csv),
        "--dataset",
        "real5",
        "--figures-dir",
        str(FIGURES),
        "--results-dir",
        str(METRICS),
    ]

    for script in ("run_profiles.py", "run_clustering.py", "run_oracle.py"):
        run([py, str(ROOT / "scripts" / script), *common])

    if not args.skip_module_metrics:
        mod_fig = FIGURES / "module_metrics"
        mod_res = METRICS / "module_metrics"
        run(
            [
                py,
                str(ROOT / "scripts" / "run_oracle_module_metrics.py"),
                "--dataset",
                "real5",
                "--raw-dir",
                str(args.module_raw_dir),
                "--gt-json",
                str(args.gt_json),
                "--figures-dir",
                str(mod_fig),
                "--results-dir",
                str(mod_res),
            ]
        )

    print(f"\nDone. Real5 baseline → metrics: {METRICS}\n                      figures: {FIGURES}")


if __name__ == "__main__":
    main()
