"""
Run pagerouter Experiments 1–4 on the OmniDocBench v1.6 hard (296-page) subset.

1. Builds CSVs via prepare_hard296_pagerouter.py (rankshift quick_match scores → data/hard296/).
2. Experiments 1–3 use only the hard slice (omni): oracle, profiles, clustering.
3. Experiment 4 trains routers on the usual full ``data/omni_predictions.csv`` (v1.5-era pages)
   and evaluates on the hard slice loaded as ``dataset=real5`` (cross-eval naming only —
   scores are still OmniDocBench-digital on those 296 pages).

Outputs go under ``results/baselines_no_train/hard296/`` (metrics + baseline figures) and
``results/routers_tabular/hard296/figures/`` (tabular routing PDF).

Usage:
  cd /path/to/pagerouter
  python scripts/run_hard296_experiments.py
  python scripts/run_hard296_experiments.py --skip-prepare
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_H = ROOT / "data" / "hard296"
BASELINES_H = ROOT / "results" / "baselines_no_train" / "hard296"
RESULTS_H = BASELINES_H / "metrics"
FIGURES_H = BASELINES_H / "figures"
ROUTING_FIGURES_H = ROOT / "results" / "routers_tabular" / "hard296" / "figures"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-prepare", action="store_true", help="Assume data/hard296/*.csv already built")
    ap.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Override rankshift scores.csv path (passed to prepare_hard296_pagerouter.py)",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Override OmniDocBench_v16_hard296.json path",
    )
    args = ap.parse_args()

    py = sys.executable
    RESULTS_H.mkdir(parents=True, exist_ok=True)
    FIGURES_H.mkdir(parents=True, exist_ok=True)
    ROUTING_FIGURES_H.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prep = [
            py,
            str(ROOT / "scripts" / "prepare_hard296_pagerouter.py"),
            "--out-dir",
            str(DATA_H),
        ]
        if args.scores is not None:
            prep += ["--scores", str(args.scores)]
        if args.gt is not None:
            prep += ["--gt", str(args.gt)]
        run(prep)

    omni_slice = DATA_H / "omni_slice.csv"
    stub = DATA_H / "empty_real5_stub.csv"
    test_real5 = DATA_H / "test_as_real5.csv"
    train_omni = ROOT / "data" / "omni_predictions.csv"

    for p in (omni_slice, stub, test_real5, train_omni):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")

    common = [
        "--figures-dir",
        str(FIGURES_H),
        "--results-dir",
        str(RESULTS_H),
    ]

    run(
        [
            py,
            str(ROOT / "scripts" / "run_profiles.py"),
            "--omni",
            str(omni_slice),
            "--real5",
            str(stub),
            "--dataset",
            "omni",
            *common,
        ]
    )
    run(
        [
            py,
            str(ROOT / "scripts" / "run_clustering.py"),
            "--omni",
            str(omni_slice),
            "--real5",
            str(stub),
            "--dataset",
            "omni",
            *common,
        ]
    )
    run(
        [
            py,
            str(ROOT / "scripts" / "run_oracle.py"),
            "--omni",
            str(omni_slice),
            "--real5",
            str(stub),
            "--dataset",
            "omni",
            *common,
        ]
    )
    run(
        [
            py,
            str(ROOT / "scripts" / "run_routing.py"),
            "--omni",
            str(train_omni),
            "--real5",
            str(test_real5),
            "--figures-dir",
            str(ROUTING_FIGURES_H),
            "--test-set-label",
            "hard296",
        ]
    )

    print(f"\nDone. Baseline figures → {FIGURES_H}\n       Baseline tables → {RESULTS_H}")
    print(f"       Tabular routing PDF → {ROUTING_FIGURES_H / 'routing_results.pdf'}")


if __name__ == "__main__":
    main()
