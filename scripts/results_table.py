"""
Aggregate MLP router runs and baselines into ``results/mlp_ablation.csv``.

Reads ``results/mlp_runs_detail.csv`` (from ``train_mlp.py``), computes mean/std
across seeds grouped by (encoder, label_type), and appends oracle / best-single /
stratum-table baselines on Real5.

Usage::

  PYTHONPATH=. python scripts/results_table.py
  PYTHONPATH=. python scripts/results_table.py --detail-csv results/mlp_runs_detail.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter.evaluate import (  # noqa: E402
    best_single_realized_ned,
    mean_ned,
    oracle_gap_recovered,
    per_page_ned,
)
from pagerouter.load import DEFAULT_OMNI, DEFAULT_REAL5, get_matrix, load_predictions  # noqa: E402
from pagerouter.mlp_labels import per_stratum_baseline_selections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail-csv", type=Path, default=ROOT / "results" / "mlp_runs_detail.csv")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "results" / "mlp_ablation.csv")
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    args = ap.parse_args()

    df = load_predictions(args.omni, args.real5)
    df["page_id"] = df["page_id"].astype(str)
    train_df = df[df["dataset"] == "omni"]
    train_mat = get_matrix(train_df, "omni")
    test_mat = get_matrix(df[df["dataset"] == "real5"], "real5")
    train_mat = train_mat.loc[train_mat.notna().any(axis=1)]
    test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
    train_mat.index = train_mat.index.astype(str)
    test_mat.index = test_mat.index.astype(str)

    oracle_ned = float(test_mat.max(axis=1).mean())
    best_single_ned, champion = best_single_realized_ned(train_mat, test_mat)
    stratum_sel = per_stratum_baseline_selections(test_mat, train_mat, df, train_df)
    stratum_ned = mean_ned(per_page_ned(stratum_sel, test_mat))

    baseline_rows = [
        {
            "kind": "baseline",
            "method": "oracle_upper_bound",
            "encoder": "",
            "label_type": "",
            "mean_ned_real5": oracle_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": 1.0,
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": "per-page max over parsers on real5",
        },
        {
            "kind": "baseline",
            "method": "best_single_train_champion",
            "encoder": "",
            "label_type": "",
            "mean_ned_real5": best_single_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(best_single_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"parser={champion} by omni mean NED, score on real5",
        },
        {
            "kind": "baseline",
            "method": "best_per_stratum_table",
            "encoder": "",
            "label_type": "",
            "mean_ned_real5": stratum_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(stratum_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": "(doc_type,layout_type) mean leaders on omni → real5",
        },
    ]

    rows: list[dict] = list(baseline_rows)

    if args.detail_csv.is_file():
        detail = pd.read_csv(args.detail_csv)
        if not detail.empty:
            g = detail.groupby(["encoder", "label_type"], as_index=False)
            agg = g.agg(
                mean_ned_real5=("mean_ned_real5", "mean"),
                std_ned_real5=("mean_ned_real5", "std"),
                mean_gap=("oracle_gap_recovered", "mean"),
                std_gap=("oracle_gap_recovered", "std"),
                n_seeds=("seed", "count"),
            )
            for _, r in agg.iterrows():
                rows.append(
                    {
                        "kind": "mlp",
                        "method": "mlp_router",
                        "encoder": r["encoder"],
                        "label_type": r["label_type"],
                        "mean_ned_real5": float(r["mean_ned_real5"]),
                        "std_ned_real5": float(r["std_ned_real5"]) if pd.notna(r["std_ned_real5"]) else 0.0,
                        "oracle_gap_recovered": float(r["mean_gap"]),
                        "std_oracle_gap_recovered": float(r["std_gap"]) if pd.notna(r["std_gap"]) else 0.0,
                        "n_seeds": int(r["n_seeds"]),
                        "notes": "frozen encoder + 2-layer MLP; omni→real5",
                    }
                )
    else:
        print(f"Warning: {args.detail_csv} not found — only baselines written.", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
