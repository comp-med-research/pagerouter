"""Aggregate validation shortlist runs (mixed visual / layout / combo rows)."""

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
from pagerouter.load import get_matrix, load_predictions  # noqa: E402
from pagerouter.mlp_labels import (  # noqa: E402
    per_stratum_baseline_selections,
    stratum_table_baseline_selections,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--omni", type=Path, required=True)
    ap.add_argument("--real5", type=Path, required=True)
    ap.add_argument("--test-set-label", default="Real5")
    args = ap.parse_args()

    detail = pd.read_csv(args.detail_csv)
    if detail.empty:
        raise SystemExit(f"Empty detail CSV: {args.detail_csv}")

    full_df = load_predictions(args.omni, args.real5)
    full_df["page_id"] = full_df["page_id"].astype(str)
    train_df = full_df[full_df["dataset"] == "omni"]
    test_label = args.test_set_label
    eval_scope = test_label.lower()

    train_mat = get_matrix(train_df, "omni")
    train_mat = train_mat.loc[train_mat.notna().any(axis=1)]
    train_mat.index = train_mat.index.astype(str)
    test_mat = get_matrix(full_df[full_df["dataset"] == "real5"], "real5")
    test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
    test_mat.index = test_mat.index.astype(str)

    oracle_ned = float(test_mat.max(axis=1).mean())
    best_single_ned, champion = best_single_realized_ned(train_mat, test_mat)
    best_fixed_on_test = float(test_mat.mean(axis=0).max())
    best_fixed_name = str(test_mat.mean(axis=0).idxmax())
    stratum_sel = per_stratum_baseline_selections(test_mat, train_mat, full_df, train_df)
    stratum_ned = mean_ned(per_page_ned(stratum_sel, test_mat))
    doc_type_sel = stratum_table_baseline_selections(test_mat, train_mat, full_df, "doc_type")
    doc_type_ned = mean_ned(per_page_ned(doc_type_sel, test_mat))
    layout_sel = stratum_table_baseline_selections(test_mat, train_mat, full_df, "layout_type")
    layout_ned = mean_ned(per_page_ned(layout_sel, test_mat))

    baseline_rows = [
        _baseline_row(
            "oracle_upper_bound",
            oracle_ned,
            1.0,
            f"per-page max over parsers on {eval_scope}",
            eval_scope,
        ),
        _baseline_row(
            "best_single_train_champion",
            best_single_ned,
            oracle_gap_recovered(best_single_ned, best_single_ned, oracle_ned),
            f"parser={champion} by omni train mean NED, score on {eval_scope}",
            eval_scope,
        ),
        _baseline_row(
            "best_fixed_on_test",
            best_fixed_on_test,
            oracle_gap_recovered(best_fixed_on_test, best_single_ned, oracle_ned),
            f"best single parser by {eval_scope} column mean ({best_fixed_name})",
            eval_scope,
        ),
        _baseline_row(
            "best_per_stratum_table",
            stratum_ned,
            oracle_gap_recovered(stratum_ned, best_single_ned, oracle_ned),
            f"(doc_type,layout_type) mean leaders on omni train → {eval_scope}",
            eval_scope,
        ),
        _baseline_row(
            "best_per_doc_type_table",
            doc_type_ned,
            oracle_gap_recovered(doc_type_ned, best_single_ned, oracle_ned),
            f"doc_type mean leaders on omni train → {eval_scope}",
            eval_scope,
        ),
        _baseline_row(
            "best_per_layout_table",
            layout_ned,
            oracle_gap_recovered(layout_ned, best_single_ned, oracle_ned),
            f"layout_type mean leaders on omni train → {eval_scope}",
            eval_scope,
        ),
    ]

    if "config_id" not in detail.columns:
        detail["config_id"] = (
            detail["feature_mode"].astype(str)
            + "|"
            + detail["encoder"].astype(str)
            + "|"
            + detail.get("feature_fusion", pd.Series(["concat"] * len(detail))).astype(str)
            + "|"
            + detail["label_type"].astype(str)
        )

    group_cols = [
        "router",
        "config_id",
        "encoder",
        "feature_mode",
        "feature_fusion",
        "label_type",
    ]
    agg = detail.groupby(group_cols, as_index=False).agg(
        mean_ned_real5=("mean_ned_real5", "mean"),
        std_ned_real5=("mean_ned_real5", "std"),
        mean_gap=("oracle_gap_recovered", "mean"),
        std_gap=("oracle_gap_recovered", "std"),
        n_seeds=("seed", "count"),
    )

    rows = list(baseline_rows)
    for _, r in agg.iterrows():
        rows.append(
            {
                "kind": str(r["router"]),
                "method": f"{r['router']}_router",
                "config_id": r["config_id"],
                "encoder": r["encoder"],
                "feature_mode": r["feature_mode"],
                "feature_fusion": r["feature_fusion"],
                "label_type": r["label_type"],
                "mean_ned_real5": float(r["mean_ned_real5"]),
                "std_ned_real5": float(r["std_ned_real5"]) if pd.notna(r["std_ned_real5"]) else 0.0,
                "oracle_gap_recovered": float(r["mean_gap"]),
                "std_oracle_gap_recovered": float(r["std_gap"]) if pd.notna(r["std_gap"]) else 0.0,
                "n_seeds": int(r["n_seeds"]),
                "notes": f"omni train → {eval_scope}",
            }
        )

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)")


def _baseline_row(
    method: str,
    ned: float,
    gap: float,
    notes: str,
    eval_scope: str,
) -> dict[str, object]:
    return {
        "kind": "baseline",
        "method": method,
        "config_id": "",
        "encoder": "",
        "feature_mode": "",
        "feature_fusion": "",
        "label_type": "",
        "mean_ned_real5": ned,
        "std_ned_real5": "",
        "oracle_gap_recovered": gap,
        "std_oracle_gap_recovered": "",
        "n_seeds": "",
        "notes": notes,
    }


if __name__ == "__main__":
    main()
