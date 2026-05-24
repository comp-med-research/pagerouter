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
from pagerouter.mlp_labels import per_stratum_baseline_selections, stratum_table_baseline_selections  # noqa: E402
from pagerouter.mlp_splits import omni_holdout_page_ids  # noqa: E402


def _is_fusion_method_detail(detail: pd.DataFrame) -> bool:
    """True when comparing fusion rules on image+layout (norm_concat, GMU, …)."""
    if "feature_fusion" not in detail.columns or "feature_mode" not in detail.columns:
        return False
    modes = set(detail["feature_mode"].dropna().astype(str).str.strip()) - {"", "nan"}
    if modes - {"image_layout"}:
        return False
    fusions = set(detail["feature_fusion"].dropna().astype(str).str.strip()) - {"", "nan", "concat"}
    return len(fusions) > 1


def _is_multimodal_detail(detail: pd.DataFrame) -> bool:
    """True when detail CSV compares multiple feature modes (image vs layout vs fusion).

    Layout-only or image-only grids keep one feature_mode but many encoders; those
    should group by encoder, not collapse into a single feature_mode row.
    """
    if "feature_mode" not in detail.columns:
        return False
    modes = set(detail["feature_mode"].dropna().astype(str).str.strip()) - {"", "nan"}
    if not modes:
        return False
    fusion_modes = {"image_layout", "image_metadata", "layout_metadata", "all"}
    if modes & fusion_modes:
        return True
    primary = modes & {"image", "layout"}
    return len(primary) > 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail-csv", type=Path, default=ROOT / "results" / "mlp_runs_detail.csv")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "results" / "mlp_ablation.csv")
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="Group MLP rows by feature_mode (multimodal ablation grid)",
    )
    ap.add_argument(
        "--merge-detail-csv",
        type=Path,
        default=None,
        help="Optional second detail CSV to merge before aggregation (e.g. existing MLP runs)",
    )
    ap.add_argument(
        "--omni-heldout-fraction",
        type=float,
        default=0.0,
        help="If >0, compute baselines on stratified OmniDoc hold-out (matches train_mlp.py).",
    )
    ap.add_argument(
        "--omni-heldout-split-seed",
        type=int,
        default=42,
        help="Split seed for --omni-heldout-fraction (default: 42).",
    )
    ap.add_argument(
        "--test-set-label",
        default=None,
        help="Label for plots / baseline notes (default: Real5, hard296, or Omni holdout).",
    )
    ap.add_argument(
        "--fusion-method",
        action="store_true",
        help="Group MLP rows by feature_fusion (image+layout fusion ablation)",
    )
    args = ap.parse_args()

    df = load_predictions(args.omni, args.real5)
    df["page_id"] = df["page_id"].astype(str)
    train_df = df[df["dataset"] == "omni"]
    omni_mat_full = get_matrix(train_df, "omni")
    omni_mat_full = omni_mat_full.loc[omni_mat_full.notna().any(axis=1)]
    omni_mat_full.index = omni_mat_full.index.astype(str)

    hold_frac = float(args.omni_heldout_fraction)
    if hold_frac > 1e-12:
        train_ids, hold_ids = omni_holdout_page_ids(
            omni_mat_full,
            df,
            holdout_frac=hold_frac,
            split_seed=int(args.omni_heldout_split_seed),
        )
        train_mat = omni_mat_full.loc[train_ids]
        test_mat = omni_mat_full.loc[hold_ids]
        eval_scope = "omni hold-out"
        default_test_label = f"Omni holdout ({hold_frac:.0%})"
        baseline_notes_suffix = f"omni hold-out ({hold_frac:g}, seed={args.omni_heldout_split_seed})"
    else:
        train_mat = omni_mat_full
        test_mat = get_matrix(df[df["dataset"] == "real5"], "real5")
        test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
        test_mat.index = test_mat.index.astype(str)
        eval_scope = "real5"
        default_test_label = "Real5"
        if "hard296" in str(args.real5):
            default_test_label = "hard296"
        baseline_notes_suffix = "real5"

    test_set_label = args.test_set_label or default_test_label

    oracle_ned = float(test_mat.max(axis=1).mean())
    best_single_ned, champion = best_single_realized_ned(train_mat, test_mat)
    best_fixed_on_test = float(test_mat.mean(axis=0).max())
    best_fixed_name = str(test_mat.mean(axis=0).idxmax())
    stratum_sel = per_stratum_baseline_selections(test_mat, train_mat, df, train_df)
    stratum_ned = mean_ned(per_page_ned(stratum_sel, test_mat))
    doc_type_sel = stratum_table_baseline_selections(test_mat, train_mat, df, "doc_type")
    doc_type_ned = mean_ned(per_page_ned(doc_type_sel, test_mat))
    layout_sel = stratum_table_baseline_selections(test_mat, train_mat, df, "layout_type")
    layout_ned = mean_ned(per_page_ned(layout_sel, test_mat))

    baseline_rows = [
        {
            "kind": "baseline",
            "method": "oracle_upper_bound",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": oracle_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": 1.0,
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"per-page max over parsers on {eval_scope}",
        },
        {
            "kind": "baseline",
            "method": "best_single_train_champion",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": best_single_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(best_single_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"parser={champion} by omni train mean NED, score on {baseline_notes_suffix}",
        },
        {
            "kind": "baseline",
            "method": "best_fixed_on_test",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": best_fixed_on_test,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(best_fixed_on_test, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"best single parser by {eval_scope} column mean ({best_fixed_name})",
        },
        {
            "kind": "baseline",
            "method": "best_per_doc_type_table",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": doc_type_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(doc_type_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"doc_type mean leaders on omni train → {baseline_notes_suffix}",
        },
        {
            "kind": "baseline",
            "method": "best_per_layout_table",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": layout_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(layout_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"layout_type mean leaders on omni train → {baseline_notes_suffix}",
        },
        {
            "kind": "baseline",
            "method": "best_per_stratum_table",
            "encoder": "",
            "feature_mode": "",
            "label_type": "",
            "mean_ned_real5": stratum_ned,
            "std_ned_real5": "",
            "oracle_gap_recovered": oracle_gap_recovered(stratum_ned, best_single_ned, oracle_ned),
            "std_oracle_gap_recovered": "",
            "n_seeds": "",
            "notes": f"(doc_type,layout_type) mean leaders on omni train → {baseline_notes_suffix}",
        },
    ]

    rows: list[dict] = list(baseline_rows)

    if args.detail_csv.is_file():
        detail = pd.read_csv(args.detail_csv)
        if args.merge_detail_csv is not None and args.merge_detail_csv.is_file():
            extra = pd.read_csv(args.merge_detail_csv)
            detail = pd.concat([detail, extra], ignore_index=True)
        if not detail.empty:
            if "router" not in detail.columns:
                detail["router"] = "mlp"
            use_fusion_method = args.fusion_method or _is_fusion_method_detail(detail)
            use_multimodal = (args.multimodal or _is_multimodal_detail(detail)) and not use_fusion_method
            if use_fusion_method:
                group_cols = ["router", "feature_fusion", "label_type"]
            elif use_multimodal:
                group_cols = ["router", "feature_mode", "label_type"]
            else:
                group_cols = ["router", "encoder", "label_type"]
            g = detail.groupby(group_cols, as_index=False)
            agg = g.agg(
                mean_ned_real5=("mean_ned_real5", "mean"),
                std_ned_real5=("mean_ned_real5", "std"),
                mean_gap=("oracle_gap_recovered", "mean"),
                std_gap=("oracle_gap_recovered", "std"),
                n_seeds=("seed", "count"),
                encoder=("encoder", "first"),
                feature_mode=("feature_mode", "first"),
                feature_fusion=("feature_fusion", "first"),
            )
            for _, r in agg.iterrows():
                router = str(r["router"]) if pd.notna(r["router"]) else "mlp"
                notes = (
                    "frozen encoder + sklearn logistic; "
                    f"omni train → {baseline_notes_suffix}"
                    if router == "logistic"
                    else f"frozen encoder + 2-layer MLP; omni train → {baseline_notes_suffix}"
                )
                rows.append(
                    {
                        "kind": router,
                        "method": f"{router}_router",
                        "encoder": r["encoder"],
                        "feature_mode": r["feature_mode"] if (use_multimodal or use_fusion_method) else "",
                        "feature_fusion": r["feature_fusion"] if use_fusion_method else "",
                        "label_type": r["label_type"],
                        "mean_ned_real5": float(r["mean_ned_real5"]),
                        "std_ned_real5": float(r["std_ned_real5"]) if pd.notna(r["std_ned_real5"]) else 0.0,
                        "oracle_gap_recovered": float(r["mean_gap"]),
                        "std_oracle_gap_recovered": float(r["std_gap"]) if pd.notna(r["std_gap"]) else 0.0,
                        "n_seeds": int(r["n_seeds"]),
                        "notes": notes,
                    }
                )
    else:
        print(f"Warning: {args.detail_csv} not found — only baselines written.", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)")

    if not out_df[out_df["kind"].isin(["mlp", "logistic"])].empty:
        from pagerouter import viz

        fig_dir = args.out_csv.parent.parent / "figures"
        test_label = test_set_label
        router_rows = out_df[out_df["kind"].isin(["mlp", "logistic"])]
        use_fusion_method = args.fusion_method or (
            "feature_fusion" in out_df.columns
            and not router_rows.empty
            and bool(set(router_rows["feature_fusion"].astype(str).str.strip()) - {"", "nan"})
        )
        use_multimodal = (args.multimodal or (
            "feature_mode" in out_df.columns
            and not router_rows.empty
            and bool(set(router_rows["feature_mode"].astype(str).str.strip()) - {"", "image", "nan"})
        )) and not use_fusion_method
        has_logistic = (out_df["kind"] == "logistic").any()
        fig_stem = args.out_csv.stem
        fig_name = f"{fig_stem}.pdf" if fig_stem.endswith("_ablation") else (
            "mlp_fusion_method_ablation.pdf" if use_fusion_method else (
                "mlp_multimodal_ablation.pdf" if use_multimodal else (
                    "mlp_vs_logistic_ablation.pdf" if has_logistic else "mlp_ablation.pdf"
                )
            )
        )
        if use_fusion_method:
            viz.plot_mlp_fusion_method_ablation(
                out_df, fig_dir / fig_name, test_set_label=test_label
            )
            print(f"Wrote {fig_dir / fig_name}")
        elif use_multimodal:
            viz.plot_mlp_multimodal_ablation(
                out_df, fig_dir / fig_name, test_set_label=test_label
            )
            print(f"Wrote {fig_dir / fig_name}")
        elif has_logistic:
            viz.plot_mlp_logistic_ablation(out_df, fig_dir / fig_name, test_set_label=test_label)
            print(f"Wrote {fig_dir / fig_name}")
        else:
            viz.plot_mlp_ablation(out_df, fig_dir / fig_name, test_set_label=test_label)
            print(f"Wrote {fig_dir / fig_name}")


if __name__ == "__main__":
    main()
