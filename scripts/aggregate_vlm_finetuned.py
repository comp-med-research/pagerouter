#!/usr/bin/env python3
"""Aggregate VLM fine-tuned router metrics (Real5 + hard296) into tables and plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter import viz  # noqa: E402
from pagerouter.load import DEFAULT_OMNI, DEFAULT_REAL5, get_matrix, load_predictions  # noqa: E402
from pagerouter.vlm_finetune.registry import PARADIGM_LABELS, PARADIGMS  # noqa: E402


def _load_metrics(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/metrics.json")):
        data = json.loads(path.read_text())
        model = path.parent.parent.name
        paradigm = path.parent.name
        row = {
            "model": model,
            "paradigm": paradigm,
            "path": str(path),
            **data,
        }
        rows.append(row)
    return rows


def _baseline_rows(split: str, eval_block: dict, *, test_label: str) -> list[dict]:
    oracle = float(eval_block["oracle_ned"])
    best_single = float(eval_block["best_single_ned"])
    champion = str(eval_block["best_single_champion"])
    return [
        {
            "kind": "baseline",
            "model": "oracle_upper_bound",
            "paradigm": "",
            "mean_ned": oracle,
            "oracle_gap_recovered": 1.0,
            "notes": f"per-page max over parsers on {test_label}",
        },
        {
            "kind": "baseline",
            "model": "best_single_train_champion",
            "paradigm": "",
            "mean_ned": best_single,
            "oracle_gap_recovered": 0.0,
            "notes": f"parser={champion} by omni train mean NED, score on {test_label}",
        },
    ]


def _best_fixed_ned(omni: Path, eval_csv: Path) -> float:
    df = load_predictions(omni, eval_csv)
    test_mat = get_matrix(df[df["dataset"] == "real5"], "real5")
    test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
    return float(test_mat.mean(axis=0).max())


def _build_split_table(
    metrics_rows: list[dict],
    split: str,
    *,
    omni: Path,
    eval_csv: Path,
    test_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    ablation_rows: list[dict] = []
    ref_eval: dict | None = None

    for row in metrics_rows:
        status = row.get("status", "ok")
        base = {
            "model": row["model"],
            "paradigm": row["paradigm"],
            "paradigm_label": row.get("paradigm_label", PARADIGM_LABELS.get(row["paradigm"], "")),
            "status": status,
            "label_type": row.get("label_type", ""),
            "trainable_params": row.get("trainable_params"),
            "n_train": row.get("n_train"),
        }
        if status == "skipped":
            detail_rows.append(
                {
                    **base,
                    f"mean_ned_{split}": None,
                    f"oracle_gap_{split}": None,
                    "error": row.get("error", ""),
                }
            )
            continue

        evals = row.get("evals") or {}
        ev = evals.get(split)
        if not ev:
            detail_rows.append({**base, f"mean_ned_{split}": None, f"oracle_gap_{split}": None})
            continue

        ref_eval = ref_eval or ev
        detail_rows.append(
            {
                **base,
                f"mean_ned_{split}": float(ev["mean_ned"]),
                f"oracle_gap_{split}": float(ev["oracle_gap_recovered"]),
                f"oracle_ned_{split}": float(ev["oracle_ned"]),
                f"best_single_ned_{split}": float(ev["best_single_ned"]),
                f"n_eval_{split}": int(ev["n_eval"]),
            }
        )
        ablation_rows.append(
            {
                "kind": "vlm_router",
                "model": row["model"],
                "paradigm": row["paradigm"],
                "paradigm_label": row.get("paradigm_label", ""),
                "mean_ned": float(ev["mean_ned"]),
                "oracle_gap_recovered": float(ev["oracle_gap_recovered"]),
                "notes": f"omni train → {test_label}",
            }
        )

    detail = pd.DataFrame(detail_rows)
    if ref_eval is None:
        return detail, pd.DataFrame()

    best_fixed = _best_fixed_ned(omni, eval_csv)
    baselines = _baseline_rows(split, ref_eval, test_label=test_label)
    baselines.append(
        {
            "kind": "baseline",
            "model": "best_fixed_on_test",
            "paradigm": "",
            "mean_ned": best_fixed,
            "oracle_gap_recovered": (best_fixed - float(ref_eval["best_single_ned"]))
            / (float(ref_eval["oracle_ned"]) - float(ref_eval["best_single_ned"])),
            "notes": f"best single parser by {test_label} column mean",
        }
    )
    ablation = pd.concat([pd.DataFrame(baselines), pd.DataFrame(ablation_rows)], ignore_index=True)
    return detail, ablation


def _plot_split(ablation: pd.DataFrame, split: str, out_path: Path, *, test_label: str) -> None:
    if ablation.empty:
        return
    routers = ablation[ablation["kind"] == "vlm_router"].copy()
    if routers.empty:
        return
    oracle_row = ablation[ablation["model"] == "oracle_upper_bound"]
    best_fixed_row = ablation[ablation["model"] == "best_fixed_on_test"]
    oracle_ned = float(oracle_row["mean_ned"].iloc[0]) if not oracle_row.empty else 1.0
    best_fixed = float(best_fixed_row["mean_ned"].iloc[0]) if not best_fixed_row.empty else None

    summaries = [
        {
            "label": f"{r['model']} | {r['paradigm']}",
            "mean_ned": float(r["mean_ned"]),
            "oracle_gap_pct": float(r["oracle_gap_recovered"]),
        }
        for _, r in routers.iterrows()
    ]
    viz.plot_routing_results(
        summaries,
        oracle_ned,
        out_path,
        best_fixed_ned=best_fixed,
        test_set_label=test_label,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results/vlm_finetuned")
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument("--hard296", type=Path, default=ROOT / "data/hard296/test_as_real5.csv")
    args = ap.parse_args()

    metrics_rows = _load_metrics(args.results_dir)
    if not metrics_rows:
        raise SystemExit(f"No metrics.json under {args.results_dir}")

    splits = [
        ("real5", args.real5, "Real5"),
        ("hard296", args.hard296, "hard296"),
    ]

    pivot: dict[str, dict] = {}
    for model_par in metrics_rows:
        key = (model_par["model"], model_par["paradigm"])
        pivot.setdefault(key, {"model": key[0], "paradigm": key[1], "status": model_par.get("status", "ok")})
        for split_name, _, _ in splits:
            ev = (model_par.get("evals") or {}).get(split_name)
            if ev:
                pivot[key][f"mean_ned_{split_name}"] = float(ev["mean_ned"])
                pivot[key][f"oracle_gap_{split_name}"] = float(ev["oracle_gap_recovered"])

    summary_path = args.results_dir / "summary" / "vlm_finetuned_pivot.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pivot.values()).sort_values(["model", "paradigm"]).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    for split_name, eval_csv, test_label in splits:
        detail, ablation = _build_split_table(
            metrics_rows,
            split_name,
            omni=args.omni,
            eval_csv=eval_csv,
            test_label=test_label,
        )
        out_metrics = args.results_dir / split_name / "metrics"
        out_figures = args.results_dir / split_name / "figures"
        out_metrics.mkdir(parents=True, exist_ok=True)
        out_figures.mkdir(parents=True, exist_ok=True)

        detail_path = out_metrics / "vlm_finetuned_detail.csv"
        detail.to_csv(detail_path, index=False)
        print(f"Wrote {detail_path}")

        if not ablation.empty:
            ablation_path = out_metrics / "vlm_finetuned_ablation.csv"
            ablation.to_csv(ablation_path, index=False)
            print(f"Wrote {ablation_path}")
            fig_path = out_figures / "vlm_finetuned_ablation.pdf"
            _plot_split(ablation, split_name, fig_path, test_label=test_label)
            print(f"Wrote {fig_path}")

        completed = detail[detail["status"] != "skipped"]
        scored = completed[completed[f"mean_ned_{split_name}"].notna()]
        if not scored.empty:
            best = scored.sort_values(f"mean_ned_{split_name}", ascending=False).iloc[0]
            print(
                f"[{test_label}] best: {best['model']} | {best['paradigm']} "
                f"ned={best[f'mean_ned_{split_name}']:.4f} "
                f"gap={best[f'oracle_gap_{split_name}']:.1%}"
            )


if __name__ == "__main__":
    main()
