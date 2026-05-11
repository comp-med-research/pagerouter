"""
Post-hoc analysis of agentic routing traces.

Sources:
  --source ablation  reads results/ablation_results.jsonl (multi-agent ablation)
  --source agentic   reads results/agentic_router_responses.jsonl (single-agent)

For ablation: produces agent agreement heatmap and disagreement report.
For agentic:  produces per-model accuracy breakdown and failure analysis.

Usage:
  python scripts/analyze_agentic_traces.py --source ablation
  python scripts/analyze_agentic_traces.py --source agentic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pagerouter import load, viz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_ablation(log_path: Path) -> pd.DataFrame:
    records = []
    with log_path.open() as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        raise ValueError(f"No valid records in {log_path}")
    return pd.DataFrame(records)


def load_agentic(log_path: Path) -> pd.DataFrame:
    records = []
    with log_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                # Normalise to ablation schema
                records.append({
                    "page_id": rec["page_id"],
                    "agent_name": "claude",
                    "parsed_model": rec.get("parsed_model"),
                    "response": rec.get("response_text", ""),
                    "timestamp": rec.get("timestamp", ""),
                })
            except (json.JSONDecodeError, KeyError):
                pass
    if not records:
        raise ValueError(f"No valid records in {log_path}")
    return pd.DataFrame(records)


# ─── Ablation analysis ────────────────────────────────────────────────────────

def analyze_ablation(df: pd.DataFrame, matrix: pd.DataFrame, out_dir: Path) -> None:
    agents = sorted(df["agent_name"].unique().tolist())
    n_pages = df["page_id"].nunique()

    print(f"\n=== Ablation trace analysis: {len(agents)} agents, {n_pages} pages ===\n")

    # ── 1. Parse failure rates ────────────────────────────────────────────────
    print("Parse failure rates (% calls where model could not be identified):")
    for agent in agents:
        sub = df[df["agent_name"] == agent]
        failures = sub["parsed_model"].isna().mean()
        print(f"  {agent:<12} {failures:.1%}")

    # ── 2. Selection distribution per agent ──────────────────────────────────
    print("\nTop-3 model selections per agent:")
    for agent in agents:
        sub = df[df["agent_name"] == agent].dropna(subset=["parsed_model"])
        top3 = sub["parsed_model"].value_counts().head(3)
        top3_str = ", ".join(f"{m}({c})" for m, c in top3.items())
        print(f"  {agent:<12} {top3_str}")

    # ── 3. Per-page disagreement ──────────────────────────────────────────────
    pivot = df.pivot_table(
        index="page_id", columns="agent_name", values="parsed_model", aggfunc="first"
    )
    # Number of distinct models chosen per page (across all agents)
    n_distinct = pivot.nunique(axis=1)
    print(f"\nPer-page agent disagreement (distinct models chosen):")
    print(f"  Mean: {n_distinct.mean():.2f} / Max possible: {len(agents)}")
    print(f"  Pages with full agreement (1 model): {(n_distinct == 1).sum()}")
    print(f"  Pages with full disagreement ({len(agents)} models): {(n_distinct == len(agents)).sum()}")

    # Most contested pages
    hardest = n_distinct.sort_values(ascending=False).head(10)
    print("\nTop 10 most contested pages (highest disagreement):")
    for pid, nd in hardest.items():
        choices = pivot.loc[pid].dropna().to_dict()
        choices_str = ", ".join(f"{a}→{m}" for a, m in sorted(choices.items()))
        print(f"  {pid[:50]:<50}  {nd} models  [{choices_str}]")

    # ── 4. Agreement heatmap ──────────────────────────────────────────────────
    heatmap_path = out_dir / "agent_agreement_heatmap.pdf"
    viz.plot_agent_agreement_heatmap(df, heatmap_path)
    print(f"\nSaved agent agreement heatmap → {heatmap_path}")

    # ── 5. Per-doc-type mean NED (if ned_score present) ──────────────────────
    if "ned_score" in df.columns and "doc_type" not in df.columns:
        # Try to join doc_type from matrix
        pass
    if "ned_score" in df.columns:
        print("\nMean NED per agent (from logged ned_scores):")
        for agent in agents:
            sub = df[df["agent_name"] == agent].dropna(subset=["ned_score"])
            print(f"  {agent:<12} {sub['ned_score'].mean():.4f}")

    if "ned_score" in df.columns and "doc_type" in df.columns:
        print("\nMean NED by doc_type (averaged across agents):")
        by_doc = df.groupby("doc_type")["ned_score"].mean().sort_values(ascending=False)
        for dt, ned in by_doc.items():
            print(f"  {dt:<25} {ned:.4f}")


# ─── Single-agent analysis ────────────────────────────────────────────────────

def analyze_agentic(df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    n_pages = df["page_id"].nunique()
    print(f"\n=== Single-agent trace analysis: {n_pages} pages ===\n")

    valid = df.dropna(subset=["parsed_model"])
    print(f"Parse success rate: {len(valid)/len(df):.1%}")

    # Per-model selection counts
    print("\nModel selection counts:")
    for model, count in valid["parsed_model"].value_counts().items():
        print(f"  {model:<25} {count:>4}")

    # Per-model oracle accuracy (was agent right?)
    valid_ids = valid["page_id"][valid["page_id"].isin(matrix.index)]
    oracle = matrix.loc[valid_ids].idxmax(axis=1)
    merged = valid.set_index("page_id")[["parsed_model"]].join(oracle.rename("oracle"))
    merged = merged.dropna()
    accuracy = (merged["parsed_model"] == merged["oracle"]).mean()
    print(f"\nOracle accuracy (agent matched oracle): {accuracy:.1%}")

    print("\nOracle accuracy by oracle model:")
    for oracle_model, grp in merged.groupby("oracle"):
        acc = (grp["parsed_model"] == grp["oracle"]).mean()
        print(f"  {oracle_model:<25} {acc:.1%}  (n={len(grp)})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["ablation", "agentic"], default="ablation")
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument(
        "--log-path", type=Path, default=None,
        help="Override default log file path",
    )
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    default_logs = {
        "ablation": RESULTS / "ablation_results.jsonl",
        "agentic":  RESULTS / "agentic_router_responses.jsonl",
    }
    log_path = args.log_path or default_logs[args.source]
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df_all = load.load_predictions(args.omni, args.real5)
    matrix = load.get_matrix(df_all[df_all["dataset"] == "omni"], "omni")

    if args.source == "ablation":
        traces = load_ablation(log_path)
        analyze_ablation(traces, matrix, args.out_dir)
    else:
        traces = load_agentic(log_path)
        analyze_agentic(traces, matrix)


if __name__ == "__main__":
    main()
