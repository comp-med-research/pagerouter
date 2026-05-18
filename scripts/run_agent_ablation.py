"""
Multi-agent ablation — compares registry VLMs (light + heavy) on a stratified Omni sample.

All agents share the same prompt (image + routing instructions). Logs stream to
``results/ablation_results.jsonl`` (resumable). Summary CSV includes latency p50/p90
and rough cost estimates so you can pick a router before running it on Real5 or a
custom CSV (same schema as ``omni_predictions.csv``).

Agent tiers (see ``pagerouter.agents.AGENT_REGISTRY``): ``heavy`` (Claude, GPT, Gemini),
``light`` (Kimi, Qwen-VL, InternVL on Together). Filter with ``--tier``.

Usage:
  python scripts/run_agent_ablation.py --yes --tier all --sample-n 100
  python scripts/run_agent_ablation.py --agents qwen internvl claude --parallel --yes
  python scripts/run_agent_ablation.py --tier light --sample-n 40 --yes

After you pick an agent, evaluate on Real5::

  PYTHONPATH=. python scripts/run_vlm_router_eval.py --agent qwen --dataset real5
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pagerouter import evaluate, load
from pagerouter.agents import AGENT_REGISTRY, VLMAgent, _ENV_KEY, available_agents
from pagerouter.routing import BestSingleRouter, MetadataRouter, MODELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

FALLBACK_MODEL = "dotsocr"


# ─── Sampling ─────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample by doc_type, exactly n pages."""
    pages = (
        df[df["dataset"] == "omni"]
        .drop_duplicates("page_id")[["page_id", "doc_type", "layout_type"]]
        .reset_index(drop=True)
    )
    total = len(pages)
    parts = []
    for _, group in pages.groupby("doc_type"):
        k = min(len(group), max(1, round(len(group) / total * n)))
        parts.append(group.sample(k, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n:
        pool = pages[~pages["page_id"].isin(sampled["page_id"])]
        extra = pool.sample(min(n - len(sampled), len(pool)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled


# ─── Cost estimate ────────────────────────────────────────────────────────────

def print_cost_estimate(active_specs: list[dict], n_pages: int) -> None:
    col_w = [10, 8, 28, 14, 12]
    header = (
        f"{'Agent':<{col_w[0]}} {'Tier':<{col_w[1]}} {'Model':<{col_w[2]}} "
        f"{'Est./call':>{col_w[3]}} {'Est. total':>{col_w[4]}}"
    )
    sep = "-" * (sum(col_w) + 8)
    print()
    print(header)
    print(sep)
    total = 0.0
    for spec in active_specs:
        cost_total = spec["cost_per_call"] * n_pages
        total += cost_total
        print(
            f"{spec['name']:<10} {spec['tier']:<8} {spec['model']:<34} "
            f"${spec['cost_per_call']:<10.4f} ${cost_total:>8.2f}"
        )
    print(sep)
    print(
        f"Total estimated cost: ~${total:.2f} for {n_pages} pages × {len(active_specs)} agents "
        "(rough; calibrate cost_per_call in agents.py from invoices)"
    )
    print()


def confirm_run(active_specs: list[dict]) -> list[str]:
    """Ask which agents to run. Returns list of agent names to run."""
    response = input("Proceed? [y/n] or enter agent names separated by commas (or 'all'): ").strip()
    if response.lower() in ("n", "no", ""):
        return []
    if response.lower() in ("y", "yes", "all"):
        return [s["name"] for s in active_specs]
    names = [n.strip() for n in response.split(",")]
    valid = {s["name"] for s in active_specs}
    unknown = set(names) - valid
    if unknown:
        print(f"Unknown agents: {unknown}. Aborting.")
        return []
    return names


# ─── Resume support ───────────────────────────────────────────────────────────

def load_completed(log_path: Path) -> set[tuple[str, str]]:
    """Return set of (page_id, agent_name) already recorded in the log."""
    done: set[tuple[str, str]] = set()
    if not log_path.exists():
        return done
    with log_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                done.add((rec["page_id"], rec["agent_name"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ─── Single page routing ─────────────────────────────────────────────────────

def route_page(
    agent: VLMAgent,
    page_id: str,
    image_path: Path,
    prompt: str,
    oracle_model: str,
    oracle_ned: float,
    matrix: pd.DataFrame,
    log_path: Path,
) -> dict:
    """Call agent, parse response, compute NED, write to log. Returns record dict."""
    parsed_model = None
    response_text = ""
    latency_s = float("nan")
    for attempt in range(4):
        try:
            response_text, latency_s = agent.call(image_path, prompt)
            parsed_model = agent.parse_model_choice(response_text)
            if parsed_model is None:
                raise ValueError(f"Could not parse model from: {response_text!r}")
            break
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < 3:
                time.sleep(wait)
            else:
                warnings.warn(
                    f"[{agent.name}] {page_id}: all retries failed ({exc}), "
                    f"falling back to {FALLBACK_MODEL}"
                )
                parsed_model = FALLBACK_MODEL

    ned_score = float(matrix.at[page_id, parsed_model]) if parsed_model in matrix.columns else float("nan")
    record = {
        "page_id": page_id,
        "agent_name": agent.name,
        "response": response_text,
        "parsed_model": parsed_model,
        "oracle_model": oracle_model,
        "is_oracle": parsed_model == oracle_model,
        "ned_score": ned_score,
        "oracle_ned": oracle_ned,
        "latency_s": latency_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


# ─── Summary ─────────────────────────────────────────────────────────────────

def compute_summary(
    log_path: Path,
    matrix: pd.DataFrame,
    omni_df: pd.DataFrame,
    active_agent_names: list[str],
) -> pd.DataFrame:
    records = []
    with log_path.open() as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        return pd.DataFrame()

    ablation_df = pd.DataFrame(records)

    # Filter to pages present in matrix
    ablation_df = ablation_df[ablation_df["page_id"].isin(matrix.index)]
    sampled_ids = ablation_df["page_id"].unique()
    sub_matrix = matrix.loc[sampled_ids]

    oracle_ned = float(sub_matrix.max(axis=1).mean())
    best_single_ned = float(sub_matrix.mean(axis=0).max())

    rows = []

    # Oracle row
    rows.append({
        "agent": "oracle",
        "tier": "—",
        "provider": "—",
        "model": "—",
        "n_calls": len(sampled_ids),
        "mean_ned": oracle_ned,
        "oracle_gap_pct": 1.0,
        "oracle_accuracy": 1.0,
        "mean_latency_s": float("nan"),
        "p50_latency_s": float("nan"),
        "p90_latency_s": float("nan"),
        "est_cost_usd": float("nan"),
    })

    # Agent rows
    for name in active_agent_names:
        ag = ablation_df[ablation_df["agent_name"] == name]
        if ag.empty:
            continue
        mean_ned = float(ag["ned_score"].mean())
        gap = evaluate.oracle_gap_recovered(mean_ned, best_single_ned, oracle_ned)
        oracle_acc = float(ag["is_oracle"].mean())
        lat = pd.to_numeric(ag["latency_s"], errors="coerce").dropna()
        mean_latency = float(lat.mean()) if len(lat) else float("nan")
        p50_lat = float(lat.quantile(0.5)) if len(lat) else float("nan")
        p90_lat = float(lat.quantile(0.9)) if len(lat) else float("nan")
        spec = next((s for s in AGENT_REGISTRY if s["name"] == name), None)
        n_calls = int(len(ag))
        est_cost = float(spec["cost_per_call"] * n_calls) if spec else float("nan")
        rows.append({
            "agent": name,
            "tier": spec["tier"] if spec else "—",
            "provider": spec["provider"] if spec else "—",
            "model": spec["model"] if spec else "—",
            "n_calls": n_calls,
            "mean_ned": mean_ned,
            "oracle_gap_pct": gap,
            "oracle_accuracy": oracle_acc,
            "mean_latency_s": mean_latency,
            "p50_latency_s": p50_lat,
            "p90_latency_s": p90_lat,
            "est_cost_usd": est_cost,
        })

    # Metadata baseline (fitted on full omni, evaluated on sampled pages)
    full_matrix = load.get_matrix(omni_df, "omni")
    for RouterCls, label in [(MetadataRouter, "metadata"), (BestSingleRouter, "best_single")]:
        router = RouterCls()
        router.fit(full_matrix, omni_df)
        sampled_df = omni_df[omni_df["page_id"].isin(sampled_ids)]
        sels = router.predict(sampled_df)
        sels = sels[sels.index.isin(sub_matrix.index)]
        summary = evaluate.routing_summary(
            sels, sub_matrix, label=label,
            oracle_ned=oracle_ned, best_single_ned=best_single_ned,
        )
        rows.append({
            "agent": label,
            "tier": "—",
            "provider": "—",
            "model": "—",
            "n_calls": len(sampled_ids),
            "mean_ned": summary["mean_ned"],
            "oracle_gap_pct": summary["oracle_gap_pct"],
            "oracle_accuracy": float("nan"),
            "mean_latency_s": float("nan"),
            "p50_latency_s": float("nan"),
            "p90_latency_s": float("nan"),
            "est_cost_usd": float("nan"),
        })

    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print()
    hdr = (
        f"{'Agent':<12} {'Tier':<7} {'NED':>8} {'Gap%':>8} {'OrclAcc':>9} "
        f"{'Latμ':>8} {'p50':>8} {'p90':>8} {'$/est':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, row in summary_df.iterrows():
        gap_str = f"{row['oracle_gap_pct']:.1%}" if not pd.isna(row["oracle_gap_pct"]) else "—"
        acc_str = f"{row['oracle_accuracy']:.1%}" if not pd.isna(row["oracle_accuracy"]) else "—"
        lat_m = f"{row['mean_latency_s']:.2f}" if not pd.isna(row["mean_latency_s"]) else "—"
        p50 = f"{row['p50_latency_s']:.2f}" if not pd.isna(row["p50_latency_s"]) else "—"
        p90 = f"{row['p90_latency_s']:.2f}" if not pd.isna(row["p90_latency_s"]) else "—"
        cost_str = f"{row['est_cost_usd']:.2f}" if not pd.isna(row["est_cost_usd"]) else "—"
        tier = str(row.get("tier", "—"))
        print(
            f"{row['agent']:<12} {tier:<7} {row['mean_ned']:>8.4f} {gap_str:>8} {acc_str:>9} "
            f"{lat_m:>8} {p50:>8} {p90:>8} {cost_str:>10}"
        )
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--prompt", type=Path, default=PROMPTS / "routing_prompt.txt")
    ap.add_argument("--image-dir", type=Path, default=DATA / "page_images")
    ap.add_argument("--log-path", type=Path, default=RESULTS / "ablation_results.jsonl")
    ap.add_argument(
        "--agents", nargs="+",
        choices=[a["name"] for a in AGENT_REGISTRY],
        default=None,
        help="Agent names to run (default: all with available API keys)",
    )
    ap.add_argument(
        "--tier",
        choices=("all", "light", "heavy"),
        default="all",
        help="Restrict to router tier: light (Kimi, Together VLMs) vs heavy (Claude, GPT, Gemini)",
    )
    ap.add_argument("--sample-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--reuse-pages", action="store_true",
        help="Load existing results/ablation_pages.csv instead of resampling",
    )
    ap.add_argument(
        "--parallel", action="store_true",
        help="Run agents in parallel per page (one thread per agent)",
    )
    ap.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip cost confirmation prompt",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not args.prompt.exists():
        raise FileNotFoundError(f"Prompt not found: {args.prompt}")
    prompt = args.prompt.read_text().strip()

    # ── Load data ────────────────────────────────────────────────────────────
    df = load.load_predictions(args.omni, args.real5)
    omni_df = df[df["dataset"] == "omni"].copy()
    matrix = load.get_matrix(omni_df, "omni")
    oracle_selections = matrix.idxmax(axis=1)
    oracle_neds = matrix.max(axis=1)

    # ── Sample pages ─────────────────────────────────────────────────────────
    pages_csv = RESULTS / "ablation_pages.csv"
    if args.reuse_pages and pages_csv.exists():
        sampled = pd.read_csv(pages_csv)
        print(f"[pages] Reusing {len(sampled)} pages from {pages_csv}")
    else:
        sampled = stratified_sample(df, args.sample_n, args.seed)
        sampled.to_csv(pages_csv, index=False)
        print(f"[pages] Sampled {len(sampled)} pages (stratified by doc_type, seed={args.seed})")

    sampled_ids = sampled["page_id"].tolist()
    sub_matrix = matrix.loc[[p for p in sampled_ids if p in matrix.index]]

    # ── Resolve active agents ────────────────────────────────────────────────
    requested_names = args.agents  # None means all
    active_specs = [
        s for s in AGENT_REGISTRY
        if (requested_names is None or s["name"] in requested_names)
        and bool(import_env(s["provider"]))
    ]
    if args.tier != "all":
        active_specs = [s for s in active_specs if s.get("tier") == args.tier]
    if not active_specs:
        print("No agents available (check API keys and --tier / --agents filters). Exiting.")
        return

    # ── Cost estimate + confirmation ─────────────────────────────────────────
    print_cost_estimate(active_specs, len(sampled_ids))
    if args.yes:
        confirmed_names = [s["name"] for s in active_specs]
    else:
        confirmed_names = confirm_run(active_specs)
    if not confirmed_names:
        print("Aborted.")
        return

    active_specs = [s for s in active_specs if s["name"] in confirmed_names]
    active_agents = [VLMAgent(s["name"], s["provider"], s["model"]) for s in active_specs]

    # ── Resume: skip already-done (page_id, agent_name) ─────────────────────
    completed = load_completed(args.log_path)
    todo = [
        (pid, agent)
        for pid in sampled_ids
        for agent in active_agents
        if (pid, agent.name) not in completed
    ]
    if completed:
        print(f"[resume] Skipping {len(completed)} already-completed calls; {len(todo)} remaining")

    # ── Run ──────────────────────────────────────────────────────────────────
    total = len(todo)
    done_count = 0

    if args.parallel:
        # Per-page: fire all agents for one page concurrently, then move to next
        todo_by_page: dict[str, list[VLMAgent]] = {}
        for pid, agent in todo:
            todo_by_page.setdefault(pid, []).append(agent)

        for page_idx, (pid, page_agents) in enumerate(todo_by_page.items()):
            image_path = args.image_dir / pid
            if not image_path.exists():
                print(f"[skip] {pid}: image not found")
                continue
            with ThreadPoolExecutor(max_workers=len(page_agents)) as pool:
                futures = {
                    pool.submit(
                        route_page, agent, pid, image_path, prompt,
                        oracle_selections[pid], float(oracle_neds[pid]),
                        sub_matrix, args.log_path,
                    ): agent.name
                    for agent in page_agents
                }
                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[error] {pid}/{agent_name}: {exc}")
                    done_count += 1
                    print(f"\r[{done_count}/{total}]", end="", flush=True)
    else:
        for pid, agent in todo:
            image_path = args.image_dir / pid
            if not image_path.exists():
                print(f"[skip] {pid}: image not found")
                continue
            try:
                route_page(
                    agent, pid, image_path, prompt,
                    oracle_selections[pid], float(oracle_neds[pid]),
                    sub_matrix, args.log_path,
                )
            except Exception as exc:
                print(f"[error] {pid}/{agent.name}: {exc}")
            done_count += 1
            print(f"\r[{done_count}/{total}] {agent.name} / {pid[:40]}", end="", flush=True)

    print(f"\nDone. Results written to {args.log_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_df = compute_summary(
        args.log_path, sub_matrix, omni_df,
        active_agent_names=[s["name"] for s in active_specs],
    )
    print_summary_table(summary_df)
    summary_csv = RESULTS / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")


def import_env(provider: str) -> str:
    """Return the API key for a provider, or empty string."""
    import os
    return os.environ.get(_ENV_KEY[provider], "")


if __name__ == "__main__":
    main()
