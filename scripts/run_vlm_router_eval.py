"""
Evaluate **one** registry VLM (see ``pagerouter.agents.AGENT_REGISTRY``) as the page router.

Uses the same prompt file as the multi-agent ablation. Typical workflow:

1. Run ``run_agent_ablation.py`` on an Omni sample; compare NED, oracle-gap %, latency p50/p90, est. cost.
2. Pick an agent name, then score it on Real5 (or Omni) at full size or ``--sample-n`` pilot:

     PYTHONPATH=. python scripts/run_vlm_router_eval.py --agent qwen --dataset real5

For your own labelled set: build a CSV with the same columns as ``data/omni_predictions.csv``
and pass ``--omni`` / ``--real5`` if you split train/test, or extend this script with a
``--predictions`` single-table loader when you are ready.

Requires the provider API key for the chosen agent (same env vars as ablation).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pagerouter import evaluate, load
from pagerouter.agents import AGENT_REGISTRY, VLMAgent, _ENV_KEY

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

FALLBACK = "dotsocr"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--dataset", choices=("omni", "real5"), default="real5")
    ap.add_argument(
        "--agent",
        required=True,
        choices=[a["name"] for a in AGENT_REGISTRY],
        help="Agent name from AGENT_REGISTRY (e.g. qwen, claude, gemini)",
    )
    ap.add_argument("--prompt", type=Path, default=PROMPTS / "routing_prompt.txt")
    ap.add_argument("--image-dir", type=Path, default=DATA / "page_images")
    ap.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Evaluate on a random subset of pages (pilot)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Append JSONL rows (page_id, response, parsed_model, latency_s)",
    )
    args = ap.parse_args()

    spec = next(s for s in AGENT_REGISTRY if s["name"] == args.agent)
    env_var = _ENV_KEY[spec["provider"]]
    if not os.environ.get(env_var):
        raise EnvironmentError(f"{env_var} must be set for agent {args.agent!r}")

    if not args.prompt.is_file():
        raise FileNotFoundError(args.prompt)
    prompt = args.prompt.read_text().strip()

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)
    test_df = df[df["dataset"] == args.dataset].copy()
    matrix = load.get_matrix(test_df, args.dataset)
    page_ids = matrix.index.tolist()
    if args.sample_n is not None:
        page_ids = (
            pd.Series(page_ids)
            .sample(n=min(args.sample_n, len(page_ids)), random_state=args.seed)
            .tolist()
        )
    sub_matrix = matrix.loc[page_ids]

    agent = VLMAgent(spec["name"], spec["provider"], spec["model"])
    log_path = Path(args.log_path) if args.log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    selections: dict[str, str] = {}
    latencies: list[float] = []

    n = len(page_ids)
    for i, pid in enumerate(page_ids):
        img = args.image_dir / pid
        if not img.exists():
            warnings.warn(f"[skip] missing image: {img}")
            continue
        t0 = time.perf_counter()
        response_text = ""
        try:
            response_text, latency_s = agent.call(img, prompt)
            latencies.append(latency_s)
            parsed = agent.parse_model_choice(response_text)
            if parsed is None:
                raise ValueError(f"unparseable: {response_text[:120]!r}")
        except Exception as exc:
            latency_s = time.perf_counter() - t0
            latencies.append(latency_s)
            warnings.warn(f"[{pid}] {exc}; using fallback {FALLBACK}")
            parsed = FALLBACK

        selections[pid] = parsed
        if log_path:
            rec = {
                "page_id": pid,
                "agent_name": args.agent,
                "response": response_text,
                "parsed_model": parsed,
                "latency_s": latency_s,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"\r[{i + 1}/{n}] {pid[:52]}", end="", flush=True)
    print()

    sel = pd.Series(selections, name="model")
    common = sel.index.intersection(sub_matrix.index)
    sel = sel.loc[common]
    m = sub_matrix.loc[common]

    oracle_ned = float(m.max(axis=1).mean())
    summ = evaluate.routing_summary(sel, m, label=args.agent)
    lat_s = pd.Series(latencies, dtype="float64")

    print(
        f"[{args.agent}] dataset={args.dataset}  mean_ned={summ['mean_ned']:.4f}  "
        f"oracle_gap_pct={summ['oracle_gap_pct']:.1%}  n_pages={summ['n_pages']}  "
        f"oracle_upper_bound={oracle_ned:.4f}"
    )
    if len(lat_s):
        print(
            f"[{args.agent}] latency mean={lat_s.mean():.2f}s  p50={lat_s.quantile(0.5):.2f}s  "
            f"p90={lat_s.quantile(0.9):.2f}s"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"vlm_router_eval_{args.agent}_{args.dataset}.csv"
    sel.to_csv(out_csv, header=True)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
