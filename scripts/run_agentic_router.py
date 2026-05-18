"""
Experiment 5 — Agentic router (headline experiment).

A VLM routing agent receives only a page image and selects the best document
parsing model. No metadata, no features, no ground truth.

Uses Anthropic by default (``claude-sonnet-4-6``); set ``--anthropic-model`` to
another vision-capable model id.

Requires:
  - ANTHROPIC_API_KEY environment variable
  - Page images in data/page_images/{page_id}.jpg (or .png)

The --sample-n flag runs a pilot on n randomly selected pages before committing
to full evaluation.

Usage:
  python scripts/run_agentic_router.py --sample-n 50
  python scripts/run_agentic_router.py
  python scripts/run_agentic_router.py --dataset omni --log-path results/agent_log.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pagerouter import evaluate, load, viz
from pagerouter.routing import AgenticRouter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument("--dataset", choices=["omni", "real5"], default="real5",
                    help="Dataset to evaluate on (default: real5 for cross-domain test)")
    ap.add_argument("--prompt", type=Path, default=PROMPTS / "routing_prompt.txt")
    ap.add_argument(
        "--anthropic-model",
        default="claude-sonnet-4-6",
        help="Anthropic vision-capable model id (default: claude-sonnet-4-6)",
    )
    ap.add_argument("--log-path", type=Path, default=RESULTS / "agentic_router_responses.jsonl")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="Run on a random sample of N pages (pilot mode)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for --sample-n sampling")
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    print(f"[agentic] Anthropic model: {args.anthropic_model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load.load_predictions(args.omni, args.real5)
    load.validate_schema(df)

    test_df = df[df["dataset"] == args.dataset].copy()
    if args.sample_n:
        test_df = test_df.drop_duplicates("page_id").sample(
            n=args.sample_n, random_state=args.seed
        )
        print(f"Pilot mode: evaluating {args.sample_n} pages (seed={args.seed})")

    test_matrix = load.get_matrix(test_df, dataset=args.dataset)

    router = AgenticRouter(
        prompt_path=args.prompt,
        log_path=args.log_path,
        model_id=args.anthropic_model,
        sample_n=args.sample_n,
    )
    router.fit(test_matrix, test_df)
    selections = router.predict(test_df)

    oracle_ned = evaluate.mean_ned(test_matrix.max(axis=1))
    oracle_selections = test_matrix.idxmax(axis=1)

    summary = evaluate.routing_summary(selections, test_matrix, label="agentic")
    print(f"Agentic router: mean_ned={summary['mean_ned']:.4f}  oracle_gap_pct={summary['oracle_gap_pct']:.1%}")

    selections.to_csv(RESULTS / f"agentic_selections_{args.dataset}.csv", header=True)

    viz.plot_agentic_confusion_matrix(
        selections,
        oracle_selections,
        out_path=args.out_dir / f"agentic_confusion_{args.dataset}.pdf",
    )
    print(f"Wrote agentic_confusion_{args.dataset}.pdf")
    print(f"Raw API responses logged to {args.log_path}")


if __name__ == "__main__":
    main()
