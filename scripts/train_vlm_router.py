#!/usr/bin/env python3
"""Fine-tune VLM / document encoders for parser routing (page classification)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter.evaluate import (  # noqa: E402
    best_single_realized_ned,
    oracle_gap_recovered,
    per_page_ned,
)
from pagerouter.load import DEFAULT_OMNI, DEFAULT_REAL5, get_matrix, load_predictions  # noqa: E402
from pagerouter.mlp_labels import model_to_index, per_stratum_labels  # noqa: E402
from pagerouter.mlp_paths import DEFAULT_IMAGE_ROOT  # noqa: E402
from pagerouter.routing import MODELS  # noqa: E402
from pagerouter.vlm_finetune.dataset import PageRouterDataset, collate_pages, page_ids_and_labels  # noqa: E402
from pagerouter.vlm_finetune.lora import apply_paradigm, trainable_param_count  # noqa: E402
from pagerouter.vlm_finetune.model import PageRouter, RouterHead, build_page_router  # noqa: E402
from pagerouter.vlm_finetune.registry import (  # noqa: E402
    PARADIGMS,
    PARADIGM_LABELS,
    VLM_ROUTER_MODELS,
)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_router(
    router: PageRouter,
    page_ids: list[str],
    test_mat: pd.DataFrame,
    device: torch.device,
    *,
    image_root: Path,
    batch_size: int,
) -> float:
    router.eval()
    labels_dummy = pd.Series([0] * len(page_ids), index=page_ids)
    ds = PageRouterDataset(page_ids, torch.zeros(len(page_ids), dtype=torch.long), image_root=image_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pages)
    pred_models: list[str] = []
    kept: list[str] = []
    for batch in loader:
        logits = router(batch["images"], device)
        idx = logits.argmax(dim=-1).cpu().tolist()
        pred_models.extend([MODELS[i] for i in idx])
        kept.extend(batch["page_ids"])
    selections = pd.Series(pred_models, index=kept, name="model")
    return float(per_page_ned(selections, test_mat).mean())


def train_one(
    router: PageRouter,
    head: RouterHead,
    train_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    grad_accum: int,
) -> None:
    params = [p for p in list(router.backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    router.train()
    head.train()

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            logits = router(batch["images"], device)
            loss = loss_fn(logits, batch["labels"].to(device)) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * grad_accum
            n_batches += 1
            pbar.set_postfix(loss=f"{running / max(n_batches, 1):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(VLM_ROUTER_MODELS.keys()), required=True)
    ap.add_argument("--paradigm", choices=PARADIGMS, required=True)
    ap.add_argument("--label-type", default="per_stratum", choices=["per_stratum"])
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument("--hard296", type=Path, default=ROOT / "data/hard296/test_as_real5.csv")
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/vlm_finetuned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-optional-on-error", action="store_true")
    args = ap.parse_args()

    spec = VLM_ROUTER_MODELS[args.model]
    out_dir = args.out_dir / spec.key / args.paradigm
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"Skip (metrics exist): {metrics_path}")
        return

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {spec.key} | {args.paradigm} | {PARADIGM_LABELS[args.paradigm]} ===")
    print(f"Device: {device}")

    try:
        router, head, backbone, _processor, embed_dim = build_page_router(spec, len(MODELS), device)
    except Exception as exc:
        if spec.optional and args.skip_optional_on_error:
            print(f"SKIP optional model {spec.key}: {exc}")
            metrics_path.write_text(json.dumps({"status": "skipped", "error": str(exc)}, indent=2))
            return
        raise

    backbone = apply_paradigm(backbone, head, spec, args.paradigm)
    router.backbone = backbone
    trainable, total = trainable_param_count(backbone, head)
    print(f"Params trainable: {trainable:,} / {total:,} | embed_dim={embed_dim}")

    # Train labels from full omni
    df_omni = load_predictions(args.omni, args.real5)
    df_omni["page_id"] = df_omni["page_id"].astype(str)
    omni_rows = df_omni[df_omni["dataset"] == "omni"]
    train_mat = get_matrix(omni_rows, "omni")
    train_mat = train_mat.loc[train_mat.notna().any(axis=1)]
    train_mat.index = train_mat.index.astype(str)

    raw_labels = per_stratum_labels(train_mat, omni_rows)
    y_series, _ = model_to_index(raw_labels)
    train_ids, y_train = page_ids_and_labels(train_mat, y_series)

    train_ds = PageRouterDataset(train_ids, y_train, image_root=args.image_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        collate_fn=collate_pages,
        num_workers=0,
    )

    train_one(
        router,
        head,
        train_loader,
        device,
        epochs=spec.epochs,
        lr=spec.lr,
        grad_accum=spec.grad_accum,
    )

    # Eval Real5 + hard296
    results: dict[str, object] = {
        "model": spec.key,
        "hf_model_id": spec.hf_model_id,
        "paradigm": args.paradigm,
        "paradigm_label": PARADIGM_LABELS[args.paradigm],
        "label_type": args.label_type,
        "trainable_params": trainable,
        "total_params": total,
        "embed_dim": embed_dim,
        "n_train": len(train_ids),
        "evals": {},
    }

    for split_name, real5_csv in (("real5", args.real5), ("hard296", args.hard296)):
        df = load_predictions(args.omni, real5_csv)
        df["page_id"] = df["page_id"].astype(str)
        test_mat = get_matrix(df[df["dataset"] == "real5"], "real5")
        test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
        test_mat.index = test_mat.index.astype(str)
        test_ids = test_mat.index.astype(str).tolist()

        oracle_ned = float(test_mat.max(axis=1).mean())
        best_single_ned, champion = best_single_realized_ned(train_mat, test_mat)
        ned = evaluate_router(
            router,
            test_ids,
            test_mat,
            device,
            image_root=args.image_root,
            batch_size=max(1, spec.batch_size),
        )
        gap = oracle_gap_recovered(ned, best_single_ned, oracle_ned)
        results["evals"][split_name] = {
            "mean_ned": ned,
            "oracle_ned": oracle_ned,
            "best_single_ned": best_single_ned,
            "best_single_champion": champion,
            "oracle_gap_recovered": gap,
            "n_eval": len(test_ids),
        }
        print(f"{split_name}: mean_ned={ned:.4f} gap={gap:.1%}")

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "backbone": backbone.state_dict(),
            "spec_key": spec.key,
            "paradigm": args.paradigm,
        },
        ckpt_dir / "router.pt",
    )
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
