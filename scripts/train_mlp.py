"""
Train a 2-layer MLP router on OmniDocBench (omni) embeddings and evaluate on Real5.

Labels:
  per_page  — oracle argmax per page on omni
  per_stratum — (doc_type, layout_type) → argmax of column means within stratum on omni

Appends one row per run to ``results/mlp_runs_detail.csv`` and optional stratum breakdown file.

Usage::

  PYTHONPATH=. python scripts/train_mlp.py --embeddings embeddings/dinov2-base.pt --label-type per_page
  PYTHONPATH=. python scripts/train_mlp.py --embeddings embeddings/dinov2-base.pt --label-type per_stratum --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter.evaluate import best_single_realized_ned, oracle_gap_recovered, per_page_ned  # noqa: E402
from pagerouter.load import (  # noqa: E402
    DEFAULT_OMNI,
    DEFAULT_REAL5,
    get_matrix,
    load_predictions,
)
from pagerouter.mlp_labels import (  # noqa: E402
    model_to_index,
    per_page_oracle_labels,
    per_stratum_labels,
)
from pagerouter.routing import MODELS  # noqa: E402


class MLPRouter(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_embedding_pt(path: Path) -> tuple[list[str], torch.Tensor, str]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    page_ids = [str(p) for p in blob["page_ids"]]
    emb = blob["embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    enc = str(blob.get("encoder", path.stem))
    return page_ids, emb.float(), enc


def align_matrix(page_ids: list[str], emb: torch.Tensor, index_order: pd.Index) -> tuple[torch.Tensor, np.ndarray]:
    """Reorder / subset rows to match ``index_order`` (drops pages missing from embeddings)."""
    pos = {p: i for i, p in enumerate(page_ids)}
    rows: list[int] = []
    kept_idx: list[str] = []
    for p in index_order.astype(str):
        if p in pos:
            rows.append(pos[p])
            kept_idx.append(str(p))
    if not rows:
        raise ValueError("No overlap between embedding page_ids and dataframe index.")
    x = emb[rows]
    return x, np.array(kept_idx, dtype=object)


def zscore_train_apply(X_train: torch.Tensor, X_other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = X_train.mean(dim=0, keepdim=True)
    sigma = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X_train - mu) / sigma, (X_other - mu) / sigma


def train_one_seed(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> MLPRouter:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = MLPRouter(X_tr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.inference_mode()
def predict_labels(model: MLPRouter, X: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return integer class indices."""
    out: list[torch.Tensor] = []
    bs = 256
    for i in range(0, len(X), bs):
        chunk = X[i : i + bs].to(device)
        logits = model(chunk)
        pred = logits.argmax(dim=-1)
        out.append(pred.cpu())
    return torch.cat(out, dim=0).numpy()


def mean_ned_breakdown(
    selections: pd.Series,
    matrix: pd.DataFrame,
    df_real5: pd.DataFrame,
    stratum_col: str,
) -> dict[str, float]:
    """Mean realized NED per stratum value (string keys JSON-safe)."""
    attrs = (
        df_real5[df_real5["dataset"] == "real5"][["page_id", stratum_col]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    scores = per_page_ned(selections, matrix)
    merged = scores.to_frame("ned").join(attrs, how="left")
    out: dict[str, float] = {}
    for val, g in merged.groupby(stratum_col):
        out[str(val)] = float(g["ned"].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, required=True, help="Path to {encoder}.pt from extract_embeddings.py")
    ap.add_argument("--label-type", choices=["per_page", "per_stratum"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument(
        "--detail-csv",
        type=Path,
        default=ROOT / "results" / "mlp_runs_detail.csv",
        help="Append one row per seed run",
    )
    ap.add_argument(
        "--stratum-csv",
        type=Path,
        default=ROOT / "results" / "mlp_ablation_by_stratum.csv",
        help="Append per–doc_type / layout_type stats per seed",
    )
    args = ap.parse_args()

    df = load_predictions(args.omni, args.real5)
    df["page_id"] = df["page_id"].astype(str)
    train_df = df[df["dataset"] == "omni"]
    test_df_view = df[df["dataset"] == "real5"]

    train_mat = get_matrix(train_df, "omni")
    test_mat = get_matrix(test_df_view, "real5")

    train_mat = train_mat.loc[train_mat.notna().any(axis=1)]
    test_mat = test_mat.loc[test_mat.notna().any(axis=1)]

    train_mat.index = train_mat.index.astype(str)
    test_mat.index = test_mat.index.astype(str)

    page_ids_file, emb_all, enc_key = load_embedding_pt(args.embeddings)

    if args.label_type == "per_page":
        raw_labels = per_page_oracle_labels(train_mat)
    else:
        raw_labels = per_stratum_labels(train_mat, train_df)

    raw_labels.index = raw_labels.index.astype(str)

    y_idx_series, _ = model_to_index(raw_labels)
    y_idx_series.index = y_idx_series.index.astype(str)

    X_train_raw, train_kept = align_matrix(page_ids_file, emb_all, train_mat.index)
    train_kept_list = train_kept.tolist()
    y_aligned = torch.tensor([int(y_idx_series.loc[pid]) for pid in train_kept_list], dtype=torch.long)

    X_test_raw, test_kept = align_matrix(page_ids_file, emb_all, test_mat.index)

    X_train_z, X_test_z = zscore_train_apply(X_train_raw, X_test_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(MODELS)

    args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    args.stratum_csv.parent.mkdir(parents=True, exist_ok=True)

    oracle_ned = float(test_mat.max(axis=1).mean())
    best_single_ned, _champion = best_single_realized_ned(train_mat, test_mat)

    detail_exists = args.detail_csv.is_file()
    stratum_exists = args.stratum_csv.is_file()

    for seed in args.seeds:
        model = train_one_seed(
            X_train_z,
            y_aligned,
            X_test_z,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=seed,
        )
        pred_idx = predict_labels(model, X_test_z, device)
        pred_models = [MODELS[i] for i in pred_idx]
        selections = pd.Series(pred_models, index=test_kept, name="model")
        router_ned = float(per_page_ned(selections, test_mat).mean())
        gap_pct = oracle_gap_recovered(router_ned, best_single_ned, oracle_ned)

        row = {
            "encoder": enc_key if enc_key else args.embeddings.stem,
            "embeddings_file": str(args.embeddings.name),
            "label_type": args.label_type,
            "seed": seed,
            "mean_ned_real5": router_ned,
            "oracle_gap_recovered": gap_pct,
            "oracle_ned": oracle_ned,
            "best_single_ned": best_single_ned,
            "best_single_champion_omni": _champion,
            "n_train": len(train_kept_list),
            "n_eval": len(test_kept),
        }
        with args.detail_csv.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not detail_exists:
                w.writeheader()
                detail_exists = True
            w.writerow(row)

        for col in ("doc_type", "layout_type"):
            br = mean_ned_breakdown(selections, test_mat, df, col)
            srow = {
                "encoder": row["encoder"],
                "label_type": args.label_type,
                "seed": seed,
                "stratum_col": col,
                "json_mean_ned_by_value": json.dumps(br, ensure_ascii=False),
            }
            with args.stratum_csv.open("a", newline="") as fh2:
                w2 = csv.DictWriter(fh2, fieldnames=list(srow.keys()))
                if not stratum_exists:
                    w2.writeheader()
                    stratum_exists = True
                w2.writerow(srow)

        print(f"seed={seed} mean_ned={router_ned:.4f} gap_rec={gap_pct:.3f}")

    print(f"Appended runs to {args.detail_csv}")


if __name__ == "__main__":
    main()
