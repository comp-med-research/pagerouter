"""
Train a 2-layer MLP router on OmniDocBench (omni) embeddings and evaluate on a second split.

By default evaluation is cross-domain OmniDoc → Real5.

For **fusion architecture ablations**, prefer in-domain OmniDoc-only selection so Real5 /
hard296 are reserved for comparing the winner only::

    PYTHONPATH=. python scripts/train_mlp.py ... --feature-mode image_layout --fusion concat \\
      --label-type per_page \\
      --omni-heldout-fraction 0.2 --omni-heldout-split-seed 42

The Real5/Hard296 CSV merged by ``load_predictions`` is unused for scoring when the hold-out
fraction is enabled (any valid ``--real5`` schema works; stub optional).

Uses stratified OmniDoc splits when possible (falls back to a plain shuffle otherwise).

Labels:
  per_page      — oracle argmax per page on the omni training pages (excluding held-out)
  per_doc_type  — doc_type → argmax of column means within page type on train
  per_layout    — layout_type → argmax of column means within layout on train
  per_stratum   — (doc_type, layout_type) → argmax of column means within stratum on train

Appends one row per run to ``results/mlp_runs_detail.csv`` and optional stratum breakdown file.

Usage::

  PYTHONPATH=. python scripts/train_mlp.py --embeddings embeddings/dinov2-base.pt --label-type per_page
  PYTHONPATH=. python scripts/train_mlp.py --feature-mode image_layout \\
      --fusion norm_concat \\
      --image-embeddings data/embeddings/dinov2_omni \\
      --layout-embeddings data/embeddings/layoutlmv3-base.pt \\
      --label-type per_page
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

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
    per_doc_type_labels,
    per_layout_labels,
    per_page_oracle_labels,
    per_stratum_labels,
)
from pagerouter.mlp_splits import omni_holdout_page_ids  # noqa: E402
from pagerouter.mlp_features import (  # noqa: E402
    FeatureMode,
    build_feature_matrix,
    fit_metadata_columns,
)
from pagerouter.multimodal_fusion import FUSION_LABELS, FeatureFusion, FusionMLPRouter, WeightedAvgFusionRouter  # noqa: E402
from pagerouter.routing import MODELS  # noqa: E402

RouterKind = Literal["mlp", "logistic"]


FEATURE_MODES: tuple[FeatureMode, ...] = (
    "image",
    "layout",
    "image_layout",
    "image_metadata",
    "layout_metadata",
    "all",
)

FUSION_CHOICES: tuple[FeatureFusion, ...] = (
    "concat",
    "norm_concat",
    "weighted_avg",
    "gmu",
    "bilinear",
)


def _needs_image(mode: FeatureMode) -> bool:
    return mode in ("image", "image_layout", "image_metadata", "all")


def _needs_layout(mode: FeatureMode) -> bool:
    return mode in ("layout", "image_layout", "layout_metadata", "all")


def _needs_metadata(mode: FeatureMode) -> bool:
    return mode in ("image_metadata", "layout_metadata", "all")


def _encoder_label(mode: FeatureMode, image_enc: str, layout_enc: str) -> str:
    if mode == "image":
        return image_enc
    if mode == "layout":
        return layout_enc
    parts: list[str] = []
    if _needs_image(mode):
        parts.append(image_enc)
    if _needs_layout(mode):
        parts.append(layout_enc)
    if _needs_metadata(mode):
        parts.append("metadata")
    return "+".join(parts)


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


def _resolve_npy_pair(path: Path) -> tuple[Path, Path, str]:
    """Resolve ``{stem}_omni.npy`` + ``{stem}_omni_ids.npy`` from several path forms."""
    path = Path(path)
    if path.name.endswith("_omni.npy"):
        npy = path
        ids_npy = path.with_name(path.name.replace("_omni.npy", "_omni_ids.npy"))
        enc_key = path.name[: -len("_omni.npy")]
    elif path.suffix == ".npy" and path.name.endswith("_omni_ids.npy"):
        ids_npy = path
        npy = path.with_name(path.name.replace("_omni_ids.npy", "_omni.npy"))
        enc_key = npy.name[: -len("_omni.npy")]
    else:
        # Legacy npy: ``dinov2_omni.npy`` + ``dinov2_omni_ids.npy`` (path arg often ``.../dinov2_omni``).
        if path.suffix == "" and (path.parent / f"{path.name}.npy").is_file():
            npy = path.parent / f"{path.name}.npy"
            ids_npy = path.parent / f"{path.name}_ids.npy"
            enc_key = path.name
        else:
            name = path.stem if path.suffix in (".npy", ".omni") else path.name
            npy = path.parent / f"{name}_omni.npy"
            ids_npy = path.parent / f"{name}_omni_ids.npy"
            enc_key = name
    if not npy.is_file():
        raise FileNotFoundError(f"Embedding array not found: {npy}")
    if not ids_npy.is_file():
        raise FileNotFoundError(f"Embedding page_ids not found: {ids_npy}")
    return npy, ids_npy, enc_key


def load_embeddings(path: Path) -> tuple[list[str], torch.Tensor, str]:
    """Load ``.pt`` from extract_embeddings.py or legacy ``*_omni.npy`` + ``*_omni_ids.npy``."""
    path = Path(path)
    if path.suffix == ".pt":
        return load_embedding_pt(path)
    npy, ids_npy, enc_key = _resolve_npy_pair(path)
    emb = torch.tensor(np.load(npy), dtype=torch.float32)
    page_ids = [str(p) for p in np.load(ids_npy, allow_pickle=True)]
    if len(page_ids) != emb.shape[0]:
        raise ValueError(f"{npy}: {len(page_ids)} page_ids vs {emb.shape[0]} rows")
    return page_ids, emb, enc_key


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
    model: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
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
def predict_labels(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return integer class indices."""
    out: list[torch.Tensor] = []
    bs = 256
    for i in range(0, len(X), bs):
        chunk = X[i : i + bs].to(device)
        logits = model(chunk)
        pred = logits.argmax(dim=-1)
        out.append(pred.cpu())
    return torch.cat(out, dim=0).numpy()


def train_logistic_one_seed(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    seed: int,
    C: float,
) -> "LogisticRegression":
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=seed, C=C)
    clf.fit(X_tr.numpy(), y_tr.numpy())
    return clf


def predict_logistic(clf: "LogisticRegression", X: torch.Tensor) -> np.ndarray:
    return clf.predict(X.numpy()).astype(np.int64)


def mean_ned_breakdown(
    selections: pd.Series,
    matrix: pd.DataFrame,
    df: pd.DataFrame,
    stratum_col: str,
    *,
    dataset_scope: str = "real5",
) -> dict[str, float]:
    """Mean realized NED per stratum value (string keys JSON-safe).

    Attributes are taken from rows with ``df['dataset'] == dataset_scope``.
    Cross-domain evaluation uses ``real5``; omni held-out splits use ``omni``.
    """
    attrs = (
        df[df["dataset"] == dataset_scope][["page_id", stratum_col]]
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
    ap.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Image embeddings (.pt or npy pair); alias for --image-embeddings",
    )
    ap.add_argument(
        "--image-embeddings",
        type=Path,
        default=None,
        help="Image encoder embeddings (dinov2)",
    )
    ap.add_argument(
        "--layout-embeddings",
        type=Path,
        default=None,
        help="Layout-from-image embeddings (layoutlmv3-base.pt)",
    )
    ap.add_argument(
        "--feature-mode",
        choices=list(FEATURE_MODES),
        default="image",
        help="Input modalities for the MLP (default: image only)",
    )
    ap.add_argument(
        "--fusion",
        choices=list(FUSION_CHOICES),
        default="concat",
        help="How image+layout are fused before the router MLP "
        "(concat/norm_concat/weighted_avg preprocess; gmu/bilinear fuse inside nn.Module)",
    )
    ap.add_argument(
        "--router",
        choices=["mlp", "logistic"],
        default="mlp",
        help="Router head: 2-layer MLP (default) or sklearn logistic regression on z-scored features",
    )
    ap.add_argument(
        "--logistic-C",
        type=float,
        default=1.0,
        help="Inverse regularization for --router logistic (sklearn LogisticRegression C)",
    )
    ap.add_argument("--hidden", type=int, default=256, help="Hidden width of router MLP")
    ap.add_argument(
        "--bilinear-out-dim",
        type=int,
        default=256,
        help="Fusion width for fusion=bilinear (nn.Bilinear D×D→out_dim)",
    )
    ap.add_argument(
        "--fusion-proj-dim",
        type=int,
        default=None,
        help="Shared projection dim for fusion=weighted_avg when image/layout dims differ "
        "(default: max(d_img, d_layout))",
    )
    ap.add_argument(
        "--test-embeddings",
        type=Path,
        default=None,
        help="Optional separate image embeddings for test pages",
    )
    ap.add_argument(
        "--test-layout-embeddings",
        type=Path,
        default=None,
        help="Optional separate layout embeddings for test pages",
    )
    ap.add_argument(
        "--label-type",
        choices=["per_page", "per_doc_type", "per_layout", "per_stratum"],
        required=True,
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument(
        "--omni-heldout-fraction",
        type=float,
        default=0.0,
        help=(
            "OmniDoc held-out fraction for fusion ablations (>0 ⇒ eval on OmniDoc hold-out rows only, "
            "not Real5; pick best fusion here, then re-run that fusion with "
            "`--omni-heldout-fraction 0` for Real5 / hard296)."
        ),
    )
    ap.add_argument(
        "--omni-heldout-split-seed",
        type=int,
        default=42,
        help="Shuffle seed stratifying OmniDoc page_id into train vs held-out (--omni-heldout-fraction).",
    )
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

    mode: FeatureMode = args.feature_mode
    fusion: FeatureFusion = args.fusion
    router: RouterKind = args.router
    if router == "logistic" and fusion in ("gmu", "bilinear", "weighted_avg"):
        raise SystemExit(
            f"--router=logistic is incompatible with --fusion={fusion!r}; use concat or norm_concat."
        )
    if fusion in ("weighted_avg", "gmu", "bilinear") and not (
        _needs_image(mode) and _needs_layout(mode)
    ):
        raise SystemExit(
            f"--fusion={fusion!r} needs both modalities: use "
            "`--feature-mode image_layout` or `all`, not `{mode}`."
        )

    image_path = args.image_embeddings or args.embeddings
    if _needs_image(mode) and image_path is None:
        raise SystemExit(f"--feature-mode={mode!r} requires --image-embeddings or --embeddings")
    if _needs_layout(mode) and args.layout_embeddings is None:
        raise SystemExit(f"--feature-mode={mode!r} requires --layout-embeddings")

    hold_frac = float(args.omni_heldout_fraction)
    omni_holdout_eval = hold_frac > 1e-12
    if omni_holdout_eval:
        if args.test_embeddings is not None or args.test_layout_embeddings is not None:
            raise SystemExit(
                "--omni-heldout-fraction > 0 is incompatible with --test-embeddings / "
                "--test-layout-embeddings (OmniDoc hold-out uses one embedding artifact for train+eval)."
            )
        if not (0.0 < hold_frac < 1.0):
            raise SystemExit("--omni-heldout-fraction must be strictly between 0 and 1.")

    df = load_predictions(args.omni, args.real5)
    df["page_id"] = df["page_id"].astype(str)
    omni_rows = df[df["dataset"] == "omni"]

    omni_mat_full = get_matrix(omni_rows, "omni")
    omni_mat_full = omni_mat_full.loc[omni_mat_full.notna().any(axis=1)]
    omni_mat_full.index = omni_mat_full.index.astype(str)

    if omni_holdout_eval:
        fit_ids, hold_ids = omni_holdout_page_ids(
            omni_mat_full,
            df,
            holdout_frac=hold_frac,
            split_seed=int(args.omni_heldout_split_seed),
        )
        train_mat = omni_mat_full.loc[fit_ids]
        test_mat = omni_mat_full.loc[hold_ids]
        stratum_dataset_scope = "omni"
        eval_split_note = (
            f"omni_heldout_frac={hold_frac:g}_split_seed={int(args.omni_heldout_split_seed)}"
        )
    else:
        train_mat = omni_mat_full
        real5_rows = df[df["dataset"] == "real5"]
        test_mat = get_matrix(real5_rows, "real5")
        test_mat = test_mat.loc[test_mat.notna().any(axis=1)]
        test_mat.index = test_mat.index.astype(str)
        stratum_dataset_scope = "real5"
        eval_split_note = "cross_domain_real5"

    image_enc = layout_enc = ""
    image_ids: list[str] | None = None
    image_all: torch.Tensor | None = None
    if _needs_image(mode):
        image_ids, image_all, image_enc = load_embeddings(image_path)

    layout_ids: list[str] | None = None
    layout_all: torch.Tensor | None = None
    if _needs_layout(mode):
        layout_ids, layout_all, layout_enc = load_embeddings(args.layout_embeddings)

    metadata_columns = fit_metadata_columns(df) if _needs_metadata(mode) else None

    if args.label_type == "per_page":
        raw_labels = per_page_oracle_labels(train_mat)
    elif args.label_type == "per_doc_type":
        raw_labels = per_doc_type_labels(train_mat, omni_rows)
    elif args.label_type == "per_layout":
        raw_labels = per_layout_labels(train_mat, omni_rows)
    else:
        raw_labels = per_stratum_labels(train_mat, omni_rows)

    raw_labels.index = raw_labels.index.astype(str)

    y_idx_series, _ = model_to_index(raw_labels)
    y_idx_series.index = y_idx_series.index.astype(str)

    train_image_raw: torch.Tensor | None = None
    test_image_raw: torch.Tensor | None = None
    train_layout_raw: torch.Tensor | None = None
    test_layout_raw: torch.Tensor | None = None

    if _needs_image(mode):
        assert image_ids is not None and image_all is not None
        train_image_raw, train_kept = align_matrix(image_ids, image_all, train_mat.index)
        if args.test_embeddings is not None:
            test_page_ids, test_emb, _ = load_embeddings(args.test_embeddings)
            test_image_raw, test_kept = align_matrix(test_page_ids, test_emb, test_mat.index)
        else:
            test_image_raw, test_kept = align_matrix(image_ids, image_all, test_mat.index)

    if _needs_layout(mode):
        assert layout_ids is not None and layout_all is not None
        train_layout_raw, train_kept_layout = align_matrix(layout_ids, layout_all, train_mat.index)
        if args.test_layout_embeddings is not None:
            test_layout_ids, test_layout_emb, _ = load_embeddings(args.test_layout_embeddings)
            test_layout_raw, test_kept_layout = align_matrix(test_layout_ids, test_layout_emb, test_mat.index)
        else:
            test_layout_raw, test_kept_layout = align_matrix(layout_ids, layout_all, test_mat.index)
        if _needs_image(mode):
            if not np.array_equal(train_kept, train_kept_layout):
                raise ValueError("Train page overlap mismatch between image and layout embeddings")
            if not np.array_equal(test_kept, test_kept_layout):
                raise ValueError("Test page overlap mismatch between image and layout embeddings")
        else:
            train_kept = train_kept_layout
            test_kept = test_kept_layout

    train_mat.index = train_mat.index.astype(str)
    train_kept_list = train_kept.tolist()
    test_kept_list = test_kept.tolist()

    X_train_raw = build_feature_matrix(
        mode,
        fusion=fusion,
        page_ids=train_kept_list,
        image=train_image_raw,
        layout=train_layout_raw,
        df=df,
        metadata_columns=metadata_columns,
    )
    X_test_raw = build_feature_matrix(
        mode,
        fusion=fusion,
        page_ids=test_kept_list,
        image=test_image_raw,
        layout=test_layout_raw,
        df=df,
        metadata_columns=metadata_columns,
    )

    y_aligned = torch.tensor([int(y_idx_series.loc[pid]) for pid in train_kept_list], dtype=torch.long)

    X_train_z, X_test_z = zscore_train_apply(X_train_raw, X_test_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(MODELS)

    args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    args.stratum_csv.parent.mkdir(parents=True, exist_ok=True)

    oracle_ned = float(test_mat.max(axis=1).mean())
    best_single_ned, _champion = best_single_realized_ned(train_mat, test_mat)

    enc_key = _encoder_label(mode, image_enc, layout_enc)

    fusion_label = FUSION_LABELS[fusion]
    if (
        fusion == "weighted_avg"
        and train_image_raw is not None
        and train_layout_raw is not None
        and int(train_image_raw.shape[1]) != int(train_layout_raw.shape[1])
    ):
        fusion_label = f"{fusion_label} (projected)"

    detail_exists = args.detail_csv.is_file()
    stratum_exists = args.stratum_csv.is_file()

    for seed in args.seeds:
        if router == "logistic":
            clf = train_logistic_one_seed(
                X_train_z,
                y_aligned,
                seed=seed,
                C=float(args.logistic_C),
            )
            pred_idx = predict_logistic(clf, X_test_z)
        elif fusion == "weighted_avg" and train_image_raw is not None and train_layout_raw is not None:
            di, dl = int(train_image_raw.shape[1]), int(train_layout_raw.shape[1])
            if di == dl:
                core_model = MLPRouter(X_train_z.shape[1], num_classes, hidden=args.hidden)
                model = train_one_seed(
                    X_train_z,
                    y_aligned,
                    X_test_z,
                    model=core_model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                    seed=seed,
                )
                pred_idx = predict_labels(model, X_test_z, device)
            else:
                d_meta = int(X_train_z.shape[1] - di - dl)
                if d_meta < 0:
                    raise RuntimeError(
                        "Feature width mismatch between z-scored X and modality slice sizes."
                    )
                core_model = WeightedAvgFusionRouter(
                    d_img=di,
                    d_layout=dl,
                    d_meta=d_meta,
                    num_classes=num_classes,
                    hidden=args.hidden,
                    proj_dim=args.fusion_proj_dim,
                )
                model = train_one_seed(
                    X_train_z,
                    y_aligned,
                    X_test_z,
                    model=core_model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                    seed=seed,
                )
                pred_idx = predict_labels(model, X_test_z, device)
        elif fusion in ("gmu", "bilinear"):
            if train_image_raw is None or train_layout_raw is None:
                raise RuntimeError("Fusion model requires aligned image/layout tensors.")
            di, dl = int(train_image_raw.shape[1]), int(train_layout_raw.shape[1])
            d_meta = int(X_train_z.shape[1] - di - dl)
            if d_meta < 0:
                raise RuntimeError(
                    "Feature width mismatch between z-scored X and modality slice sizes."
                )
            core_model: nn.Module = FusionMLPRouter(
                fusion=fusion,
                d_img=di,
                d_layout=dl,
                d_meta=d_meta,
                num_classes=num_classes,
                hidden=args.hidden,
                bilinear_out_dim=args.bilinear_out_dim,
            )
            model = train_one_seed(
                X_train_z,
                y_aligned,
                X_test_z,
                model=core_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                seed=seed,
            )
            pred_idx = predict_labels(model, X_test_z, device)
        else:
            core_model = MLPRouter(X_train_z.shape[1], num_classes, hidden=args.hidden)
            model = train_one_seed(
                X_train_z,
                y_aligned,
                X_test_z,
                model=core_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                seed=seed,
            )
            pred_idx = predict_labels(model, X_test_z, device)
        pred_models = [MODELS[i] for i in pred_idx]
        selections = pd.Series(pred_models, index=test_kept_list, name="model")
        router_ned = float(per_page_ned(selections, test_mat).mean())
        gap_pct = oracle_gap_recovered(router_ned, best_single_ned, oracle_ned)

        row = {
            "router": router,
            "encoder": enc_key,
            "feature_mode": mode,
            "feature_fusion": fusion,
            "fusion_label": fusion_label,
            "eval_split": eval_split_note,
            "omni_heldout_fraction": str(hold_frac),
            "embeddings_file": str(image_path.name if image_path else ""),
            "layout_embeddings_file": str(args.layout_embeddings.name if args.layout_embeddings else ""),
            "label_type": args.label_type,
            "seed": seed,
            "mean_ned_real5": router_ned,
            "oracle_gap_recovered": gap_pct,
            "oracle_ned": oracle_ned,
            "best_single_ned": best_single_ned,
            "best_single_champion_omni": _champion,
            "n_train": len(train_kept_list),
            "n_eval": len(test_kept_list),
            "input_dim": int(X_train_z.shape[1]),
        }
        with args.detail_csv.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not detail_exists:
                w.writeheader()
                detail_exists = True
            w.writerow(row)

        for col in ("doc_type", "layout_type"):
            br = mean_ned_breakdown(
                selections,
                test_mat,
                df,
                col,
                dataset_scope=stratum_dataset_scope,
            )
            srow = {
                "router": router,
                "encoder": row["encoder"],
                "feature_mode": mode,
                "feature_fusion": fusion,
                "eval_split": eval_split_note,
                "omni_heldout_fraction": str(hold_frac),
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
