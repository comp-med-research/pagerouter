"""
Extract DocLayout-YOLO detections and derived layout embeddings.

Writes under ``--out-dir``:
  doclayout_detections.pt   — cached boxes per page
  layoutlmv3-detected.pt    — LayoutLMv3 pooled over detected boxes (no OCR)
  doclayout-yolo-stats.pt   — fixed-size detector summary vectors

Usage::

  PYTHONPATH=. python scripts/extract_detected_layout.py \\
      --omni data/omni_predictions.csv \\
      --real5 data/hard296/omni_slice.csv \\
      --out-dir data/embeddings/omni_hard296
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagerouter.doclayout import (  # noqa: E402
    DOCLAYOUT_STATS_DIM,
    blob_to_detections,
    detect_pages,
    detections_to_blob,
    doclayout_stats_vector,
    load_yolo,
)
from pagerouter.embed_backends import load_encoder, load_page_image, pooled_layoutlmv3_embedding  # noqa: E402
from pagerouter.embed_backends import ENCODER_REGISTRY  # noqa: E402
from pagerouter.load import DEFAULT_OMNI, DEFAULT_REAL5, load_predictions  # noqa: E402
from pagerouter.mlp_paths import resolve_page_image  # noqa: E402


def all_page_ids(df) -> list[str]:
    return sorted(df["page_id"].astype(str).unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "embeddings" / "omni_hard296")
    ap.add_argument("--image-root", type=Path, default=ROOT / "data" / "page_images")
    ap.add_argument("--batch-size", type=int, default=8, help="Pages per YOLO batch (sequential per page)")
    ap.add_argument("--layout-batch-size", type=int, default=4)
    ap.add_argument("--skip-detect", action="store_true", help="Reuse existing doclayout_detections.pt")
    ap.add_argument("--skip-stats", action="store_true", help="Reuse existing doclayout-yolo-stats.pt")
    ap.add_argument("--layout-only", action="store_true", help="Only build layoutlmv3-detected.pt (needs detections cache)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    df = load_predictions(args.omni, args.real5)
    page_ids = all_page_ids(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    det_path = args.out_dir / "doclayout_detections.pt"
    stats_path = args.out_dir / "doclayout-yolo-stats.pt"
    layout_path = args.out_dir / "layoutlmv3-detected.pt"

    if args.layout_only:
        if not det_path.is_file():
            raise SystemExit(f"--layout-only requires {det_path}")
        blob = torch.load(det_path, map_location="cpu", weights_only=False)
        cached_ids, dets = blob_to_detections(blob)
        pos = {p: i for i, p in enumerate(cached_ids)}
        missing = [p for p in page_ids if p not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} page_ids missing from {det_path}")
        dets = [dets[pos[p]] for p in page_ids]
    elif args.skip_detect and det_path.is_file():
        blob = torch.load(det_path, map_location="cpu", weights_only=False)
        cached_ids, dets = blob_to_detections(blob)
        pos = {p: i for i, p in enumerate(cached_ids)}
        missing = [p for p in page_ids if p not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} page_ids missing from {det_path} (e.g. {missing[0]})")
        dets = [dets[pos[p]] for p in page_ids]
    else:
        yolo, yolo_device = load_yolo(device=args.device)
        all_dets = []
        for i in tqdm(range(0, len(page_ids), args.batch_size), desc="doclayout-yolo"):
            batch_ids = page_ids[i : i + args.batch_size]
            images = [load_page_image(resolve_page_image(pid, args.image_root)) for pid in batch_ids]
            all_dets.extend(
                detect_pages(yolo, images, device=yolo_device, conf=0.2, imgsz=1024, max_boxes=64)
            )
        torch.save(detections_to_blob(page_ids, all_dets), det_path)
        dets = all_dets
        print(f"Wrote {det_path}")

    if not args.layout_only and not (args.skip_stats and stats_path.is_file()):
        stats_rows = [doclayout_stats_vector(det) for det in dets]
        stats = torch.stack(stats_rows, dim=0).float()
        torch.save(
            {
                "encoder": "doclayout-yolo-stats",
                "hf_model_id": "doclayout_yolo_docstructbench",
                "page_ids": page_ids,
                "embeddings": stats,
                "dim": DOCLAYOUT_STATS_DIM,
            },
            stats_path,
        )
        print(f"Wrote {stats_path} ({stats.shape})")
    elif stats_path.is_file():
        print(f"Skip stats (exists): {stats_path}")

    if layout_path.is_file() and not args.layout_only:
        print(f"Skip layoutlmv3-detected (exists): {layout_path}")
        print("Done.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ENCODER_REGISTRY["layoutlmv3-base"]
    model, processor = load_encoder(spec, device)
    layout_rows: list[torch.Tensor] = []
    boxes_batch = [d.boxes_1000 for d in dets]
    words_batch = [d.words for d in dets]
    for i in tqdm(range(0, len(page_ids), args.layout_batch_size), desc="layoutlmv3-detected"):
        images = [
            load_page_image(resolve_page_image(pid, args.image_root))
            for pid in page_ids[i : i + args.layout_batch_size]
        ]
        b = i + len(images)
        vec = pooled_layoutlmv3_embedding(
            model,
            processor,
            images,
            device,
            boxes_per_image=boxes_batch[i:b],
            words_per_image=words_batch[i:b],
        )
        layout_rows.append(vec)
    layout_emb = torch.cat(layout_rows, dim=0).float()
    layout_path = args.out_dir / "layoutlmv3-detected.pt"
    torch.save(
        {
            "encoder": "layoutlmv3-detected",
            "hf_model_id": spec.hf_model_id,
            "page_ids": page_ids,
            "embeddings": layout_emb,
        },
        layout_path,
    )
    print(f"Wrote {layout_path} ({layout_emb.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
