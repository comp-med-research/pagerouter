"""
Extract frozen vision embeddings for all pages in omni ∪ real5.

Each encoder run writes ``{out_dir}/{encoder_key}.pt`` with keys:
  encoder, hf_model_id, page_ids, embeddings (float32 tensor [N, dim]).

Vision encoders (HuggingFace): see ``pagerouter.embed_backends.ENCODER_REGISTRY``.

Usage::

  PYTHONPATH=. python scripts/extract_embeddings.py --encoder qwen3-vl-8b-instruct
  PYTHONPATH=. python scripts/extract_embeddings.py --encoder all --out-dir embeddings
  PYTHONPATH=. python scripts/extract_embeddings.py --encoder all_with_vlm --out-dir embeddings
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

from pagerouter.embed_backends import (  # noqa: E402
    ENCODER_REGISTRY,
    list_encoder_keys,
    load_encoder,
    load_page_image,
    pooled_embedding,
    pooled_jina_embedding,
    pooled_colbert_doc_embedding,
    pooled_layoutlmv3_embedding,
    pooled_siglip2_embedding,
    pooled_vlm_embedding,
)
from pagerouter.load import DATA_DIR, DEFAULT_OMNI, DEFAULT_REAL5, load_predictions  # noqa: E402
from pagerouter.mlp_paths import resolve_page_image  # noqa: E402


def all_page_ids(df) -> list[str]:
    return sorted(df["page_id"].astype(str).unique().tolist())


@torch.inference_mode()
def run_one_encoder(
    encoder_key: str,
    page_ids: list[str],
    out_path: Path,
    batch_size: int,
    device: torch.device,
    image_root: Path,
) -> None:
    spec = ENCODER_REGISTRY[encoder_key]
    model, processor = load_encoder(spec, device)
    embs: list[torch.Tensor] = []
    for i in tqdm(range(0, len(page_ids), batch_size), desc=f"{encoder_key}"):
        batch_ids = page_ids[i : i + batch_size]
        images = []
        for pid in batch_ids:
            img_path = resolve_page_image(pid, image_root)
            images.append(load_page_image(img_path))
        if spec.family in ("qwen3_vl", "gemma3"):
            vec = pooled_vlm_embedding(model, spec, processor, images, device).detach().cpu().float()
        elif spec.family in ("jina_clip", "jina_embeddings_v4"):
            vec = pooled_jina_embedding(model, spec, images, device).detach().cpu().float()
        elif spec.family in ("colqwen2", "colpali", "nemotron_colembed"):
            vec = pooled_colbert_doc_embedding(model, spec, processor, images, device).detach().cpu().float()
        elif spec.family == "layoutlmv3":
            vec = pooled_layoutlmv3_embedding(model, processor, images, device)
        elif spec.family == "siglip2":
            vec = pooled_siglip2_embedding(model, processor, images, device)
        else:
            pix = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            vec = pooled_embedding(model, spec.family, pix).detach().cpu()
        embs.append(vec)
    stacked = torch.cat(embs, dim=0).float()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder_key,
            "hf_model_id": spec.hf_model_id,
            "page_ids": page_ids,
            "embeddings": stacked,
        },
        out_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--encoder",
        default="dinov2-base",
        help="Registry key, or presets: all (patch only), all_with_vlm (+ Qwen3-VL & Gemma 3).",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "embeddings")
    ap.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    ap.add_argument("--real5", type=Path, default=DEFAULT_REAL5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--layout-batch-size",
        type=int,
        default=4,
        help="Batch size for LayoutLMv3 (lower memory than patch encoders)",
    )
    ap.add_argument(
        "--heavy-batch-size",
        type=int,
        default=1,
        help="Batch size for heavy VLMs / Jina encoders (one image at a time recommended)",
    )
    ap.add_argument(
        "--siglip2-batch-size",
        type=int,
        default=4,
        help="Batch size for SigLIP 2 NaFlex",
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        default=DATA_DIR / "page_images",
        help="Override image directory",
    )
    args = ap.parse_args()

    df = load_predictions(args.omni, args.real5)
    page_ids = all_page_ids(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoder == "all":
        keys = list_encoder_keys(include_heavy_vlm=False)
    elif args.encoder == "all_with_vlm":
        keys = list_encoder_keys(include_heavy_vlm=True)
    else:
        keys = [args.encoder]
    for k in keys:
        if k not in ENCODER_REGISTRY:
            raise SystemExit(
                f"Unknown encoder {k!r}. Choose from: {list_encoder_keys()} "
                "(or presets: all, all_with_vlm)"
            )
        out_path = args.out_dir / f"{k}.pt"
        print(f"Writing {out_path} ({len(page_ids)} pages, device={device})")
        bs = args.layout_batch_size if ENCODER_REGISTRY[k].family == "layoutlmv3" else args.batch_size
        if ENCODER_REGISTRY[k].family in (
            "jina_clip",
            "jina_embeddings_v4",
            "qwen3_vl",
            "gemma3",
            "colqwen2",
            "colpali",
            "nemotron_colembed",
        ):
            bs = min(bs, args.heavy_batch_size)
        if ENCODER_REGISTRY[k].family == "siglip2":
            bs = min(bs, args.siglip2_batch_size)
        run_one_encoder(k, page_ids, out_path, bs, device, args.image_root)
    print("Done.")


if __name__ == "__main__":
    main()
