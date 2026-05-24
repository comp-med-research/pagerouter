"""Frozen vision encoders (Hugging Face) for MLP routing features.

**Patch encoders:** DINOv2, CLIP, SigLIP, DiT — standard ``pixel_values`` → vision backbone.

**VLM encoders (vision-only pooling for routing):**
  - **Qwen3-VL** — use ``get_image_features``; pool per-image token features to one vector.
  - **Gemma 3** — use ``get_image_features``; pool projected multimodal tokens.

Suggested **fine-tune / LoRA targets** (not wired here): same checkpoints with a small head or LoRA on
top of these pooled vectors for parser classification.

Requires a recent ``transformers`` with ``Qwen3VLForConditionalGeneration`` and
``Gemma3ForConditionalGeneration`` (install/upgrade if imports fail).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPVisionModel,
    SiglipVisionModel,
)

try:
    from transformers import (
        Gemma3ForConditionalGeneration,
        LayoutLMv3Model,
        LayoutLMv3Processor,
        Qwen3VLForConditionalGeneration,
        Siglip2VisionModel,
    )
except ImportError:  # pragma: no cover — optional until transformers version catches up
    Gemma3ForConditionalGeneration = None  # type: ignore[misc, assignment]
    LayoutLMv3Model = None  # type: ignore[misc, assignment]
    LayoutLMv3Processor = None  # type: ignore[misc, assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[misc, assignment]
    Siglip2VisionModel = None  # type: ignore[misc, assignment]

EncoderFamily = Literal[
    "dinov2",
    "clip",
    "siglip",
    "siglip2",
    "dit",
    "layoutlmv3",
    "qwen3_vl",
    "gemma3",
    "jina_clip",
    "jina_embeddings_v4",
    "colqwen2",
    "colpali",
    "nemotron_colembed",
]


def apply_transformers_compat_patches() -> None:
    """Best-effort shims for Jina remote code on newer ``transformers`` builds."""
    try:
        import transformers.models.clip.modeling_clip as clip_mod

        if not hasattr(clip_mod, "clip_loss"):

            def _clip_loss_stub(*args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("clip_loss stub — inference-only")

            clip_mod.clip_loss = _clip_loss_stub  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS and "linear" in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["linear"]
    except Exception:
        pass


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    hf_model_id: str
    family: EncoderFamily


# Keys match CLI `--encoder` in extract_embeddings.py / train_mlp.py
ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "dinov2-small": EncoderSpec("dinov2-small", "facebook/dinov2-small", "dinov2"),
    "dinov2-base": EncoderSpec("dinov2-base", "facebook/dinov2-base", "dinov2"),
    "dinov2-large": EncoderSpec("dinov2-large", "facebook/dinov2-large", "dinov2"),
    "clip-l-14": EncoderSpec("clip-l-14", "openai/clip-vit-large-patch14", "clip"),
    "siglip-l16": EncoderSpec(
        "siglip-l16",
        "google/siglip-large-patch16-384",
        "siglip",
    ),
    "siglip2-base-naflex": EncoderSpec(
        "siglip2-base-naflex",
        "google/siglip2-base-patch16-naflex",
        "siglip2",
    ),
    "jina-clip-v2": EncoderSpec(
        "jina-clip-v2",
        "jinaai/jina-clip-v2",
        "jina_clip",
    ),
    "jina-embeddings-v4": EncoderSpec(
        "jina-embeddings-v4",
        "jinaai/jina-embeddings-v4",
        "jina_embeddings_v4",
    ),
    "dit-base": EncoderSpec("dit-base", "microsoft/dit-base", "dit"),
    "dit-large": EncoderSpec("dit-large", "microsoft/dit-large", "dit"),
    "layoutlmv3-base": EncoderSpec(
        "layoutlmv3-base",
        "microsoft/layoutlmv3-base",
        "layoutlmv3",
    ),
    # VLMs (heavy — use bfloat16 on GPU; smaller checkpoints recommended for ablations)
    "qwen3-vl-8b-instruct": EncoderSpec(
        "qwen3-vl-8b-instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3_vl",
    ),
    "gemma-3-4b-it": EncoderSpec(
        "gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "gemma3",
    ),
    # Document-retrieval VLMs (ColBERT-style multi-vector → mean-pooled for routing MLP)
    "colqwen2-v1.0": EncoderSpec(
        "colqwen2-v1.0",
        "vidore/colqwen2-v1.0",
        "colqwen2",
    ),
    "colpali-v1.3": EncoderSpec(
        "colpali-v1.3",
        "vidore/colpali-v1.3",
        "colpali",
    ),
    "nemotron-colembed-vl-8b-v2": EncoderSpec(
        "nemotron-colembed-vl-8b-v2",
        "nvidia/nemotron-colembed-vl-8b-v2",
        "nemotron_colembed",
    ),
}

# Excluded from ``--encoder all`` (use ``all_with_vlm`` or name keys explicitly).
HEAVY_VLM_KEYS: frozenset[str] = frozenset(
    {
        "qwen3-vl-8b-instruct",
        "gemma-3-4b-it",
        "jina-clip-v2",
        "jina-embeddings-v4",
        "colqwen2-v1.0",
        "colpali-v1.3",
        "nemotron-colembed-vl-8b-v2",
    }
)


def list_encoder_keys(*, include_heavy_vlm: bool = True) -> list[str]:
    keys = sorted(ENCODER_REGISTRY.keys())
    if not include_heavy_vlm:
        keys = [k for k in keys if k not in HEAVY_VLM_KEYS]
    return keys


def _vlm_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_encoder(spec: EncoderSpec, device: torch.device) -> tuple[nn.Module, Any]:
    """Load frozen encoder + processor (``AutoImageProcessor`` or ``AutoProcessor`` for VLMs)."""
    apply_transformers_compat_patches()
    if spec.family == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is missing; upgrade transformers "
                "(e.g. pip install -U 'transformers>=4.51')."
            )
        dtype = _vlm_dtype(device)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(spec.hf_model_id, trust_remote_code=True)
    elif spec.family == "gemma3":
        if Gemma3ForConditionalGeneration is None:
            raise ImportError(
                "Gemma3ForConditionalGeneration is missing; upgrade transformers "
                "(e.g. pip install -U 'transformers>=4.49')."
            )
        dtype = _vlm_dtype(device)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(spec.hf_model_id)
    elif spec.family == "clip":
        model = CLIPVisionModel.from_pretrained(spec.hf_model_id)
        processor = AutoImageProcessor.from_pretrained(spec.hf_model_id)
    elif spec.family == "siglip":
        model = SiglipVisionModel.from_pretrained(spec.hf_model_id)
        processor = AutoImageProcessor.from_pretrained(spec.hf_model_id)
    elif spec.family == "siglip2":
        if Siglip2VisionModel is None:
            raise ImportError("Siglip2VisionModel missing; upgrade transformers (>=4.50).")
        model = Siglip2VisionModel.from_pretrained(spec.hf_model_id)
        processor = AutoImageProcessor.from_pretrained(spec.hf_model_id)
    elif spec.family in ("jina_clip", "jina_embeddings_v4"):
        dtype = _vlm_dtype(device)
        model = AutoModel.from_pretrained(
            spec.hf_model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = None
    elif spec.family == "colqwen2":
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as e:
            raise ImportError(
                "colpali-engine is required for colqwen2-v1.0 "
                "(pip install 'colpali-engine>=0.3.4')."
            ) from e
        dtype = _vlm_dtype(device)
        model = ColQwen2.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = ColQwen2Processor.from_pretrained(spec.hf_model_id)
    elif spec.family == "colpali":
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as e:
            raise ImportError(
                "colpali-engine is required for colpali-v1.3 "
                "(pip install 'colpali-engine>=0.3.4')."
            ) from e
        dtype = _vlm_dtype(device)
        model = ColPali.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = ColPaliProcessor.from_pretrained(spec.hf_model_id)
    elif spec.family == "nemotron_colembed":
        dtype = _vlm_dtype(device)
        model = AutoModel.from_pretrained(
            spec.hf_model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = None
    elif spec.family == "layoutlmv3":
        if LayoutLMv3Model is None or LayoutLMv3Processor is None:
            raise ImportError("LayoutLMv3Model/Processor missing; upgrade transformers.")
        model = LayoutLMv3Model.from_pretrained(spec.hf_model_id)
        processor = LayoutLMv3Processor.from_pretrained(spec.hf_model_id, apply_ocr=False)
    else:
        model = AutoModel.from_pretrained(spec.hf_model_id)
        processor = AutoImageProcessor.from_pretrained(spec.hf_model_id)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def prepare_qwen3_vl_inputs_one(
    processor: Any,
    image: Image.Image,
    device: torch.device,
) -> Any:
    """One image → processor batch (with ``pixel_values``, ``image_grid_thw``, …)."""
    messages = _qwen3_messages_single(image)
    batch_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(batch_inputs, "to"):
        return batch_inputs.to(device)
    return batch_inputs


def _qwen3_messages_single(image: Image.Image, *, short_prompt: str = ".") -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": short_prompt},
            ],
        }
    ]


def prepare_gemma3_inputs_one(
    processor: Any,
    image: Image.Image,
    device: torch.device,
) -> Any:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "."},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    out = processor(text=rendered, images=image, return_tensors="pt", padding=False)
    if hasattr(out, "to"):
        return out.to(device)
    return out


@torch.inference_mode()
def pooled_embedding(
    model: nn.Module,
    family: EncoderFamily,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Map a batch of `pixel_values` to one vector per image (B, D) — patch encoders only."""
    outputs = model(pixel_values=pixel_values)
    if family == "dinov2":
        return outputs.last_hidden_state[:, 0, :]
    if family == "clip":
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]
    if family == "siglip":
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]
    if family == "siglip2":
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]
    if family == "dit":
        return outputs.last_hidden_state[:, 0, :]
    raise ValueError(f"pooled_embedding: unsupported family {family!r} (use pooled_vlm_embedding for VLMs)")


@torch.inference_mode()
def pooled_vlm_embedding(
    model: nn.Module,
    spec: EncoderSpec,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Vision-only embedding (B, D) for Qwen3-VL or Gemma 3 (one forward per image)."""
    rows: list[torch.Tensor] = []
    if spec.family == "qwen3_vl":
        for im in images:
            inputs = prepare_qwen3_vl_inputs_one(processor, im, device)
            pixel_values = inputs["pixel_values"]  # type: ignore[index]
            image_grid_thw = inputs.get("image_grid_thw")  # type: ignore[union-attr]
            if pixel_values is None:
                raise ValueError("Qwen3-VL processor did not return pixel_values")
            vo = model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
            chunks = vo.pooler_output
            if isinstance(chunks, tuple):
                if len(chunks) != 1:
                    vec = torch.cat([c.float().reshape(-1, c.shape[-1]) for c in chunks], dim=0).mean(dim=0)
                else:
                    vec = chunks[0].float().mean(dim=0)
            else:
                vec = chunks.float().mean(dim=0)
            rows.append(vec)
        return torch.stack(rows, dim=0)

    if spec.family == "gemma3":
        for im in images:
            inputs = prepare_gemma3_inputs_one(processor, im, device)
            pixel_values = inputs["pixel_values"]  # type: ignore[index]
            vo = model.get_image_features(pixel_values)
            feats = vo.pooler_output
            if feats is None:
                raise ValueError("Gemma 3 get_image_features returned no pooler_output")
            f = feats.float()
            if f.dim() == 3:
                rows.append(f.mean(dim=1).squeeze(0))
            elif f.dim() == 2:
                rows.append(f.mean(dim=0))
            else:
                rows.append(f.reshape(-1))
        return torch.stack(rows, dim=0)

    raise ValueError(f"Unknown VLM family: {spec.family!r}")


def _normalize_embedding_batch(raw: Any) -> torch.Tensor:
    """Coerce ``encode_image`` outputs (tensor, list, or ndarray) to ``(B, D)`` float32 CPU."""
    if torch.is_tensor(raw):
        t = raw.detach().float().cpu()
    elif isinstance(raw, list):
        rows = []
        for item in raw:
            if torch.is_tensor(item):
                rows.append(item.detach().float().reshape(-1).cpu())
            else:
                rows.append(torch.as_tensor(item, dtype=torch.float32).reshape(-1))
        t = torch.stack(rows, dim=0)
    else:
        t = torch.as_tensor(raw, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


@torch.inference_mode()
def pooled_jina_embedding(
    model: nn.Module,
    spec: EncoderSpec,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Vision-only embedding (B, D) for Jina CLIP v2 or Jina Embeddings v4."""
    rows: list[torch.Tensor] = []
    if spec.family == "jina_clip":
        for im in images:
            raw = model.encode_image([im])
            rows.append(_normalize_embedding_batch(raw).squeeze(0))
        return torch.stack(rows, dim=0)

    if spec.family == "jina_embeddings_v4":
        for im in images:
            raw = model.encode_image(images=[im], task="retrieval")
            rows.append(_normalize_embedding_batch(raw).squeeze(0))
        return torch.stack(rows, dim=0)

    raise ValueError(f"Unknown Jina family: {spec.family!r}")


@torch.inference_mode()
def pooled_siglip2_embedding(
    model: nn.Module,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """SigLIP 2 NaFlex vision vector (B, D) — processor supplies ``spatial_shapes`` etc."""
    batch = processor(images=images, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output.detach().cpu().float()
    return outputs.last_hidden_state[:, 0, :].detach().cpu().float()


def _pool_colbert_tokens(raw: Any) -> torch.Tensor:
    """ColBERT / late-interaction token grid → one vector per item (B, D)."""
    if hasattr(raw, "embeddings"):
        raw = raw.embeddings
    if torch.is_tensor(raw):
        t = raw.detach().float()
    else:
        t = torch.as_tensor(raw, dtype=torch.float32)
    if t.dim() == 3:
        return t.mean(dim=1)
    if t.dim() == 2:
        return t
    raise ValueError(f"Expected ColBERT tensor dim 2 or 3, got shape {tuple(t.shape)}")


@torch.inference_mode()
def pooled_colbert_doc_embedding(
    model: nn.Module,
    spec: EncoderSpec,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Document-page embedding from ColQwen2, ColPali, or Nemotron Colembed."""
    if spec.family in ("colqwen2", "colpali"):
        batch = processor.process_images(images).to(device)
        out = model(**batch)
        return _pool_colbert_tokens(out).cpu()

    if spec.family == "nemotron_colembed":
        out = model.forward_images(images, batch_size=len(images))
        return _pool_colbert_tokens(out).cpu()

    raise ValueError(f"Unknown ColBERT doc family: {spec.family!r}")


@torch.inference_mode()
def pooled_layoutlmv3_embedding(
    model: nn.Module,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
    *,
    boxes_per_image: list[list[list[int]]] | None = None,
    words_per_image: list[list[str]] | None = None,
) -> torch.Tensor:
    """Layout-from-image vector (B, D) via LayoutLMv3 — no OCR.

    Default: single full-page box. Pass ``boxes_per_image`` + ``words_per_image`` for
    detector boxes (one token word per box).
    """
    rows: list[torch.Tensor] = []
    for i, im in enumerate(images):
        if boxes_per_image is not None and words_per_image is not None:
            boxes = boxes_per_image[i]
            words = words_per_image[i]
        else:
            boxes = [[0, 0, 1000, 1000]]
            words = ["page"]
        enc = processor(
            im,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        rows.append(out.last_hidden_state.float().mean(dim=1).squeeze(0).cpu())
    return torch.stack(rows, dim=0)


def load_page_image(path: str | bytes | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        im = path.convert("RGB")
    else:
        im = Image.open(path).convert("RGB")
    return im
