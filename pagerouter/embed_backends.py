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
    from transformers import Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover — optional until transformers version catches up
    Gemma3ForConditionalGeneration = None  # type: ignore[misc, assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[misc, assignment]

EncoderFamily = Literal["dinov2", "clip", "siglip", "dit", "qwen3_vl", "gemma3"]


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
    "dit-base": EncoderSpec("dit-base", "microsoft/dit-base", "dit"),
    "dit-large": EncoderSpec("dit-large", "microsoft/dit-large", "dit"),
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
}

# Excluded from ``--encoder all`` (use ``all_with_vlm`` or name keys explicitly).
HEAVY_VLM_KEYS: frozenset[str] = frozenset({"qwen3-vl-8b-instruct", "gemma-3-4b-it"})


def list_encoder_keys(*, include_heavy_vlm: bool = True) -> list[str]:
    keys = sorted(ENCODER_REGISTRY.keys())
    if not include_heavy_vlm:
        keys = [k for k in keys if k not in HEAVY_VLM_KEYS]
    return keys


def _vlm_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_encoder(spec: EncoderSpec, device: torch.device) -> tuple[nn.Module, Any]:
    """Load frozen encoder + processor (``AutoImageProcessor`` or ``AutoProcessor`` for VLMs)."""
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


def load_page_image(path: str | bytes | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        im = path.convert("RGB")
    else:
        im = Image.open(path).convert("RGB")
    return im
