"""Registry of models and fine-tuning paradigms for VLM routers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouterParadigm = Literal["freeze", "lora_connector", "lora_partial_vit", "lora_full"]

PARADIGMS: tuple[RouterParadigm, ...] = (
    "freeze",
    "lora_connector",
    "lora_partial_vit",
    "lora_full",
)

PARADIGM_LABELS: dict[RouterParadigm, str] = {
    "freeze": "frozen backbone + trainable cls head",
    "lora_connector": "LoRA on connector/LM (vision frozen)",
    "lora_partial_vit": "LoRA on top vision + connector/LM",
    "lora_full": "LoRA end-to-end (rank 8)",
}


@dataclass(frozen=True)
class VlmRouterSpec:
    key: str
    hf_model_id: str
    family: Literal["doc_encoder", "vlm", "colpali", "udop"]
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    epochs: int = 5
    lora_r: int = 16
    lora_alpha: int = 32
    optional: bool = False


VLM_ROUTER_MODELS: dict[str, VlmRouterSpec] = {
    "layoutlmv3-base": VlmRouterSpec(
        "layoutlmv3-base",
        "microsoft/layoutlmv3-base",
        "doc_encoder",
        batch_size=8,
        grad_accum=2,
        lr=3e-5,
        epochs=8,
    ),
    "udop-large": VlmRouterSpec(
        "udop-large",
        "microsoft/udop-large",
        "udop",
        batch_size=2,
        grad_accum=4,
        lr=2e-5,
        epochs=6,
    ),
    "qwen3-vl-8b-instruct": VlmRouterSpec(
        "qwen3-vl-8b-instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "vlm",
        batch_size=1,
        grad_accum=8,
        lr=1e-4,
        epochs=4,
    ),
    "internvl3-8b": VlmRouterSpec(
        "internvl3-8b",
        "OpenGVLab/InternVL3-8B",
        "vlm",
        batch_size=1,
        grad_accum=8,
        lr=1e-4,
        epochs=4,
        optional=True,
    ),
    "gemma-3-4b-it": VlmRouterSpec(
        "gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "vlm",
        batch_size=1,
        grad_accum=8,
        lr=1e-4,
        epochs=4,
    ),
    "colpali-v1.3": VlmRouterSpec(
        "colpali-v1.3",
        "vidore/colpali-v1.3",
        "colpali",
        batch_size=2,
        grad_accum=4,
        lr=2e-4,
        epochs=5,
    ),
}

DEFAULT_QUEUE: tuple[str, ...] = (
    "layoutlmv3-base",
    "udop-large",
    "qwen3-vl-8b-instruct",
    "internvl3-8b",
    "gemma-3-4b-it",
    "colpali-v1.3",
)
