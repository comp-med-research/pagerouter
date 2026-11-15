"""Apply PEFT configs per model family and paradigm."""

from __future__ import annotations

import re
from typing import Literal

import torch
import torch.nn as nn

from pagerouter.vlm_finetune.registry import RouterParadigm, VlmRouterSpec

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError("peft is required for VLM router fine-tuning") from e

ParamRegion = Literal["lm", "vision", "connector", "patch", "other"]

PARTIAL_VIT_TOP_BLOCKS = 4

# Per-model LoRA module suffixes (PEFT matches on leaf module names).
_LORA_PROFILES: dict[str, dict[str, list[str]]] = {
    "qwen3-vl-8b-instruct": {
        "lm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "vision": ["qkv", "proj", "linear_fc1", "linear_fc2"],
    },
    "gemma-3-4b-it": {
        "lm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "vision": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    },
    "internvl3-8b": {
        "lm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "vision": ["qkv", "proj", "fc1", "fc2"],
    },
    "colpali-v1.3": {
        "lm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "vision": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "connector": ["linear", "custom_text_proj"],
    },
}


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_head(head: nn.Module) -> None:
    for p in head.parameters():
        p.requires_grad = True


def merge_colpali_pretrained_lora(model: nn.Module) -> None:
    """Fold ColPali checkpoint LoRA into base weights so router paradigms start clean."""
    for module in model.modules():
        if not hasattr(module, "base_layer") or not hasattr(module, "lora_A"):
            continue
        lora_a = getattr(module.lora_A, "default", None)
        lora_b = getattr(module.lora_B, "default", None)
        if lora_a is None or lora_b is None:
            continue
        with torch.no_grad():
            delta = lora_b.weight @ lora_a.weight
            scale = module.scaling.get("default", 1.0) if hasattr(module, "scaling") else 1.0
            base_w = module.base_layer.weight
            if delta.device != base_w.device:
                delta = delta.to(base_w.device)
            if delta.dtype != base_w.dtype:
                delta = delta.to(dtype=base_w.dtype)
            base_w.add_(delta * scale)
            lora_a.weight.zero_()
            lora_b.weight.zero_()



def _lora_target_modules(family: str) -> list[str]:
    if family == "doc_encoder":
        return ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    if family == "udop":
        return ["q", "k", "v", "o", "wi", "wo"]
    if family == "colpali":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _profile_targets(spec: VlmRouterSpec) -> tuple[list[str], list[str], list[str]]:
    profile = _LORA_PROFILES.get(spec.key)
    if profile is None:
        lm = _lora_target_modules(spec.family if spec.family != "vlm" else "vlm")
        return lm, [], []
    return profile["lm"], profile.get("vision", []), profile.get("connector", [])


def _param_region(name: str) -> ParamRegion:
    low = name.lower()
    if re.search(r"(?:^|\.)visual\.blocks|vision_tower|vision_model\.encoder", low):
        return "vision"
    if re.search(r"(merger|multi_modal_projector|mm_projector|(?:^|\.)mlp1\.|custom_text_proj)", low):
        return "connector"
    if re.search(r"(patch_embed|embed_patches|embeddings\.patch_embedding)", low):
        return "patch"
    if re.search(r"language_model\.(?:layers|model\.layers)|text_model\.layers|(?:^|\.)decoder\.block", low):
        return "lm"
    if re.search(r"(?:^|\.)encoder\.block", low):
        return "vision"
    return "other"


def _vision_block_index(name: str) -> int | None:
    for pat in (
        r"visual\.blocks\.(\d+)",
        r"vision_tower\.encoder\.layers\.(\d+)",
        r"vision_model\.encoder\.layers\.(\d+)",
        r"encoder\.block\.(\d+)",
    ):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def _max_vision_block(model: nn.Module) -> int:
    mx = -1
    for n, _ in model.named_parameters():
        idx = _vision_block_index(n)
        if idx is not None:
            mx = max(mx, idx)
    return mx


def _targets_for_paradigm(spec: VlmRouterSpec, paradigm: RouterParadigm) -> list[str]:
    if spec.family == "doc_encoder":
        return _lora_target_modules(spec.family)

    if spec.family == "udop":
        return _lora_target_modules("udop")

    lm_targets, vision_targets, connector_targets = _profile_targets(spec)
    if spec.family == "colpali" and spec.key not in _LORA_PROFILES:
        lm_targets = _lora_target_modules("colpali")
        vision_targets = []
        connector_targets = []

    if spec.family == "colpali":
        if paradigm == "lora_connector":
            return sorted(set(lm_targets + connector_targets))
        if paradigm == "lora_partial_vit":
            return sorted(set(lm_targets + connector_targets + vision_targets))
        return sorted(set(lm_targets + connector_targets + vision_targets))

    # Vision suffixes are included for connector paradigms too (e.g. Qwen merger
    # linears share names with ViT blocks); block-level vision LoRA is masked off.
    return sorted(set(lm_targets + vision_targets))


def _apply_udop_paradigm_masks(model: nn.Module, paradigm: RouterParadigm) -> None:
    """UDoP routing uses encoder() only — never train decoder or patch embeddings."""
    n_layers = _max_vision_block(model) + 1
    top_cutoff = max(0, n_layers - PARTIAL_VIT_TOP_BLOCKS)
    mid_cutoff = n_layers // 2

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "decoder." in n or re.search(r"(?:^|\.)patch_embed|embed_patches", n):
            p.requires_grad = False
            continue
        idx = _vision_block_index(n)
        if idx is None:
            continue
        if paradigm == "lora_connector" and idx < mid_cutoff:
            p.requires_grad = False
        elif paradigm == "lora_partial_vit" and idx < top_cutoff:
            p.requires_grad = False


def _apply_paradigm_masks(model: nn.Module, spec: VlmRouterSpec, paradigm: RouterParadigm) -> None:
    if paradigm == "freeze":
        return

    if spec.family == "udop":
        _apply_udop_paradigm_masks(model, paradigm)
        return

    max_v = _max_vision_block(model)
    top_cutoff = max(0, max_v - PARTIAL_VIT_TOP_BLOCKS + 1)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        region = _param_region(n)

        if paradigm == "lora_connector":
            if region in {"vision", "patch"}:
                p.requires_grad = False
            continue

        if paradigm == "lora_partial_vit":
            if region == "patch":
                p.requires_grad = False
            elif region == "vision":
                idx = _vision_block_index(n)
                if idx is not None and idx < top_cutoff:
                    p.requires_grad = False
            continue

        if paradigm == "lora_full" and region == "patch":
            p.requires_grad = False


def _unfreeze_colpali_trainable_lora(model: nn.Module) -> None:
    """Re-enable merged checkpoint LoRA slots for router fine-tuning."""
    for n, p in model.named_parameters():
        if re.search(r"\.lora_[ab]\.", n, flags=re.IGNORECASE):
            p.requires_grad = True


def _unfreeze_colpali_connector(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if _param_region(n) == "connector":
            p.requires_grad = True


def _apply_colpali_paradigm(
    backbone: nn.Module,
    head: nn.Module,
    spec: VlmRouterSpec,
    paradigm: RouterParadigm,
) -> nn.Module:
    """ColPali ships with LM LoRA baked in; merge it, then train adapters per paradigm."""
    merge_colpali_pretrained_lora(backbone)
    _freeze_all(backbone)
    _unfreeze_head(head)

    if paradigm == "freeze":
        return backbone

    rank = 8 if paradigm == "lora_full" else spec.lora_r
    alpha = spec.lora_alpha if paradigm != "lora_full" else 16

    _unfreeze_colpali_trainable_lora(backbone)
    _unfreeze_colpali_connector(backbone)

    _, vision_targets, _ = _profile_targets(spec)
    if paradigm in {"lora_partial_vit", "lora_full"}:
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=vision_targets,
            bias="none",
        )
        wrapped = get_peft_model(backbone, cfg)
        _apply_paradigm_masks(wrapped, spec, paradigm)
        return wrapped

    return backbone

def apply_paradigm(
    backbone: nn.Module,
    head: nn.Module,
    spec: VlmRouterSpec,
    paradigm: RouterParadigm,
) -> nn.Module:
    """Return trainable module (backbone may be wrapped with PEFT). Head always trainable."""
    if spec.family == "colpali":
        return _apply_colpali_paradigm(backbone, head, spec, paradigm)

    _freeze_all(backbone)
    _unfreeze_head(head)

    if paradigm == "freeze":
        return backbone

    rank = 8 if paradigm == "lora_full" else spec.lora_r
    alpha = spec.lora_alpha if paradigm != "lora_full" else 16

    if spec.family == "doc_encoder":
        n_layers = getattr(getattr(backbone, "config", None), "num_hidden_layers", 12)
        if paradigm == "lora_connector":
            layers = list(range(max(0, n_layers // 2), n_layers))
        elif paradigm == "lora_partial_vit":
            layers = list(range(max(0, n_layers - PARTIAL_VIT_TOP_BLOCKS), n_layers))
        else:
            layers = None
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=_lora_target_modules(spec.family),
            layers_to_transform=layers,
            bias="none",
        )
        return get_peft_model(backbone, cfg)

    targets = _targets_for_paradigm(spec, paradigm)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        bias="none",
    )
    wrapped = get_peft_model(backbone, cfg)
    _apply_paradigm_masks(wrapped, spec, paradigm)
    return wrapped


def trainable_param_count(model: nn.Module, head: nn.Module) -> tuple[int, int]:
    trainable = sum(
        p.numel() for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad
    )
    total = sum(p.numel() for p in list(model.parameters()) + list(head.parameters()))
    return trainable, total
