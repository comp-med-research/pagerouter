"""Unit tests for VLM LoRA paradigm targeting helpers."""

from __future__ import annotations

from pagerouter.vlm_finetune.lora import (
    _param_region,
    _targets_for_paradigm,
)
from pagerouter.vlm_finetune.registry import VLM_ROUTER_MODELS


def test_colpali_connector_targets_exclude_vision():
    spec = VLM_ROUTER_MODELS["colpali-v1.3"]
    connector = _targets_for_paradigm(spec, "lora_connector")
    partial = _targets_for_paradigm(spec, "lora_partial_vit")
    assert "out_proj" in partial
    assert "fc1" in partial
    assert "out_proj" not in connector
    assert "fc1" not in connector
    assert "custom_text_proj" in connector
    assert partial != connector


def test_udop_and_colpali_param_regions():
    assert _param_region("encoder.block.12.layer.0.SelfAttention.q.weight") == "vision"
    assert _param_region("decoder.block.3.layer.0.SelfAttention.q.weight") == "lm"
    assert _param_region("model.model.multi_modal_projector.linear.weight") == "connector"
    assert _param_region("custom_text_proj.base_layer.weight") == "connector"


def test_qwen_targets_still_include_vision_for_connector():
    spec = VLM_ROUTER_MODELS["qwen3-vl-8b-instruct"]
    targets = _targets_for_paradigm(spec, "lora_connector")
    assert "qkv" in targets
    assert "q_proj" in targets
