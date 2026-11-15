"""Load backbones and router classification heads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from pagerouter.embed_backends import (
    _pool_colbert_tokens,
    apply_transformers_compat_patches,
    load_encoder,
    patch_internvl_post_init,
    prepare_gemma3_inputs_one,
    prepare_qwen3_vl_inputs_one,
)
from pagerouter.vlm_finetune.registry import VlmRouterSpec


class RouterHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PageRouter(nn.Module):
    """Image → logits router wrapping a HF backbone + MLP head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: RouterHead,
        spec: VlmRouterSpec,
        processor: Any,
        *,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.spec = spec
        self.processor = processor
        self.embed_dim = embed_dim

    def encode_images(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        family = self.spec.family
        if family == "doc_encoder":
            return self._encode_layoutlm(images, device)
        if family == "udop":
            return self._encode_udop(images, device)
        if family == "colpali":
            return self._encode_colpali(images, device)
        return self._encode_vlm(images, device)

    def _encode_layoutlm(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for im in images:
            enc = self.processor(
                im,
                text=["page"],
                boxes=[[0, 0, 1000, 1000]],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.backbone(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            rows.append(hidden.float().mean(dim=1).squeeze(0))
        return torch.stack(rows, dim=0)

    def _encode_udop(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for im in images:
            enc = self.processor(
                im,
                text=["page"],
                boxes=[[0, 0, 1000, 1000]],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            enc_out = self.backbone.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                bbox=enc["bbox"],
                pixel_values=enc.get("pixel_values"),
            )
            h = enc_out.last_hidden_state
            rows.append(h.float().mean(dim=1).squeeze(0))
        return torch.stack(rows, dim=0)

    def _encode_colpali(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        batch = self.processor.process_images(images).to(device)
        out = self.backbone(**batch)
        return _pool_colbert_tokens(out).to(device=device, dtype=torch.float32)

    def _encode_vlm(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        if self.spec.key == "internvl3-8b":
            rows: list[torch.Tensor] = []
            dtype = next(self.backbone.parameters()).dtype
            for im in images:
                pv = self.processor.image_processor(images=im, return_tensors="pt")[
                    "pixel_values"
                ].to(device=device, dtype=dtype)
                feats = self.backbone.extract_feature(pv)
                rows.append(feats.float().mean(dim=1).squeeze(0))
            return torch.stack(rows, dim=0)

        rows: list[torch.Tensor] = []
        if self.spec.key.startswith("qwen3-vl"):
            for im in images:
                inputs = prepare_qwen3_vl_inputs_one(self.processor, im, device)
                pixel_values = inputs["pixel_values"]
                image_grid_thw = inputs.get("image_grid_thw")
                vo = self.backbone.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
                chunks = vo.pooler_output
                if isinstance(chunks, tuple):
                    if len(chunks) != 1:
                        vec = torch.cat(
                            [c.float().reshape(-1, c.shape[-1]) for c in chunks], dim=0
                        ).mean(dim=0)
                    else:
                        vec = chunks[0].float().mean(dim=0)
                else:
                    vec = chunks.float().mean(dim=0)
                rows.append(vec)
            return torch.stack(rows, dim=0)

        if self.spec.key.startswith("gemma"):
            for im in images:
                inputs = prepare_gemma3_inputs_one(self.processor, im, device)
                pixel_values = inputs["pixel_values"]
                vo = self.backbone.get_image_features(pixel_values)
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

        raise ValueError(f"Unsupported VLM key for training: {self.spec.key}")

    def forward(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        feats = self.encode_images(images, device)
        return self.head(feats)


def _probe_embed_dim(
    router: PageRouter,
    device: torch.device,
    *,
    image_size: tuple[int, int] = (384, 384),
) -> int:
    dummy = Image.new("RGB", image_size, color=(255, 255, 255))
    with torch.no_grad():
        vec = router.encode_images([dummy], device)
    return int(vec.shape[-1])


def build_page_router(
    spec: VlmRouterSpec,
    num_classes: int,
    device: torch.device,
) -> tuple[PageRouter, RouterHead, nn.Module, Any, int]:
    apply_transformers_compat_patches()

    if spec.family == "udop":
        from transformers import UdopProcessor, UdopModel

        processor = UdopProcessor.from_pretrained(spec.hf_model_id, apply_ocr=False)
        backbone = UdopModel.from_pretrained(spec.hf_model_id)
        backbone = backbone.to(device)
    elif spec.family == "colpali":
        from colpali_engine.models import ColPali, ColPaliProcessor

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        backbone = ColPali.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            device_map={"" : device},
        )
        processor = ColPaliProcessor.from_pretrained(spec.hf_model_id)
    elif spec.key == "internvl3-8b":
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        patch_internvl_post_init(spec.hf_model_id)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        processor = SimpleNamespace(
            tokenizer=AutoProcessor.from_pretrained(spec.hf_model_id, trust_remote_code=True),
            image_processor=AutoImageProcessor.from_pretrained(
                spec.hf_model_id, trust_remote_code=True
            ),
        )
        backbone = AutoModel.from_pretrained(
            spec.hf_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"" : device},
        )
    else:
        from pagerouter.embed_backends import ENCODER_REGISTRY

        enc_spec = ENCODER_REGISTRY[spec.key]
        backbone, processor = load_encoder(enc_spec, device)
        backbone = backbone.to(device)

    head = RouterHead(512, num_classes)  # placeholder in_dim; resized below
    router = PageRouter(backbone, head, spec, processor, embed_dim=512)
    dim = _probe_embed_dim(router, device)
    head = RouterHead(dim, num_classes).to(device)
    router = PageRouter(backbone, head, spec, processor, embed_dim=dim)
    return router, head, backbone, processor, dim
