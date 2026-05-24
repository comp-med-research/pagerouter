"""Multimodal fusion blocks for router ablations.

Supported **pre-MLP** (no extra parameters, applied in ``mlp_features``):

- ``concat`` — concatenate modality vectors (baseline).
- ``norm_concat`` — L2-normalize each continuous block, then concatenate.
- ``weighted_avg`` — L2-normalize each of two embeddings and take the arithmetic mean (requires equal dim).

**Learned** fusion (applied inside ``train_mlp`` before the classifier MLP; trained end-to-end):

- ``gmu`` — gated blend of two modalities (same dim), similar spirit to multimodal gated units (*e.g.*
  Arevalo et al.; we use Tanh projections + sigmoid gate on ``[u;v]``).
- ``bilinear`` — ``torch.nn.Bilinear`` map ``(u, v) -> out_dim`` (full bilinear interaction; distinct from low-rank
  MLB variants but a standard bilinear ablation).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

FeatureFusion = Literal["concat", "norm_concat", "weighted_avg", "gmu", "bilinear"]

FUSION_LABELS: dict[str, str] = {
    "concat": "concat",
    "norm_concat": "normalized concat",
    "weighted_avg": "weighted average (0.5·L2)",
    "gmu": "GMU",
    "bilinear": "bilinear (nn.Bilinear)",
}


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


class GMUFusion(nn.Module):
    """Two same-dim vectors → one D-dim vector with learnable gates."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin_u = nn.Linear(d, d, bias=True)
        self.lin_v = nn.Linear(d, d, bias=True)
        self.lin_z = nn.Linear(2 * d, d, bias=True)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        hu = torch.tanh(self.lin_u(u))
        hv = torch.tanh(self.lin_v(v))
        z = torch.sigmoid(self.lin_z(torch.cat([u, v], dim=-1)))
        return z * hu + (1.0 - z) * hv


class BilinearFusion(nn.Module):
    """Bilinear map (u, v) → R^{out_dim}."""

    def __init__(self, d: int, out_dim: int) -> None:
        super().__init__()
        self.b = nn.Bilinear(d, d, out_dim, bias=True)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.b(u, v)


class FusionMLPRouter(nn.Module):
    """Optional bilinear/GMU on (image, layout) blocks, then concat metadata, then 2-layer MLP."""

    def __init__(
        self,
        *,
        fusion: FeatureFusion,
        d_img: int,
        d_layout: int,
        d_meta: int,
        num_classes: int,
        hidden: int = 256,
        dropout: float = 0.1,
        bilinear_out_dim: int = 256,
    ) -> None:
        super().__init__()
        if d_img != d_layout:
            raise ValueError(f"GMU/bilinear require d_img==d_layout, got {d_img} vs {d_layout}")
        d = d_img
        self.fusion = fusion
        self.d_img = d_img
        self.d_layout = d_layout
        self.d_meta = d_meta

        if fusion == "gmu":
            self.fuse = GMUFusion(d)
            fuse_out = d
        elif fusion == "bilinear":
            self.fuse = BilinearFusion(d, bilinear_out_dim)
            fuse_out = bilinear_out_dim
        else:
            raise ValueError(f"FusionMLPRouter expects fusion in ('gmu','bilinear'), got {fusion!r}")

        in_dim = fuse_out + d_meta
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x[:, : self.d_img]
        v = x[:, self.d_img : self.d_img + self.d_layout]
        meta = x[:, self.d_img + self.d_layout :]
        z = self.fuse(u, v)
        if self.d_meta > 0:
            z = torch.cat([z, meta], dim=-1)
        return self.head(z)
