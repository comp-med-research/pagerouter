"""Shared detection structs and layout stat vectors (Docling / layout encoders)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PageDetections:
    boxes_1000: list[list[int]]
    class_ids: list[int]
    scores: list[float]
    words: list[str]


def detector_stats_vector(det: PageDetections, num_classes: int) -> torch.Tensor:
    """Class histogram + box geometry stats."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for cid in det.class_ids:
        if 0 <= cid < num_classes:
            counts[cid] += 1.0
    counts = counts / counts.sum() if counts.sum() > 0 else counts

    areas: list[float] = []
    cx: list[float] = []
    cy: list[float] = []
    for box in det.boxes_1000:
        x0, y0, x1, y1 = box
        w = max(0, x1 - x0) / 1000.0
        h = max(0, y1 - y0) / 1000.0
        areas.append(w * h)
        cx.append((x0 + x1) / 2000.0)
        cy.append((y0 + y1) / 2000.0)

    n = max(len(det.boxes_1000), 1)
    area_t = torch.tensor(areas, dtype=torch.float32)
    cx_t = torch.tensor(cx, dtype=torch.float32)
    cy_t = torch.tensor(cy, dtype=torch.float32)
    extras = torch.tensor(
        [
            float(torch.log1p(torch.tensor(float(n)))),
            float(area_t.mean()) if areas else 0.0,
            float(area_t.std(unbiased=False)) if areas else 0.0,
            float(area_t.max()) if areas else 0.0,
            float(cx_t.std(unbiased=False)) if len(cx) else 0.0,
            float(cy_t.std(unbiased=False)) if len(cy) else 0.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([counts, extras], dim=0)