"""DocLayout-YOLO detection and layout feature vectors (no OCR)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

DOCLAYOUT_PYTHON = os.environ.get(
    "DOCLAYOUT_PYTHON",
    "/home/halimatmac/projects/rankshift/.venv-monkey/bin/python3",
)
DEFAULT_YOLO_WEIGHTS = Path(
    os.environ.get(
        "DOCLAYOUT_YOLO_WEIGHTS",
        "/home/halimatmac/projects/rankshift/models/models--echo840--MonkeyOCR-pro-3B/"
        "snapshots/c42d9c37e7e546a30e619ad109daa0d17373aed4/Structure/"
        "doclayout_yolo_docstructbench_imgsz1280_2501.pt",
    )
)

DOCLAYOUT_CLASS_NAMES: dict[int, str] = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

NUM_DOCLAYOUT_CLASSES = len(DOCLAYOUT_CLASS_NAMES)
# Class histogram (10) + log1p(n_boxes) + area mean/std/max + x/y center std.
DOCLAYOUT_STATS_DIM = NUM_DOCLAYOUT_CLASSES + 6


@dataclass
class PageDetections:
    boxes_1000: list[list[int]]
    class_ids: list[int]
    scores: list[float]
    words: list[str]


def load_yolo(weights: Path | str = DEFAULT_YOLO_WEIGHTS, device: str = "cuda:0") -> Any:
    from doclayout_yolo import YOLOv10

    try:
        from doclayout_yolo.nn.tasks import YOLOv10DetectionModel

        torch.serialization.add_safe_globals([YOLOv10DetectionModel])
    except Exception:
        pass

    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"DocLayout-YOLO weights not found: {path}")
    return YOLOv10(str(path)), device


def _xyxy_to_1000(box: torch.Tensor, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box.tolist()
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def detect_page(
    model: Any,
    image: Image.Image,
    *,
    device: str = "cuda:0",
    conf: float = 0.2,
    imgsz: int = 1024,
    max_boxes: int = 64,
) -> PageDetections:
    """Run DocLayout-YOLO on one page; return top boxes in 0–1000 coords."""
    width, height = image.size
    results = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return PageDetections(
            boxes_1000=[[0, 0, 1000, 1000]],
            class_ids=[0],
            scores=[1.0],
            words=["page"],
        )

    order = torch.argsort(r.boxes.conf, descending=True)
    order = order[:max_boxes]
    boxes_1000: list[list[int]] = []
    class_ids: list[int] = []
    scores: list[float] = []
    words: list[str] = []
    for idx in order:
        i = int(idx.item())
        cid = int(r.boxes.cls[i].item())
        boxes_1000.append(_xyxy_to_1000(r.boxes.xyxy[i], width, height))
        class_ids.append(cid)
        scores.append(float(r.boxes.conf[i].item()))
        words.append(DOCLAYOUT_CLASS_NAMES.get(cid, f"cls{cid}").replace(" ", "_"))

    return PageDetections(boxes_1000=boxes_1000, class_ids=class_ids, scores=scores, words=words)


def detect_pages(
    model: Any,
    images: list[Image.Image],
    *,
    device: str = "cuda:0",
    conf: float = 0.2,
    imgsz: int = 1024,
    max_boxes: int = 64,
) -> list[PageDetections]:
    return [
        detect_page(model, im, device=device, conf=conf, imgsz=imgsz, max_boxes=max_boxes)
        for im in images
    ]


def doclayout_stats_vector(det: PageDetections) -> torch.Tensor:
    """Fixed-size layout summary from detector outputs (no OCR)."""
    return detector_stats_vector(det, NUM_DOCLAYOUT_CLASSES)


def detector_stats_vector(det: PageDetections, num_classes: int) -> torch.Tensor:
    """Class histogram + box geometry stats (shared by YOLO / RT-DETR layout detectors)."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for cid in det.class_ids:
        if 0 <= cid < num_classes:
            counts[cid] += 1.0
    n = max(len(det.boxes_1000), 1)
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

    area_t = torch.tensor(areas, dtype=torch.float32)
    cx_t = torch.tensor(cx, dtype=torch.float32)
    cy_t = torch.tensor(cy, dtype=torch.float32)
    extras = torch.tensor(
        [
            float(torch.log1p(torch.tensor(float(n)))),
            float(area_t.mean()) if len(areas) else 0.0,
            float(area_t.std(unbiased=False)) if len(areas) else 0.0,
            float(area_t.max()) if len(areas) else 0.0,
            float(cx_t.std(unbiased=False)) if len(cx) else 0.0,
            float(cy_t.std(unbiased=False)) if len(cy) else 0.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([counts, extras], dim=0)


def detections_to_blob(page_ids: list[str], dets: list[PageDetections]) -> dict[str, Any]:
    return {
        "page_ids": page_ids,
        "detections": [
            {
                "boxes_1000": d.boxes_1000,
                "class_ids": d.class_ids,
                "scores": d.scores,
                "words": d.words,
            }
            for d in dets
        ],
    }


def blob_to_detections(blob: dict[str, Any]) -> tuple[list[str], list[PageDetections]]:
    page_ids = [str(p) for p in blob["page_ids"]]
    dets: list[PageDetections] = []
    for raw in blob["detections"]:
        dets.append(
            PageDetections(
                boxes_1000=[list(map(int, b)) for b in raw["boxes_1000"]],
                class_ids=[int(c) for c in raw["class_ids"]],
                scores=[float(s) for s in raw["scores"]],
                words=[str(w) for w in raw["words"]],
            )
        )
    return page_ids, dets
