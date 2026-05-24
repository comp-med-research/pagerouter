"""Docling Heron layout detection (RT-DETRv2) for layout-from-image features."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from pagerouter.doclayout import PageDetections, detector_stats_vector

HERON_101_MODEL_ID = "docling-project/docling-layout-heron-101"

HERON_101_CLASS_NAMES: dict[int, str] = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}

NUM_HERON_101_CLASSES = len(HERON_101_CLASS_NAMES)
HERON_101_STATS_DIM = NUM_HERON_101_CLASSES + 6


def load_heron_101(device: torch.device | str = "cuda") -> tuple[Any, Any]:
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    processor = RTDetrImageProcessor.from_pretrained(HERON_101_MODEL_ID)
    model = RTDetrV2ForObjectDetection.from_pretrained(HERON_101_MODEL_ID)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    model.to(dev)
    return model, processor


def _xyxy_to_1000(box: torch.Tensor, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box.tolist()
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


@torch.inference_mode()
def detect_page_heron_101(
    model: Any,
    processor: Any,
    image: Image.Image,
    *,
    device: torch.device,
    threshold: float = 0.6,
    max_boxes: int = 64,
) -> PageDetections:
    """Run Docling Heron-101 on one page; return top boxes in 0–1000 coords."""
    width, height = image.size
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[height, width]], device=device),
        threshold=threshold,
    )
    result = results[0]
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]

    if scores.numel() == 0:
        return PageDetections(
            boxes_1000=[[0, 0, 1000, 1000]],
            class_ids=[0],
            scores=[1.0],
            words=["page"],
        )

    order = torch.argsort(scores, descending=True)[:max_boxes]
    boxes_1000: list[list[int]] = []
    class_ids: list[int] = []
    det_scores: list[float] = []
    words: list[str] = []
    for idx in order:
        i = int(idx.item())
        cid = int(labels[i].item())
        boxes_1000.append(_xyxy_to_1000(boxes[i], width, height))
        class_ids.append(cid)
        det_scores.append(float(scores[i].item()))
        name = HERON_101_CLASS_NAMES.get(cid, f"cls{cid}")
        words.append(name.replace(" ", "_").replace("-", "_"))

    return PageDetections(
        boxes_1000=boxes_1000,
        class_ids=class_ids,
        scores=det_scores,
        words=words,
    )


def detect_pages_heron_101(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    *,
    device: torch.device,
    threshold: float = 0.6,
    max_boxes: int = 64,
) -> list[PageDetections]:
    return [
        detect_page_heron_101(
            model,
            processor,
            im,
            device=device,
            threshold=threshold,
            max_boxes=max_boxes,
        )
        for im in images
    ]


def heron_101_stats_vector(det: PageDetections) -> torch.Tensor:
    return detector_stats_vector(det, NUM_HERON_101_CLASSES)
