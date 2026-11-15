"""Page image dataset for VLM router training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from pagerouter.embed_backends import load_page_image
from pagerouter.mlp_paths import resolve_page_image


class PageRouterDataset(Dataset):
    def __init__(
        self,
        page_ids: list[str],
        labels: torch.Tensor,
        *,
        image_root: Path,
    ) -> None:
        self.page_ids = page_ids
        self.labels = labels
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.page_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        pid = self.page_ids[idx]
        path = resolve_page_image(pid, self.image_root)
        image = load_page_image(path)
        return {
            "page_id": pid,
            "image": image,
            "label": int(self.labels[idx].item()),
        }


def collate_pages(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "page_ids": [str(b["page_id"]) for b in batch],
        "images": [b["image"] for b in batch],  # type: ignore[misc]
        "labels": torch.tensor([int(b["label"]) for b in batch], dtype=torch.long),
    }


def page_ids_and_labels(
    matrix: pd.DataFrame,
    label_series: pd.Series,
) -> tuple[list[str], torch.Tensor]:
    kept = [str(p) for p in matrix.index if str(p) in label_series.index]
    if not kept:
        raise ValueError("No overlap between score matrix and label series.")
    y = torch.tensor([int(label_series.loc[pid]) for pid in kept], dtype=torch.long)
    return kept, y
