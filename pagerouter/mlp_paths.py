"""Resolve `page_id` to an image file under ``data/page_images``."""

from __future__ import annotations

from pathlib import Path

from pagerouter.load import DATA_DIR

DEFAULT_IMAGE_ROOT = DATA_DIR / "page_images"

_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")


def resolve_page_image(page_id: str, image_root: Path | None = None) -> Path:
    root = Path(image_root) if image_root is not None else DEFAULT_IMAGE_ROOT
    # page_id may already include suffix as stored in CSV
    direct = root / str(page_id)
    if direct.is_file():
        return direct
    for suf in _SUFFIXES:
        p = root / f"{page_id}{suf}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No image for page_id={page_id!r} under {root}")
