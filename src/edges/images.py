"""Filesystem and image IO helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageReadError(RuntimeError):
    """Raised when an image cannot be read from disk."""


def iter_image_paths(input_dir: Path, extensions: Iterable[str], *, recursive: bool) -> Iterator[Path]:
    """Yield image paths with selected extensions."""
    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)
    allowed = {ext.lower() for ext in extensions}
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def read_image(path: Path) -> NDArray[np.uint8]:
    """Read image in BGR format."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {path}"
        raise ImageReadError(msg)
    return cast(NDArray[np.uint8], image)


def save_grayscale(path: Path, image: NDArray[np.uint8]) -> None:
    """Save grayscale image as uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        msg = f"Expected uint8 grayscale image, got {image.dtype}"
        raise ValueError(msg)
    if not cv2.imwrite(str(path), image):
        msg = f"Could not write grayscale image to: {path}"
        raise OSError(msg)


def save_color(path: Path, image: NDArray[np.uint8]) -> None:
    """Save color image as uint8 BGR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        msg = f"Expected uint8 color image, got {image.dtype}"
        raise ValueError(msg)
    if not cv2.imwrite(str(path), image):
        msg = f"Could not write color image to: {path}"
        raise OSError(msg)


def make_output_path(input_dir: Path, output_dir: Path, image_path: Path, suffix: str) -> Path:
    """Build output path preserving relative directory structure."""
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{image_path.stem}_{suffix}.png"
