"""Typed configuration models for processing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MethodT = Literal["sobel", "laplacian", "both"]
ThresholdT = int | Literal["auto"]
MAX_UINT8 = 255

DEFAULT_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


def validate_odd_kernel(value: int, name: str) -> None:
    """Validate odd positive kernel size."""
    if value <= 0 or value % 2 == 0:
        msg = f"{name} must be odd and positive, got: {value}"
        raise ValueError(msg)


def validate_blur_kernel(value: int) -> None:
    """Validate blur kernel size (0 or odd positive)."""
    if value == 0:
        return
    validate_odd_kernel(value, "blur_kernel")


def normalize_extensions(extensions: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize extensions to lowercase dot-prefixed values."""
    normalized = []
    for extension in extensions:
        ext = extension.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    if not normalized:
        msg = "extensions cannot be empty"
        raise ValueError(msg)
    return tuple(sorted(set(normalized)))


@dataclass(slots=True, kw_only=True)
class ProcessingConfig:
    """Runtime processing configuration."""

    method: MethodT = "both"
    blur_kernel: int = 3
    sobel_kernel: int = 3
    laplacian_kernel: int = 3
    threshold: ThresholdT = "auto"
    recursive: bool = True
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    draw_contours: bool = True
    min_contour_area: float = 0.0
    max_images: int | None = None
    overwrite: bool = True

    def __post_init__(self) -> None:
        """Validate processing configuration values."""
        if self.method not in {"sobel", "laplacian", "both"}:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)
        validate_blur_kernel(self.blur_kernel)
        validate_odd_kernel(self.sobel_kernel, "sobel_kernel")
        validate_odd_kernel(self.laplacian_kernel, "laplacian_kernel")
        if self.threshold != "auto":
            if not isinstance(self.threshold, int):
                msg = "threshold must be int or 'auto'"
                raise ValueError(msg)
            if self.threshold < 0 or self.threshold > MAX_UINT8:
                msg = "threshold must be in [0, 255]"
                raise ValueError(msg)
        if self.min_contour_area < 0:
            msg = "min_contour_area must be non-negative"
            raise ValueError(msg)
        if self.max_images is not None and self.max_images <= 0:
            msg = "max_images must be positive"
            raise ValueError(msg)
        self.extensions = normalize_extensions(self.extensions)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable config mapping."""
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class ImageResult:
    """Aggregated result of processing a single image."""

    source_path: Path
    sobel_path: Path | None
    laplacian_path: Path | None
    binary_path: Path
    contours_path: Path | None
    width: int
    height: int
    edge_pixel_ratio: float
    contour_count: int
    largest_contour_area: float
    processing_ms: float
    method: MethodT

    def __post_init__(self) -> None:
        """Validate result values."""
        if self.width <= 0 or self.height <= 0:
            msg = "Image dimensions must be positive"
            raise ValueError(msg)
        if self.edge_pixel_ratio < 0 or self.edge_pixel_ratio > 1:
            msg = "edge_pixel_ratio must be in [0, 1]"
            raise ValueError(msg)
        if self.contour_count < 0:
            msg = "contour_count must be non-negative"
            raise ValueError(msg)
        if self.largest_contour_area < 0:
            msg = "largest_contour_area must be non-negative"
            raise ValueError(msg)
        if self.processing_ms < 0:
            msg = "processing_ms must be non-negative"
            raise ValueError(msg)

    def to_record(self) -> dict[str, object]:
        """Return row-like dict for CSV/JSON reporting."""
        return {
            "source_path": str(self.source_path),
            "sobel_path": str(self.sobel_path) if self.sobel_path else None,
            "laplacian_path": str(self.laplacian_path) if self.laplacian_path else None,
            "binary_path": str(self.binary_path),
            "contours_path": str(self.contours_path) if self.contours_path else None,
            "width": self.width,
            "height": self.height,
            "method": self.method,
            "edge_pixel_ratio": self.edge_pixel_ratio,
            "contour_count": self.contour_count,
            "largest_contour_area": self.largest_contour_area,
            "processing_ms": self.processing_ms,
        }
