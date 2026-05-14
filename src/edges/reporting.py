"""Report generation utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from edges.config import ImageResult, ProcessingConfig


def write_summary_csv(results: Sequence[ImageResult], path: Path) -> None:
    """Write processing results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "sobel_path",
        "laplacian_path",
        "binary_path",
        "contours_path",
        "width",
        "height",
        "method",
        "edge_pixel_ratio",
        "contour_count",
        "largest_contour_area",
        "processing_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_record())


def write_summary_json(results: Sequence[ImageResult], path: Path) -> None:
    """Write processing results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [result.to_record() for result in results]
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)


def write_config_json(config: ProcessingConfig, path: Path) -> None:
    """Write runtime configuration to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_safe(config.to_dict()), output_file, ensure_ascii=False, indent=2)


def _json_safe(value: Any) -> Any:
    """Convert potentially non-JSON values to JSON-friendly ones."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value
