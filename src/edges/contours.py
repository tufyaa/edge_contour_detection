"""Contour extraction and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True, kw_only=True)
class ContourStats:
    """Aggregated contour metrics."""

    count: int = 0
    largest_area: float = 0.0
    mean_area: float = 0.0
    mean_perimeter: float = 0.0


def find_contours(
    binary: NDArray[np.uint8], mode: int = cv2.RETR_EXTERNAL, min_area: float = 0.0
) -> list[NDArray[np.int32]]:
    """Find contours in binary image."""
    if binary.ndim != 2:
        msg = f"Binary image must be 2D, got shape {binary.shape}"
        raise ValueError(msg)
    contour_result = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
    typed_contours = [cast(NDArray[np.int32], contour) for contour in contours]
    if min_area <= 0:
        return typed_contours
    return [contour for contour in typed_contours if cv2.contourArea(contour) >= min_area]


def draw_contours(
    image: NDArray[np.uint8],
    contours: Sequence[NDArray[np.int32]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> NDArray[np.uint8]:
    """Draw contours over source image copy."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    cv2.drawContours(canvas, list(contours), -1, color, thickness)
    return canvas


def contour_stats(contours: Sequence[NDArray[np.int32]]) -> ContourStats:
    """Calculate contour metrics."""
    if not contours:
        return ContourStats()
    areas = [float(cv2.contourArea(contour)) for contour in contours]
    perimeters = [float(cv2.arcLength(contour, closed=True)) for contour in contours]
    return ContourStats(
        count=len(contours),
        largest_area=max(areas),
        mean_area=float(np.mean(areas)),
        mean_perimeter=float(np.mean(perimeters)),
    )
