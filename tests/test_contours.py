from __future__ import annotations

import numpy as np
import pytest

from edges.contours import contour_stats, draw_contours, find_contours


def test_square_has_at_least_one_contour(synthetic_square_image: np.ndarray) -> None:
    contours = find_contours(synthetic_square_image)
    assert len(contours) >= 1


def test_empty_binary_has_no_contours() -> None:
    binary = np.zeros((32, 32), dtype=np.uint8)
    contours = find_contours(binary)
    assert contours == []


def test_contour_stats_are_computed(synthetic_circle_image: np.ndarray) -> None:
    contours = find_contours(synthetic_circle_image)
    stats = contour_stats(contours)
    assert stats.count >= 1
    assert stats.largest_area > 0
    assert stats.mean_perimeter > 0


def test_draw_contours_preserves_shape(synthetic_square_image: np.ndarray) -> None:
    contours = find_contours(synthetic_square_image)
    drawn = draw_contours(synthetic_square_image, contours)
    assert drawn.shape[:2] == synthetic_square_image.shape


def test_find_contours_rejects_non_2d_image() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        find_contours(np.zeros((8, 8, 3), dtype=np.uint8))


def test_find_contours_filters_by_min_area() -> None:
    binary = np.zeros((32, 32), dtype=np.uint8)
    binary[1:4, 1:4] = 255
    binary[10:25, 10:25] = 255
    contours = find_contours(binary, min_area=20.0)
    assert len(contours) == 1


def test_contour_stats_on_empty_contours() -> None:
    stats = contour_stats([])
    assert stats.count == 0
    assert stats.largest_area == 0.0


def test_draw_contours_on_color_image_returns_color() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 255
    binary = np.zeros((16, 16), dtype=np.uint8)
    binary[4:12, 4:12] = 255
    contours = find_contours(binary)
    drawn = draw_contours(image, contours)
    assert drawn.shape == image.shape
