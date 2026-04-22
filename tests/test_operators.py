from __future__ import annotations

import numpy as np
import pytest

from edges.operators import (
    apply_blur,
    laplacian_edges,
    normalize_to_uint8,
    sobel_edges,
    threshold_edges,
    to_grayscale,
)


def test_to_grayscale_keeps_2d_shape(synthetic_square_image: np.ndarray) -> None:
    result = to_grayscale(synthetic_square_image)
    assert result.shape == synthetic_square_image.shape


def test_sobel_returns_zeros_for_uniform_image() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    result = sobel_edges(image, 3)
    assert np.count_nonzero(result) == 0


def test_sobel_detects_vertical_edge() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 255
    result = sobel_edges(image, 3)
    assert np.count_nonzero(result) > 0


def test_laplacian_returns_zeros_for_uniform_image() -> None:
    image = np.full((64, 64), 120, dtype=np.uint8)
    result = laplacian_edges(image, 3)
    assert np.count_nonzero(result) == 0


def test_threshold_auto_binary_values(gradient_image: np.ndarray) -> None:
    binary = threshold_edges(gradient_image, "auto")
    assert set(np.unique(binary)).issubset({0, 255})


def test_even_kernel_size_rejected() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd and positive"):
        sobel_edges(image, 2)


def test_blur_kernel_zero_returns_copy(synthetic_square_image: np.ndarray) -> None:
    blurred = apply_blur(synthetic_square_image, 0)
    assert np.array_equal(blurred, synthetic_square_image)


def test_to_grayscale_from_bgr_image() -> None:
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    color[:, :, 2] = 255
    gray = to_grayscale(color)
    assert gray.shape == (8, 8)


def test_to_grayscale_rejects_invalid_shape() -> None:
    invalid = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        to_grayscale(invalid)


def test_apply_blur_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd and positive"):
        apply_blur(np.zeros((5, 5), dtype=np.uint8), 2)


def test_normalize_to_uint8_handles_empty_input() -> None:
    values = np.array([], dtype=np.float64)
    normalized = normalize_to_uint8(values)
    assert normalized.size == 0


def test_normalize_to_uint8_handles_constant_input() -> None:
    values = np.full((2, 2), 10.0, dtype=np.float64)
    normalized = normalize_to_uint8(values)
    assert np.array_equal(normalized, np.zeros((2, 2), dtype=np.uint8))


def test_threshold_edges_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        threshold_edges(np.zeros((4, 4), dtype=np.uint8), 256)


def test_threshold_edges_fixed_threshold() -> None:
    edge_map = np.array([[0, 10], [200, 255]], dtype=np.uint8)
    binary = threshold_edges(edge_map, 100)
    assert np.array_equal(binary, np.array([[0, 0], [255, 255]], dtype=np.uint8))
