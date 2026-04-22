"""Image operators: grayscale, Sobel, Laplacian, threshold."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

ThresholdT = int | Literal["auto"]
MAX_UINT8 = 255


def _validate_odd_kernel(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        msg = f"Kernel size must be odd and positive, got: {kernel_size}"
        raise ValueError(msg)


def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR image to grayscale."""
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    msg = f"Unsupported image shape: {image.shape}"
    raise ValueError(msg)


def apply_blur(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]:
    """Apply gaussian blur to grayscale image."""
    if kernel_size == 0:
        return gray.copy()
    _validate_odd_kernel(kernel_size)
    return cast(NDArray[np.uint8], cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0))


def normalize_to_uint8(values: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Normalize float map to uint8 range [0, 255]."""
    if values.size == 0:
        return np.array([], dtype=np.uint8)
    max_value = float(np.max(values))
    min_value = float(np.min(values))
    if max_value == min_value:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - min_value) * (255.0 / (max_value - min_value))
    return normalized.astype(np.uint8)


def sobel_edges(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]:
    """Compute Sobel edge magnitude map."""
    _validate_odd_kernel(kernel_size)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = np.hypot(grad_x, grad_y)
    return normalize_to_uint8(magnitude)


def laplacian_edges(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]:
    """Compute Laplacian edge map."""
    _validate_odd_kernel(kernel_size)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    return normalize_to_uint8(np.abs(lap))


def threshold_edges(edge_map: NDArray[np.uint8], threshold: ThresholdT) -> NDArray[np.uint8]:
    """Build binary edge map from grayscale edge map."""
    if threshold == "auto":
        _, binary = cv2.threshold(edge_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cast(NDArray[np.uint8], binary)
    if threshold < 0 or threshold > MAX_UINT8:
        msg = "threshold must be in [0, 255]"
        raise ValueError(msg)
    _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)
    return cast(NDArray[np.uint8], binary)
