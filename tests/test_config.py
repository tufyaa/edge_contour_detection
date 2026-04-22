from __future__ import annotations

from pathlib import Path

import pytest

from edges.config import ImageResult, ProcessingConfig, normalize_extensions


def test_processing_config_normalizes_extensions() -> None:
    config = ProcessingConfig(extensions=("JPG", ".png", "jpg"))
    assert config.extensions == (".jpg", ".png")


def test_processing_config_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd and positive"):
        ProcessingConfig(sobel_kernel=4)


def test_processing_config_accepts_zero_blur() -> None:
    config = ProcessingConfig(blur_kernel=0)
    assert config.blur_kernel == 0


def test_processing_config_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        ProcessingConfig(threshold=300)


def test_processing_config_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unsupported method"):
        ProcessingConfig(method="x")  # type: ignore[arg-type]


def test_processing_config_rejects_non_int_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be int"):
        ProcessingConfig(threshold=1.5)  # type: ignore[arg-type]


def test_processing_config_rejects_negative_min_contour_area() -> None:
    with pytest.raises(ValueError, match="min_contour_area"):
        ProcessingConfig(min_contour_area=-0.1)


def test_processing_config_rejects_non_positive_max_images() -> None:
    with pytest.raises(ValueError, match="max_images"):
        ProcessingConfig(max_images=0)


def test_normalize_extensions_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="extensions cannot be empty"):
        normalize_extensions(("",))


def test_image_result_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        ImageResult(
            source_path=Path("a.jpg"),
            sobel_path=None,
            laplacian_path=None,
            binary_path=Path("a_binary.png"),
            contours_path=None,
            width=0,
            height=10,
            edge_pixel_ratio=0.1,
            contour_count=0,
            largest_contour_area=0,
            processing_ms=1.0,
            method="both",
        )


def test_image_result_rejects_edge_ratio_gt_one() -> None:
    with pytest.raises(ValueError, match="edge_pixel_ratio"):
        ImageResult(
            source_path=Path("a.jpg"),
            sobel_path=None,
            laplacian_path=None,
            binary_path=Path("a_binary.png"),
            contours_path=None,
            width=10,
            height=10,
            edge_pixel_ratio=1.1,
            contour_count=0,
            largest_contour_area=0,
            processing_ms=1.0,
            method="both",
        )


def test_image_result_rejects_negative_counters() -> None:
    with pytest.raises(ValueError, match="contour_count"):
        ImageResult(
            source_path=Path("a.jpg"),
            sobel_path=None,
            laplacian_path=None,
            binary_path=Path("a_binary.png"),
            contours_path=None,
            width=10,
            height=10,
            edge_pixel_ratio=0.1,
            contour_count=-1,
            largest_contour_area=0,
            processing_ms=1.0,
            method="both",
        )


def test_image_result_to_record_handles_none_paths() -> None:
    result = ImageResult(
        source_path=Path("a.jpg"),
        sobel_path=None,
        laplacian_path=None,
        binary_path=Path("a_binary.png"),
        contours_path=None,
        width=10,
        height=10,
        edge_pixel_ratio=0.2,
        contour_count=1,
        largest_contour_area=5.0,
        processing_ms=1.0,
        method="sobel",
    )
    record = result.to_record()
    assert record["sobel_path"] is None
    assert record["laplacian_path"] is None
    assert record["contours_path"] is None
    with pytest.raises(ValueError, match="largest_contour_area"):
        ImageResult(
            source_path=Path("a.jpg"),
            sobel_path=None,
            laplacian_path=None,
            binary_path=Path("a_binary.png"),
            contours_path=None,
            width=10,
            height=10,
            edge_pixel_ratio=0.1,
            contour_count=0,
            largest_contour_area=-1.0,
            processing_ms=1.0,
            method="both",
        )
    with pytest.raises(ValueError, match="processing_ms"):
        ImageResult(
            source_path=Path("a.jpg"),
            sobel_path=None,
            laplacian_path=None,
            binary_path=Path("a_binary.png"),
            contours_path=None,
            width=10,
            height=10,
            edge_pixel_ratio=0.1,
            contour_count=0,
            largest_contour_area=0.0,
            processing_ms=-0.1,
            method="both",
        )
