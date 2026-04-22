"""Main image processing pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from edges.config import ImageResult, ProcessingConfig
from edges.contours import contour_stats, draw_contours, find_contours
from edges.images import iter_image_paths, make_output_path, read_image, save_color, save_grayscale
from edges.operators import apply_blur, laplacian_edges, sobel_edges, threshold_edges, to_grayscale
from edges.reporting import write_config_json, write_summary_csv, write_summary_json


def _combined_edge_map(sobel_map: np.ndarray | None, laplacian_map: np.ndarray | None) -> np.ndarray:
    """Combine available edge maps into one map for binary thresholding."""
    if sobel_map is not None and laplacian_map is not None:
        return np.maximum(sobel_map, laplacian_map)
    if sobel_map is not None:
        return sobel_map
    if laplacian_map is not None:
        return laplacian_map
    msg = "No edge map produced; unsupported method configuration"
    raise ValueError(msg)


def process_image(image_path: Path, input_dir: Path, output_dir: Path, config: ProcessingConfig) -> ImageResult:
    """Process one image and save all requested outputs."""
    t_start = perf_counter()
    image = read_image(image_path)
    gray = to_grayscale(image)
    prepared = apply_blur(gray, config.blur_kernel)

    sobel_map: np.ndarray | None = None
    laplacian_map: np.ndarray | None = None
    sobel_path: Path | None = None
    laplacian_path: Path | None = None
    contours_path: Path | None = None

    if config.method in {"sobel", "both"}:
        sobel_map = sobel_edges(prepared, config.sobel_kernel)
        sobel_path = make_output_path(input_dir, output_dir / "sobel", image_path, "sobel")
        save_grayscale(sobel_path, sobel_map)

    if config.method in {"laplacian", "both"}:
        laplacian_map = laplacian_edges(prepared, config.laplacian_kernel)
        laplacian_path = make_output_path(input_dir, output_dir / "laplacian", image_path, "laplacian")
        save_grayscale(laplacian_path, laplacian_map)

    edge_map = _combined_edge_map(sobel_map, laplacian_map)
    binary = threshold_edges(edge_map, config.threshold)
    binary_path = make_output_path(input_dir, output_dir / "binary", image_path, "binary")
    save_grayscale(binary_path, binary)

    contours = find_contours(binary, min_area=config.min_contour_area)
    stats = contour_stats(contours)

    if config.draw_contours:
        overlay = draw_contours(image, contours)
        contours_path = make_output_path(input_dir, output_dir / "contours", image_path, "contours")
        save_color(contours_path, overlay)

    t_end = perf_counter()
    edge_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    return ImageResult(
        source_path=image_path,
        sobel_path=sobel_path,
        laplacian_path=laplacian_path,
        binary_path=binary_path,
        contours_path=contours_path,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        edge_pixel_ratio=edge_ratio,
        contour_count=stats.count,
        largest_contour_area=stats.largest_area,
        processing_ms=(t_end - t_start) * 1000.0,
        method=config.method,
    )


def process_directory(input_dir: Path, output_dir: Path, config: ProcessingConfig) -> list[ImageResult]:
    """Process all matching images in input directory."""
    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    image_paths = list(iter_image_paths(input_dir, config.extensions, recursive=config.recursive))
    if config.max_images is not None:
        image_paths = image_paths[: config.max_images]

    results = [process_image(image_path, input_dir, output_dir, config) for image_path in image_paths]

    write_summary_csv(results, output_dir / "summary.csv")
    write_summary_json(results, output_dir / "summary.json")
    write_config_json(config, output_dir / "config.json")
    return results
