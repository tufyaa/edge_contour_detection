from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from edges.config import ProcessingConfig
from edges.pipeline import _combined_edge_map, process_directory, process_image


def test_process_image_creates_outputs(image_folder: Path, tmp_path: Path) -> None:
    input_image = image_folder / "a.jpg"
    config = ProcessingConfig(method="both", threshold="auto", draw_contours=True)
    result = process_image(input_image, image_folder, tmp_path, config)

    assert result.binary_path.exists()
    assert result.sobel_path is not None
    assert result.sobel_path.exists()
    assert result.laplacian_path is not None
    assert result.laplacian_path.exists()
    assert result.contours_path is not None
    assert result.contours_path.exists()


def test_process_directory_respects_limit(image_folder: Path, tmp_path: Path) -> None:
    config = ProcessingConfig(max_images=1, recursive=True)
    results = process_directory(image_folder, tmp_path, config)
    assert len(results) == 1


def test_process_directory_writes_summary_files(image_folder: Path, tmp_path: Path) -> None:
    config = ProcessingConfig(recursive=True)
    results = process_directory(image_folder, tmp_path, config)
    assert len(results) == 2
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "config.json").exists()

    with (tmp_path / "summary.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2


def test_process_directory_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Input directory does not exist"):
        process_directory(tmp_path / "missing", tmp_path / "out", ProcessingConfig())


def test_combined_edge_map_branches() -> None:
    sobel = np.ones((2, 2), dtype=np.uint8)
    lap = np.zeros((2, 2), dtype=np.uint8)
    both = _combined_edge_map(sobel, lap)
    assert np.array_equal(both, sobel)
    assert np.array_equal(_combined_edge_map(sobel, None), sobel)
    assert np.array_equal(_combined_edge_map(None, lap), lap)
    with pytest.raises(ValueError, match="No edge map produced"):
        _combined_edge_map(None, None)


def test_process_image_without_contours_output(image_folder: Path, tmp_path: Path) -> None:
    input_image = image_folder / "a.jpg"
    config = ProcessingConfig(method="sobel", draw_contours=False)
    result = process_image(input_image, image_folder, tmp_path, config)
    assert result.sobel_path is not None
    assert result.sobel_path.exists()
    assert result.laplacian_path is None
    assert result.contours_path is None


def test_process_image_laplacian_only(image_folder: Path, tmp_path: Path) -> None:
    input_image = image_folder / "a.jpg"
    config = ProcessingConfig(method="laplacian", draw_contours=False)
    result = process_image(input_image, image_folder, tmp_path, config)
    assert result.sobel_path is None
    assert result.laplacian_path is not None
