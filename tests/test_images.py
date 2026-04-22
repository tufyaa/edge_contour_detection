from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from edges.images import ImageReadError, iter_image_paths, make_output_path, read_image, save_color, save_grayscale


def test_iter_image_paths_non_recursive(image_folder: Path) -> None:
    found = list(iter_image_paths(image_folder, (".jpg", ".png"), recursive=False))
    assert len(found) == 1
    assert found[0].name == "a.jpg"


def test_iter_image_paths_recursive(image_folder: Path) -> None:
    found = list(iter_image_paths(image_folder, (".jpg", ".png"), recursive=True))
    assert [path.name for path in found] == ["a.jpg", "b.png"]


def test_make_output_path_preserves_relative_path(image_folder: Path, tmp_path: Path) -> None:
    image_path = image_folder / "nested" / "b.png"
    output = make_output_path(image_folder, tmp_path / "out" / "sobel", image_path, "sobel")
    assert output.as_posix().endswith("nested/b_sobel.png")


def test_read_image_raises_on_invalid_file(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.jpg"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageReadError):
        read_image(invalid)


def test_save_functions_write_files(tmp_path: Path) -> None:
    gray = np.zeros((10, 10), dtype=np.uint8)
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    color_path = tmp_path / "color.png"
    save_grayscale(gray_path, gray)
    save_color(color_path, color)
    assert cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE) is not None
    assert cv2.imread(str(color_path), cv2.IMREAD_COLOR) is not None


def test_iter_image_paths_rejects_invalid_input_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Input directory does not exist"):
        list(iter_image_paths(tmp_path / "missing", (".jpg",), recursive=True))


def test_save_grayscale_rejects_non_uint8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected uint8 grayscale"):
        save_grayscale(tmp_path / "x.png", np.zeros((5, 5), dtype=np.float32))


def test_save_color_rejects_non_uint8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected uint8 color"):
        save_color(tmp_path / "x.png", np.zeros((5, 5, 3), dtype=np.float32))


def test_save_grayscale_raises_when_imwrite_fails(tmp_path: Path, mocker) -> None:
    mocker.patch("edges.images.cv2.imwrite", return_value=False)
    with pytest.raises(OSError, match="Could not write grayscale image"):
        save_grayscale(tmp_path / "bad.png", np.zeros((5, 5), dtype=np.uint8))


def test_save_color_raises_when_imwrite_fails(tmp_path: Path, mocker) -> None:
    mocker.patch("edges.images.cv2.imwrite", return_value=False)
    with pytest.raises(OSError, match="Could not write color image"):
        save_color(tmp_path / "bad.png", np.zeros((5, 5, 3), dtype=np.uint8))
