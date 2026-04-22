from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from numpy.typing import NDArray


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def synthetic_square_image() -> NDArray[np.uint8]:
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(image, (32, 32), (96, 96), 255, thickness=-1)
    return image


@pytest.fixture
def synthetic_circle_image() -> NDArray[np.uint8]:
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(image, (64, 64), 28, 255, thickness=-1)
    return image


@pytest.fixture
def gradient_image() -> NDArray[np.uint8]:
    return np.tile(np.arange(0, 128, dtype=np.uint8), (128, 1))


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    root = tmp_path / "input"
    nested = root / "nested"
    nested.mkdir(parents=True)

    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(img1, (8, 8), (24, 24), (255, 255, 255), thickness=-1)
    img2 = np.zeros((40, 20, 3), dtype=np.uint8)
    cv2.circle(img2, (10, 20), 8, (255, 255, 255), thickness=-1)

    cv2.imwrite(str(root / "a.jpg"), img1)
    cv2.imwrite(str(nested / "b.png"), img2)
    (root / "ignore.txt").write_text("not image", encoding="utf-8")
    return root
