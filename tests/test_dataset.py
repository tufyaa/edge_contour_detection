from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Self

import cv2
import numpy as np
import pytest

import edges.dataset as dataset
from edges.dataset import copy_sample, download_bsds500, find_bsds_images


def test_find_bsds_images_for_split(tmp_path: Path) -> None:
    base = tmp_path / "BSR" / "BSDS500" / "data" / "images" / "train"
    base.mkdir(parents=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(base / "a.jpg"), image)
    cv2.imwrite(str(base / "b.jpg"), image)

    paths = find_bsds_images(tmp_path, "train")
    assert [path.name for path in paths] == ["a.jpg", "b.jpg"]


def test_copy_sample_limits_result_count(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        cv2.imwrite(str(source / name), image)

    target = tmp_path / "target"
    copied = copy_sample(source, target, limit=2)
    assert len(copied) == 2
    assert all(path.exists() for path in copied)


def test_download_returns_existing_path_without_network(tmp_path: Path) -> None:
    existing = tmp_path / "BSR" / "BSDS500"
    existing.mkdir(parents=True)
    assert download_bsds500(tmp_path) == existing


def test_find_bsds_images_all_split(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = tmp_path / "BSR" / "BSDS500" / "data" / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(split_dir / f"{split}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    paths = find_bsds_images(tmp_path, "all")
    assert len(paths) == 3


def test_copy_sample_rejects_non_positive_limit(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    with pytest.raises(ValueError, match="limit must be positive"):
        copy_sample(source, tmp_path / "dst", 0)


def test_find_bsds_images_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not resolve BSDS500 image root"):
        find_bsds_images(tmp_path, "train")


def test_download_bsds500_uses_pooch_branch(tmp_path: Path, monkeypatch) -> None:
    extracted_root = tmp_path / "BSR" / "BSDS500"

    class DummyTar:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def getmembers(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="BSR/BSDS500")]

        def extract(self, member: SimpleNamespace, path: Path, **kwargs: str) -> None:
            assert kwargs["filter"] == "data"
            (path / member.name).mkdir(parents=True, exist_ok=True)

    class DummyPooch:
        @staticmethod
        def retrieve(**_: object) -> None:
            return None

    monkeypatch.setattr(dataset.pooch, "retrieve", DummyPooch.retrieve)
    monkeypatch.setattr(dataset.tarfile, "open", lambda *_args, **_kwargs: DummyTar())

    result = download_bsds500(tmp_path)
    assert result == extracted_root


def test_download_bsds500_rejects_archive_path_traversal(tmp_path: Path, monkeypatch) -> None:
    class DummyTar:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def getmembers(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="../escaped")]

        def extract(self, member: SimpleNamespace, path: Path, **kwargs: str) -> None:
            pytest.fail("Unsafe archive member must not be extracted")

    monkeypatch.setattr(dataset.pooch, "retrieve", lambda **_: None)
    monkeypatch.setattr(dataset.tarfile, "open", lambda *_args, **_kwargs: DummyTar())

    with pytest.raises(RuntimeError, match="Archive member escapes target directory"):
        download_bsds500(tmp_path)


def test_download_bsds500_raises_when_extract_target_missing(tmp_path: Path, monkeypatch) -> None:
    class DummyTar:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def getmembers(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="somewhere")]

        def extract(self, member: SimpleNamespace, path: Path, **kwargs: str) -> None:
            assert kwargs["filter"] == "data"
            (path / "somewhere").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dataset.pooch, "retrieve", lambda **_: None)
    monkeypatch.setattr(dataset.tarfile, "open", lambda *_args, **_kwargs: DummyTar())

    with pytest.raises(FileNotFoundError, match="Expected extracted dataset"):
        download_bsds500(tmp_path)
