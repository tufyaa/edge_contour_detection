"""Dataset helpers for BSDS500."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Literal

import pooch

BSDS500_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
SplitT = Literal["train", "val", "test", "all"]


def download_bsds500(target_dir: Path) -> Path:
    """Download and extract BSDS500 dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "BSR_bsds500.tgz"
    extracted_root = target_dir / "BSR" / "BSDS500"

    if extracted_root.exists():
        return extracted_root

    _download_archive(archive_path, target_dir)
    _extract_archive(archive_path, target_dir)

    if not extracted_root.exists():
        msg = f"Expected extracted dataset at {extracted_root}"
        raise FileNotFoundError(msg)
    return extracted_root


def _download_archive(archive_path: Path, target_dir: Path) -> None:
    """Download BSDS500 archive with pooch."""
    pooch.retrieve(
        url=BSDS500_URL,
        known_hash=None,
        path=str(target_dir),
        fname=archive_path.name,
        progressbar=False,
    )


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract dataset archive while rejecting paths outside target_dir."""
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_target = (target_root / member.name).resolve()
            if not member_target.is_relative_to(target_root):
                msg = f"Archive member escapes target directory: {member.name}"
                raise RuntimeError(msg)
            archive.extract(member, path=target_root, filter="data")


def find_bsds_images(dataset_dir: Path, split: SplitT) -> list[Path]:
    """Find BSDS500 image files for selected split."""
    images_root = _resolve_images_root(dataset_dir)
    if split == "all":
        paths = [
            *images_root.joinpath("train").glob("*.jpg"),
            *images_root.joinpath("val").glob("*.jpg"),
            *images_root.joinpath("test").glob("*.jpg"),
        ]
    else:
        paths = list(images_root.joinpath(split).glob("*.jpg"))
    return sorted(paths)


def copy_sample(source_dir: Path, target_dir: Path, limit: int) -> list[Path]:
    """Copy first N images from source tree preserving relative paths."""
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    target_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    selected = source_images[:limit]

    copied: list[Path] = []
    for source_image in selected:
        relative = source_image.relative_to(source_dir)
        target_path = target_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_path)
        copied.append(target_path)
    return copied


def _resolve_images_root(dataset_dir: Path) -> Path:
    """Resolve BSDS500 images root in downloaded folder structure."""
    candidates = [
        dataset_dir / "data" / "images",
        dataset_dir / "BSR" / "BSDS500" / "data" / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = f"Could not resolve BSDS500 image root from {dataset_dir}"
    raise FileNotFoundError(msg)
