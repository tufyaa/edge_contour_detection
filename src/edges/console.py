"""Click CLI for edge and contour detection."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import cast

import click

from edges import __version__
from edges.config import MethodT, ProcessingConfig, ThresholdT
from edges.dataset import copy_sample, download_bsds500
from edges.pipeline import process_directory


def _parse_threshold(_: click.Context, __: click.Parameter, value: str) -> int | str:
    """Parse threshold option from string argument."""
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = "threshold must be integer or 'auto'"
        raise click.BadParameter(msg) from exc
    return parsed


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Batch edge and contour detection tool."""


@main.command("process")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--method", type=click.Choice(["sobel", "laplacian", "both"]), default="both", show_default=True)
@click.option("--threshold", callback=_parse_threshold, default="auto", show_default=True)
@click.option("--blur-kernel", type=int, default=3, show_default=True)
@click.option("--sobel-kernel", type=int, default=3, show_default=True)
@click.option("--laplacian-kernel", type=int, default=3, show_default=True)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--contours/--no-contours", "draw_contours", default=True, show_default=True)
@click.option("--min-contour-area", type=float, default=0.0, show_default=True)
@click.option("--extension", "extensions", multiple=True, default=(".png", ".jpg", ".jpeg", ".bmp"), show_default=True)
@click.option("--limit", "max_images", type=int, default=None)
def process_command(
    input_dir: Path,
    output_dir: Path,
    *,
    method: str,
    threshold: int | str,
    blur_kernel: int,
    sobel_kernel: int,
    laplacian_kernel: int,
    recursive: bool,
    draw_contours: bool,
    min_contour_area: float,
    extensions: tuple[str, ...],
    max_images: int | None,
) -> None:
    """Process all images in INPUT_DIR and save outputs to OUTPUT_DIR."""
    try:
        config = ProcessingConfig(
            method=cast(MethodT, method),
            blur_kernel=blur_kernel,
            sobel_kernel=sobel_kernel,
            laplacian_kernel=laplacian_kernel,
            threshold=cast(ThresholdT, threshold),
            recursive=recursive,
            extensions=extensions,
            draw_contours=draw_contours,
            min_contour_area=min_contour_area,
            max_images=max_images,
        )
        results = process_directory(input_dir, output_dir, config)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Processed images: {len(results)}")
    click.echo(f"Summary: {output_dir / 'summary.csv'}")


@main.group("dataset")
def dataset_group() -> None:
    """Dataset utilities."""


@dataset_group.command("download")
@click.argument("target_dir", type=click.Path(file_okay=False, path_type=Path))
def dataset_download_command(target_dir: Path) -> None:
    """Download BSDS500 dataset into TARGET_DIR."""
    try:
        extracted = download_bsds500(target_dir)
    except (RuntimeError, ValueError, OSError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Dataset available at: {extracted}")


@dataset_group.command("sample")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("target_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--limit", type=int, default=20, show_default=True)
def dataset_sample_command(source_dir: Path, target_dir: Path, limit: int) -> None:
    """Copy N sample images from SOURCE_DIR to TARGET_DIR."""
    try:
        copied = copy_sample(source_dir, target_dir, limit)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Copied images: {len(copied)}")


@main.command("benchmark")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--limit", type=int, default=50, show_default=True)
def benchmark_command(input_dir: Path, output_dir: Path, limit: int) -> None:
    """Run processing benchmark with default parameters."""
    try:
        config = ProcessingConfig(max_images=limit)
        started = perf_counter()
        results = process_directory(input_dir, output_dir, config)
        elapsed = perf_counter() - started
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Processed images: {len(results)}")
    click.echo(f"Elapsed seconds: {elapsed:.3f}")
