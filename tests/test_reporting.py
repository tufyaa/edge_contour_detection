from __future__ import annotations

import json
from pathlib import Path

import edges.reporting as reporting
from edges.config import ImageResult, ProcessingConfig
from edges.reporting import write_config_json, write_summary_csv, write_summary_json


def _result(path: Path) -> ImageResult:
    return ImageResult(
        source_path=path,
        sobel_path=path.with_name("x_sobel.png"),
        laplacian_path=path.with_name("x_laplacian.png"),
        binary_path=path.with_name("x_binary.png"),
        contours_path=path.with_name("x_contours.png"),
        width=64,
        height=64,
        edge_pixel_ratio=0.1,
        contour_count=3,
        largest_contour_area=12.0,
        processing_ms=1.5,
        method="both",
    )


def test_write_summary_csv_and_json(tmp_path: Path) -> None:
    results = [_result(tmp_path / "source.jpg")]
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    write_summary_csv(results, csv_path)
    write_summary_json(results, json_path)
    assert csv_path.exists()
    assert json_path.exists()
    assert "source_path" in csv_path.read_text(encoding="utf-8")
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert records[0]["contour_count"] == 3


def test_write_config_json(tmp_path: Path) -> None:
    config = ProcessingConfig()
    path = tmp_path / "config.json"
    write_config_json(config, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["method"] == "both"


def test_json_safe_handles_nested_values() -> None:
    value = {"path": Path("x"), "tuple": (1, 2), "list": [Path("y")]}
    encoded = reporting._json_safe(value)
    assert encoded["path"] == "x"
    assert encoded["tuple"] == [1, 2]
    assert encoded["list"] == ["y"]
