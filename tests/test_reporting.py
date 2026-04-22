from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import edges.reporting as reporting
from edges.config import ImageResult, ProcessingConfig
from edges.reporting import (
    plot_contour_counts,
    plot_processing_times,
    write_config_json,
    write_summary_csv,
    write_summary_json,
)


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


def test_plot_functions_raise_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    original_import = builtins.__import__

    def failing_import(name: str, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)
    with pytest.raises(RuntimeError, match="matplotlib is required"):
        plot_processing_times([_result(tmp_path / "s.jpg")], tmp_path / "plot.png")
    with pytest.raises(RuntimeError, match="matplotlib is required"):
        plot_contour_counts([_result(tmp_path / "s.jpg")], tmp_path / "plot2.png")


def test_plot_functions_with_fake_matplotlib(tmp_path: Path, monkeypatch) -> None:
    fake_plot = _FakePyplot()
    fake_matplotlib = ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plot)
    plot_processing_times([_result(tmp_path / "a.jpg")], tmp_path / "t.png")
    plot_contour_counts([_result(tmp_path / "a.jpg")], tmp_path / "c.png")
    assert fake_plot.saved_paths == [tmp_path / "t.png", tmp_path / "c.png"]


class _FakeAxis:
    def bar(self, *_args, **_kwargs) -> None:
        return None

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return None

    def set_title(self, *_args, **_kwargs) -> None:
        return None

    def tick_params(self, *_args, **_kwargs) -> None:
        return None


class _FakeFigure:
    def __init__(self, collector: list[Path]) -> None:
        self._collector = collector

    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path) -> None:
        self._collector.append(path)


class _FakePyplot:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def subplots(self, **_kwargs) -> tuple[_FakeFigure, _FakeAxis]:
        return _FakeFigure(self.saved_paths), _FakeAxis()

    def close(self, *_args, **_kwargs) -> None:
        return None
