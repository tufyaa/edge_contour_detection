from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from edges.console import main


def test_help_command(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output


def test_version_command(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_process_command_calls_pipeline(runner: CliRunner, image_folder: Path, tmp_path: Path, mocker) -> None:
    mock_process = mocker.patch("edges.console.process_directory", return_value=[])
    result = runner.invoke(main, ["process", str(image_folder), str(tmp_path)])
    assert result.exit_code == 0
    mock_process.assert_called_once()


def test_process_command_rejects_invalid_threshold(runner: CliRunner, image_folder: Path, tmp_path: Path) -> None:
    result = runner.invoke(main, ["process", str(image_folder), str(tmp_path), "--threshold", "bad"])
    assert result.exit_code != 0
    assert "threshold must be integer" in result.output


def test_process_command_accepts_numeric_threshold(
    runner: CliRunner, image_folder: Path, tmp_path: Path, mocker
) -> None:
    mock_process = mocker.patch("edges.console.process_directory", return_value=[])
    result = runner.invoke(main, ["process", str(image_folder), str(tmp_path), "--threshold", "100"])
    assert result.exit_code == 0
    mock_process.assert_called_once()


def test_process_command_converts_runtime_error_to_click_error(
    runner: CliRunner, image_folder: Path, tmp_path: Path, mocker
) -> None:
    mocker.patch("edges.console.process_directory", side_effect=ValueError("boom"))
    result = runner.invoke(main, ["process", str(image_folder), str(tmp_path)])
    assert result.exit_code != 0
    assert "boom" in result.output


def test_dataset_download_command_calls_download(runner: CliRunner, tmp_path: Path, mocker) -> None:
    mock_download = mocker.patch("edges.console.download_bsds500", return_value=tmp_path / "BSR" / "BSDS500")
    result = runner.invoke(main, ["dataset", "download", str(tmp_path)])
    assert result.exit_code == 0
    mock_download.assert_called_once()


def test_dataset_sample_command_calls_copy(runner: CliRunner, tmp_path: Path, mocker) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = tmp_path / "dst"
    mock_copy = mocker.patch("edges.console.copy_sample", return_value=[])
    result = runner.invoke(main, ["dataset", "sample", str(source), str(target), "--limit", "5"])
    assert result.exit_code == 0
    mock_copy.assert_called_once_with(source, target, 5)


def test_dataset_download_command_wraps_errors(runner: CliRunner, tmp_path: Path, mocker) -> None:
    mocker.patch("edges.console.download_bsds500", side_effect=FileNotFoundError("missing"))
    result = runner.invoke(main, ["dataset", "download", str(tmp_path)])
    assert result.exit_code != 0
    assert "missing" in result.output


def test_dataset_sample_command_wraps_value_error(runner: CliRunner, tmp_path: Path, mocker) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = tmp_path / "dst"
    mocker.patch("edges.console.copy_sample", side_effect=ValueError("bad limit"))
    result = runner.invoke(main, ["dataset", "sample", str(source), str(target), "--limit", "1"])
    assert result.exit_code != 0
    assert "bad limit" in result.output


def test_benchmark_command(runner: CliRunner, image_folder: Path, tmp_path: Path, mocker) -> None:
    mock_process = mocker.patch("edges.console.process_directory", return_value=[object(), object()])
    result = runner.invoke(main, ["benchmark", str(image_folder), str(tmp_path), "--limit", "2"])
    assert result.exit_code == 0
    assert "Processed images: 2" in result.output
    mock_process.assert_called_once()


def test_benchmark_command_wraps_errors(runner: CliRunner, image_folder: Path, tmp_path: Path, mocker) -> None:
    mocker.patch("edges.console.process_directory", side_effect=ValueError("benchmark failed"))
    result = runner.invoke(main, ["benchmark", str(image_folder), str(tmp_path)])
    assert result.exit_code != 0
    assert "benchmark failed" in result.output
