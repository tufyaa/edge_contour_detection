from __future__ import annotations

import runpy


def test_main_module_executes_console_main(mocker) -> None:
    mock_main = mocker.patch("edges.console.main")
    runpy.run_module("edges.__main__", run_name="__main__")
    mock_main.assert_called_once()
