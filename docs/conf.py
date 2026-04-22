"""Sphinx configuration."""

import sys
from pathlib import Path

from sphinx_pyproject import SphinxConfig

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

config = SphinxConfig("../pyproject.toml", globalns=globals())
project = config.name
author = "IPPI Student Project"
version = config.version
release = config.version
