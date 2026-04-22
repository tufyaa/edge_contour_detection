# ippi-edges

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![Tests](https://img.shields.io/badge/tests-pytest%2079%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Docs](https://img.shields.io/badge/docs-Sphinx%20%2B%20Furo-blue)

Edge and contour detection project for image folders (Sobel + Laplacian + contours).

## Main Features

- Batch processing of input directory images
- Sobel and Laplacian edge maps
- Binary edge map with fixed threshold or Otsu (`auto`)
- Contour extraction and optional contour overlay
- CSV/JSON summary for processed images
- CLI based on `click`

## Quick Start

1. Install dependencies with `uv`:
   `uv sync --all-groups`
1. Run CLI help:
   `uv run edges --help`
1. Process directory:
   `uv run edges process <input_dir> <output_dir> --method both --threshold auto --contours`

By design, large datasets and generated outputs are expected outside the repo root:

- `../data/`
- `../results/`
- `../reports/generated/`

## Dataset

Download BSDS500:

`uv run edges dataset download ../data/raw`

Prepare small sample:

`uv run edges dataset sample ../data/raw/BSR/BSDS500/data/images ../data/sample --limit 50`

## Documentation

Build docs:

`uv run poe build_doc`

Generated site:

`docs/_build/html/index.html`

## Development

For this workspace, the shared virtual environment is intentionally outside the git repository:
`C:\proga\ippi_python\.venv`. Activate it, sync dependencies into the active environment, and run
Poe with the simple executor to avoid creating `edges/.venv`.

- Run tests:
  `uv run poe test`
- Run full local CI through the shared parent venv:
  `uv sync --active --all-groups && poe --executor simple ci`
- Run coverage-only command:
  `uv run poe test-coverage`
- Run lint and format checks:
  `uv run poe lint`
- Run type checks:
  `uv run poe typecheck`
- Run full local CI pipeline:
  `uv run poe ci`
