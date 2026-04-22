# Usage

## Workspace Environment

In this workspace, the virtual environment is intentionally outside the git repository:
`C:\proga\ippi_python\.venv`. Activate it, sync dependencies into the active environment, and run Poe
with the simple executor to avoid creating `edges/.venv`:

```powershell
C:\proga\ippi_python\.venv\Scripts\Activate.ps1
uv sync --active --all-groups
poe --executor simple ci
```

## Basic Processing

```bash
uv run edges process ../data/sample ../results/demo --method both --threshold auto --contours
```

Outputs are written into the selected output directory:

- `sobel/*.png`
- `laplacian/*.png`
- `binary/*.png`
- `contours/*.png`
- `summary.csv`
- `summary.json`
- `config.json`

## Example Input/Output

Input tree:

```text
../data/sample/
  img01.jpg
  img02.jpg
```

Result tree:

```text
../results/demo/
  sobel/img01_sobel.png
  laplacian/img01_laplacian.png
  binary/img01_binary.png
  contours/img01_contours.png
  summary.csv
  summary.json
  config.json
```

Example CLI output:

```text
Processed images: 2
Summary: ../results/demo/summary.csv
```

## Dataset Commands

```bash
uv run edges dataset download ../data/raw
uv run edges dataset sample ../data/raw/BSR/BSDS500/data/images ../data/sample --limit 50
```

## Benchmark

```bash
uv run edges benchmark ../data/sample ../results/benchmark --limit 50
```
