# Architecture

The project follows a layered package layout similar to the reference project:

- `edges.console`: Click CLI and argument handling.
- `edges.pipeline`: Directory-level orchestration.
- `edges.images`: File-system and image IO.
- `edges.operators`: Sobel/Laplacian/threshold pure image transforms.
- `edges.contours`: Contour extraction, drawing, and contour metrics.
- `edges.reporting`: CSV/JSON output and performance plots.
- `edges.dataset`: BSDS500 download and sample preparation.

## Processing Flow

1. Collect image paths from input directory.
1. Read image and convert to grayscale.
1. Optional blur (`--blur-kernel`).
1. Compute edge maps (`sobel`, `laplacian`, or `both`).
1. Build binary edge map with fixed or Otsu threshold.
1. Extract contours from binary map.
1. Save outputs and aggregate per-image metrics.

## Output Layout

```text
output_dir/
  sobel/
  laplacian/
  binary/
  contours/
  summary.csv
  summary.json
  config.json
```
