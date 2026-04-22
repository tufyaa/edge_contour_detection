# Dataset

Primary dataset: BSDS500.

Official sources:

- https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
- https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/

Download command:

```bash
uv run edges dataset download ../data/raw
```

Create a smaller sample:

```bash
uv run edges dataset sample ../data/raw/BSR/BSDS500/data/images ../data/sample --limit 50
```

Recommended split for quick checks:

- `../data/raw/BSR/BSDS500/data/images/test`
- first 20-50 images for fast iteration
