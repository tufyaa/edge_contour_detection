# Performance

This page describes how to measure runtime and how to report bottlenecks.

## Benchmark Command

```bash
uv run edges benchmark ../data/sample ../results/benchmark --limit 50
```

## Profiling

```bash
python -m cProfile -o ../reports/generated/profile.pstats -m edges process ../data/sample ../results/profile
```

Recommended flow:

1. Run baseline benchmark on a fixed image subset.
1. Save total runtime and average runtime per image.
1. Profile the run and identify dominant functions.
1. Optimize one bottleneck at a time.
1. Re-run benchmark with identical input set.
1. Compare before/after metrics.

## Runtime Charts

The package can produce charts from `ImageResult` records:

- `plot_processing_times(...)`
- `plot_contour_counts(...)`

Store generated figures under `../reports/generated/figures/` and include them
in the LaTeX report.

## Required Report Artifacts

- Runtime table (before/after).
- At least one generated performance chart.
- Short explanation of the selected bottleneck.
- Description of data sample used for tests.
