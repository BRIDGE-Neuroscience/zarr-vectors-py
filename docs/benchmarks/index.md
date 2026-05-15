# Benchmarks

```{admonition} Stub
:class: warning

Authoritative benchmark numbers for `zarr-vectors-py` are not yet
published. This page describes how to run the benchmark notebooks
shipped with the repository and what each one measures. **Results
tables and plots will be added in a future release** once a fixed
benchmark harness, reference hardware, and reproducibility protocol
have been agreed.
```

## Notebooks

Three small Jupyter notebooks live under
[`benchmarks/`](https://github.com/BRIDGE-Neuroscience/zarr-vectors-py/tree/main/benchmarks)
in the source tree. Each follows the same shape — **setup → sweep
→ table → plot** — and is intentionally small (~10 cells, one
plot per notebook).

| Notebook | Axis swept | Fixed |
|----------|-----------|-------|
| `01_size_scaling.ipynb` | vertex count `N ∈ {10³, 10⁴, 10⁵, 10⁶}` | point cloud, local backend |
| `02_data_types.ipynb`   | geometry type (all seven) | `N = 50 000`, local backend |
| `03_backends.ipynb`     | storage backend (`local`, `obstore`, `fsspec`) | point cloud, `N = 100 000` |

The notebooks are designed as "what to expect on my machine"
sanity references, not as cross-format comparisons.

## Running locally

```bash
pip install zarr-vectors pandas matplotlib
jupyter lab benchmarks/
```

Then open one notebook and run all cells. Expected runtime on a
laptop:

- `01_size_scaling`: a few minutes (the 1 M-vertex case dominates).
- `02_data_types`: ~30 seconds.
- `03_backends`: ~10 seconds without cloud, longer with.

## Cloud-backend mode

Notebook `03_backends.ipynb` benchmarks the `obstore` and `fsspec`
cloud backends **only when** the `ZV_BENCH_S3_URL` env var is set:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
pip install "zarr-vectors[obstore]"   # preferred
# or
pip install "zarr-vectors[cloud]"     # fsspec fallback
```

Without the env var the cloud rows are skipped and a one-row
local-only result is reported.

## Results

*To be added.* Numbers will be published here once a reproducibility
protocol (hardware spec, OS, dependency versions, dataset seeds) has
been frozen and the harness has been re-run against the locked
target. Until then, the notebooks themselves are the only available
reference and they are machine-dependent — do not treat them as
published metrics.

## Caveats

- These notebooks measure **disk bytes and wall time** only — no
  memory profiling.
- No CI gating and no `pytest-benchmark` integration: regressions
  are not caught automatically.
- Different geometry types do genuinely different work; do not
  cross-compare rows of `02_data_types`.
- Numbers are not directly comparable across hardware, file systems,
  or cloud regions.

To regenerate the notebooks from the source recipe:

```bash
python benchmarks/_build.py
```
