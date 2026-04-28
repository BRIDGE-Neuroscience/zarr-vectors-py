# Point clouds

Point clouds are the simplest ZVF geometry type: a collection of spatial
positions with optional per-vertex scalar or vector attributes. They arise
in synchrotron absorption imaging (HiP-CT, micro-CT), single-molecule
localisation microscopy (STORM, PALM, MINFLUX), spatial transcriptomics
(Visium HD, Xenium, MERFISH), and lidar scanning.

This tutorial covers writing, reading, spatial querying, attribute handling,
ingesting from external formats, and building multi-resolution pyramids.
All examples use synthetic data and require only `zarr-vectors` (base
install) except the ingest examples, which require `zarr-vectors[ingest]`.

---

## Writing a point cloud

### Minimal write

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(42)

# 100 000 points in a 1 000³ µm volume
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),   # one chunk file per 200³ µm
    bin_shape=(50.0, 50.0, 50.0),        # spatial index at 50³ µm
)
```

After writing, inspect the store with the CLI:

```bash
zarr-vectors info scan.zarrvectors
# geometry_type:  point_cloud
# spatial_dims:   3
# chunk_shape:    [200.0, 200.0, 200.0]
# bin_shape:      [50.0, 50.0, 50.0]
# resolution_0:   100000 vertices, 125 chunks
```

### Write with per-vertex attributes

Any number of named float or integer attribute arrays can be attached.
The arrays must have the same length as `positions` (one value per vertex):

```python
rng = np.random.default_rng(42)
n = 100_000
positions   = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
intensity   = rng.uniform(0, 1, n).astype(np.float32)          # absorption
label       = rng.integers(0, 8, n).astype(np.int32)           # class label
rgb         = rng.integers(0, 256, (n, 3)).astype(np.uint8)    # colour
confidence  = rng.uniform(0.5, 1, n).astype(np.float32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    attributes={
        "intensity":  intensity,
        "label":      label,
        "color":      rgb,        # vector attribute: shape (N, 3)
        "confidence": confidence,
    },
    coordinate_system="RAS",
    axis_units="micrometer",
)
```

### Choosing `chunk_shape` and `bin_shape`

A practical starting point: `chunk_shape` should be large enough that
each chunk contains ~10 000–100 000 vertices; `bin_shape` should be
roughly `chunk_shape / 4` per axis.

```python
# Estimate expected vertices per chunk
total_vertices    = 10_000_000
volume            = 4000 ** 3        # µm³
chunk_volume      = 500 ** 3         # µm³ with chunk_shape = 500
expected_per_chunk = total_vertices * chunk_volume / volume
# ≈ 7 812 — within the target range

write_points(
    "large_scan.zarrvectors",
    positions_10M,
    chunk_shape=(500.0, 500.0, 500.0),
    bin_shape=(125.0, 125.0, 125.0),   # 4×4×4 = 64 bins/chunk
)
```

---

## Reading a point cloud

### Read all data

```python
from zarr_vectors.types.points import read_points

result = read_points("scan.zarrvectors")

print(result["vertex_count"])                     # 100000
print(result["positions"].shape)                  # (100000, 3)
print(result["attributes"]["intensity"].shape)    # (100000,)
print(result["attributes"]["label"].dtype)        # int32
print(result["attributes"]["color"].shape)        # (100000, 3)
```

### Read a specific level

If the store has a multi-resolution pyramid, pass `level=N` to read a
coarser representation:

```python
coarse = read_points("scan.zarrvectors", level=1)
print(coarse["vertex_count"])     # fewer vertices — spatially coarsened
print(coarse["level"])            # 1
```

### Read a subset of attributes

For large stores, reading only the needed attributes avoids loading
attribute data for unused arrays:

```python
result = read_points(
    "scan.zarrvectors",
    attributes=["intensity"],         # only load intensity; skip others
)
assert "label" not in result["attributes"]
```

---

## Spatial bounding-box queries

ZVF queries target individual bins — not full chunks — so the amount of
data loaded is proportional to the query volume, not the chunk volume.

```python
lo = np.array([100.0, 100.0, 100.0])
hi = np.array([200.0, 200.0, 200.0])

result = read_points(
    "scan.zarrvectors",
    bbox=(lo, hi),
)
print(result["vertex_count"])   # ≈ 100 (100³/1000³ × 100 000)
print(result["positions"].min(axis=0))  # all >= lo
print(result["positions"].max(axis=0))  # all <= hi
```

The query is exact: only vertices within the half-open interval
`[lo, hi)` per axis are returned.

### Combining bbox and level

A coarser level with bbox is the key pattern for overview-first rendering:
load a low-resolution overview of the full volume, then switch to the
finer level only for the region of interest.

```python
# Overview: coarse level, full volume
overview = read_points("scan.zarrvectors", level=2)

# Detail: full resolution, small region
detail = read_points(
    "scan.zarrvectors",
    level=0,
    bbox=(np.array([400., 400., 400.]),
          np.array([600., 600., 600.])),
)
```

---

## Multi-resolution pyramids

### Building a pyramid

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2)},   # level 1: 8× fewer vertices
        {"bin_ratio": (4, 4, 4)},   # level 2: 64× fewer vertices
    ],
    attribute_aggregation={
        "intensity": "mean",
        "label":     "majority",
        "color":     "mean",
        "confidence":"min",
    },
)
```

After building, verify:

```bash
zarr-vectors info scan.zarrvectors
# resolution_0:  100000 vertices  (bin_ratio 1×1×1)
# resolution_1:  12890 vertices   (bin_ratio 2×2×2)
# resolution_2:  1613 vertices    (bin_ratio 4×4×4)
```

### Anisotropic pyramids

For data with anisotropic resolution (e.g. 4×4×25 nm voxels), use an
anisotropic `bin_ratio` that coarsens proportionally in each axis:

```python
# Anisotropic: coarsen z 2× less than x,y (data is 6× coarser in z already)
build_pyramid(
    "aniso.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 1)},    # level 1: coarsen only x,y
        {"bin_ratio": (4, 4, 2)},    # level 2: coarsen z by 2 total
    ],
)
```

---

## Ingesting from external formats

### LAS / LAZ point clouds

```python
from zarr_vectors.ingest.las import ingest_las   # requires zarr-vectors[ingest]

ingest_las(
    "survey.las",
    "survey.zarrvectors",
    chunk_shape=(100.0, 100.0, 50.0),
    bin_shape=(25.0, 25.0, 12.5),
    # LAS attributes auto-detected: intensity, return_number, classification, …
)
```

To select specific LAS attributes:

```python
ingest_las(
    "survey.las",
    "survey.zarrvectors",
    chunk_shape=(100.0, 100.0, 50.0),
    attributes=["intensity", "return_number"],   # subset of LAS dimensions
)
```

### CSV / XYZ files

```python
from zarr_vectors.ingest.csv_points import ingest_csv

ingest_csv(
    "measurements.csv",
    "measurements.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    position_columns=["x", "y", "z"],
    attribute_columns=["temperature", "pressure", "gene_count"],
)
```

For XYZ files (space-separated, no header):

```python
ingest_csv(
    "cloud.xyz",
    "cloud.zarrvectors",
    chunk_shape=(200., 200., 200.),
    position_columns=[0, 1, 2],     # column indices for headerless files
    attribute_columns=[3, 4],
    sep=" ",
    header=None,
)
```

### PLY files

```python
from zarr_vectors.ingest.ply import ingest_ply

ingest_ply(
    "scan.ply",
    "scan.zarrvectors",
    chunk_shape=(200., 200., 200.),
    # Vertex properties auto-detected from PLY header
)
```

### CLI ingest

```bash
zarr-vectors ingest points scan.las scan.zarrvectors --chunk-shape 100,100,50
zarr-vectors ingest points measurements.csv measurements.zarrvectors \
    --chunk-shape 50,50,50 \
    --position-columns x,y,z \
    --attribute-columns temperature,pressure
```

---

## Exporting

```python
from zarr_vectors.export.csv_points import export_csv
from zarr_vectors.export.ply import export_ply

export_csv("scan.zarrvectors", "scan_export.csv")
export_ply("scan.zarrvectors", "scan_export.ply")
```

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   42 passed, 0 warnings, 0 errors
```

---

## Common pitfalls

**`bin_shape` does not divide `chunk_shape`.**
The writer raises `ValueError` immediately. Check that
`chunk_shape[d] / bin_shape[d]` is an integer for every axis.

```python
# This will raise ValueError:
write_points("bad.zarrvectors", positions,
             chunk_shape=(200., 200., 200.),
             bin_shape=(60., 60., 60.))   # 200/60 = 3.33... ✗
```

**Writing float64 positions.**
ZVF stores positions as float32 by default. If your coordinates require
float64 precision (sub-nanometre accuracy at kilometre scale), pass
`dtype=np.float64` to preserve precision:

```python
write_points("precise.zarrvectors", positions_f64,
             chunk_shape=(200., 200., 200.),
             dtype=np.float64)
```

Be aware that float64 doubles storage size and reduces Blosc compression
ratio.

**Attribute array has wrong length.**
The write function checks that each attribute array has the same length as
`positions`. If you have a mismatch, check whether your data pipeline
dropped or duplicated rows.

**Reading a large store without bbox.**
`read_points` without `bbox` loads all vertices into memory. For stores
with > 50M vertices this may exhaust RAM. Use a bbox query or the lazy
API (see [Lazy loading](../multiscale/lazy_loading.md)).
