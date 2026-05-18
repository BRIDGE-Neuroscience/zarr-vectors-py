# Quickstart

This page covers the most common operations in `zarr-vectors`: writing a
point cloud, reading it back, performing a spatial bounding-box query, and
writing a set of streamlines with group labels. All examples use synthetic
data and require only the base install.

## Point clouds

### Writing

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(42)

# 100 000 points uniformly distributed in a 1 000³ μm volume
positions  = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
intensity  = rng.uniform(0, 1,    size=100_000).astype(np.float32)
label      = rng.integers(0, 5,   size=100_000).astype(np.int32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),   # I/O unit — one file per chunk
    bin_shape=(50.0, 50.0, 50.0),        # spatial index unit — 64 bins/chunk
    attributes={"intensity": intensity, "label": label},
)
```

`chunk_shape` and `bin_shape` are in the same physical units as `positions`
(here, micrometres). The store is written as a directory tree called
`scan.zarrvectors/`. See [Concepts](concepts.md) for an explanation of the
chunk/bin distinction.

### Reading all data

```python
from zarr_vectors.types.points import read_points

result = read_points("scan.zarrvectors")

print(result["vertex_count"])          # 100000
print(result["positions"].shape)       # (100000, 3)
print(result["attributes"]["intensity"].shape)  # (100000,)
```

### Spatial bounding-box query

Spatial queries target individual bins rather than entire chunks, so only
the data in the requested region is loaded from disk.

```python
result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100.0, 100.0, 100.0]),
          np.array([200.0, 200.0, 200.0])),
)

print(result["vertex_count"])   # number of points in the 100³ μm box
print(result["positions"].shape)
```

### Reading a coarser resolution level

If you have built a multi-resolution pyramid (see
[Building pyramids](../tutorials/multiscale/building_pyramids.md)), pass
`level=` to read from a coarser level:

```python
result = read_points("scan.zarrvectors", level=1)
print(result["vertex_count"])   # fewer points — coarser bin resolution
```

---

## Streamlines

### Writing

```python
from zarr_vectors.types.polylines import write_polylines

rng = np.random.default_rng(0)

# 500 streamlines, each with 40 vertices, walking through 3-D space
streamlines = [
    rng.normal(0, 50, size=(40, 3)).cumsum(axis=0).astype(np.float32)
    for _ in range(500)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    # Optional: assign streamlines to named groups
    groups={0: list(range(250)), 1: list(range(250, 500))},
)
```

### Reading by group

```python
from zarr_vectors.types.polylines import read_polylines

result = read_polylines("tracts.zarrvectors", group_ids=[0])
print(result["polyline_count"])   # 250
```

### Reading a single object by ID

```python
result = read_polylines("tracts.zarrvectors", object_ids=[42])
print(result["polyline_count"])   # 1
print(result["polylines"][0].shape)  # (40, 3) — the 40 vertices of streamline 42
```

---

## Format converters

Ingesting from third-party formats (LAS, PLY, CSV, TRK, TCK, TRX, SWC,
GraphML, OBJ, STL) and exporting back to them lives in the companion
package **`zarr-vectors-tools`**, alongside the `zarr-vectors` CLI.

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   42 passed, 0 warnings, 0 errors
```

Validation levels 1–5 check progressively deeper properties of the store.
See [Validation](../tutorials/io/validation_and_repair.md) for details.

---

## Next steps

- **[Concepts](concepts.md)** — understand chunks, bins, fragments, and
  the multiscale pyramid before working with larger datasets.
- **[Data type tutorials](../tutorials/data_types/point_clouds.md)** — deeper
  walkthroughs for each geometry type.
- **[Building pyramids](../tutorials/multiscale/building_pyramids.md)** — add
  multi-resolution levels for level-of-detail rendering.
- **[Neuroglancer integration](../tutorials/neuroglancer/overview.md)** — visualise
  your stores in Neuroglancer using `zv-ngtools`.
