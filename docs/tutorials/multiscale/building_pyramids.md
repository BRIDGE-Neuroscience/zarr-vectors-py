# Building multi-resolution pyramids

A multi-resolution pyramid stores the same dataset at progressively coarser
spatial resolutions. Viewers and analysis pipelines select the appropriate
level based on viewport size, memory budget, or query scale — loading only
the data density they need.

This tutorial covers pyramid construction for point clouds, streamlines,
and skeletons; manual level management; choosing bin ratios and sparsity
values; and verifying the resulting OME-Zarr metadata.

---

## Concepts recap

**Bin ratio** scales the supervoxel bin size at each level. A `bin_ratio`
of `(2, 2, 2)` at level 1 means each bin covers `2 × base_bin_shape` per
axis — 8× more volume — so up to 8× fewer metanodes per unit volume.

**Object sparsity** thins discrete objects (streamlines, skeletons, meshes)
at coarser levels. A value of `0.5` retains 50 % of objects.

**Total reduction** at level N = vertex_reduction × object_reduction =
`product(bin_ratio) × (1 / object_sparsity)`.

---

## Point cloud pyramids

Point clouds use spatial coarsening only; `object_sparsity` is always 1.0.

### Quick three-level pyramid

```python
import numpy as np
from zarr_vectors.types.points import write_points
from zarr_vectors.multiresolution.coarsen import build_pyramid

rng = np.random.default_rng(0)
positions  = rng.uniform(0, 2000, (500_000, 3)).astype(np.float32)
intensity  = rng.uniform(0, 1, 500_000).astype(np.float32)
label      = rng.integers(0, 16, 500_000).astype(np.int32)

write_points(
    "synchrotron.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    attributes={"intensity": intensity, "label": label},
)

build_pyramid(
    "synchrotron.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2)},   # level 1: 8× reduction
        {"bin_ratio": (4, 4, 4)},   # level 2: 64× reduction
        {"bin_ratio": (8, 8, 8)},   # level 3: 512× reduction
    ],
    attribute_aggregation={
        "intensity": "mean",
        "label":     "majority",
    },
)
```

Verify the result:

```bash
zarr-vectors info synchrotron.zarrvectors
# resolution_0:  500000 vertices  (bin_ratio [1,1,1])
# resolution_1:  63241 vertices   (bin_ratio [2,2,2])
# resolution_2:  8022 vertices    (bin_ratio [4,4,4])
# resolution_3:  1018 vertices    (bin_ratio [8,8,8])
```

The actual vertex count at each level is less than the theoretical maximum
(`500000 / 8^N`) because bins near the data boundary may have fewer
vertices to merge.

### Choosing bin ratios

A balanced pyramid reduces the vertex count by a consistent factor per
level. For 3-D data the factor equals `product(bin_ratio)`.

| Target reduction/level | Isotropic `bin_ratio` | Anisotropic example |
|-----------------------|----------------------|---------------------|
| 8× | (2, 2, 2) | — |
| 27× | (3, 3, 3) | — |
| 64× | (4, 4, 4) | — |
| ~12× (anisotropic) | — | (2, 2, 3) |

For synchrotron data with anisotropic voxels (e.g. 1 µm × 1 µm × 4 µm),
apply a proportional bin ratio so the spatial resolution remains roughly
isotropic across levels:

```python
build_pyramid(
    "aniso.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 1)},    # coarsen x,y only first
        {"bin_ratio": (4, 4, 2)},    # now coarsen z too
        {"bin_ratio": (8, 8, 4)},
    ],
)
```

### Attribute aggregation strategies

Different attributes benefit from different aggregation rules:

```python
build_pyramid(
    "scan.zarrvectors",
    level_configs=[{"bin_ratio": (2, 2, 2)}],
    attribute_aggregation={
        "intensity":   "mean",      # average fluorescence signal
        "label":       "majority",  # most frequent class wins
        "confidence":  "min",       # conservative: take lowest confidence
        "count":       "sum",       # accumulate counts across merged points
        "peak_signal": "max",       # preserve peak values
    },
)
```

---

## Streamline pyramids

Streamlines use both spatial coarsening (vertex metanodes) and object
thinning (dropping individual streamlines at coarser levels).

### Two-stage pyramid with increasing thinning

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.multiresolution.coarsen import build_pyramid

rng = np.random.default_rng(0)
streamlines = [
    rng.normal(0, 30, (rng.integers(30, 120), 3)).cumsum(0).astype(np.float32)
    for _ in range(10_000)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    geometry_type="streamline",
)

build_pyramid(
    "tracts.zarrvectors",
    level_configs=[
        # Level 1: keep all streamlines, 8× fewer vertices
        {
            "bin_ratio":        (2, 2, 2),
            "object_sparsity":  1.0,
        },
        # Level 2: keep 25% of streamlines, 64× fewer vertices/stream
        {
            "bin_ratio":        (4, 4, 4),
            "object_sparsity":  0.25,
            "sparsity_strategy": "spatial_coverage",
        },
        # Level 3: keep 5% of streamlines, 512× fewer vertices/stream
        {
            "bin_ratio":        (8, 8, 8),
            "object_sparsity":  0.05,
            "sparsity_strategy": "spatial_coverage",
        },
    ],
)
```

Expected output:

```
resolution_0:  10000 streamlines, ~750 000 vertices
resolution_1:  10000 streamlines, ~96 500 vertices   (8× vertex reduction)
resolution_2:  2500 streamlines,  ~3 100 vertices    (64× vertex × 4× object)
resolution_3:  500 streamlines,   ~155 vertices      (512× vertex × 20× object)
```

### Comparing sparsity strategies

The `sparsity_strategy` controls which objects are kept at coarser levels.

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity
import time

n = 10_000
midpoints = np.array([s[len(s) // 2] for s in streamlines])
lengths   = np.array([np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))
                      for s in streamlines], dtype=np.float32)

for strategy in ["spatial_coverage", "length", "random"]:
    t0 = time.perf_counter()
    kept = apply_sparsity(n, sparsity=0.1, strategy=strategy,
                          representative_points=midpoints if strategy=="spatial_coverage" else None,
                          lengths=lengths if strategy=="length" else None,
                          bin_shape=(50., 50., 50.),
                          seed=42)
    elapsed = time.perf_counter() - t0
    print(f"{strategy:20s}  kept={len(kept)}  time={elapsed:.3f}s")
```

For most tractography datasets, `spatial_coverage` produces the most
visually representative thinned set because it samples proportionally from
every spatial region. `length` is useful when scientific importance
correlates with streamline length.

### Building a pyramid for a pre-grouped store

Groups (bundles) are preserved across levels. The thinning strategy is
applied uniformly across the entire store, not per-group:

```python
write_polylines(
    "bundled.zarrvectors",
    streamlines,
    chunk_shape=(100., 100., 100.),
    groups={"CST": list(range(3000)), "AF": list(range(3000, 10000))},
)

build_pyramid(
    "bundled.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2), "object_sparsity": 0.5,
         "sparsity_strategy": "spatial_coverage"},
    ],
)

# Groups are accessible at coarser levels
from zarr_vectors.types.polylines import read_polylines
result = read_polylines("bundled.zarrvectors", level=1, group_ids=["CST"])
print(result["polyline_count"])   # ≈ 1500 (50% of 3000)
```

---

## Manual level management

For fine-grained control, add and populate levels one at a time.

### Listing existing levels

```python
from zarr_vectors.core.store import open_store, list_resolution_levels, list_available_ratios

root = open_store("synchrotron.zarrvectors", mode="r")
print(list_resolution_levels(root))      # [0, 1, 2, 3]
print(list_available_ratios(root))       # [(1,1,1), (2,2,2), (4,4,4), (8,8,8)]
```

### Adding a level manually

```python
from zarr_vectors.core.store import open_store, add_resolution_level
from zarr_vectors.multiresolution.coarsen import coarsen_level

root = open_store("synchrotron.zarrvectors", mode="r+")

# Create the empty level shell (metadata only)
add_resolution_level(root, level_index=4, bin_ratio=(16, 16, 16))

# Populate the level by coarsening from level 3
coarsen_level(
    "synchrotron.zarrvectors",
    source_level=3,
    target_level=4,
    bin_ratio=(16, 16, 16),
    attribute_aggregation={"intensity": "mean", "label": "majority"},
    position_strategy="centroid",
)
```

### Removing a level

```python
from zarr_vectors.core.store import remove_resolution_level

root = open_store("synchrotron.zarrvectors", mode="r+")
remove_resolution_level(root, level_index=4)
print(list_resolution_levels(root))   # [0, 1, 2, 3]
```

---

## Reading pyramid levels

```python
from zarr_vectors.types.points import read_points

# Read each level and compare vertex counts
for level in range(4):
    result = read_points("synchrotron.zarrvectors", level=level)
    print(f"Level {level}: {result['vertex_count']:>8d} vertices")
```

Reading a specific level with a bounding box:

```python
# Quick overview: coarsest level, full volume
overview = read_points("synchrotron.zarrvectors", level=3)

# Drill-down: finest level, region of interest
detail = read_points(
    "synchrotron.zarrvectors",
    level=0,
    bbox=(np.array([400., 400., 400.]),
          np.array([600., 600., 600.])),
)
```

---

## Verifying OME-Zarr multiscale metadata

After building a pyramid, confirm the metadata is correct:

```python
from zarr_vectors.core.multiscale import (
    get_level_scale,
    get_level_translation,
    write_multiscale_metadata,
)
from zarr_vectors.core.store import open_store

root = open_store("synchrotron.zarrvectors", mode="r")

for level in range(4):
    scale       = get_level_scale(root, level)
    translation = get_level_translation(root, level)
    print(f"Level {level}:  scale={scale}  translation={translation}")
# Level 0:  scale=[1.0,1.0,1.0]  translation=[25.0,25.0,25.0]
# Level 1:  scale=[2.0,2.0,2.0]  translation=[50.0,50.0,50.0]
# Level 2:  scale=[4.0,4.0,4.0]  translation=[100.0,100.0,100.0]
# Level 3:  scale=[8.0,8.0,8.0]  translation=[200.0,200.0,200.0]
```

If you have manually modified the `.zattrs` or added levels programmatically,
regenerate the metadata:

```python
root = open_store("synchrotron.zarrvectors", mode="r+")
write_multiscale_metadata(root)
```

---

## Performance tips for large datasets

**Build levels from finest to coarsest.** `coarsen_level(source=N-1,
target=N)` is faster than `coarsen_level(source=0, target=N)` because
the source data is smaller. `build_pyramid` does this automatically.

**Parallelise across chunks.** For HPC environments, set `n_workers`
to use multiprocessing:

```python
build_pyramid(
    "large.zarrvectors",
    level_configs=[{"bin_ratio": (2, 2, 2)}],
    n_workers=16,    # uses multiprocessing.Pool
)
```

**Skip fine-level attributes when only coarse is needed.** If coarse
levels are only used for rendering (not attribute analysis), omit
attributes to save storage and compute:

```python
build_pyramid(
    "large.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2)},
        {"bin_ratio": (4, 4, 4)},
    ],
    attribute_names=None,   # do not coarsen attributes at any level
)
```
