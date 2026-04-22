> [!NOTE]
> This package is under development and will change. It will also be migrated to another location once completed.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />

**Tools for Zarr Vectors Data**

`zarr-vectors-py` is a Python package for reading, writing, and managing large-scale vector geometry data in the zarr vectors format — a chunked, cloud-native format built on Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons, and meshes.

The package supports supervoxel-level spatial binning with separated chunk and bin sizes, per-level object sparsity for balanced multi-resolution pyramids, and OME-Zarr-compatible multiscale metadata.

*Aligned to the Zarr Vectors specification by Forest Collman, Allen Institute for Brain Sciences*
[Link to specification GitHub](https://github.com/AllenInstitute/zarr_vectors)

## Install

```bash
pip install zarr-vectors
```

With optional dependencies:

```bash
pip install zarr-vectors[ingest]     # LAS, PLY, TRK, TCK, TRX, SWC, GraphML, OBJ, STL
pip install zarr-vectors[draco]      # Draco mesh compression
pip install zarr-vectors[cloud]      # S3/GCS remote stores
pip install zarr-vectors[all]        # everything
```

## Quick Start

### Point clouds

```python
import numpy as np
from zarr_vectors.types.points import write_points, read_points

positions = np.random.default_rng(42).uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
intensity = np.random.default_rng(42).uniform(0, 1, size=100_000).astype(np.float32)

# Write with 200³ spatial chunks and 50³ supervoxel bins (64 bins per chunk)
write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    attributes={"intensity": intensity},
)

# Read all
result = read_points("scan.zarrvectors")
print(result["vertex_count"])  # 100000

# Spatial query — targets individual bins, not entire chunks
result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100, 100, 100]), np.array([200, 200, 200])),
)
```

### Streamlines

```python
from zarr_vectors.types.polylines import write_polylines, read_polylines

streamlines = [
    np.random.default_rng(i).normal(0, 50, size=(40, 3)).cumsum(axis=0).astype(np.float32)
    for i in range(500)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    groups={0: list(range(250)), 1: list(range(250, 500))},
)

# Read by group or object ID
result = read_polylines("tracts.zarrvectors", group_ids=[0])
print(result["polyline_count"])  # 250

result = read_polylines("tracts.zarrvectors", object_ids=[42])
print(result["polyline_count"])  # 1
```

### Skeletons (SWC)

```python
from zarr_vectors.ingest.swc import ingest_swc
from zarr_vectors.types.graphs import read_graph
from zarr_vectors.export.swc import export_swc

ingest_swc("neuron.swc", "neuron.zarrvectors", chunk_shape=(200.0, 200.0, 200.0))
result = read_graph("neuron.zarrvectors")
print(result["node_count"], "nodes,", result["edge_count"], "edges")

export_swc("neuron.zarrvectors", "neuron_out.swc")
```

### Meshes (OBJ)

```python
from zarr_vectors.ingest.obj import ingest_obj
from zarr_vectors.types.meshes import read_mesh

ingest_obj("brain.obj", "brain.zarrvectors", chunk_shape=(100.0, 100.0, 100.0))
result = read_mesh("brain.zarrvectors")
print(result["vertex_count"], "vertices,", result["face_count"], "faces")
```

### CSV point clouds

```python
from zarr_vectors.ingest.csv_points import ingest_csv
from zarr_vectors.export.csv_points import export_csv

ingest_csv(
    "measurements.csv",
    "measurements.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    position_columns=["x", "y", "z"],
    attribute_columns=["temperature", "pressure"],
)

export_csv("measurements.zarrvectors", "output.csv")
```

## Multi-Resolution Pyramids

### Automatic pyramid

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

# Legacy mode — auto-select levels by 8× vertex reduction
build_pyramid("scan.zarrvectors")
```

### Explicit levels with object sparsity

```python
# Two levels: 8× vertex reduction at level 1, 64× at level 2 with 50% object thinning
build_pyramid(
    "tracts.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2), "object_sparsity": 1.0},   # 8× from binning
        {"bin_ratio": (4, 4, 4), "object_sparsity": 0.5},   # 64× binning × 2× sparsity = 128×
    ],
)

# Read from a coarser level
from zarr_vectors.types.points import read_points
coarse = read_points("scan.zarrvectors", level=1)
```

### Manual single-level coarsening

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

# Add one level at 2× per axis (8× vertex reduction), keeping 30% of objects
coarsen_level(
    "scan.zarrvectors",
    source_level=0,
    target_level=1,
    bin_ratio=(2, 2, 2),
    object_sparsity=0.3,
    sparsity_strategy="spatial_coverage",
)
```

### Manual level management

```python
from zarr_vectors.core.store import (
    open_store, add_resolution_level, remove_resolution_level,
    list_available_ratios,
)

root = open_store("scan.zarrvectors", mode="r+")

# Create an empty level at 4× per axis
level_group = add_resolution_level(root, level_index=2, bin_ratio=(4, 4, 4))

# Query existing ratios
print(list_available_ratios(root))  # [(1,1,1), (2,2,2), (4,4,4)]

# Remove a level
remove_resolution_level(root, level_index=2)
```

### OME-Zarr multiscale metadata

```python
from zarr_vectors.core.multiscale import (
    write_multiscale_metadata, get_level_scale, get_level_translation,
)

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)

print(get_level_scale(root, 1))        # [2.0, 2.0, 2.0]
print(get_level_translation(root, 1))  # [50.0, 50.0, 50.0]
```

## Object Sparsity Strategies

When thinning objects for coarser levels, four selection strategies are available:

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity

# Keep 50% of 1000 objects, selected by spatial coverage
kept = apply_sparsity(
    1000, sparsity=0.5, strategy="spatial_coverage",
    representative_points=midpoints, bin_shape=(50.0, 50.0, 50.0),
)

# By length (keep longest streamlines)
kept = apply_sparsity(1000, 0.5, "length", lengths=streamline_lengths)

# By attribute (keep highest FA values)
kept = apply_sparsity(1000, 0.5, "attribute", attribute_values=fa_values)

# Random (reproducible)
kept = apply_sparsity(1000, 0.5, "random", seed=42)
```

## Validation

Five conformance levels, each building on the previous:

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   42 passed, 0 warnings, 0 errors
```

Levels: L1 structure, L2 metadata (bin/chunk divisibility, sparsity range), L3 consistency (vertex counts, bin bounds), L4 geometry conformance, L5 multi-resolution pyramid (ratio monotonicity, vertex non-increase).

## CLI

```bash
# Ingest
zarr-vectors ingest points  scan.las     scan.zarrvectors   --chunk-shape 100,100,50
zarr-vectors ingest streams tracts.trk   tracts.zarrvectors --chunk-shape 50,50,50
zarr-vectors ingest skeleton neuron.swc  neuron.zarrvectors --chunk-shape 200,200,200
zarr-vectors ingest mesh    brain.obj    brain.zarrvectors  --chunk-shape 100,100,100

# Export
zarr-vectors export csv  scan.zarrvectors   output.csv
zarr-vectors export obj  brain.zarrvectors  brain_out.obj
zarr-vectors export swc  neuron.zarrvectors neuron_out.swc
zarr-vectors export trk  tracts.zarrvectors tracts_out.trk

# Multi-resolution
zarr-vectors build-pyramid scan.zarrvectors --reduction-factor 8

# Validate
zarr-vectors validate scan.zarrvectors --level 5

# Info
zarr-vectors info scan.zarrvectors
```

## Key Concepts

### Chunks vs Bins

**Chunk shape** controls disk partitioning — each chunk is a file on disk (or a range request in cloud storage). Chunk shape is constant across all resolution levels.

**Bin shape** controls vertex grouping within chunks. Multiple bins tile each chunk. At coarser resolution levels, bins grow larger (fewer, bigger vertex groups per chunk) while the chunk grid stays fixed. This keeps data volume per chunk balanced across levels.

```
chunk_shape = (200, 200, 200)   # disk files
bin_shape   = (50, 50, 50)      # vertex groups within each file
→ 4×4×4 = 64 bins per chunk
```

When `bin_shape` is omitted, it defaults to `chunk_shape` (one bin per chunk), which is backward compatible with stores that don't use supervoxel binning.

### Object Sparsity

For geometry types with discrete objects (streamlines, skeletons, meshes), multi-resolution can reduce data volume through two independent channels:

- **Vertex reduction** from spatial binning (metanodes): controlled by `bin_ratio`
- **Object reduction** from thinning: controlled by `object_sparsity`

Total volume reduction = vertex_reduction × object_reduction. For example, `bin_ratio=(2,2,2)` gives 8× vertex reduction and `object_sparsity=0.5` gives 2× object reduction, for 16× total.

## Supported Formats

| Direction | Points | Streamlines | Skeletons | Meshes |
|-----------|--------|-------------|-----------|--------|
| **Ingest** | LAS, PLY, CSV, XYZ | TRK, TCK, TRX | SWC, GraphML | OBJ, STL |
| **Export** | CSV, PLY | TRK, TRX | SWC | OBJ |

## Store Layout

```
dataset.zarrvectors/
├── .zattrs                          # root metadata: SID, CRS, conventions, base_bin_shape
├── resolution_0/                    # full resolution (bin_ratio = 1,1,1)
│   ├── vertices/                    # spatial positions (ragged per bin)
│   ├── vertex_group_offsets/        # byte offsets for sub-bin random access
│   ├── links/                       # connectivity (edges, faces, parents)
│   ├── attributes/                  # per-vertex data
│   │   ├── intensity/
│   │   └── gene_expression/
│   ├── object_index/                # object ID → (chunk, vertex_group) mapping
│   ├── object_attributes/           # per-object metadata
│   ├── groupings/                   # group ID → [object IDs]
│   ├── groupings_attributes/        # per-group metadata
│   └── cross_chunk_links/           # connectivity across chunk boundaries
├── resolution_1/                    # coarsened (bin_ratio, object_sparsity in .zattrs)
│   └── [same arrays]
├── parametric/                      # algebraic objects (planes, spheres)
│   ├── objects/
│   └── object_attributes/
└── metadata.json
```

## License

BSD-3-Clause
