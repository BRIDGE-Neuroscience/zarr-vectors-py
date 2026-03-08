
> [!NOTE]
> This package is under development and will change. It will also be migrated to another location once completed.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />


**Tools for Zarr Vectors Data**

`zarr-vectors-py` is a Python package for reading, writing, and managing large-scale vector geometry data in the zarr vectors format — a chunked format built on mirror Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons, and meshes.

*Aligned to the Zarr_vectors specification by Forest Collman, Allen Institute for Brain Sciences*
[Link to specification GitHub](https://github.com/AllenInstitute/zarr_vectors)

## Install

```bash
pip install zarr-vectors
```

## Quick Start

### A simple example: Write a point cloud 

```python
import numpy as np
from zarr_vectors.types.points import write_points, read_points

# 10,000 random 3D points with an intensity attribute
rng = np.random.default_rng(42)
positions = rng.uniform(0, 1000, size=(10_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, size=10_000).astype(np.float32)

# Write to a ZVF store with 200×200×200 spatial chunks
write_points(
    "my_points.zarr",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    attributes={"intensity": intensity},
)
```

### Read it back

```python
# Read all points
result = read_points("my_points.zarr")
print(result["vertex_count"])        # 10000
print(result["positions"].shape)     # (10000, 3)

# Read only points within a bounding box
result = read_points(
    "my_points.zarr",
    bbox=(np.array([100, 100, 100]), np.array([300, 300, 300])),
)
print(result["vertex_count"])        # ~roughly 800 points
```

### Points with object identity and groups (i.e. spatial transcriptomics)

```python
# Spatial transcriptomics: each point is a cell, cells grouped by type
cell_positions = rng.uniform(0, 500, size=(5000, 3)).astype(np.float32)
cell_types = rng.integers(0, 3, size=5000).astype(np.int64)  # 3 cell types

# Gene expression: 100 genes per cell
gene_expr = rng.uniform(0, 10, size=(5000, 100)).astype(np.float32)

# Each cell is its own object
object_ids = np.arange(5000, dtype=np.int64)

# Group cells by type
groups = {
    0: np.where(cell_types == 0)[0].tolist(),  # glutamatergic
    1: np.where(cell_types == 1)[0].tolist(),  # GABAergic
    2: np.where(cell_types == 2)[0].tolist(),  # glial
}

group_names = np.array([0, 1, 2], dtype=np.float32)  # group label IDs

write_points(
    "cells.zarr",
    cell_positions,
    chunk_shape=(100.0, 100.0, 100.0),
    attributes={"gene_expression": gene_expr},
    object_ids=object_ids,
    groups=groups,
    group_attributes={"type_id": group_names},
)

# Read all cells of one type
result = read_points("cells.zarr", group_ids=[1])
print(f"GABAergic cells: {result['vertex_count']}")

# Read a single cell by object ID
result = read_points("cells.zarr", object_ids=[42])
```

### Ingest from CSV (i.e. graph network)

```python
from zarr_vectors.ingest.csv_points import ingest_csv

ingest_csv(
    "measurements.csv",           # x, y, z, temperature, pressure
    "measurements.zarr",
    chunk_shape=(50.0, 50.0, 50.0),
    position_columns=["x", "y", "z"],
    attribute_columns=["temperature", "pressure"],
)
```

### Export to CSV

```python
from zarr_vectors.export.csv_points import export_csv

export_csv(
    "measurements.zarr",
    "output.csv",
    bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
)
```

## Format Overview

A zarr vectors store is a directory tree built on ragged zarr arrays:

```
dataset.zarr/
├── .zattrs                          # root metadata: SID, CRS, conventions
├── resolution_0/                    # full resolution
│   ├── vertices/                    # spatial positions (SID-shaped, ragged)
│   ├── vertex_group_offsets/        # byte offsets for sub-chunk access
│   ├── links/                       # connectivity (edges, faces, parents)
│   ├── attributes/                  # per-vertex data (gene expression, etc.)
│   │   ├── intensity/
│   │   └── gene_expression/
│   ├── object_index/                # object ID → (chunk, vertex_group) mapping
│   ├── object_attributes/           # per-object metadata
│   ├── groupings/                   # group ID → [object IDs]
│   ├── groupings_attributes/        # per-group metadata
│   └── cross_chunk_links/           # connectivity across chunk boundaries
├── resolution_1/                    # coarsened (adaptive threshold)
│   └── [same structure]
├── parametric/                      # algebraic objects (planes, lines)
│   ├── objects/
│   └── object_attributes/
└── metadata.json
```

