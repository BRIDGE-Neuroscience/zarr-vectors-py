
> [!NOTE]
> This package is under development and will change. It will also be migrated to another location once completed.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />


**Tools for Zarr Vectors Data**

`zarr-vectors-py` is a Python package for reading, writing, and managing large-scale vector geometry data in the zarr vectors format — a chunked format built to mirror Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons, and meshes.

*Aligned to the Zarr_vectors specification by Forest Collman, Allen Institute for Brain Sciences*
[Link to specification GitHub](https://github.com/AllenInstitute/zarr_vectors)

## Install

```bash
pip install zarr-vectors
```

## Quick Start

### Example 1: A simple point cloud 

```python
import numpy as np
from zarr_vectors.types.points import write_points, read_points

# 10,000 random 3D points with an intensity attribute
rng = np.random.default_rng(42)
positions = rng.uniform(0, 1000, size=(10_000, 3)).astype(np.float32)   # spatial data 
intensity = rng.uniform(0, 1, size=10_000).astype(np.float32)           # attributes data 

# Write to a zarr vectors store with 20×20×20 spatial chunks
write_points(
    "random_points.zarrvectors",
    positions,
    chunk_shape=(20.0, 20.0, 20.0),
    attributes={"intensity": intensity},
)
```

Read it back:

```python
# Read all points
result = read_points("my_points.zarrvectors")
print(result["vertex_count"])        # 10000
print(result["positions"].shape)     # (10000, 3)

# Read only points within a bounding box
result = read_points(
    "my_points.zarrvectors",
    bbox=(np.array([100, 100, 100]), np.array([300, 300, 300])),
)
print(result["vertex_count"])        # ~roughly 800 points
```


### Example 2: A graph network (.csv)

```python
from zarr_vectors.ingest.csv_points import ingest_csv

ingest_csv(
    "measurements.csv",           # x, y, z, temperature, pressure
    "measurements.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    position_columns=["x", "y", "z"],
    attribute_columns=["temperature", "pressure"],
)
```

Export back to .csv

```python
from zarr_vectors.export.csv_points import export_csv

export_csv(
    "measurements.zarrvectors",
    "output.csv",
    bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
)
```

## Format Overview

A zarr vectors store is a directory tree built on ragged zarr arrays:

```
dataset.zarrvectors/
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

