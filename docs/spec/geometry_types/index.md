# Geometry types

ZVF defines seven geometry types. Each type is identified by a string
constant stored in the root `.zattrs` under `"geometry_type"`. A single
ZVF store holds exactly one geometry type; multi-type datasets require
multiple stores.

## Type constants

```python
from zarr_vectors.constants import (
    GEOM_POINT_CLOUD,   # "point_cloud"
    GEOM_LINE,          # "line"
    GEOM_POLYLINE,      # "polyline"
    GEOM_STREAMLINE,    # "streamline"
    GEOM_GRAPH,         # "graph"
    GEOM_SKELETON,      # "skeleton"
    GEOM_MESH,          # "mesh"
)
```

## Comparison matrix

| Property | `point_cloud` | `line` | `polyline` | `streamline` | `graph` | `skeleton` | `mesh` |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Discrete objects | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `links/edges/` | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `links/faces/` | — | — | — | — | — | — | ✓ |
| `object_index/` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cross_chunk_links/` | — | — | ✓ | ✓ | ✓ | ✓ | — |
| `object_attributes/` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `groupings/` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Per-vertex attributes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Object sparsity | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multiscale pyramid | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Draco compression | — | — | — | — | — | — | ✓ |
| `is_directed` flag | — | — | — | — | ✓ | — | — |
| `is_tree` flag | — | — | — | — | ✓ | ✓ | — |

## Type selection guide

| Data description | Recommended type |
|-----------------|-----------------|
| Lidar, synchrotron point scan, single-molecule localisation | `point_cloud` |
| Pairs of connected points (short segments, contact sites) | `line` |
| Ordered vertex sequences without tractography metadata | `polyline` |
| MRI/synchrotron tractography streamlines with step size / seeding metadata | `streamline` |
| General connectivity graph (not necessarily a tree) | `graph` |
| Neuronal morphology, vascular tree (tree topology, SWC-compatible) | `skeleton` |
| 3-D surface mesh (cell boundary, brain surface, organelle hull) | `mesh` |

## Per-type documentation

```{toctree}
:maxdepth: 1

point_cloud
line
polyline
streamline
graph
skeleton
mesh
```
