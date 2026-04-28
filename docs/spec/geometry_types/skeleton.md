# Skeleton (`skeleton`)

## Terms

**Skeleton**
: A tree-structured graph representing the branching morphology of a
  biological structure — typically a neuron, blood vessel, or other
  tubular object. Skeletons are acyclic and connected; each vertex has
  exactly one parent except the root, which has none.

**`GEOM_SKELETON`**
: The geometry type constant `"skeleton"`.

**SWC convention**
: The de facto standard file format for neuronal morphology, defined by
  Cannon et al. (1998). An SWC file stores one vertex per row with columns:
  `id, type, x, y, z, radius, parent_id`. ZVF skeletons follow the SWC
  vertex type taxonomy and store `radius` and `type` as per-vertex
  attributes.

**Vertex type**
: An integer label from the SWC taxonomy identifying the anatomical role
  of a vertex: 1 = undefined, 2 = soma, 3 = axon, 4 = basal dendrite,
  5 = apical dendrite, 6 = custom, 7 = custom. Stored as per-vertex
  attribute `"swc_type"` (int32).

**Radius**
: The estimated radius of the tubular structure at a given vertex.
  Stored as per-vertex attribute `"radius"` (float32). Unit follows
  `axis_units` from root `.zattrs`.

**Root vertex**
: The vertex with no parent, representing the origin of the tree (typically
  the soma centre for a neuron). Identified by a parent edge index of `-1`.

---

## Introduction

The `skeleton` type is a specialisation of `graph` with `is_tree = true`,
aligned to the SWC morphology convention. It is the appropriate type for
neuronal morphologies, vascular trees, and any other branching tree-
structured shape that will be read by or compared to SWC-compatible tools.

ZVF skeletons store all the information present in an SWC file (position,
radius, type, parent relationship) plus the spatial indexing and multi-
resolution features of ZVF. The `ingest_swc` and `export_swc` functions
provide lossless round-tripping between SWC and ZVF for standard SWC files.

---

## Technical reference

### Arrays present

Identical to `graph` with `is_tree = true`.

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Node positions (x, y, z) |
| `vertex_group_offsets/` | Yes | VG index |
| `links/edges/` | Yes | Parent–child pairs `[child, parent]`; parent = `-1` for root |
| `object_index/` | Yes | One entry per skeleton (one component = one object) |
| `cross_chunk_links/` | Yes* | Edges crossing chunk boundaries |
| `attributes/swc_type/` | Recommended | int32 per vertex: SWC compartment type |
| `attributes/radius/` | Recommended | float32 per vertex: estimated radius |
| `attributes/swc_id/` | Optional | int64 per vertex: original SWC row ID (for round-trips) |
| `object_attributes/<name>/` | No | Per-skeleton attributes |

### Root `.zattrs` type-specific keys

```json
{
  "geometry_type":  "skeleton",
  "is_tree":        true,
  "swc_compatible": true
}
```

| Key | Type | Description |
|-----|------|-------------|
| `is_tree` | `bool` | Must be `true` for skeleton type. |
| `swc_compatible` | `bool` | When `true`, `swc_type` and `radius` attributes use standard SWC conventions and the store can be exported to SWC without loss. |

### `links/edges/` encoding for skeletons

Each row is `[child_local_idx, parent_local_idx]` where both indices are
local to the chunk. For the root vertex, `parent_local_idx = -1`.

For intra-chunk parent–child pairs this is stored directly. For cross-
chunk parent–child relationships (a child in chunk A whose parent is in
chunk B), the edge is stored in `cross_chunk_links/` as a global vertex
ID pair `[child_global_id, parent_global_id]`.

This is a key distinction from `polyline` cross-chunk links, where the
link direction encodes traversal order. For skeletons, the link direction
encodes the child→parent relationship in the tree.

### SWC type taxonomy

| Code | Anatomical meaning |
|------|--------------------|
| 0 | Undefined |
| 1 | Soma |
| 2 | Axon |
| 3 | Basal dendrite |
| 4 | Apical dendrite |
| 5 | Custom (fork point) |
| 6 | Custom (end point) |
| 7 | Custom (unspecified) |

Note: some SWC conventions use `1` for soma and `2` for axon. The mapping
above follows the NeuroMorpho / SWC+ convention. The `ingest_swc` function
detects the convention from the input file header.

### Ingest and export

```python
from zarr_vectors.ingest.swc import ingest_swc
from zarr_vectors.export.swc import export_swc

# Ingest
ingest_swc(
    "neuron.swc",
    "neuron.zarrvectors",
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
)

# Export — lossless if swc_compatible = true
export_swc("neuron.zarrvectors", "neuron_out.swc")
```

For multi-skeleton stores (many neurons in one ZVF store):

```python
from zarr_vectors.ingest.swc import ingest_swc_directory

ingest_swc_directory(
    "morphologies/",          # directory of .swc files
    "neurons.zarrvectors",
    chunk_shape=(500., 500., 500.),
)
```

### Write API

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

# positions: (n_nodes, 3), edges: (n_nodes-1, 2) [child, parent]; root has parent=-1
write_graph(
    "neuron.zarrvectors",
    positions=node_positions,
    edges=parent_child_pairs,
    chunk_shape=(200., 200., 200.),
    geometry_type="skeleton",
    is_tree=True,
    vertex_attributes={
        "radius":   radii,       # float32
        "swc_type": swc_types,   # int32
    },
)
```

### Multi-skeleton stores

For connectome-scale datasets with thousands of neurons in a shared
coordinate space, a single `skeleton` store is more efficient than
separate per-neuron files:

```python
from zarr_vectors.types.graphs import read_graph

# Read one skeleton by ID
result = read_graph("connectome.zarrvectors", object_ids=[1042])
print(result["node_count"])    # nodes in skeleton 1042
print(result["attributes"]["radius"])

# Spatial query — returns all skeletons with nodes in the region
result = read_graph(
    "connectome.zarrvectors",
    bbox=(np.array([1000., 2000., 500.]),
          np.array([1200., 2200., 700.])),
)
```

### Validation

L1, L2, L3: Same as `graph` with `is_tree = true`.

L4 (skeleton-specific):
- Each skeleton (connected component / object) is a valid tree:
  connected, acyclic, exactly one root vertex (parent = -1).
- If `swc_compatible = true`: `swc_type` attribute exists and all values
  are in `[0, 7]`. `radius` attribute exists and all values are ≥ 0.

L3 (cross-chunk links specific to skeleton):
- All cross-chunk links reference valid global vertex IDs.
- The direction `[child, parent]` is consistent with the tree structure
  (the child's depth is one greater than the parent's depth).
