# Graph (`graph`)

## Terms

**Graph**
: A collection of vertices (nodes) and edges connecting pairs of vertices.
  Edges may be directed or undirected. The graph may be disconnected
  (multiple connected components). Cycles are permitted.

**`GEOM_GRAPH`**
: The geometry type constant `"graph"`.

**`is_directed`**
: A boolean flag in root `.zattrs` indicating whether edges have direction.
  When `true`, edge `[i, j]` represents a directed connection from vertex
  `i` to vertex `j`; the reverse is not implied. When `false` (default),
  edges are undirected.

**`is_tree`**
: A boolean flag in root `.zattrs` indicating that the graph is a tree
  (connected, acyclic, exactly `n_vertices - 1` edges). When `true`, the
  store may omit edges that can be inferred from the parent-child
  relationship stored in `links/edges/`. Enabling `is_tree` also enables
  tree-specific validation (cycle detection, connectivity check).

**Root vertex**
: For tree graphs (`is_tree = true`), the root is the vertex with no parent.
  Its entry in `links/edges/` has the parent index set to `-1`.

---

## Introduction

The `graph` type stores an arbitrary vertex–edge graph, spatially chunked
like all other ZVF types. It is appropriate for connectivity data that does
not fit the stricter topology of `skeleton` (which requires a tree): vascular
networks with anastomoses, synaptic connectivity matrices embedded in 3-D
space, or any general graph with cycles.

`graph` and `skeleton` share the same underlying array schema. The
distinction is semantic and enforced by metadata flags and validation:
`skeleton` enforces tree topology and aligns to the SWC convention;
`graph` is unconstrained.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Node positions |
| `vertex_group_offsets/` | Yes | VG index |
| `links/edges/` | Yes | Vertex pairs; shape `(E, 2)` int32 per chunk |
| `object_index/` | Yes | Object (component) ID → primary chunk + VG offset |
| `cross_chunk_links/` | Yes* | Inter-chunk edges |
| `attributes/<name>/` | No | Per-vertex attributes |
| `object_attributes/<name>/` | No | Per-component attributes |
| `groupings/` | No | Group assignment |

*Required when any edge connects vertices in different chunks.

### Root `.zattrs` type-specific keys

```json
{
  "geometry_type": "graph",
  "is_directed":   false,
  "is_tree":       false
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `is_directed` | `bool` | `false` | Whether edges are directed. |
| `is_tree` | `bool` | `false` | Whether graph topology is a tree. Enables tree validation. |

### Edge encoding

Edges in `links/edges/` are local-chunk vertex index pairs `[i, j]`. For
undirected graphs, each edge is stored once in canonical form `[min(i,j),
max(i,j)]`; readers must treat `[i,j]` and `[j,i]` as the same edge.

For directed graphs, edges are stored in `[source, destination]` order.
The direction is significant; readers must not reverse edges.

Cross-chunk edges (edges whose two endpoints are in different chunks) are
stored in `cross_chunk_links/`. Each entry is a pair of global vertex IDs.
See [Cross-chunk links](../object_model/cross_chunk_links.md).

### Object model for graphs

Each *connected component* of the graph is treated as one object, identified
by an integer object ID. The `object_index/` maps each component's ID to
its primary chunk and VG offset.

For single-component graphs (a common case), there is exactly one object
(object ID 0). For multi-component graphs (e.g. a store containing many
disconnected subgraphs), each component has its own ID.

### Write API

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes = 5000
positions = rng.uniform(0, 1000, (n_nodes, 3)).astype(np.float32)

# Random sparse graph: ~3 edges per node
src = rng.integers(0, n_nodes, 7500)
dst = rng.integers(0, n_nodes, 7500)
edges = np.column_stack([src, dst]).astype(np.int32)

write_graph(
    "network.zarrvectors",
    positions=positions,
    edges=edges,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    is_directed=False,
    is_tree=False,
)
```

### Write API — tree mode

```python
write_graph(
    "tree.zarrvectors",
    positions=positions,
    edges=edges,         # (n-1, 2) parent→child pairs
    chunk_shape=(200., 200., 200.),
    is_tree=True,        # validates tree topology at write time
)
```

With `is_tree=True`, `write_graph` validates that:
- The graph is connected.
- The graph is acyclic.
- Exactly one vertex has no parent (the root).

### Read API

```python
from zarr_vectors.types.graphs import read_graph

result = read_graph("network.zarrvectors")
print(result["node_count"])           # int
print(result["edge_count"])           # int
print(result["positions"].shape)      # (N, D)
print(result["edges"].shape)          # (E, 2)

# Single component
result = read_graph("network.zarrvectors", object_ids=[0])

# Spatial bbox
result = read_graph(
    "network.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([200., 200., 200.])),
)
# Returns all nodes in bbox; edges where both endpoints are in bbox.
# Use include_boundary_edges=True to include edges crossing the bbox boundary.
```

### Multi-graph stores

A single `graph` store may contain many disconnected components
(e.g. one per cell in a connectome). Each component is one object. Read
individual components by object ID:

```python
result = read_graph("connectome.zarrvectors", object_ids=[42, 107, 318])
```

### Validation

L1: `vertices/`, `vertex_group_offsets/`, `links/edges/`, `object_index/`
exist.

L2:
- `is_directed` is a boolean.
- `is_tree` is a boolean.

L3:
- All edge vertex indices are in `[0, N_chunk)`.
- No self-loops: `edges[i,0] != edges[i,1]` for all `i`.
- For undirected graphs: no duplicate edges (both `[i,j]` and `[j,i]`).
- `cross_chunk_links/` entries reference valid global vertex IDs.

L4 (if `is_tree = true`):
- Graph is connected (single component or each declared component is
  individually connected).
- Graph is acyclic.
- Exactly one vertex per component has parent index `-1` (the root).
- Number of edges equals `n_vertices - n_components`.
