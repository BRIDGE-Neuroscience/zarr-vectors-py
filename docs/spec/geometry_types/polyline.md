# Polyline (`polyline`)

## Terms

**Polyline**
: An ordered sequence of vertices connected by consecutive edges:
  `V_0 — V_1 — V_2 — … — V_{n-1}`. A polyline store holds a collection
  of such sequences, each identified by an integer object ID.

**`GEOM_POLYLINE`**
: The geometry type constant `"polyline"`.

**Object ID**
: A non-negative integer uniquely identifying a polyline within the store.
  Object IDs are assigned at write time and are stable across reads.

**Object manifest**
: The set of `(chunk_coord, vg_indices)` pairs that together contain all
  vertices of a given object. Encoded in the `object_index/` array. See
  [Object manifest](../object_model/object_manifest.md).

**Cross-chunk link**
: A stored connection between the last vertex of a polyline segment in
  chunk A and the first vertex of the continuation segment in chunk B.
  Required when a polyline spans multiple chunks. See
  [Cross-chunk links](../object_model/cross_chunk_links.md).

---

## Introduction

The `polyline` type extends `line` with an object model: each polyline is
a named, addressable entity that can be retrieved by ID. Polylines may
span multiple spatial chunks; the `cross_chunk_links/` array preserves
inter-chunk connectivity.

Use `polyline` over `line` whenever you need to:
- Retrieve individual paths by ID.
- Assign paths to named groups.
- Preserve path topology across chunk boundaries.
- Store per-path metadata (length, label, confidence).

Use `streamline` instead of `polyline` when the paths are MRI tractography
streamlines and you need to store tractography-specific metadata (step size,
seeding strategy, propagation algorithm).

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions |
| `vertex_group_offsets/` | Yes | VG index |
| `links/edges/` | Yes | Intra-chunk consecutive vertex pairs |
| `object_index/` | Yes | Object ID → primary chunk + VG offset |
| `cross_chunk_links/` | Yes* | Inter-chunk vertex connections |
| `attributes/<name>/` | No | Per-vertex attributes |
| `object_attributes/<name>/` | No | Per-polyline attributes |
| `groupings/` | No | Group ID → [object IDs] |

*`cross_chunk_links/` must be present if any polyline spans more than one
chunk. It may be absent for stores where all polylines are confined to
a single chunk each.

### Vertex ordering within a chunk

For a polyline that contributes vertices to multiple bins within a chunk,
its vertices appear in multiple VGs. Within each VG, vertices are stored in
traversal order (from the start of the polyline toward the end).

Across VGs within the same chunk, vertices may be interleaved with vertices
from other polylines. The `links/edges/` array stores the consecutive pairs
for all polylines in the chunk, regardless of VG assignment.

### `links/edges/` for polylines

Each row in `links/edges/` is a pair `[i, j]` where `i` and `j` are
local-chunk vertex indices and vertex `i` immediately precedes vertex `j`
in the traversal order of some polyline. For a polyline contributing
vertices at local indices `[3, 7, 11, 4]` (in traversal order), the edges
are `[[3,7], [7,11], [11,4]]`.

Only *intra-chunk* edges are stored here. The connection between the last
vertex of a segment in chunk A and the first vertex of the continuation in
chunk B is stored in `cross_chunk_links/`.

### Write API

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines

rng = np.random.default_rng(0)
polylines = [
    rng.normal(0, 50, (rng.integers(10, 60), 3)).cumsum(0).astype(np.float32)
    for _ in range(1000)
]

write_polylines(
    "paths.zarrvectors",
    polylines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    # Optional group assignment
    groups={
        "group_A": list(range(500)),
        "group_B": list(range(500, 1000)),
    },
    # Optional per-polyline attributes
    object_attributes={
        "length": np.array([np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
                            for p in polylines], dtype=np.float32),
    },
)
```

### Read API

```python
from zarr_vectors.types.polylines import read_polylines

# All polylines
result = read_polylines("paths.zarrvectors")
print(result["polyline_count"])        # 1000
print(len(result["polylines"]))        # list of (N_i, D) arrays

# By object ID
result = read_polylines("paths.zarrvectors", object_ids=[0, 5, 42])
print(result["polyline_count"])        # 3

# By group
result = read_polylines("paths.zarrvectors", group_ids=["group_A"])
print(result["polyline_count"])        # 500

# Spatial bbox
result = read_polylines(
    "paths.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
)
# Returns all polylines that have at least one vertex in the bbox.
# Full polyline geometry is returned (not clipped to bbox).
```

### Spatial query semantics

A bbox query on a polyline store returns the complete geometry of every
polyline that has at least one vertex in the bbox. The full vertex sequence
(including portions outside the bbox) is returned. This is necessary to
preserve path continuity.

To clip polylines to the bbox (return only the vertices inside), pass
`clip=True`:

```python
result = read_polylines("paths.zarrvectors",
                        bbox=(lo, hi), clip=True)
```

Clipped polylines that cross the bbox boundary are split at the boundary;
the result may contain more polylines than the input if a single path
enters and exits the bbox multiple times.

### Groupings

Groups are named collections of object IDs. The `groupings/` array stores
the group → object mapping. Group IDs may be integers or strings (stored
as a separate string lookup table in `groupings_attributes/name/`).

### Validation

L1: `vertices/`, `vertex_group_offsets/`, `links/edges/`, `object_index/`
exist.

L3:
- Every object ID in `object_index/` is in `[0, n_objects)`.
- For each object, all referenced VGs exist and contain at least one vertex.
- `cross_chunk_links/` entries reference valid global vertex IDs.
- No polyline has a gap (a vertex with no outgoing edge unless it is the
  last vertex of the polyline).
