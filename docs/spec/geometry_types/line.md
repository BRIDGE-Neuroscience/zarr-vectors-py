# Line (`line`)

## Terms

**Line segment**
: A pair of vertices `(A, B)` connected by a straight edge. A `line` store
  holds an unordered collection of such pairs. Unlike polylines, there is
  no sequential ordering between segments; each segment is independent.

**`GEOM_LINE`**
: The geometry type constant `"line"`. Stored in root `.zattrs` under
  `"geometry_type"`.

**`links/edges/`**
: The array storing pairs of local-chunk vertex indices for each line
  segment in a chunk. Shape `(E, 2)` int32 per chunk, where `E` is the
  number of segments in the chunk. Both endpoints of a segment must lie
  in the same chunk (cross-chunk segments are not supported for this type).

---

## Introduction

The `line` type stores independent line segments: pairs of vertices with
no sequential relationship between segments. It is appropriate for contact
sites (two endpoints of a synapse), short connectors in circuit diagrams,
pairwise distance annotations, or any other dataset where the fundamental
unit is a two-point segment rather than an ordered path.

`line` is simpler than `polyline` because segments do not span chunks
(no `cross_chunk_links/` needed) and there is no object model
(no `object_index/`). It is therefore faster to write and read than
`polyline` for data that genuinely consists of independent segments.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Endpoint positions, shape `(N, D)` float32 per chunk |
| `vertex_group_offsets/` | Yes | VG index |
| `links/edges/` | Yes | Segment pairs, shape `(E, 2)` int32 per chunk |
| `attributes/<name>/` | No | Per-vertex attributes |

No `object_index/`, `cross_chunk_links/`, or `object_attributes/` arrays.

### Constraint: both endpoints in the same chunk

A line segment's two vertices must fall within the same ZVF spatial chunk.
Segments that cross a chunk boundary are not supported by this type; use
`polyline` with `cross_chunk_links` for such data.

At write time, `write_lines` raises `ValueError` if any segment has
endpoints in different chunks. To handle cross-chunk segments, pass
`split_cross_chunk=True`, which splits the segment at the chunk boundary
and inserts a midpoint vertex; this changes the geometry slightly and is
not always appropriate.

### Vertex ordering in `links/edges/`

Edge indices are local to the chunk (0-indexed within the chunk's vertex
slice). An edge `[i, j]` in chunk `(cx, cy, cz)` refers to vertices at
positions `vertices[cx, cy, cz, i]` and `vertices[cx, cy, cz, j]`.

Edges are stored in the same VG order as their source vertex (the vertex
with lower array index): edges whose first endpoint is in bin 0 come first,
then bin 1, etc.

### Write API

```python
import numpy as np
from zarr_vectors.types.lines import write_lines

# n_segments × 2 × D array: each row is a segment, columns are endpoints
segments = np.random.default_rng(0).uniform(0, 500, (10_000, 2, 3)).astype(np.float32)

# Reshape to flat vertices and edge pairs
n = len(segments)
verts = segments.reshape(n * 2, 3)            # (2n, D)
edges = np.column_stack([                      # (n, 2)
    np.arange(0, 2*n, 2),
    np.arange(1, 2*n, 2),
]).astype(np.int32)

write_lines(
    "contacts.zarrvectors",
    vertices=verts,
    edges=edges,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
)
```

Alternatively, pass a list of `(A, B)` tuples:

```python
from zarr_vectors.types.lines import write_line_pairs

pairs = [(rng.uniform(0, 500, 3), rng.uniform(0, 500, 3)) for _ in range(10_000)]
write_line_pairs("contacts.zarrvectors", pairs,
                 chunk_shape=(200., 200., 200.))
```

### Read API

```python
from zarr_vectors.types.lines import read_lines

result = read_lines("contacts.zarrvectors")
print(result["segment_count"])       # int
print(result["vertices"].shape)      # (2N, D)
print(result["edges"].shape)         # (N, 2)

# Return as (N, 2, D) array of segment endpoint pairs
result = read_lines("contacts.zarrvectors", return_pairs=True)
print(result["pairs"].shape)         # (N, 2, D)

# Spatial query
result = read_lines(
    "contacts.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([200., 200., 200.])),
)
```

### Relationship to `polyline`

`line` and `polyline` share the `links/edges/` array schema. The
distinction is:

| Property | `line` | `polyline` |
|----------|--------|-----------|
| Segment order | Independent | Sequential (ordered path) |
| Cross-chunk segments | Not supported | Supported via `cross_chunk_links/` |
| Object index | No | Yes |
| Object model | No | Yes |

If your data has ordered paths (e.g. vessel centrelines from which individual
segment pairs were extracted), `polyline` preserves the ordering and enables
object-level queries. Use `line` only for genuinely unordered, independent
segment pairs.

### Validation

L1: `vertices/`, `vertex_group_offsets/`, `links/edges/` exist.

L2: No `object_index/` or `cross_chunk_links/` present.

L3:
- For every chunk, all edge vertex indices are in `[0, N_chunk)` where
  `N_chunk` is the vertex count in that chunk.
- No edge has both indices equal (no self-loops).
