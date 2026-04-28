# Cross-chunk links

## Terms

**Cross-chunk link**
: A stored connection between two vertices in different spatial chunks.
  For polylines and streamlines, a link connects the last vertex of a
  segment in chunk A to the first vertex of the continuation in chunk B.
  For graphs and skeletons, a link connects any two vertices in different
  chunks that share an edge.

**Global vertex ID**
: A store-wide unique integer identifier for a vertex, encoding both its
  chunk location and its local index within the chunk:
  `global_id = chunk_flat × N_max + local_index`.

**`cross_chunk_links/`**
: The Zarr array storing all cross-chunk links as pairs of global vertex
  IDs. Shape `(n_links, 2)` int64. Each row `[src_global, dst_global]`
  represents one directed link from the source vertex to the destination.

**Link direction**
: For polylines/streamlines: `src` is the last vertex of the departing
  segment; `dst` is the first vertex of the arriving segment (traversal
  direction). For graphs/skeletons: `src` is the child and `dst` is the
  parent (tree direction), or either endpoint for undirected graphs.

**Link map**
: An in-memory hash table built from `cross_chunk_links/` that maps
  `src_global_id → dst_global_id`. Used during manifest reconstruction.

---

## Introduction

When a polyline or streamline crosses a chunk boundary, its vertex
sequence is split: some vertices are in chunk A, the rest are in chunk B
(or further chunks). The `links/edges/` array within each chunk stores
only intra-chunk edges; the connection across the boundary must be stored
separately.

`cross_chunk_links/` fills this role. It is a flat array of global vertex
ID pairs; each pair records a single cross-chunk connection. The array is
global to the store (not per-chunk), which allows efficient bulk lookup.

Understanding the generation algorithm is important for contributors: the
rules for when a link is generated (between different chunks, not between
different bins in the same chunk) and how global vertex IDs are computed
are both non-obvious and error-prone.

---

## Technical reference

### Global vertex ID encoding

```
global_id = chunk_flat_index × N_max + local_vertex_index
```

where:
- `chunk_flat_index = ravel_multi_index(chunk_coord, chunk_grid_shape)`
- `local_vertex_index` is the vertex's position within the VG-sorted
  `vertices/` chunk slice (0-indexed)
- `N_max` is the maximum vertex count per chunk as declared in the Zarr
  array shape

**Example:** in a store with `chunk_grid_shape = (5, 5, 5)` and `N_max =
65536`, a vertex at local index 1024 in chunk `(2, 3, 1)`:

```python
chunk_flat = np.ravel_multi_index((2, 3, 1), (5, 5, 5))  # = 2*25 + 3*5 + 1 = 66
global_id  = 66 * 65536 + 1024  # = 4326400
```

**Decoding** a global ID:

```python
chunk_flat = global_id // N_max
local_idx  = global_id %  N_max
chunk_coord = np.unravel_index(chunk_flat, chunk_grid_shape)
```

### `cross_chunk_links/` array schema

```
path:        resolution_<N>/cross_chunk_links/
shape:       (n_links, 2)
dtype:       int64
chunk_shape: (65536, 2)
fill_value:  -1
```

Row `k`: `[src_global_id, dst_global_id]` representing a directed link
from vertex `src_global_id` to vertex `dst_global_id`.

Links are stored in no guaranteed order. Readers must build an in-memory
lookup table (link map) for efficient access.

### Generation algorithm for polylines and streamlines

A cross-chunk link is generated when consecutive vertices in a polyline
(vertex `v_i` followed by `v_{i+1}`) are in **different chunks**, i.e.
their chunk coordinates differ in at least one axis:

```python
def generate_cross_chunk_links(polylines, chunk_shape, chunk_grid_shape, N_max):
    links = []
    for polyline in polylines:
        for i in range(len(polyline) - 1):
            coord_a = floor(polyline[i]   / chunk_shape).astype(int)
            coord_b = floor(polyline[i+1] / chunk_shape).astype(int)
            if not np.array_equal(coord_a, coord_b):
                # Different chunks — generate a cross-chunk link
                flat_a  = ravel(coord_a, chunk_grid_shape)
                flat_b  = ravel(coord_b, chunk_grid_shape)
                local_a = local_index_of(polyline[i],   flat_a, ...)
                local_b = local_index_of(polyline[i+1], flat_b, ...)
                src_global = flat_a * N_max + local_a
                dst_global = flat_b * N_max + local_b
                links.append([src_global, dst_global])
    return np.array(links, dtype=np.int64)
```

**Key constraint:** a cross-chunk link is generated only when vertices
are in **different chunks**, not merely different bins within the same chunk.
Two vertices in different bins of the same chunk are connected by an intra-
chunk edge in `links/edges/`. Generating a cross-chunk link for same-chunk
vertices would violate the invariant that `cross_chunk_links/` contains only
inter-chunk connections, causing double-counting during manifest
reconstruction.

### Generation algorithm for graphs and skeletons

For undirected graphs, a cross-chunk link is generated for every edge
whose two endpoints are in different chunks:

```python
for (u, v) in edges:
    chunk_u = floor(positions[u] / chunk_shape)
    chunk_v = floor(positions[v] / chunk_shape)
    if not np.array_equal(chunk_u, chunk_v):
        # Store canonical form [min(global_u, global_v), max(...)]
        links.append(sorted([global_id(u), global_id(v)]))
```

For directed graphs and skeletons, the direction `[child, parent]` is
preserved.

### Link resolution during manifest reconstruction

During `read_object_vertices`, the link map is built once from the full
`cross_chunk_links/` array and reused for all objects:

```python
link_map = {}
for src, dst in cross_chunk_links:
    link_map[src] = dst
```

For large stores with millions of streamlines, `cross_chunk_links/` may
contain hundreds of millions of entries. `zarr-vectors-py` uses a lazy
chunked read and builds the link map incrementally.

### Link count estimation

The expected number of cross-chunk links for a polyline store depends on
the mean streamline length and the chunk size:

```
expected_links ≈ n_streamlines × mean_vertices_per_streamline
                 × p(consecutive_pair_crosses_boundary)
```

where `p(crossing) ≈ 1 - exp(-mean_segment_length / harmonic_mean_chunk_edge_length)`.

For typical DWI tractography (step size 0.5 mm, chunk size 50 mm, 80 vertices
per streamline):

```
p(crossing) ≈ 80 × 0.5/50 = 0.8 crossings per streamline on average
```

For a 1M-streamline dataset: ~800 000 cross-chunk links. At 16 bytes per
link, this is ~13 MB — negligible.

### Validation

L1: `cross_chunk_links/` exists for polyline, streamline, graph, skeleton
types where any object spans multiple chunks. May be absent only if all
objects are confined to single chunks.

L3:
- All global vertex IDs are in `[0, total_vertices_in_store)`.
- `src` and `dst` in each link are in different chunks
  (`src // N_max != dst // N_max`).
- For undirected graphs: no duplicate links (both `[a,b]` and `[b,a]`).
- For polylines/streamlines: the link graph forms a set of disjoint paths
  (no vertex has more than one outgoing link and more than one incoming
  link in the link map).

L4:
- Every object whose primary VG is not the only VG in the store has at
  least one cross-chunk link.
- Following cross-chunk links from every object's primary VG reaches all
  vertices declared in `object_index/`.
