# Chunk arrays

## Terms

**Chunk array**
: A Zarr array within a resolution level group whose logical shape is
  expressed in terms of the chunk grid. The first `D` dimensions of a chunk
  array's shape correspond to the chunk grid dimensions; later dimensions
  carry per-chunk payload data (vertex coordinates, edge indices, etc.).

**Ragged array**
: An array where the size of the payload dimension varies per chunk. For
  example, `vertices/` has shape `(Cx, Cy, Cz, N_max, D)` in the Zarr
  metadata, but most chunks contain fewer than `N_max` vertices. Unused
  positions are filled with the array's fill value.

**`N_max`**
: The maximum number of vertices (or edges, faces) per chunk as declared in
  the Zarr array's `shape`. This is a soft upper bound; `zarr-vectors-py`
  resizes the array as needed when writing.

**Chunk grid shape**
: The number of chunks along each spatial axis. For a store with spatial
  extent `[E_0, E_1, …, E_{D-1}]` and `chunk_shape = [C_0, …, C_{D-1}]`,
  the chunk grid shape is `[ceil(E_i / C_i) for i in range(D)]`.

---

## Introduction

Every vertex, edge, face, and attribute value in a ZVF store is stored in a
Zarr array whose first dimensions index the chunk grid. Within a given chunk,
data is stored as a dense payload slice. Because different chunks contain
different numbers of vertices, the payload dimension is ragged: the Zarr
array declares a fixed maximum, but only the used portion is written (the
rest is fill-value padded or absent if the chunk is empty).

This page documents the dtype, shape, chunk grid, and fill value for every
array defined by the ZVF spec, for each geometry type.

---

## Technical reference

### `vertices/`

Stores the spatial positions of all vertices in the store, one Zarr chunk
per spatial chunk.

| Property | Value |
|----------|-------|
| Dtype | `float32` |
| Logical shape | `(*chunk_grid_shape, N_max, D)` |
| Zarr chunk shape | `(1, 1, …, 1, N_chunk_max, D)` — one Zarr chunk per spatial chunk |
| Fill value | `0.0` |
| Codec | `bytes → blosc(zstd, bitshuffle)` (default) |

`N_max` is set conservatively at write time based on the expected vertex
density and chunk volume; the array is resized if a chunk exceeds the
declared maximum.

Within each spatial chunk the vertices are stored in **VG order**: all
vertices of bin (0,0,0) first, then bin (0,0,1), etc., in C-order bin
index. The `vertex_group_offsets/` array encodes the start and length of
each bin's vertices within this ordering (see [VG index arrays](vg_index_arrays.md)).

**Example:** a 3-D store with a chunk grid of shape `(5, 6, 4)` and up to
65 536 vertices per chunk:

```json
{
  "shape": [5, 6, 4, 65536, 3],
  "data_type": "float32",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1, 1, 1, 65536, 3]}
  },
  "fill_value": 0.0
}
```

### `vertex_group_offsets/`

Stores the byte offset and length of each VG within the corresponding
`vertices/` chunk slice. See [VG index arrays](vg_index_arrays.md) for a
detailed description of the VG index structure.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(*chunk_grid_shape, B_per_chunk, 2)` |
| Zarr chunk shape | `(1, 1, …, 1, B_per_chunk, 2)` |
| Fill value | `-1` (indicates empty bin) |
| Codec | `bytes → blosc(zstd, byteshuffle)` |

`B_per_chunk = product(chunk_shape[i] / bin_shape[i] for i in range(D))`.

The two values per bin are `[offset, count]` where `offset` is the index
of the first vertex of this bin within the chunk's vertex slice, and
`count` is the number of vertices in this bin.

### `links/edges/`

Present for: polyline, streamline, graph, skeleton.

Stores pairs of vertex indices representing graph edges or polyline segment
connections, within a single chunk.

| Property | Value |
|----------|-------|
| Dtype | `int32` |
| Logical shape | `(*chunk_grid_shape, E_max, 2)` |
| Zarr chunk shape | `(1, 1, …, 1, E_max, 2)` |
| Fill value | `-1` |
| Codec | `bytes → blosc(zstd, byteshuffle)` |

Vertex indices in `edges/` are **local to the chunk**: index `k` refers to
the `k`-th vertex in the `vertices/` chunk slice. Inter-chunk connections are
stored separately in `cross_chunk_links/`.

For polylines and streamlines, edges are stored in traversal order: edge `i`
connects vertex `i` to vertex `i+1` along the polyline. For graphs and
skeletons, edge order is not semantically significant.

### `links/faces/`

Present for: mesh only.

Stores triangular face definitions as triplets of vertex indices, local to
the chunk.

| Property | Value |
|----------|-------|
| Dtype | `int32` |
| Logical shape | `(*chunk_grid_shape, F_max, 3)` |
| Zarr chunk shape | `(1, 1, …, 1, F_max, 3)` |
| Fill value | `-1` |
| Codec | `bytes → blosc(zstd, byteshuffle)` or `draco` |

Vertex winding order is consistent within a store (default: counter-clockwise
when viewed from outside the surface, i.e. outward-facing normals). The
winding order convention is stored in root `.zattrs` under `"winding_order"`:
`"ccw"` (default) or `"cw"`.

### `attributes/<name>/`

One sub-group per named per-vertex attribute. The attribute name is a
valid Python identifier (alphanumeric and underscores only).

| Property | Value |
|----------|-------|
| Dtype | Any numeric dtype declared in `zarr.json` |
| Logical shape | `(*chunk_grid_shape, N_max)` for scalar attributes; `(*chunk_grid_shape, N_max, K)` for vector attributes of width K |
| Zarr chunk shape | `(1, 1, …, 1, N_max)` or `(1, 1, …, 1, N_max, K)` |
| Fill value | `0` or `NaN` (declared per array) |
| Codec | Varies; default is `bytes → blosc(zstd, bitshuffle)` |

The vertex ordering within an attribute chunk must match the vertex ordering
in the corresponding `vertices/` chunk exactly. That is, attribute value `k`
in chunk `(i,j,l)` of `attributes/intensity/` corresponds to vertex `k` in
chunk `(i,j,l)` of `vertices/`.

### `object_index/`

Present for: polyline, streamline, graph, skeleton, mesh.

Maps each object ID to a `(chunk_flat_index, vg_offset)` pair identifying
where the object's first VG is located.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(n_objects, 2)` |
| Zarr chunk shape | `(65536, 2)` |
| Fill value | `-1` |
| Codec | `bytes → blosc(zstd, byteshuffle)` |

`chunk_flat_index` is the C-order flat index of the object's primary chunk in
the chunk grid. `vg_offset` is the index of the first VG belonging to this
object within that chunk's `vertex_group_offsets/` slice.

See [Object manifest](../object_model/object_manifest.md) for a detailed
description of the object indexing mechanism.

### `object_attributes/<name>/`

One sub-group per named per-object attribute. Shape `(n_objects,)` for
scalar attributes.

| Property | Value |
|----------|-------|
| Dtype | Any numeric dtype |
| Logical shape | `(n_objects,)` or `(n_objects, K)` |
| Zarr chunk shape | `(65536,)` or `(65536, K)` |
| Fill value | `0` or `NaN` |

### `groupings/`

Maps group IDs to lists of object IDs.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(n_groups, max_group_size)` |
| Zarr chunk shape | `(1, max_group_size)` |
| Fill value | `-1` (padding for groups smaller than `max_group_size`) |

### `cross_chunk_links/`

Present for: polyline, streamline.

Stores pairs of global vertex IDs representing connections that cross a
chunk boundary.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(n_links, 2)` |
| Zarr chunk shape | `(65536, 2)` |
| Fill value | `-1` |

Each row is a `(src_global_vertex_id, dst_global_vertex_id)` pair where
`src` is the last vertex of a polyline segment in chunk A and `dst` is the
first vertex of the continuation segment in chunk B.

See [Cross-chunk links](../object_model/cross_chunk_links.md) for the
encoding of global vertex IDs and the reconstruction algorithm.
