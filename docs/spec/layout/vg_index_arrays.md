# VG index arrays

## Terms

**Vertex group (VG)**
: The set of all vertices within a single spatial bin within a single
  spatial chunk. A VG is identified by a `(chunk_coord, bin_flat_index)`
  pair. The vertices of a VG are stored as a contiguous slice of the
  `vertices/` array for that chunk.

**VG index** (also: `vertex_group_offsets/`)
: The Zarr array that maps each `(chunk_coord, bin_flat_index)` pair to
  an `(offset, count)` pair in the `vertices/` chunk slice. This is the
  primary spatial index structure of ZVF: a bounding-box query resolves to
  a set of `(chunk_coord, bin_flat_index)` pairs, which are then looked up
  in the VG index to retrieve the exact vertex slice.

**`offset`**
: The index of the first vertex of a VG within the `vertices/` chunk slice.
  Combined with `count`, this gives the vertex slice `vertices[chunk][offset : offset + count]`.

**`count`**
: The number of vertices in a VG. A `count` of `0` indicates an empty bin
  (no vertices in this bin at this chunk). A `count` of `-1` (fill value)
  indicates a bin that was never written (the chunk is sparse).

**Bin flat index**
: The C-order (row-major) ravel of the D-dimensional bin coordinate within
  a chunk. For a chunk with `(4, 4, 4)` bins, the bin at coordinate
  `(i, j, k)` has flat index `16i + 4j + k`.

**VG order**
: The ordering of vertices within a `vertices/` chunk slice: bin (0,0,…,0)
  first, then (0,0,…,1), etc., in C-order bin index. Vertices within a
  single bin may be in any order.

---

## Introduction

The VG index is the core spatial acceleration structure of ZVF. Without it,
a bounding-box query would require loading all vertices in every overlapping
chunk and filtering client-side — equivalent to a full table scan. With it,
the query engine knows exactly which byte ranges within each chunk contain
relevant vertices and reads only those.

The index is deliberately simple: a flat array of `(offset, count)` pairs,
one per bin per chunk. This makes it fast to read and easy to implement, at
the cost of a modest storage overhead (proportional to `bins_per_chunk`, not
to the number of vertices).

This page describes the VG index encoding, the algorithm for resolving a
bounding-box query using the index, and the rules for maintaining the index
during writes.

---

## Technical reference

### Array schema

The `vertex_group_offsets/` array is a standard Zarr array with:

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(*chunk_grid_shape, B_per_chunk, 2)` |
| Zarr chunk shape | `(1, 1, …, 1, B_per_chunk, 2)` |
| Fill value | `-1` |

`B_per_chunk = product(chunk_shape[i] // bin_shape[i] for i in range(D))`.

For a 3-D store with `chunk_shape = (200, 200, 200)` and `bin_shape =
(50, 50, 50)`: `B_per_chunk = 4 × 4 × 4 = 64`.

Each chunk of `vertex_group_offsets/` corresponds one-to-one with the same
chunk in `vertices/`. The chunk at grid coordinate `(i, j, k)` in
`vertex_group_offsets/` describes the bin layout of the vertices in chunk
`(i, j, k)` of `vertices/`.

### Entry encoding

For each bin at flat index `b` within a chunk, the VG index entry is:

```
vertex_group_offsets[*chunk_coord, b, 0] = offset  # first vertex index in this bin
vertex_group_offsets[*chunk_coord, b, 1] = count   # number of vertices in this bin
```

Special values:

| `count` value | Meaning |
|---------------|---------|
| `> 0` | Bin is non-empty; `count` vertices starting at `offset` |
| `0` | Bin is empty (was written, but no vertices fell here) |
| `-1` (fill value) | Bin was never written; treat as empty |

`offset` is only meaningful when `count > 0`. When `count <= 0`, `offset`
should be ignored by readers (it may be the fill value `-1`).

### VG order invariant

The vertices within a `vertices/` chunk slice are stored in VG order. This
is a hard invariant that must be preserved by all write operations:

1. Group vertices by their bin flat index (computed from their position and
   the bin grid).
2. Sort the groups in ascending bin flat index order.
3. Concatenate the groups: bin 0 first, then bin 1, …, then bin B-1.
4. Write the result as the `vertices/` chunk slice.
5. Compute `offset` and `count` for each bin and write to
   `vertex_group_offsets/`.

A bin flat index is computed from a vertex position `p` as follows:

```python
def bin_flat_index(p, chunk_coord, chunk_shape, bin_shape):
    # Position within the chunk (local coordinates)
    local = p - np.array(chunk_coord) * np.array(chunk_shape)
    # Bin coordinate within the chunk
    bin_coord = np.floor(local / np.array(bin_shape)).astype(int)
    # Bins-per-chunk shape
    bpc_shape = tuple(int(c / b) for c, b in zip(chunk_shape, bin_shape))
    # C-order ravel
    return np.ravel_multi_index(bin_coord, bpc_shape)
```

### Bounding-box query algorithm

Given a bounding box `(bbox_min, bbox_max)`:

**Step 1 — Compute overlapping chunks.**
```python
chunk_min = np.floor(bbox_min / chunk_shape).astype(int)
chunk_max = np.floor(bbox_max / chunk_shape).astype(int)
# Iterate over all chunk coordinates in the range [chunk_min, chunk_max]
```

**Step 2 — For each overlapping chunk, compute overlapping bins.**
```python
# Lower-left corner of this chunk in physical space
chunk_origin = chunk_coord * chunk_shape
# Clip bbox to this chunk
local_min = np.maximum(bbox_min - chunk_origin, 0)
local_max = np.minimum(bbox_max - chunk_origin, chunk_shape)
# Bin range
bin_min = np.floor(local_min / bin_shape).astype(int)
bin_max = np.floor(local_max / bin_shape).astype(int)
# Iterate over all bin coordinates in [bin_min, bin_max]
```

**Step 3 — Read VG index entries for overlapping bins.**
```python
vg_index = zarr_open(f"resolution_0/vertex_group_offsets")[*chunk_coord]
# Shape: (B_per_chunk, 2)
for bin_coord in overlapping_bins:
    b = ravel_multi_index(bin_coord, bpc_shape)
    offset, count = vg_index[b]
    if count > 0:
        verts = zarr_open("resolution_0/vertices")[*chunk_coord, offset:offset+count, :]
        yield verts
```

**Step 4 — Optional point-in-box filter.** Because bins may extend slightly
beyond the requested bbox, a final point-in-box test filters vertices to
strictly within `[bbox_min, bbox_max]`. This is a client-side operation
after the I/O is complete.

### Write-time invariants

When a `vertices/` chunk slice is extended (e.g. because additional vertices
are added to an existing chunk), the `vertex_group_offsets/` must be
recomputed entirely for that chunk. It is not possible to append to a
`vertices/` chunk and update only the affected VG index entries without
re-sorting and re-offsetting the entire chunk.

This is why ZVF write functions buffer all vertices for a chunk in memory
and write the chunk in a single pass. Streaming per-vertex writes to an
on-disk ZVF store are not supported by `zarr-vectors-py`.

### Attribute VG ordering

The VG order invariant applies identically to all `attributes/<name>/` chunk
slices: the `k`-th element of an attribute chunk corresponds to the `k`-th
vertex in the `vertices/` chunk. No separate VG index is maintained for
attributes; they piggyback on the `vertex_group_offsets/` of their parent
resolution level.

### Memory layout summary

For a 3-D chunk with 64 bins and 4 200 vertices:

```
vertices/  chunk (i,j,k):
  [0:350]    → bin (0,0,0), 350 vertices
  [350:701]  → bin (0,0,1), 351 vertices
  [701:701]  → bin (0,0,2), 0 vertices (empty)
  [701:1048] → bin (0,0,3), 347 vertices
  …
  [4050:4200]→ bin (3,3,3), 150 vertices

vertex_group_offsets/ chunk (i,j,k):
  [0]  = [0,   350]   → bin (0,0,0)
  [1]  = [350, 351]   → bin (0,0,1)
  [2]  = [0,   0]     → bin (0,0,2) — empty (offset ignored)
  [3]  = [701, 347]   → bin (0,0,3)
  …
  [63] = [4050, 150]  → bin (3,3,3)
```
