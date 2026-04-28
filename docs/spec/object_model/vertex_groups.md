# Vertex groups

## Terms

**Vertex group (VG)**
: The set of all vertices in one spatial bin within one spatial chunk.
  A VG is the atomic unit of the ZVF spatial index. It is identified by
  a `(chunk_coord, bin_flat_index)` pair and addressed via the
  `vertex_group_offsets/` array.

**VG address**
: The pair `(chunk_flat_index, bin_flat_index)` that uniquely identifies
  a VG. `chunk_flat_index` is the C-order ravel of the chunk coordinate;
  `bin_flat_index` is the C-order ravel of the bin coordinate within the
  chunk.

**VG slice**
: The contiguous range `[offset, offset + count)` within a `vertices/`
  chunk slice that contains the vertices of a given VG.

**VG count**
: The number of VGs in a chunk equal to `B_per_chunk` (bins per chunk),
  which is fixed for the store. Empty VGs (containing no vertices) are
  stored with `count = 0`.

**Object VG list**
: The ordered list of VG addresses belonging to a given object. For a
  polyline, the order is the traversal order of the path through space.
  Encoded in the `object_index/` extended manifest (see
  [Object manifest](object_manifest.md)).

---

## Introduction

Vertex groups are the bridge between the chunk-level I/O of Zarr and the
bin-level spatial queries of ZVF. Each VG is a slice of one chunk's vertex
array; the VG index (`vertex_group_offsets/`) allows any VG to be read
without loading the entire chunk.

This page describes the VG concept in detail: how VGs are created during
writing, how they are addressed, and how the object model (for discrete-
object types) uses VG lists to reconstruct complete objects.

---

## Technical reference

### VG creation during write

When a chunk is written, its vertices are first sorted into bin order and
then the VG offsets are computed:

```python
def build_vg_index(positions, chunk_coord, chunk_shape, bin_shape):
    """Compute VG order and offset table for one chunk's vertices."""
    D          = len(chunk_shape)
    bpc_shape  = tuple(int(c / b) for c, b in zip(chunk_shape, bin_shape))
    B          = int(np.prod(bpc_shape))
    N          = len(positions)

    # 1. Compute bin flat index for each vertex
    local      = positions - np.array(chunk_coord) * np.array(chunk_shape)
    bin_coords = np.floor(local / np.array(bin_shape)).astype(int)
    bin_flats  = np.ravel_multi_index(bin_coords.T, bpc_shape)

    # 2. Sort vertices into bin order (stable sort within each bin)
    order      = np.argsort(bin_flats, kind="stable")
    sorted_pos = positions[order]
    sorted_bin = bin_flats[order]

    # 3. Build offset table
    offsets = np.full((B, 2), -1, dtype=np.int64)  # fill = -1 (empty)
    unique_bins, first_occ, counts = np.unique(
        sorted_bin, return_index=True, return_counts=True
    )
    offsets[unique_bins, 0] = first_occ
    offsets[unique_bins, 1] = counts

    return sorted_pos, order, offsets
```

The `order` array is used to reorder attribute arrays in the same way:
`sorted_attributes = attributes[order]`.

### VG addressing

A VG is uniquely identified within a store by:

1. **`chunk_flat_index`**: C-order flat index of the chunk within the
   chunk grid.

   ```python
   chunk_flat = np.ravel_multi_index(chunk_coord, chunk_grid_shape)
   ```

2. **`bin_flat_index`**: C-order flat index of the bin within the chunk.

   ```python
   bpc_shape  = tuple(int(c / b) for c, b in zip(chunk_shape, bin_shape))
   bin_flat   = np.ravel_multi_index(bin_coord, bpc_shape)
   ```

Together these form the VG address `(chunk_flat, bin_flat)`, which is the
format used in `object_index/` entries.

### VG access pattern

Reading a specific VG given its address:

```python
def read_vg(store_root, level, chunk_flat, bin_flat, chunk_grid_shape):
    """Read the vertices of a single VG."""
    chunk_coord = np.unravel_index(chunk_flat, chunk_grid_shape)
    
    # Read VG index for this chunk
    vg_idx = zarr_open(f"{store_root}/resolution_{level}/vertex_group_offsets")
    offset, count = vg_idx[*chunk_coord, bin_flat]
    
    if count <= 0:
        return np.empty((0, D), dtype=np.float32)
    
    # Read only the VG slice from the vertices array
    verts = zarr_open(f"{store_root}/resolution_{level}/vertices")
    return verts[*chunk_coord, offset:offset+count, :]
```

Because each chunk of `vertices/` is independently compressed, reading a
VG slice requires decompressing the entire chunk. The chunk is then
returned from an in-process cache for subsequent VG reads from the same
chunk.

### VGs and the object model

For discrete-object types (polyline, streamline, graph, skeleton), the
`object_index/` stores the first VG address of each object. The full set
of VGs for an object is reconstructed by following the object's vertex
connectivity:

- For **polylines and streamlines**, the VG list is ordered by traversal
  direction. All VGs in the same chunk are read together; then
  `cross_chunk_links/` is used to find the continuation VG in the next
  chunk.

- For **graphs and skeletons**, the VG list is unordered (connectivity is
  given by `links/edges/`). The full set of chunks containing the object's
  vertices is determined from the object's bounding box and the
  `cross_chunk_links/`.

### VGs across multiple chunks

A single object typically spans multiple chunks and therefore multiple
VGs. The `object_index/` stores only the *primary* VG (the first VG in
traversal order for sequential types, or the VG containing the root vertex
for trees). The full object is reconstructed by following `cross_chunk_links/`
from the primary VG.

For objects that span many chunks (long streamlines, large meshes), the
reconstruction walk can be expensive. `zarr-vectors-py` caches the per-
object VG list after the first access.

### VG count at coarser levels

At coarser resolution levels, the effective bin shape is larger
(`bin_shape × bin_ratio`), so `B_per_chunk` is smaller:

```
Level 0: bpc_shape = (4, 4, 4) → B_per_chunk = 64
Level 1: bpc_shape = (2, 2, 2) → B_per_chunk = 8  (for bin_ratio = (2,2,2))
Level 2: bpc_shape = (1, 1, 1) → B_per_chunk = 1  (for bin_ratio = (4,4,4))
```

At level 2, each chunk is one VG containing all vertices in the chunk. The
VG index degenerates to a single-entry table per chunk.

### Invariants maintained by the writer

These invariants must be preserved by any code that writes to a ZVF store:

1. Vertices within a chunk are stored in ascending bin flat index order
   (VG order).
2. `vertex_group_offsets[chunk][b, 0]` = start index of bin `b` in the
   chunk's vertex slice; `vertex_group_offsets[chunk][b, 1]` = count.
3. Attribute arrays are sorted with the same `order` permutation as
   `vertices/` — attribute `k` corresponds to vertex `k`.
4. `sum(max(0, vg_offsets[b, 1]) for b in range(B))` = total vertex count
   in the chunk.
5. Empty bins have `vg_offsets[b, 1] = 0` (written explicitly) or
   `-1` (fill value, not written).
