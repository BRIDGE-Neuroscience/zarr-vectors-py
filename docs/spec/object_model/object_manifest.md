# Object manifest

## Terms

**Object manifest**
: The complete description of where an object's data is stored across the
  ZVF store. For a polyline spanning N chunks, the manifest lists the N
  `(chunk_flat_index, vg_index)` pairs — one per chunk — that together
  contain all the object's vertices.

**`object_index/`**
: The Zarr array that stores the *primary entry* of each object's manifest:
  the `(chunk_flat_index, vg_flat_index)` of the first VG belonging to
  the object. Shape `(n_objects, 2)` int64.

**Primary VG**
: The first vertex group of an object in traversal order (for sequential
  types) or the VG containing the root vertex (for tree types). This is
  the entry stored in `object_index/`.

**Manifest reconstruction**
: The process of expanding a primary entry in `object_index/` into the full
  ordered list of VG addresses for an object. Performed by following
  `cross_chunk_links/` from the primary VG until no more links are found.

---

## Introduction

The object manifest is the mechanism that enables efficient single-object
retrieval in ZVF. Without it, fetching a single streamline from a large
tractography store would require scanning every chunk to find the
streamline's vertices. With it, the reader knows exactly which chunks to
fetch and which VG slices within those chunks belong to the object.

The manifest is stored in compact form in `object_index/` (one row per
object) and reconstructed on demand by following cross-chunk links. This
lazy reconstruction keeps the `object_index/` array small while still
supporting arbitrary multi-chunk objects.

---

## Technical reference

### `object_index/` schema

```
shape:      (n_objects, 2)
dtype:      int64
chunk_shape: (65536, 2)
fill_value: -1
```

Row `k` stores `[chunk_flat_index, vg_flat_index]` for object `k`:

```
chunk_flat_index  = ravel_multi_index(primary_chunk_coord, chunk_grid_shape)
vg_flat_index     = ravel_multi_index(primary_bin_coord,   bpc_shape)
```

For an object whose primary VG is in chunk `(1, 2, 3)` of a
`(10, 12, 8)` chunk grid, and in bin `(0, 1, 2)` of a `(4, 4, 4)` bin
grid:

```python
chunk_flat = np.ravel_multi_index((1, 2, 3), (10, 12, 8))  # = 1*96 + 2*8 + 3 = 115
bin_flat   = np.ravel_multi_index((0, 1, 2), (4, 4, 4))    # = 0*16 + 1*4 + 2 = 6
object_index[k] = [115, 6]
```

### Object ID assignment

Object IDs are non-negative integers assigned at write time. They are
dense (no gaps) and start at 0. The maximum object ID is `n_objects - 1`.

Object IDs are stable across reads but may change if the store is rechunked
(rechunking rebuilds `object_index/` and may reassign IDs if the primary
chunk changes). For stable long-term identifiers, store a custom ID as a
per-object attribute.

### Manifest reconstruction algorithm

Given an object ID `k`, reconstructing its full VG list:

```python
def get_object_vgs(store, level, object_id):
    """Return ordered list of (chunk_flat, vg_flat) for an object."""
    obj_idx = read_array(f"resolution_{level}/object_index")
    chunk_flat, vg_flat = obj_idx[object_id]

    vg_list = [(chunk_flat, vg_flat)]
    visited_chunks = {chunk_flat}

    # Follow cross-chunk links until no more continuations exist
    links = read_array(f"resolution_{level}/cross_chunk_links")  # (L, 2)
    link_map = build_link_map(links)   # global_vertex_id → global_vertex_id

    # Find the last vertex of the current VG
    current_last_vertex = get_last_vertex_of_vg(store, level, chunk_flat, vg_flat)

    while current_last_vertex in link_map:
        next_vertex_global = link_map[current_last_vertex]
        next_chunk_flat, next_vg_flat = global_to_vg_address(
            store, level, next_vertex_global
        )
        vg_list.append((next_chunk_flat, next_vg_flat))
        current_last_vertex = get_last_vertex_of_vg(
            store, level, next_chunk_flat, next_vg_flat
        )
        if next_chunk_flat in visited_chunks:
            break   # safety: cycle guard
        visited_chunks.add(next_chunk_flat)

    return vg_list
```

`zarr-vectors-py` caches the reconstructed VG list per object in an
LRU cache. For workloads that read many objects sequentially, pre-warming
the cache with `prefetch_object_manifests([ids], ...)` reduces latency.

### Reading a complete object

```python
def read_object_vertices(store, level, object_id):
    """Read and concatenate all vertices of an object, in traversal order."""
    vg_list = get_object_vgs(store, level, object_id)
    chunks  = [read_vg(store, level, cf, bf) for cf, bf in vg_list]
    return np.concatenate(chunks, axis=0)
```

For polylines and streamlines, the concatenated vertices are in traversal
order. For graphs and skeletons, the order is determined by the VG list;
use `links/edges/` and `cross_chunk_links/` to reconstruct the topology.

### Object count and density

The number of objects in a store is `object_index.shape[0]`. This can be
read efficiently without scanning the full array:

```python
from zarr_vectors.core.store import open_store
root = open_store("tracts.zarrvectors", mode="r")
n = root["resolution_0"]["object_index"].shape[0]
```

### Object index at coarser levels

At coarser resolution levels, the `object_index/` is recomputed to
reference the primary VG of each object's *coarsened* representation.
Because bin sizes change between levels, the VG address of an object's
primary VG may differ between levels.

If `object_sparsity < 1.0` at a level, not all objects from level 0 are
present. The `object_index/` at that level contains only the retained
objects, with IDs re-mapped to `[0, n_retained)`. The mapping from
coarse-level IDs to level-0 IDs is stored in
`object_attributes/level0_object_id/` at the coarser level.

### Validation

L1: `object_index/` exists for all discrete-object types.

L2:
- `object_index.shape[1] == 2`.
- `object_index.dtype` is `int64`.

L3:
- For every row `[cf, bf]` in `object_index/`: `cf` is a valid flat chunk
  index (in `[0, product(chunk_grid_shape))`); `bf` is in
  `[0, B_per_chunk)`.
- The VG at address `(cf, bf)` is non-empty (count > 0).
- No two objects share the same primary VG address.
