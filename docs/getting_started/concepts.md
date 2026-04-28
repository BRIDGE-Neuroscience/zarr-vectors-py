# Core concepts

This page explains the key ideas behind the Zarr Vector Format (ZVF) and
`zarr-vectors-py`. It is intended as a mental-model introduction — enough
to make informed decisions about chunk sizes, bin sizes, and resolution
pyramids before writing your first large dataset. The
[Specification](../spec/index.md) gives the full technical treatment of
each concept.

---

## What is the Zarr Vector Format?

ZVF is a chunked, cloud-native storage format for *spatial vector geometry*
data: point clouds, streamlines, graphs, skeletons, and meshes. It is built
on **Zarr v3** and follows a directory-tree layout in which every array
(vertex positions, connectivity, attributes) is stored as a spatially
chunked Zarr array.

The format was originally specified by Forest Collman at the Allen Institute
for Brain Sciences. `zarr-vectors-py` is a Python implementation of that
specification, extended with separated chunk/bin sizes, per-level object
sparsity, and OME-Zarr-compatible multiscale metadata.

---

## The store is a directory

A ZVF store is an ordinary directory on disk (or a prefix in cloud object
storage). Its name conventionally ends in `.zarrvectors`:

```
scan.zarrvectors/
├── .zattrs              ← root metadata (geometry type, bin shape, CRS, …)
├── resolution_0/        ← full-resolution level
│   ├── vertices/
│   ├── vertex_group_offsets/
│   ├── links/
│   ├── attributes/
│   ├── object_index/
│   └── cross_chunk_links/
├── resolution_1/        ← coarser level (bin_ratio = [2,2,2])
│   └── …
└── metadata.json
```

Each sub-directory is a Zarr group. Arrays within a group are themselves
directories containing one file per chunk (or shard). Nothing is binary-
proprietary; the entire store can be inspected with `zarr`, `h5py`-style
tools, or a plain file browser.

---

## Chunks

A **chunk** is the unit of I/O. When `zarr-vectors` reads a spatial region
it determines which chunks overlap that region, issues one read per chunk
(or one HTTP range request in cloud storage), and returns the merged result.

`chunk_shape` defines the size of each chunk in physical space. It is given
in the same units as your coordinate data (e.g. micrometres, voxels) and
applies uniformly across all resolution levels:

```python
write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),   # each chunk covers 200³ μm
)
```

Choosing `chunk_shape`:

- Larger chunks → fewer files, lower open/seek overhead, larger individual
  reads. Good for sequential access patterns and network file systems.
- Smaller chunks → faster targeted queries for small spatial regions. Good
  for interactive visualisation.
- A common starting point for 3-D biological data: **200–500 physical units**
  per axis, adjusted so each chunk contains ~10 000–100 000 vertices.

---

## Supervoxel bins

A **bin** is a finer spatial subdivision *within* a chunk. Bins are the
unit of the spatial index: a bounding-box query resolves to a set of bins,
not a set of chunks. This means you can retrieve a small spatial region
without loading an entire chunk from disk.

`bin_shape` must evenly divide `chunk_shape` in every dimension:

```python
write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),    # 4×4×4 = 64 bins per chunk
)
```

The bin grid within a chunk looks like this (2-D cross-section shown):

```
┌──────────────────────────────────────┐
│ chunk (200 × 200)                    │
│ ┌────────┬────────┬────────┬────────┐│
│ │ bin    │ bin    │ bin    │ bin    ││
│ │(50×50) │(50×50) │(50×50) │(50×50) ││
│ ├────────┼────────┼────────┼────────┤│
│ │ bin    │ bin    │ bin    │ bin    ││
│ │        │        │        │        ││
│ ├────────┼────────┼────────┼────────┤│
│ │  …                               ││
└──────────────────────────────────────┘
```

When `bin_shape` is omitted it defaults to `chunk_shape` (one bin per
chunk), which gives backward-compatible behaviour equivalent to a pure
chunk-indexed store.

**Bin shape vs chunk shape — the key difference:**
`chunk_shape` is fixed and controls the on-disk file layout. `bin_shape`
controls spatial query granularity and grows proportionally with the
`bin_ratio` at each coarser resolution level, so coarser levels have
fewer, larger bins while the chunk grid stays the same. This is what keeps
per-chunk data volume roughly constant across the pyramid.

---

## Vertex groups (VGs)

A **vertex group** (VG) is the set of vertices in a given bin within a
given chunk. The `vertex_group_offsets` array stores the byte offset and
length of each VG within the flat `vertices` array for that chunk, enabling
direct random access without scanning all vertices.

In plain terms: when you issue a bbox query, `zarr-vectors` computes which
chunks intersect the bbox, identifies the specific bins within those chunks,
looks up the VG offsets, and reads only the relevant vertex slices. No full
chunk scan occurs.

---

## Object model

For geometry types that have discrete *objects* (streamlines, skeletons,
meshes), ZVF stores an additional `object_index` array that maps each
object ID to the list of VGs (and therefore chunks) that contain its
vertices. This enables efficient single-object retrieval:

```python
result = read_polylines("tracts.zarrvectors", object_ids=[42])
# Internally: look up object 42 in object_index → fetch only the
# relevant VG slices from the relevant chunks.
```

When an object spans multiple chunks (e.g. a streamline that crosses chunk
boundaries), the segments are linked via `cross_chunk_links`, which stores
the connecting edges between the last vertex of one segment and the first
vertex of the next.

---

## Multi-resolution pyramids

ZVF stores can contain multiple resolution levels under `resolution_0/`,
`resolution_1/`, etc. Each level is generated by spatial coarsening
(*binning*) and, for discrete-object types, optional *object thinning*.

**Bin ratio** controls vertex reduction. A `bin_ratio` of `(2, 2, 2)` at
level 1 means the bin shape at that level is `2 × base_bin_shape`, so each
bin is 8× larger and contains roughly 8× more vertices merged into one
metanode:

```
Level 0: bin_shape = (50, 50, 50)    → 64 bins/chunk  (full resolution)
Level 1: bin_shape = (100, 100, 100) → 8 bins/chunk   (2×2×2 coarser)
Level 2: bin_shape = (200, 200, 200) → 1 bin/chunk    (4×4×4 coarser)
```

**Object sparsity** controls object reduction for discrete-object types.
Setting `object_sparsity=0.5` at level 1 retains 50 % of the objects,
selected by one of four strategies: spatial coverage, object length,
per-object attribute value, or random sampling.

Total data volume reduction at a level = vertex reduction × object
reduction. For example, `bin_ratio=(2,2,2)` gives 8× vertex reduction; if
also `object_sparsity=0.5`, total reduction = 8× × 2× = **16×**.

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2), "object_sparsity": 1.0},  # level 1: 8×
        {"bin_ratio": (4, 4, 4), "object_sparsity": 0.5},  # level 2: 128×
    ],
)
```

---

## OME-Zarr multiscale metadata

ZVF borrows the `multiscales` JSON block from the
[OME-Zarr NGFF specification](https://ngff.openmicroscopy.org/). This
means any OME-Zarr-aware viewer can discover the resolution pyramid and
read coordinate transforms from a ZVF store without modification. The
`scale` and `translation` fields in each level encode the bin ratio and
centroid offset respectively, so viewers that understand OME-Zarr can
perform correct physical-space alignment at each level.

ZVF-specific fields (`bin_ratio`, `object_sparsity`, `base_bin_shape`) are
added alongside the standard OME-Zarr keys. A viewer that does not
understand them will simply ignore them.

---

## Geometry types

ZVF supports seven geometry types, each identified by a string constant:

| Constant | Description |
|----------|-------------|
| `GEOM_POINT_CLOUD` | Unconnected vertices with optional per-vertex attributes |
| `GEOM_LINE` | Pairs of vertices (line segments) |
| `GEOM_POLYLINE` | Ordered vertex sequences; supports cross-chunk links |
| `GEOM_STREAMLINE` | Polylines with tractography-specific metadata (step size, seeding) |
| `GEOM_GRAPH` | Arbitrary vertex–edge graph; directed or undirected |
| `GEOM_SKELETON` | Tree-structured graph aligned to the SWC convention |
| `GEOM_MESH` | Triangulated surface mesh with face arrays |

All types share the same chunked spatial layout. Types with discrete objects
(polyline, streamline, graph, skeleton, mesh) additionally have an
`object_index` and optionally `cross_chunk_links`.

---

## Coordinate systems and physical units

ZVF stores a `coordinate_system` key in `.zattrs` (e.g. `"RAS"`,
`"LPS"`, `"voxel"`). Coordinates are always stored in the declared system;
no implicit conversion is performed. `chunk_shape` and `bin_shape` are in
the same units as the coordinates.

The OME-Zarr multiscale metadata carries axis names and units, enabling
downstream tools that understand NGFF to display data with correct physical
scaling.

---

## Summary

| Concept | What it is | Configured by |
|---------|-----------|---------------|
| Store | A `.zarrvectors` directory | `store_path` argument |
| Chunk | I/O unit; one file per chunk | `chunk_shape` |
| Bin | Spatial query unit within a chunk | `bin_shape` |
| VG | Vertices in one bin in one chunk | Computed automatically |
| Object index | Maps object IDs to VGs | Written automatically for applicable types |
| Resolution level | Coarsened copy of the data | `build_pyramid()` |
| Bin ratio | Coarsening factor per level | `bin_ratio` in `level_configs` |
| Object sparsity | Object thinning fraction per level | `object_sparsity` in `level_configs` |
