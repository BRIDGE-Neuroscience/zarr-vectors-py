# Mesh (`mesh`)

## Terms

**Mesh**
: A piecewise-linear surface represented by a set of vertices and a set of
  triangular faces. Each face is a triplet of vertex indices. ZVF stores
  closed or open surface meshes; there is no requirement of watertightness.

**`GEOM_MESH`**
: The geometry type constant `"mesh"`.

**`links/faces/`**
: The array storing triplets of local-chunk vertex indices for each
  triangle in a chunk. Shape `(F, 3)` int32 per chunk.

**Winding order**
: The orientation convention for face normals. ZVF defaults to
  counter-clockwise winding (CCW) when viewed from outside the surface,
  which implies outward-facing normals. Declared in root `.zattrs` under
  `"winding_order"`.

**Draco compression**
: An optional codec applied to mesh geometry (`vertices/` and `links/faces/`)
  that can achieve 6–15× compression over uncompressed float32 data.
  Requires `zarr-vectors[draco]`.

**Boundary face**
: A triangle in one chunk that references a vertex stored in a different
  chunk. Mesh stores do not use `cross_chunk_links/`; instead, boundary
  faces are handled via global vertex ID remapping at read time.

---

## Introduction

The `mesh` type stores triangulated surface meshes in the ZVF spatial
chunking framework. Like other geometry types, the mesh is partitioned
into spatial chunks; each chunk holds the vertices that fall within its
spatial extent and the faces whose centroid falls within that extent.

Mesh chunking introduces a subtlety that does not arise for point clouds
or streamlines: a face may reference vertices in multiple chunks (the face
straddles a chunk boundary). ZVF handles this by storing boundary faces in
the chunk containing the face centroid and using global vertex IDs to
reference vertices in other chunks. The `object_index/` maps each mesh
object (distinct connected surface) to its constituent chunks.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions, shape `(N, D)` float32 per chunk |
| `vertex_group_offsets/` | Yes | VG index |
| `links/faces/` | Yes | Triangle vertex triplets, shape `(F, 3)` int32 per chunk |
| `object_index/` | Yes | Object (mesh surface) ID → primary chunk |
| `attributes/<name>/` | No | Per-vertex attributes (normals, UVs, colours) |
| `object_attributes/<name>/` | No | Per-mesh attributes (volume, surface area) |

No `links/edges/` and no `cross_chunk_links/`. Face-level cross-chunk
references use the global vertex ID mechanism described below.

### Root `.zattrs` type-specific keys

```json
{
  "geometry_type": "mesh",
  "winding_order":  "ccw",
  "closed_surface": true
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `winding_order` | `string` | `"ccw"` | Face winding order: `"ccw"` (counter-clockwise) or `"cw"` (clockwise). |
| `closed_surface` | `bool` | `false` | Whether each mesh object is a closed watertight surface. If `true`, L4 validation checks for watertightness. |

### Face encoding and boundary faces

**Intra-chunk faces:** All three vertices are in the same chunk. Vertex
indices are local to the chunk (0-indexed within the chunk's vertex slice).

**Boundary faces:** One or more vertices are in a different chunk. For
boundary faces, vertex indices that refer to other chunks use a *negative
sentinel* encoding:

- Vertices in the current chunk: positive local index `[0, N_chunk)`.
- Vertices in other chunks: stored as a negative value `-(global_vertex_id + 1)`,
  where `global_vertex_id` is the vertex's position in the global vertex
  ID space.

At read time, the reader resolves negative indices by looking up the
corresponding global vertex IDs and fetching those vertices from their
respective chunks.

**Global vertex ID** for a vertex at local index `k` in chunk `(cx, cy, cz)`:

```
global_id = chunk_flat_index * N_max + k
```

where `chunk_flat_index = ravel_multi_index((cx, cy, cz), chunk_grid_shape)`
and `N_max` is the maximum vertices per chunk declared in the Zarr array shape.

### Face assignment to chunks

Each triangular face is assigned to the chunk containing its centroid:

```python
centroid = (vertices[i] + vertices[j] + vertices[k]) / 3
face_chunk = floor(centroid / chunk_shape).astype(int)
```

This ensures each face is stored exactly once. All three vertices of a face
are guaranteed to be within at most one chunk-width of the face's chunk (faces
cannot span more than two chunks in any dimension if all vertices are within
the face's chunk neighbourhood).

### Draco compression

```python
from zarr_vectors.ingest.obj import ingest_obj

ingest_obj(
    "brain.obj",
    "brain.zarrvectors",
    chunk_shape=(100., 100., 100.),
    use_draco=True,
    draco_quantization=11,
)
```

When Draco is enabled, the `vertices/` and `links/faces/` arrays use the
`draco` codec. Reading requires `zarr-vectors[draco]`. See
[Codec pipeline](../foundations/codec_pipeline.md) for quantisation
precision details.

Draco compression is applied per-chunk. The Draco codec receives the
combined (vertices, faces) data for one chunk and compresses them jointly,
exploiting vertex-face correlations for additional compression beyond what
independent array compression achieves.

### Write API

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

write_mesh(
    "brain.zarrvectors",
    vertices=vertices,   # (N, 3) float32
    faces=faces,         # (F, 3) int32 — global vertex indices
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    winding_order="ccw",
)
```

### Ingest and export

```python
from zarr_vectors.ingest.obj import ingest_obj
from zarr_vectors.ingest.stl import ingest_stl   # zarr-vectors[ingest]
from zarr_vectors.ingest.ply import ingest_ply
from zarr_vectors.export.obj import export_obj
from zarr_vectors.export.ply import export_ply

ingest_obj("surface.obj", "surface.zarrvectors", chunk_shape=(100., 100., 100.))
ingest_stl("surface.stl", "surface.zarrvectors", chunk_shape=(100., 100., 100.))
ingest_ply("surface.ply", "surface.zarrvectors", chunk_shape=(100., 100., 100.))

export_obj("surface.zarrvectors", "surface_out.obj")
export_ply("surface.zarrvectors", "surface_out.ply")
```

### Read API

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("brain.zarrvectors")
print(result["vertex_count"])    # int
print(result["face_count"])      # int
print(result["vertices"].shape)  # (N, 3)
print(result["faces"].shape)     # (F, 3) global vertex indices

# Spatial query — returns faces whose centroid is in bbox
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([500., 500., 500.])),
)
```

### Multi-mesh stores

A single `mesh` store may contain many distinct mesh objects (e.g. one per
cell or organelle). Each object is one connected surface:

```python
result = read_mesh("cells.zarrvectors", object_ids=[42, 107])
```

### Validation

L1: `vertices/`, `vertex_group_offsets/`, `links/faces/`, `object_index/`
exist.

L3:
- All positive face vertex indices are in `[0, N_chunk)`.
- No degenerate faces (all three vertex indices distinct).
- Negative face vertex indices decode to valid global vertex IDs.

L4 (if `closed_surface = true`):
- Each mesh object is watertight: every edge is shared by exactly two faces.
- No boundary edges.
