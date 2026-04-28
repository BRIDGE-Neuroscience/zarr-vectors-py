# Meshes

The `mesh` type stores triangulated 3-D surface meshes: cell boundaries
from electron microscopy segmentation, brain or organ surfaces from MRI
reconstruction, organelle hulls from fluorescence segmentation, or any
other closed or open triangulated surface.

ZVF meshes support optional Draco compression for significant size
reductions, per-vertex attributes (normals, UV coordinates, scalars), and
multi-mesh stores that pack thousands of mesh objects into a single
spatially indexed store.

All examples require `zarr-vectors` base install. OBJ/STL/PLY ingest and
Draco compression require `zarr-vectors[ingest]` and `zarr-vectors[draco]`
respectively.

---

## Writing a mesh

### Write from vertices and faces

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

# Generate a simple icosphere (demonstration only)
# In practice, load from OBJ, STL, PLY, or a segmentation pipeline
vertices, faces = generate_icosphere(radius=200.0, subdivisions=4)
# vertices: (N, 3) float32   — vertex positions in µm
# faces:    (F, 3) int32     — triangle vertex index triplets (0-indexed, global)

write_mesh(
    "cell.zarrvectors",
    vertices=vertices.astype(np.float32),
    faces=faces.astype(np.int32),
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    winding_order="ccw",        # counter-clockwise = outward normals
    coordinate_system="RAS",
    axis_units="micrometer",
)
```

### Write with per-vertex attributes

Common mesh attributes include normals, curvature, UV texture coordinates,
and scalar overlays (e.g. cortical thickness):

```python
rng = np.random.default_rng(0)
n_verts = len(vertices)

# Compute vertex normals (simplified; use trimesh or open3d in practice)
normals   = compute_vertex_normals(vertices, faces)   # (N, 3) float32
curvature = rng.uniform(-0.1, 0.1, n_verts).astype(np.float32)
thickness = rng.uniform(1.5, 4.5, n_verts).astype(np.float32)

write_mesh(
    "brain_surface.zarrvectors",
    vertices=vertices,
    faces=faces,
    chunk_shape=(10.0, 10.0, 10.0),     # smaller chunks for dense mesh
    bin_shape=(5.0, 5.0, 5.0),
    winding_order="ccw",
    attributes={
        "normal":    normals,           # vector attribute: (N, 3)
        "curvature": curvature,         # scalar attribute: (N,)
        "thickness": thickness,
    },
)
```

---

## Ingesting from external formats

All ingest functions require `zarr-vectors[ingest]`.

### OBJ

```python
from zarr_vectors.ingest.obj import ingest_obj

ingest_obj(
    "brain.obj",
    "brain.zarrvectors",
    chunk_shape=(10.0, 10.0, 10.0),
    bin_shape=(2.5, 2.5, 2.5),
)
```

Multi-object OBJ files (multiple `o` groups) produce a multi-mesh store
with one ZVF object per OBJ group.

### STL

```python
from zarr_vectors.ingest.stl import ingest_stl

ingest_stl(
    "organ.stl",
    "organ.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
)
```

STL files store per-face normals. These are converted to per-vertex normals
(averaged over adjacent faces) and stored in `attributes/normal/`.

### PLY

```python
from zarr_vectors.ingest.ply import ingest_ply

ingest_ply(
    "scan.ply",
    "scan.zarrvectors",
    chunk_shape=(25.0, 25.0, 25.0),
    # Vertex properties (x, y, z, nx, ny, nz, red, green, blue, …)
    # are auto-detected from the PLY header
)
```

### CLI ingest

```bash
zarr-vectors ingest mesh brain.obj brain.zarrvectors --chunk-shape 10,10,10
zarr-vectors ingest mesh organ.stl organ.zarrvectors --chunk-shape 50,50,50
zarr-vectors ingest mesh scan.ply  scan.zarrvectors  --chunk-shape 25,25,25
```

---

## Draco compression

Draco is a geometry compression library that exploits vertex-face
correlations for significantly better compression than general-purpose
codecs on mesh data. Requires `zarr-vectors[draco]`.

```python
from zarr_vectors.ingest.obj import ingest_obj

ingest_obj(
    "brain.obj",
    "brain_draco.zarrvectors",
    chunk_shape=(10.0, 10.0, 10.0),
    use_draco=True,
    draco_quantization=11,    # 11-bit quantisation: 1/2048 of bbox precision
)
```

### Compression ratio guidance

| `draco_quantization` | Precision per axis | Typical compression vs float32 |
|--------------------|--------------------|-------------------------------|
| 8 | 1 / 256 of bbox | 12–18× |
| 11 | 1 / 2048 of bbox | 7–12× |
| 14 | 1 / 16384 of bbox | 4–7× |

For nanometre-resolution EM segmentation meshes with a bbox of ~100 µm,
11-bit quantisation gives sub-50 nm precision — more than sufficient for
most visualisation and analysis workflows.

```python
# Check if a store uses Draco
from zarr_vectors.core.store import open_store
root = open_store("brain_draco.zarrvectors", mode="r")
print(root.attrs.get("draco_compressed", False))   # True
```

**Important:** Draco-compressed stores are not readable without
`zarr-vectors[draco]` installed. Communicate the compression requirement
clearly when distributing stores.

---

## Reading a mesh

### Read all data

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("brain.zarrvectors")

print(result["vertex_count"])     # int
print(result["face_count"])       # int
print(result["vertices"].shape)   # (N, 3) float32
print(result["faces"].shape)      # (F, 3) int32 — global vertex indices
```

The returned `faces` array contains global vertex indices (0-indexed into
the `vertices` array). Face indices are consistent: face `k` is defined by
`vertices[faces[k, 0]]`, `vertices[faces[k, 1]]`, `vertices[faces[k, 2]]`.

### Read per-vertex attributes

```python
result = read_mesh("brain_surface.zarrvectors",
                   attributes=["curvature", "thickness"])
curvature = result["attributes"]["curvature"]   # (N,)
thickness = result["attributes"]["thickness"]   # (N,)
normals   = result["attributes"]["normal"]      # (N, 3) if stored
```

### Spatial bbox query

A bbox query returns faces whose centroid is within the bounding box, plus
all vertices referenced by those faces (which may lie slightly outside the
bbox):

```python
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([50., 50., 50.])),
)
print(result["face_count"])      # faces with centroid in the bbox region
```

---

## Multi-mesh stores

For segmentation datasets with thousands of cell objects, a single ZVF
store is far more efficient than per-cell OBJ files:

### Writing a multi-mesh store

```python
from zarr_vectors.types.meshes import write_mesh

all_vertices = []    # list of (N_i, 3) arrays
all_faces    = []    # list of (F_i, 3) arrays (local vertex indices)
cell_volumes = []    # per-cell scalar attribute

for cell_id, (verts, faces) in enumerate(cell_meshes):
    all_vertices.append(verts)
    all_faces.append(faces)
    cell_volumes.append(compute_volume(verts, faces))

write_mesh(
    "cells.zarrvectors",
    vertices=all_vertices,      # list of per-object vertex arrays
    faces=all_faces,            # list of per-object face arrays
    chunk_shape=(50., 50., 50.),
    object_attributes={
        "volume":       np.array(cell_volumes, dtype=np.float32),
        "cell_type":    np.array(cell_types,   dtype=np.int32),
    },
)
```

### Reading individual cells

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("cells.zarrvectors", object_ids=[42, 107])
print(result["object_ids"])       # [42, 107]
print(result["vertex_count"])     # combined vertex count
```

### Spatial query in a multi-mesh store

```python
result = read_mesh(
    "cells.zarrvectors",
    bbox=(np.array([500., 500., 200.]),
          np.array([600., 600., 300.])),
    return_object_ids=True,
)
print(result["object_ids"])       # IDs of cells with faces in the region
print(result["face_count"])
```

---

## Exporting

```python
from zarr_vectors.export.obj import export_obj
from zarr_vectors.export.ply import export_ply

# Export all mesh data to OBJ
export_obj("brain.zarrvectors", "brain_out.obj")

# Export with PLY (preserves per-vertex attributes)
export_ply("brain.zarrvectors", "brain_out.ply")
```

For multi-mesh stores, pass `object_ids` to export a subset:

```python
export_obj("cells.zarrvectors", "cell_42.obj", object_ids=[42])
```

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("brain.zarrvectors", level=4)
print(result.summary())
# Level 4 validation: PASS
#   29 passed, 0 warnings, 0 errors
```

For closed surfaces, level 4 additionally checks watertightness (every
edge shared by exactly two faces). Enable this check by setting
`closed_surface = true` in root `.zattrs`:

```python
from zarr_vectors.core.store import open_store

root = open_store("cell.zarrvectors", mode="r+")
root.attrs["closed_surface"] = True
# Now zarr-vectors validate cell.zarrvectors --level 4 checks watertightness
```

---

## Common pitfalls

**Face indices are global, not local.**
When calling `write_mesh` with a single mesh, `faces` must use 0-based
indices into the `vertices` array you are passing. When calling with
a list of per-object arrays, each face array uses local indices into
its own per-object `vertices` array; the writer handles global index
conversion automatically.

**Draco changes vertex positions slightly.**
Draco quantises vertex positions to integers before compression. Even at
the highest precision level (14 bits), there is a small rounding error.
Do not use Draco if you need exact float32 round-trip fidelity (e.g. for
downstream numerical computation on vertex coordinates). For visualisation,
11-bit quantisation is imperceptible.

**Winding order inconsistency between files.**
Different mesh tools use different winding conventions. `ingest_obj`
defaults to CCW (the OBJ standard) but some exporters produce CW meshes
without declaring it. If your rendered normals point inward, pass
`winding_order="cw"` at ingest time or flip normals post-hoc:

```python
ingest_obj("inverted.obj", "inverted.zarrvectors",
           chunk_shape=(10., 10., 10.),
           winding_order="cw")
```

**Boundary face resolution requires fetching extra chunks.**
A face whose centroid is in chunk A but one vertex is in chunk B requires
fetching chunk B to resolve the vertex position. The reader does this
automatically, but it means a bbox query may issue slightly more chunk
reads than the number of chunks in the bbox. This is expected behaviour.
