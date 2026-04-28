# Ingest formats

`zarr-vectors-py` can ingest data from a range of external formats into ZVF
stores. This page provides a format-by-format reference: what each format
stores, what is preserved in the ZVF output, what is dropped or converted,
and the one-liner Python call and CLI equivalent.

All ingest functions require `zarr-vectors[ingest]` unless marked otherwise.

---

## Format summary

| Format | Ext | Direction | ZVF type | Extra required |
|--------|-----|-----------|----------|---------------|
| LAS / LAZ | `.las`, `.laz` | ingest | `point_cloud` | `[ingest]` |
| PLY | `.ply` | ingest + export | `point_cloud`, `mesh` | `[ingest]` |
| CSV / XYZ | `.csv`, `.xyz`, `.txt` | ingest + export | `point_cloud` | base |
| TRK (TrackVis) | `.trk` | ingest + export | `streamline` | `[ingest]` |
| TCK (MRtrix) | `.tck` | ingest | `streamline` | `[ingest]` |
| TRX | `.trx` | ingest + export | `streamline` | `[ingest]` |
| SWC | `.swc` | ingest + export | `skeleton` | base |
| GraphML | `.graphml` | ingest | `graph` | `[ingest]` |
| OBJ | `.obj` | ingest + export | `mesh` | `[ingest]` |
| STL | `.stl` | ingest | `mesh` | `[ingest]` |

---

## LAS / LAZ point clouds

**Library:** `laspy` (installed with `zarr-vectors[ingest]`).

LAS is the standard format for lidar and photogrammetry point clouds.
LAZ is its losslessly compressed variant. All standard LAS point data
record formats (PDRFs 0–10) are supported.

### What is preserved

| LAS data | ZVF storage |
|----------|-------------|
| `x, y, z` | `vertices/` (coordinates de-scaled from LAS integer encoding) |
| `intensity` | `attributes/intensity/` (uint16) |
| `return_number`, `number_of_returns` | `attributes/return_number/`, `attributes/num_returns/` |
| `classification` | `attributes/classification/` (uint8) |
| `scan_angle_rank` | `attributes/scan_angle/` |
| `rgb` (if present in PDRF) | `attributes/color/` (uint16, shape (N, 3)) |
| LAS header CRS | `root .zattrs["coordinate_system"]` (WKT string) |

### What is dropped

Global encoding flags, file creation date, generating software, and point
source IDs are stored in `.zattrs["custom_metadata"]` for provenance but
not used by `zarr-vectors-py`.

### Usage

```python
from zarr_vectors.ingest.las import ingest_las

ingest_las(
    "survey.las",
    "survey.zarrvectors",
    chunk_shape=(100.0, 100.0, 50.0),
    bin_shape=(25.0, 25.0, 12.5),
    attributes=["intensity", "classification", "return_number"],
    # pass None to ingest all standard attributes
)

# LAZ (compressed LAS) — same API
ingest_las("survey.laz", "survey.zarrvectors", chunk_shape=(100., 100., 50.))
```

```bash
zarr-vectors ingest points survey.las survey.zarrvectors \
    --chunk-shape 100,100,50 --bin-shape 25,25,12.5
```

---

## PLY

**Library:** `plyfile` (installed with `zarr-vectors[ingest]`).

PLY supports both point clouds and meshes. The geometry type is inferred
from whether the file has a `face` element.

### Point cloud PLY

```python
from zarr_vectors.ingest.ply import ingest_ply

ingest_ply("scan.ply", "scan.zarrvectors", chunk_shape=(200., 200., 200.))
# Auto-detects: x, y, z → vertices; nx, ny, nz → attributes/normal;
#               red, green, blue → attributes/color; scalar props → attributes/<name>
```

### Mesh PLY

```python
ingest_ply("surface.ply", "surface.zarrvectors", chunk_shape=(25., 25., 25.))
# Face element detected → mesh type; vertex properties → per-vertex attributes
```

### Export to PLY

```python
from zarr_vectors.export.ply import export_ply

export_ply("scan.zarrvectors", "scan_out.ply")      # point cloud
export_ply("surface.zarrvectors", "surface_out.ply") # mesh
```

---

## CSV / XYZ

**Library:** built-in (no extra required).

CSV ingest handles any delimited text file with spatial coordinate columns.
XYZ files (space-separated x y z, optionally with additional columns) are
a special case.

```python
from zarr_vectors.ingest.csv_points import ingest_csv

# Named columns
ingest_csv(
    "cells.csv",
    "cells.zarrvectors",
    chunk_shape=(50., 50., 50.),
    position_columns=["x", "y", "z"],
    attribute_columns=["gene_count", "cell_type", "volume"],
    dtype_map={"cell_type": "int32"},   # override auto-detected dtype
)

# Column-index based (headerless XYZ)
ingest_csv(
    "cloud.xyz",
    "cloud.zarrvectors",
    chunk_shape=(200., 200., 200.),
    position_columns=[0, 1, 2],
    attribute_columns=[3],
    sep=" ",
    header=None,
    skiprows=1,          # skip any comment lines
)
```

```bash
zarr-vectors ingest points cells.csv cells.zarrvectors \
    --chunk-shape 50,50,50 \
    --position-columns x,y,z \
    --attribute-columns gene_count,cell_type
```

### Export to CSV

```python
from zarr_vectors.export.csv_points import export_csv

export_csv("cells.zarrvectors", "cells_out.csv")
# Writes: x,y,z,gene_count,cell_type,volume
```

---

## TRK (TrackVis)

**Library:** `nibabel` (installed with `zarr-vectors[ingest]`).

TRK is the TrackVis tractography format. It stores streamline vertex
coordinates and per-vertex (scalars, properties) data alongside a header
with voxel dimensions and a voxel-to-RAS affine.

### What is preserved

| TRK field | ZVF storage |
|-----------|-------------|
| Vertex coordinates (after affine) | `vertices/` |
| Per-vertex scalars (`.scalars`) | `attributes/<name>/` |
| Per-vertex properties (`.properties`) | `attributes/<name>/` |
| Header voxel dimensions | `root .zattrs["custom_metadata"]["vox_to_ras"]` |

### Usage

```python
from zarr_vectors.ingest.trk import ingest_trk

ingest_trk(
    "tracts.trk",
    "tracts.zarrvectors",
    chunk_shape=(50., 50., 50.),
    apply_affine=True,    # transform voxel → RAS mm using header affine
)
```

```bash
zarr-vectors ingest streams tracts.trk tracts.zarrvectors \
    --chunk-shape 50,50,50 --apply-affine
```

### Export to TRK

```python
from zarr_vectors.export.trk import export_trk

export_trk("tracts.zarrvectors", "tracts_out.trk")
# Restores the original vox-to-RAS affine from custom_metadata if available
```

---

## TCK (MRtrix)

**Library:** `nibabel` (installed with `zarr-vectors[ingest]`).

TCK is the MRtrix tractography format. Coordinates are in scanner (RAS mm)
space; no affine transform is needed.

```python
from zarr_vectors.ingest.tck import ingest_tck

ingest_tck("tracts.tck", "tracts.zarrvectors", chunk_shape=(50., 50., 50.))
```

```bash
zarr-vectors ingest streams tracts.tck tracts.zarrvectors --chunk-shape 50,50,50
```

TCK files do not store per-vertex or per-streamline attributes. If you
have SIFT2 weights or per-streamline metrics from MRtrix, add them as
per-object attributes after ingest:

```python
from zarr_vectors.core.attributes import add_object_attribute
import numpy as np

weights = np.loadtxt("sift2_weights.txt", dtype=np.float32)
add_object_attribute("tracts.zarrvectors", "sift2_weight", weights)
```

---

## TRX (Tractography Exchange)

**Library:** `trx-python` (installed with `zarr-vectors[ingest]`).

TRX is the richest tractography ingest format. It preserves `dps`
(per-streamline scalars and matrices) and `dpp` (per-vertex scalars),
group assignments, and an affine transform.

### What is preserved

| TRX element | ZVF storage |
|-------------|-------------|
| `streamlines/data` + `offsets` | `vertices/` (per-VG) + `links/edges/` |
| `dps/<name>` | `object_attributes/<name>/` |
| `dpp/<name>` | `attributes/<name>/` |
| `groups/<name>` | `groupings/` |
| Header affine | Applied to coordinates (if `apply_affine=True`) |

```python
from zarr_vectors.ingest.trx import ingest_trx

ingest_trx(
    "tracts.trx",
    "tracts.zarrvectors",
    chunk_shape=(50., 50., 50.),
    apply_affine=True,
)
```

```bash
zarr-vectors ingest streams tracts.trx tracts.zarrvectors \
    --chunk-shape 50,50,50
```

### Export to TRX

```python
from zarr_vectors.export.trx import export_trx

export_trx("tracts.zarrvectors", "tracts_out.trx")
# Preserves all object_attributes (→ dps) and attributes (→ dpp)
```

---

## SWC (neuronal morphology)

**Library:** built-in (no extra required).

SWC is the standard format for neuronal morphology files. Each row
represents one node with columns: `id type x y z radius parent_id`.

### What is preserved

| SWC column | ZVF storage |
|-----------|-------------|
| `x, y, z` | `vertices/` |
| `radius` | `attributes/radius/` (float32) |
| `type` | `attributes/swc_type/` (int32) |
| `id` | `attributes/swc_id/` (int64) |
| `parent_id` | Encoded in `links/edges/` as `[child, parent]` |

```python
from zarr_vectors.ingest.swc import ingest_swc, ingest_swc_directory

ingest_swc("neuron.swc", "neuron.zarrvectors",
           chunk_shape=(200., 200., 200.))

# Directory of SWC files → multi-skeleton store
ingest_swc_directory("morphologies/", "connectome.zarrvectors",
                     chunk_shape=(500., 500., 500.))
```

```bash
zarr-vectors ingest skeleton neuron.swc neuron.zarrvectors \
    --chunk-shape 200,200,200
```

### Export to SWC

```python
from zarr_vectors.export.swc import export_swc

export_swc("neuron.zarrvectors", "neuron_out.swc")     # single skeleton
export_swc("connectome.zarrvectors", "neuron_42.swc",  # specific skeleton
           object_id=42)
```

---

## GraphML

**Library:** `networkx` (installed with `zarr-vectors[ingest]`).

GraphML is an XML-based graph exchange format. Spatial coordinates must
be stored as node attributes.

```python
from zarr_vectors.ingest.graphml import ingest_graphml

ingest_graphml(
    "network.graphml",
    "network.zarrvectors",
    chunk_shape=(100., 100., 100.),
    coordinate_attributes=("x", "y", "z"),
    # All other node/edge attributes → vertex/object attributes
)
```

---

## OBJ (Wavefront)

**Library:** built-in parser (no extra for OBJ; PLY/STL require `[ingest]`).

OBJ supports multi-object files (`o` groups), materials (ignored), and
texture coordinates (stored as `attributes/uv/` if present).

```python
from zarr_vectors.ingest.obj import ingest_obj

ingest_obj("brain.obj", "brain.zarrvectors", chunk_shape=(10., 10., 10.))
```

```bash
zarr-vectors ingest mesh brain.obj brain.zarrvectors --chunk-shape 10,10,10
```

### Export to OBJ

```python
from zarr_vectors.export.obj import export_obj

export_obj("brain.zarrvectors", "brain_out.obj")
export_obj("cells.zarrvectors", "cell_42.obj", object_ids=[42])
```

---

## STL

**Library:** `numpy-stl` (installed with `zarr-vectors[ingest]`).

STL files contain only geometry (vertices and faces); no attributes,
materials, or normals beyond per-face normals which are converted to
per-vertex attributes.

```python
from zarr_vectors.ingest.stl import ingest_stl

ingest_stl("organ.stl", "organ.zarrvectors", chunk_shape=(50., 50., 50.))
```

```bash
zarr-vectors ingest mesh organ.stl organ.zarrvectors --chunk-shape 50,50,50
```

STL does not support export from ZVF (one-way ingest only).

---

## Checking what extras are installed

```python
from zarr_vectors.extras import check_extras

print(check_extras())
# {
#   "ingest":  True,   # zarr-vectors[ingest] installed
#   "draco":   False,  # zarr-vectors[draco] not installed
#   "cloud":   True,   # zarr-vectors[cloud] installed
# }
```

Attempting to use an ingest function without the required extra raises
`ImportError` with a message indicating which `pip install` command
is needed.
