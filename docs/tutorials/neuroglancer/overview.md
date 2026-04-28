# Neuroglancer integration overview

[Neuroglancer](https://github.com/google/neuroglancer) is a WebGL-based
viewer for petascale volumetric data, widely used in connectomics, brain
imaging, and synchrotron science. It natively understands several data
formats (OME-Zarr image volumes, Neuroglancer precomputed, N5), but does
not have native support for the Zarr Vector Format.

[`zv-ngtools`](https://github.com/BRIDGE-Neuroscience/zv-ngtools) bridges
this gap. It is a fork of
[`ngtools`](https://github.com/neuroscales/ngtools) — a collection of
Neuroglancer utilities — extended with a `zarr_vectors` layer type that
translates ZVF stores into Neuroglancer layers on the fly.

---

## Why Neuroglancer cannot read ZVF natively

Neuroglancer's data source plugins expect data in specific binary formats
(precomputed, N5, OME-Zarr). ZVF stores vertex data as spatially chunked
Zarr arrays with a VG index — a structure Neuroglancer does not understand
without a mediating translation layer.

`zv-ngtools` provides that layer: it runs a local HTTP file server that
intercepts Neuroglancer's chunk requests, reads the requested spatial
region from a ZVF store using `zarr-vectors-py`, and returns the data in
a format Neuroglancer expects.

---

## Two integration paths

### Path A — Local serving via `LocalNeuroglancer`

A Python process (`zv-ngtools`) runs a local Tornado HTTP server alongside
a Neuroglancer browser tab. The server:

1. Receives chunk requests from Neuroglancer (HTTP range requests).
2. Uses `zarr-vectors-py` to read the requested VG slices.
3. Translates the ZVF data to the Neuroglancer layer protocol.
4. Returns the response to the browser.

LOD is driven by the ZVF resolution pyramid: as the user zooms out in
Neuroglancer, the server switches to coarser levels automatically, using
the `bin_shape` at each level to select the appropriate resolution.

```
Python process                         Browser
┌───────────────────────────────┐     ┌─────────────────────────────┐
│  LocalNeuroglancer             │     │  Neuroglancer (WebGL)        │
│  ├─ Tornado HTTP server        │◄────│  chunk requests              │
│  │   (port 9123)               │────►│  rendered layers             │
│  └─ zarr-vectors reader        │     │                              │
│      ├─ scan.zarrvectors       │     └─────────────────────────────┘
│      └─ tracts.zarrvectors     │
└───────────────────────────────┘
```

This path is best for local analysis and collaborative review sessions
where the data lives on local disk or a mounted network share.

### Path B — Precomputed export for static hosting

Convert a ZVF store to the Neuroglancer precomputed format and upload to
a public HTTP server (S3, GCS, nginx). Neuroglancer fetches the data
directly without any intermediary Python process.

This path is best for sharing published datasets with collaborators who
will not install `zv-ngtools`, or for embedding Neuroglancer links in
publications and websites.

See [Precomputed export](precomputed_export.md) for the conversion
workflow.

---

## What `zv-ngtools` adds over upstream `ngtools`

`zv-ngtools` is a fork of `ngtools` (neuroscales/ngtools). All upstream
`ngtools` functionality is preserved; the fork adds:

| Feature | Upstream `ngtools` | `zv-ngtools` |
|---------|-------------------|-------------|
| Load OME-Zarr image volumes | ✓ | ✓ |
| Load TRK / TCK tractography | ✓ | ✓ |
| Load NIfTI, MGH, TIFF | ✓ | ✓ |
| Load `.zarrvectors` stores | ✗ | ✓ |
| ZVF-aware LOD selection | ✗ | ✓ |
| `zarr_vectors` layer type | ✗ | ✓ |
| Precomputed export from ZVF | ✗ | ✓ |
| Point cloud layer rendering | ✗ | ✓ |
| Streamline layer rendering | ✗ (only TRK/TCK) | ✓ (from ZVF) |

The `zarr://` URL scheme (for OME-Zarr stores) works identically in both
forks. Only the `zarr_vectors://` scheme is new.

---

## Layer type mapping

| ZVF geometry type | Neuroglancer layer type | Notes |
|------------------|------------------------|-------|
| `point_cloud` | `annotation` (point) | Rendered as 3-D points; size and colour from attributes |
| `line` | `annotation` (line) | Rendered as line segments |
| `polyline` / `streamline` | `annotation` (line) | Rendered as connected line sequences |
| `graph` / `skeleton` | `segmentation` (mesh-less) or custom skeleton layer | SWC-style rendering |
| `mesh` | `segmentation` (mesh) or `annotation` (surface) | Draco-compressed meshes supported |

For types without a perfect Neuroglancer native equivalent (e.g. general
graphs), `zv-ngtools` uses the closest Neuroglancer layer type and adds a
server-side translation step.

---

## Coordinate system alignment

ZVF stores and Neuroglancer image volumes can be displayed together when
they share a coordinate system. Neuroglancer uses a global coordinate space
for all layers; each layer specifies its own coordinate transform.

`zv-ngtools` reads the `coordinate_system` and `multiscales.axes` fields
from the ZVF store's `.zattrs` to set the Neuroglancer layer transform.
Stores declared in `RAS` (Right-Anterior-Superior) space are automatically
aligned with Neuroglancer's default RAS coordinate system.

For stores in voxel space, pass an explicit affine when loading:

```python
viewer.add(
    "scan.zarrvectors",
    transform=np.array([
        [0.004, 0,     0,     0],     # 4 µm per voxel
        [0,     0.004, 0,     0],
        [0,     0,     0.025, 0],     # 25 µm z-step
        [0,     0,     0,     1],
    ]),
)
```

---

## Quick-start checklist

Before working through the detailed tutorials:

- [ ] `pip install git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git`
- [ ] A web browser that supports WebGL 2 (Chrome, Firefox, Edge)
- [ ] A `.zarrvectors` store with at least a base level (level 0)

Optionally, for best performance:

- [ ] A multi-level pyramid (run `build_pyramid` first)
- [ ] Consolidated metadata (`zarr.consolidate_metadata`)
