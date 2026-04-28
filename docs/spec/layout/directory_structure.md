# Directory structure

## Terms

**Store root**
: The top-level directory (or object-store prefix) of a ZVF store.
  Conventionally named with a `.zarrvectors` extension. Contains the root
  `zarr.json`, root `.zattrs`, all resolution level groups, and
  `metadata.json`.

**Resolution level group**
: A Zarr group at path `resolution_<N>/` within the store root, where `N`
  is a non-negative integer. Level 0 is always the full-resolution level.
  Higher levels are progressively coarser.

**Array group**
: A Zarr group within a resolution level that holds a single logical array
  (e.g. `vertices/`, `attributes/intensity/`). The group contains a
  `zarr.json` with the array metadata and one file per chunk in the `c/`
  sub-tree.

**`metadata.json`**
: A plain-text JSON file at the store root containing human-readable
  summary information about the store (total vertex count, bounding box,
  creation timestamp). Not used by the read/write API; present for
  inspection and provenance.

**`parametric/`**
: An optional sub-group at the store root for algebraic (non-vertex-based)
  geometry objects such as planes, spheres, and ellipsoids. Not chunked
  spatially; stores a single array of object parameter tuples.

---

## Introduction

The on-disk layout of a ZVF store follows a clear hierarchy: store root →
resolution levels → array groups → chunk files. Every path in the hierarchy
has a defined meaning; there are no opaque binary blobs. This page documents
every node in the tree for each supported geometry type.

Understanding the directory structure is essential for contributors
implementing new geometry types, validation tools, or custom readers. It is
also useful for debugging: if a store fails validation, the first step is
often to inspect the directory tree directly.

---

## Technical reference

### Full annotated tree (point cloud)

```
dataset.zarrvectors/
│
├── zarr.json                    # Zarr v3 root group metadata
├── .zattrs                      # ZVF root metadata (see root_metadata.md)
├── metadata.json                # human-readable summary
│
├── resolution_0/                # full-resolution level
│   ├── zarr.json                # Zarr v3 group metadata
│   ├── .zattrs                  # per-level metadata (bin_ratio, sparsity)
│   │
│   ├── vertices/                # spatial positions — shape (N_chunk, D)
│   │   ├── zarr.json
│   │   └── c/
│   │       ├── 0/0/0            # chunk at grid coord (0,0,0)
│   │       ├── 0/0/1            # chunk at grid coord (0,0,1)
│   │       └── …
│   │
│   ├── vertex_group_offsets/    # VG index — shape (B_chunk, 2) per chunk
│   │   ├── zarr.json
│   │   └── c/ …
│   │
│   └── attributes/              # per-vertex attribute arrays
│       ├── intensity/           # one sub-group per named attribute
│       │   ├── zarr.json
│       │   └── c/ …
│       └── label/
│           ├── zarr.json
│           └── c/ …
│
└── resolution_1/                # coarser level (bin_ratio declared in .zattrs)
    └── [same structure as resolution_0]
```

### Full annotated tree (streamline / polyline)

The streamline tree adds connectivity and object-model arrays:

```
tracts.zarrvectors/
│
├── zarr.json
├── .zattrs
├── metadata.json
│
└── resolution_0/
    ├── zarr.json
    ├── .zattrs
    │
    ├── vertices/                # vertex positions
    ├── vertex_group_offsets/    # VG index
    │
    ├── links/                   # connectivity
    │   └── edges/               # (N_seg, 2) int32 — consecutive vertex pairs
    │       ├── zarr.json
    │       └── c/ …
    │
    ├── attributes/              # per-vertex attributes (e.g. FA, MD)
    │
    ├── object_index/            # object ID → (chunk_flat, vg_index) mapping
    │   ├── zarr.json            # shape (n_objects, 2) int64
    │   └── c/ …
    │
    ├── object_attributes/       # per-object scalars (e.g. mean FA)
    │   ├── mean_fa/
    │   └── tract_length/
    │
    ├── groupings/               # group ID → [object IDs]
    │   ├── zarr.json
    │   └── c/ …
    │
    ├── groupings_attributes/    # per-group metadata
    │
    └── cross_chunk_links/       # inter-chunk vertex connections
        ├── zarr.json            # shape (n_links, 2) int64
        └── c/ …
```

### Full annotated tree (graph / skeleton)

```
neuron.zarrvectors/
├── zarr.json
├── .zattrs
├── metadata.json
└── resolution_0/
    ├── vertices/
    ├── vertex_group_offsets/
    ├── links/
    │   └── edges/               # (n_edges, 2) int32 or int64
    ├── attributes/
    ├── object_index/
    ├── object_attributes/
    └── cross_chunk_links/
```

### Full annotated tree (mesh)

```
brain.zarrvectors/
├── zarr.json
├── .zattrs
├── metadata.json
└── resolution_0/
    ├── vertices/
    ├── vertex_group_offsets/
    ├── links/
    │   └── faces/               # (n_faces, 3) int32 — triangle vertex indices
    ├── attributes/
    ├── object_index/
    └── object_attributes/
```

### Parametric objects

The optional `parametric/` group is not spatially chunked. It holds
algebraic objects (planes, spheres, ellipsoids) as a flat array of parameter
tuples:

```
dataset.zarrvectors/
├── …
└── parametric/
    ├── zarr.json
    ├── objects/                 # (n_parametric, param_dim) float64
    │   ├── zarr.json
    │   └── c/0
    └── object_attributes/
        └── label/
```

### Naming rules

Resolution level directories must be named `resolution_<N>` where `N` is a
non-negative integer. There is no requirement that levels be contiguous (a
store may have `resolution_0` and `resolution_2` without `resolution_1`),
but contiguous numbering from 0 is strongly recommended.

Array group names within a level are fixed by this specification. Custom
arrays may not be added at the array group level without a spec extension.
Per-vertex and per-object custom attributes must be placed under
`attributes/` and `object_attributes/` respectively.

### Required vs optional nodes

| Path | Required for | Notes |
|------|-------------|-------|
| `zarr.json` (root) | All types | Zarr v3 group node |
| `.zattrs` (root) | All types | ZVF root metadata |
| `metadata.json` | All types | Recommended; not read by API |
| `resolution_0/` | All types | At least one level required |
| `vertices/` | All types | |
| `vertex_group_offsets/` | All types | Required for spatial queries |
| `links/edges/` | polyline, streamline, graph, skeleton | |
| `links/faces/` | mesh | |
| `attributes/` | All types | Optional if no per-vertex attributes |
| `object_index/` | polyline, streamline, graph, skeleton, mesh | |
| `object_attributes/` | Any type | Optional |
| `groupings/` | Any discrete-object type | Optional |
| `cross_chunk_links/` | polyline, streamline | Required when objects span chunks |
| `parametric/` | Any type | Optional |
