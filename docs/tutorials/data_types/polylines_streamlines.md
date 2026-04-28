# Polylines and streamlines

Polylines and streamlines store ordered vertex sequences — paths through
3-D space. Use `polyline` for general ordered paths (vascular centrelines,
cell migration tracks, traced axons). Use `streamline` when the paths come
from MRI or synchrotron tractography and you need to store propagation
metadata (step size, seeding strategy, reference image).

The two types are technically identical in their on-disk arrays; the
distinction is in the `geometry_type` constant and the optional
tractography-specific metadata keys. All write/read functions in this
tutorial apply equally to both.

All examples use `zarr-vectors` base install except the TRK/TCK/TRX ingest
sections, which require `zarr-vectors[ingest]`.

---

## Writing streamlines

### Minimal write

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines

rng = np.random.default_rng(0)

# 500 streamlines, each a random walk of 40 steps in a 200³ µm volume
streamlines = [
    rng.normal(0, 25, (40, 3)).cumsum(axis=0).astype(np.float32)
    for _ in range(500)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    geometry_type="streamline",    # declares tractography type
)
```

### Write with groups and attributes

Groups allow you to tag streamlines as belonging to named bundles or
experimental conditions. Per-streamline attributes (e.g. mean FA, length)
are stored in `object_attributes/`.

```python
n = 1000
streamlines = [
    rng.normal(0, 30, (rng.integers(20, 80), 3)).cumsum(0).astype(np.float32)
    for _ in range(n)
]

# Compute arc lengths for use as an attribute
lengths = np.array([
    np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))
    for s in streamlines
], dtype=np.float32)

# Per-streamline FA (simulated)
mean_fa = rng.uniform(0.2, 0.8, n).astype(np.float32)

# Per-vertex FA (simulated) — one value per vertex along each streamline
per_vertex_fa = [rng.uniform(0.1, 0.9, len(s)).astype(np.float32)
                 for s in streamlines]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    geometry_type="streamline",
    groups={
        "CST":  list(range(300)),           # corticospinal tract
        "AF":   list(range(300, 600)),      # arcuate fasciculus
        "UF":   list(range(600, 1000)),     # uncinate fasciculus
    },
    object_attributes={
        "length":  lengths,
        "mean_fa": mean_fa,
    },
    attributes={
        "fa": per_vertex_fa,               # list of per-vertex arrays
    },
    streamline_metadata={
        "step_size":             0.5,
        "step_size_unit":        "mm",
        "propagation_algorithm": "probabilistic",
        "seeding_strategy":      "wm_mask",
    },
)
```

---

## Reading streamlines

### Read all streamlines

```python
from zarr_vectors.types.polylines import read_polylines

result = read_polylines("tracts.zarrvectors")
print(result["polyline_count"])           # 1000
print(len(result["polylines"]))           # 1000 — list of (N_i, 3) arrays
print(result["polylines"][0].shape)       # (N_0, 3) — first streamline
```

### Read by object ID

Object IDs are stable integer identifiers assigned at write time (0-indexed):

```python
result = read_polylines("tracts.zarrvectors", object_ids=[0, 42, 99])
print(result["polyline_count"])            # 3
print(result["polylines"][1].shape)        # shape of streamline 42
```

### Read by group

```python
result = read_polylines("tracts.zarrvectors", group_ids=["CST"])
print(result["polyline_count"])            # 300

# Read from multiple groups at once
result = read_polylines("tracts.zarrvectors", group_ids=["CST", "AF"])
print(result["polyline_count"])            # 600
```

### Read object attributes without fetching vertices

For large stores where you only need the per-streamline metadata (e.g. to
filter by length before loading geometry):

```python
from zarr_vectors.core.store import open_store

root = open_store("tracts.zarrvectors", mode="r")
lengths  = root["resolution_0"]["object_attributes"]["length"][:]
mean_fa  = root["resolution_0"]["object_attributes"]["mean_fa"][:]

# Select long, high-FA streamlines
good_ids = np.where((lengths > 100) & (mean_fa > 0.4))[0]

# Now fetch only those streamlines
result = read_polylines("tracts.zarrvectors", object_ids=good_ids.tolist())
print(result["polyline_count"])
```

---

## Spatial bounding-box queries

A bbox query returns all streamlines that have **at least one vertex**
in the bounding box. The full streamline geometry is returned (not clipped
to the bbox), preserving path continuity.

```python
lo = np.array([-50.0, -50.0, -50.0])
hi = np.array([ 50.0,  50.0,  50.0])

result = read_polylines(
    "tracts.zarrvectors",
    bbox=(lo, hi),
)
print(result["polyline_count"])   # streamlines passing through the bbox
```

To clip streamlines to the bbox boundary (splitting paths that enter and
exit multiple times), pass `clip=True`. This changes the number of
polylines returned and is not lossless:

```python
result = read_polylines("tracts.zarrvectors", bbox=(lo, hi), clip=True)
```

### Combining bbox and object attributes

A common analysis pattern: spatial query to find candidates, then filter
by attributes:

```python
# Step 1: find streamlines in a region of interest
result = read_polylines(
    "tracts.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
    include_object_attributes=True,
)

# Step 2: filter by FA
high_fa_mask = result["object_attributes"]["mean_fa"] > 0.5
high_fa_ids  = np.array(result["object_ids"])[high_fa_mask]

print(f"{len(high_fa_ids)} high-FA streamlines in region")
```

---

## Multi-resolution pyramids

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zarrvectors",
    level_configs=[
        # Level 1: 8× vertex reduction, all streamlines retained
        {"bin_ratio": (2, 2, 2), "object_sparsity": 1.0,
         "sparsity_strategy": "spatial_coverage"},
        # Level 2: 64× vertex reduction, keep best-coverage 25%
        {"bin_ratio": (4, 4, 4), "object_sparsity": 0.25,
         "sparsity_strategy": "spatial_coverage"},
    ],
    attribute_aggregation={"fa": "mean"},
)
```

After building:

```bash
zarr-vectors info tracts.zarrvectors
# resolution_0:  1000 streamlines, ~40 000 vertices
# resolution_1:  1000 streamlines, ~5 800 vertices (8× reduction)
# resolution_2:  250 streamlines,  ~365 vertices   (64× × 4× = 256× total)
```

---

## Ingesting from tractography formats

All ingest functions require `zarr-vectors[ingest]`.

### TRK (TrackVis)

```python
from zarr_vectors.ingest.trk import ingest_trk

ingest_trk(
    "tracts.trk",
    "tracts.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    bin_shape=(10.0, 10.0, 10.0),
    apply_affine=True,   # transform voxel → RAS mm using TRK header affine
)
```

TRK stores coordinates in voxel space with an affine in the header.
`apply_affine=True` (default) applies the affine before writing. Pass
`apply_affine=False` to keep voxel coordinates.

### TCK (MRtrix)

```python
from zarr_vectors.ingest.tck import ingest_tck

ingest_tck(
    "tracts.tck",
    "tracts.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
)
```

TCK coordinates are in scanner mm (no affine transform needed).

### TRX (Tractography Exchange format)

TRX is the richest ingest format — it preserves `dps` (per-streamline)
and `dpp` (per-vertex) attribute arrays, and group assignments:

```python
from zarr_vectors.ingest.trx import ingest_trx

ingest_trx(
    "tracts.trx",
    "tracts.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    # dps attributes → object_attributes/
    # dpp attributes → attributes/
    # groups → groupings/
)
```

After ingest, inspect preserved attributes:

```python
result = read_polylines("tracts.zarrvectors", include_object_attributes=True)
print(list(result["object_attributes"].keys()))  # ["mean_fa", "cluster_id", …]
print(list(result["attributes"].keys()))          # ["fa", "md", …]
```

### CLI ingest

```bash
zarr-vectors ingest streams tracts.trk tracts.zarrvectors \
    --chunk-shape 50,50,50 --apply-affine

zarr-vectors ingest streams tracts.tck tracts.zarrvectors \
    --chunk-shape 50,50,50

zarr-vectors ingest streams tracts.trx tracts.zarrvectors \
    --chunk-shape 50,50,50
```

---

## Exporting

```python
from zarr_vectors.export.trk import export_trk
from zarr_vectors.export.trx import export_trx

# Export level 0 to TRK
export_trk("tracts.zarrvectors", "tracts_out.trk")

# Export to TRX — preserves all attributes and groups
export_trx("tracts.zarrvectors", "tracts_out.trx")
```

---

## Common pitfalls

**Cross-chunk links not generated for same-chunk vertices.**
`cross_chunk_links/` only stores connections between different chunks.
If two consecutive vertices of a streamline happen to fall in the same
chunk, the connection is stored in `links/edges/` (intra-chunk), not in
`cross_chunk_links/`. Manually inspecting `cross_chunk_links/` will not
show all edges — use `read_polylines(object_ids=[k])` to retrieve the
complete vertex sequence.

**Streamlines entirely outside the bbox are not returned.**
A bbox query returns streamlines with at least one vertex inside the bbox.
A streamline that passes through the bbox interior but has no vertices
inside (because vertices are spaced far apart) will not be returned. Reduce
`step_size` or `bin_shape` relative to the streamline vertex spacing to
ensure all passing streamlines are captured.

**Group IDs vs group names.**
Groups can be specified by integer ID or by string name. String names are
stored in `groupings_attributes/name/`. Pass string names to `group_ids`
for readability; the reader resolves them to integer IDs internally.

**Lost attributes after rechunking.**
Rechunking reorders vertices within each chunk. Per-vertex attribute arrays
are reordered with the same permutation automatically. However, if you
manually wrote attribute arrays without going through the write functions,
the reordering will not be applied and attributes will be misaligned after
rechunking. Always use the provided write API.
