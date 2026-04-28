# Choosing chunk shape and bin shape

`chunk_shape` and `bin_shape` are the two most important parameters when
writing a ZVF store. This guide gives practical heuristics and worked
examples for choosing them well.

---

## The quick version

If you are unsure, start here and tune later:

```python
# Rule of thumb for 3-D biological data
# chunk_shape ≈ L such that expected vertices per chunk ≈ 50 000
# bin_shape   ≈ chunk_shape / 4 per axis

# Point cloud, 10M vertices in a 4000³ µm volume:
# expected/chunk at chunk=500 ≈ 10M × (500³/4000³) ≈ 48 800 ✓
write_points(..., chunk_shape=(500., 500., 500.), bin_shape=(125., 125., 125.))

# Streamlines, 100k tracts in a 200³ mm volume:
# target 200–500 streamlines per chunk
write_polylines(..., chunk_shape=(50., 50., 50.), bin_shape=(10., 10., 10.))
```

---

## Chunk shape guidance

### Primary consideration: I/O unit size

Each chunk is one file on disk or one object in cloud storage. Optimise
`chunk_shape` so that:

- **Typical queries load 1–8 chunks.** If a query always hits exactly
  one chunk, chunk_shape is well-matched to the query size.
- **Each chunk is 50 KB–50 MB compressed.** Chunks smaller than ~50 KB
  have disproportionate per-request overhead; chunks larger than ~50 MB
  load more data than needed for small queries.

### Estimating expected vertices per chunk

```python
def estimate_chunk_vertices(total_vertices, total_volume, chunk_shape):
    chunk_volume = chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
    return total_vertices * chunk_volume / total_volume

# 10M vertices, 4000³ µm volume, 500³ µm chunks
n = estimate_chunk_vertices(10_000_000, 4000**3, (500,500,500))
# n ≈ 48 828 — good
```

Target 10 000–100 000 vertices per chunk for point clouds. For sparser
geometry types (streamlines, skeletons), target 100–500 objects per chunk.

### Chunk shape by use case

| Use case | Suggested `chunk_shape` | Rationale |
|----------|------------------------|-----------|
| Interactive local viewer | 100–200 physical units | Small enough for fast partial loads |
| Cloud serving (S3/GCS) | 300–500 physical units | Fewer objects → lower request cost |
| HPC batch analysis | 500–2000 physical units | Large chunks reduce file count overhead |
| Neuroglancer (fine mesh) | 10–50 physical units | Meshes are dense; small regions needed |
| Synchrotron tractography | 50–100 mm | Typical white-matter query region |

### Anisotropic data

For data with anisotropic voxels (e.g. 1 µm × 1 µm × 4 µm), choose
chunk_shape so that each chunk covers roughly equal physical extents
in all dimensions:

```python
# 1×1×4 µm voxels: make chunks ~200 µm in x,y and ~200 µm in z
chunk_shape = (200., 200., 200.)   # 200×200×50 voxels
```

---

## Bin shape guidance

### Primary consideration: query granularity

`bin_shape` controls the finest spatial resolution of a bounding-box query.
A query that requests a 50³ µm region will load only the bins that overlap
it — not the full chunk.

**Rule of thumb:** set `bin_shape` so that a typical query spans 2–8 bins
per axis.

```python
# Typical query size 50³ µm, chunk_shape 200³ µm:
# 200 / bin_shape ≥ query_size / bin_shape ≥ 2
# → bin_shape ≤ 100 µm and bin_shape ≥ 25 µm
# → bin_shape = 50 µm (4 bins per axis) ✓
```

### Bins per chunk and index overhead

The VG index is `B_per_chunk × 16 bytes` per chunk. For `bin_shape =
chunk_shape / 4` in 3-D: `B_per_chunk = 64`, index size = 1 KB per chunk.
This is negligible compared to vertex data.

| `chunk/bin` ratio per axis | `B_per_chunk` (3-D) | Index size per chunk |
|---------------------------|--------------------|--------------------|
| 2 | 8 | 128 bytes |
| 4 | 64 | 1 KB |
| 8 | 512 | 8 KB |
| 16 | 4096 | 64 KB |

Ratios above 8 per axis are rarely needed and add non-trivial index overhead.

### One bin per chunk (no sub-chunk indexing)

Omit `bin_shape` or set it equal to `chunk_shape` when:

- All queries read entire chunks.
- The store is written for sequential batch processing only.
- You need maximum compatibility with tools that do not understand bins.

```python
write_points(..., chunk_shape=(200., 200., 200.))
# bin_shape defaults to chunk_shape → 1 bin per chunk
```

---

## Worked examples

### Synchrotron point cloud (HiP-CT)

```
Dataset:     200M vertices in 8000³ µm HiP-CT scan
Query size:  ~100³ µm (interactive viewport at high zoom)
Platform:    S3, Neuroglancer serving
```

```python
# 500³ chunk → ~245k vertices/chunk (dense dataset)
# Reduce to 200³ for ~16k vertices/chunk — more manageable
# bin_shape = 50³ → 4³ = 64 bins/chunk, query granularity = 50 µm ✓

write_points(
    store,
    positions,
    chunk_shape=(200., 200., 200.),
    bin_shape=(50., 50., 50.),
)
```

### DWI tractography (1M streamlines, 50-vertex average)

```
Dataset:     1M streamlines × 50 vertices = 50M vertices
             Streamlines span a 180³ mm MRI volume
Query size:  Typically one white-matter bundle (~30×30×80 mm)
Platform:    Local analysis + Neuroglancer
```

```python
# At chunk_shape=50³ mm: 50³/180³ × 1M ≈ 2143 streams/chunk ✓
# (target 200–500; slightly high but acceptable)
# bin_shape = 10³ mm → 125 bins/chunk; query granularity = 10 mm ✓

write_polylines(
    store,
    streamlines,
    chunk_shape=(50., 50., 50.),
    bin_shape=(10., 10., 10.),
)
```

### EM connectome skeletons (10 000 neurons)

```
Dataset:     10 000 skeletons, avg 5000 nodes/neuron = 50M nodes
             Data in a 1000³ µm EM volume
Query:       Single neuron retrieval by ID (object_index lookup)
             Spatial query by region (~100³ µm)
Platform:    Local analysis
```

```python
# chunk_shape = 100³ µm → 100³/1000³ × 50M ≈ 50k vertices/chunk ✓
# bin_shape = 25³ µm → 4³ = 64 bins/chunk, 25 µm resolution ✓

write_graph(
    store,
    positions,
    edges,
    chunk_shape=(100., 100., 100.),
    bin_shape=(25., 25., 25.),
    geometry_type="skeleton",
)
```

---

## After writing: profile and tune

Run the info command and check the distribution of vertices per chunk:

```bash
zarr-vectors info scan.zarrvectors --verbose
# resolution_0:  100000 vertices, 125 chunks
#   vertices/chunk:  min=124  median=812  max=1843  p95=1602
```

If `median << 10000`, your `chunk_shape` is too small relative to the
data density — increase it. If `p95 >> 100000`, chunks may be too large
for interactive use — decrease it.
