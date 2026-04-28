# Lazy loading

The ZVF read functions (`read_points`, `read_polylines`, etc.) are eager:
they fetch and return all requested data immediately. For large stores or
remote datasets, an eager read of the full store is impractical.

The **lazy API** provides a `ZarrVectorStore` object that opens the store
metadata without reading any array data. Array slices are fetched on demand
— only when accessed. This is the recommended access pattern for:

- Stores too large to fit in memory.
- Remote stores (S3, GCS) where each array fetch is a network request.
- Interactive viewers that need the coarsest level first and finer levels
  on demand.
- Analysis pipelines that filter by metadata before deciding which data
  to load.

---

## Opening a store lazily

```python
from zarr_vectors.lazy import ZarrVectorStore

# Opens metadata only — no vertex data fetched
store = ZarrVectorStore("synchrotron.zarrvectors")

print(store.geometry_type)           # "point_cloud"
print(store.spatial_dims)            # 3
print(store.chunk_shape)             # (200.0, 200.0, 200.0)
print(store.levels)                  # [0, 1, 2, 3]
print(store.vertex_count(level=0))   # 500000 (from metadata, no data read)
print(store.vertex_count(level=2))   # 8022
print(store.bounding_box)            # (array([0,0,0]), array([2000,2000,2000]))
```

Opening a remote store is identical — pass an fsspec URL:

```python
import s3fs
from zarr_vectors.lazy import ZarrVectorStore

fs    = s3fs.S3FileSystem(anon=True)
store = ZarrVectorStore(
    fs.get_mapper("s3://open-neuro/synchrotron.zarrvectors")
)
print(store.vertex_count(level=0))   # metadata only — one S3 LIST request
```

---

## Level-of-detail reads

### Automatic level selection

`auto_level` selects the coarsest level whose bin size is smaller than
a given target resolution:

```python
# Load the coarsest level adequate for a 200 µm resolution viewport
result = store.read(target_resolution=200.0)
print(result["level"])           # 2 (bin_shape = [200, 200, 200] at level 2)
print(result["vertex_count"])    # 8022

# Load for a detailed 50 µm view
result = store.read(target_resolution=50.0)
print(result["level"])           # 0 (finest level; bin_shape = [50, 50, 50])
```

`target_resolution` is compared against `bin_shape` at each level. The
selected level is the highest level `N` such that
`max(bin_shape[N]) ≤ target_resolution`.

### Bbox + level-of-detail

Combine spatial restriction with level selection for viewport-driven reads:

```python
# Viewport: 500³ µm region, medium detail
result = store.read(
    bbox=(np.array([800., 800., 800.]), np.array([1300., 1300., 1300.])),
    target_resolution=100.0,
)
print(result["level"])           # level where bin_shape ≤ 100 µm
print(result["vertex_count"])
```

### Explicit level override

```python
result = store.read(level=1, bbox=(lo, hi))
```

---

## Streaming large stores chunk by chunk

For datasets that do not fit in memory, iterate over chunks instead of
loading the full store at once:

### Iterate over all chunks at a level

```python
for chunk_coord, chunk_data in store.iter_chunks(level=0):
    positions  = chunk_data["positions"]    # (N_chunk, 3) float32
    attributes = chunk_data["attributes"]
    # Process this chunk — e.g. compute statistics, write to database
    yield process_chunk(positions, attributes)
```

### Iterate over chunks in a bounding box

```python
lo = np.array([0., 0., 0.])
hi = np.array([500., 500., 500.])

for chunk_coord, chunk_data in store.iter_chunks(level=0, bbox=(lo, hi)):
    print(chunk_coord, chunk_data["positions"].shape)
```

This yields only the chunks that overlap the bbox, in row-major order.
Each chunk is fetched and decompressed exactly once.

### Streaming statistics over a large point cloud

```python
total_count = 0
intensity_sum = 0.0

for _, chunk_data in store.iter_chunks(level=0):
    n = len(chunk_data["positions"])
    total_count   += n
    intensity_sum += chunk_data["attributes"]["intensity"].sum()

mean_intensity = intensity_sum / total_count
print(f"Mean intensity over {total_count} points: {mean_intensity:.4f}")
```

---

## Lazy array access

The `ZarrVectorStore` exposes each array as a lazy `zarr.Array` that can
be sliced directly:

```python
# Access the raw vertices array for level 0 without reading it
verts_array = store.raw_array("vertices", level=0)
print(verts_array.shape)    # (Cx, Cy, Cz, N_max, 3)
print(verts_array.dtype)    # float32

# Read one specific chunk (fetches only that chunk from disk/S3)
chunk_verts = verts_array[2, 3, 1]   # chunk at grid coord (2,3,1)
print(chunk_verts.shape)    # (N_max, 3) — may include fill-value rows

# Access the VG index
vg_offsets = store.raw_array("vertex_group_offsets", level=0)
offsets_chunk = vg_offsets[2, 3, 1]   # (B_per_chunk, 2)
```

### Accessing object attributes without loading vertices

```python
# Read all per-streamline FA values without fetching vertex data
fa_array = store.raw_array("object_attributes/mean_fa", level=0)
fa_values = fa_array[:]   # shape (n_objects,) — one request

# Filter objects by FA
high_fa_ids = np.where(fa_values > 0.5)[0]
print(f"{len(high_fa_ids)} high-FA streamlines")
```

---

## Remote stores (S3 / GCS)

### S3 with credentials

```python
import s3fs
from zarr_vectors.lazy import ZarrVectorStore

fs = s3fs.S3FileSystem(
    key="...",
    secret="...",
    # or: profile_name="my-aws-profile"
)
store = ZarrVectorStore(fs.get_mapper("s3://my-bucket/dataset/tracts.zarrvectors"))
```

### GCS

```python
import gcsfs
from zarr_vectors.lazy import ZarrVectorStore

fs    = gcsfs.GCSFileSystem(project="my-gcp-project")
store = ZarrVectorStore(fs.get_mapper("gs://my-bucket/tracts.zarrvectors"))
```

### Performance on object stores

Remote stores have per-request latency (~50–200 ms for S3). The lazy API
minimises requests by:

1. Reading consolidated metadata in a single request (if available).
2. Batching chunk reads for spatial queries (multiple chunks fetched in
   parallel if `n_workers > 1`).
3. Caching decompressed chunks in an LRU cache (configurable size).

```python
store = ZarrVectorStore(
    fs.get_mapper("s3://my-bucket/tracts.zarrvectors"),
    cache_size=256,     # cache up to 256 decompressed chunks in memory
    n_workers=8,        # fetch up to 8 chunks in parallel
)
```

### Prefetching for sequential access

When iterating over chunks sequentially, enable prefetch to overlap
decompression of future chunks with processing of the current one:

```python
for chunk_coord, chunk_data in store.iter_chunks(level=1, prefetch=4):
    # Process current chunk while next 4 are fetching in the background
    yield analyse(chunk_data)
```

---

## ZarrVectorStore API summary

```python
from zarr_vectors.lazy import ZarrVectorStore

store = ZarrVectorStore(path_or_store)

# Metadata (no data I/O)
store.geometry_type           # str
store.spatial_dims            # int
store.chunk_shape             # tuple
store.bin_shape               # tuple (at level 0)
store.levels                  # list[int]
store.n_objects               # int (for discrete-object types)
store.bounding_box            # (lo, hi) arrays

# Per-level metadata
store.vertex_count(level)     # int
store.bin_shape_at(level)     # tuple
store.bin_ratio_at(level)     # tuple
store.object_count_at(level)  # int (discrete-object types)

# Data reads
store.read(level, bbox, target_resolution, attributes)
store.iter_chunks(level, bbox, prefetch, n_workers)
store.raw_array(array_path, level)

# Utilities
store.close()
store.__enter__() / store.__exit__()   # context manager
```

### Using as a context manager

```python
with ZarrVectorStore("scan.zarrvectors") as store:
    result = store.read(level=2)
# Store is closed and cache is freed on exit
```

---

## Common patterns

### Thumbnail generation

Load the coarsest level for a quick full-volume thumbnail:

```python
store    = ZarrVectorStore("scan.zarrvectors")
coarsest = store.levels[-1]
result   = store.read(level=coarsest)
# Use result["positions"] to render a low-density overview
```

### Memory-bounded streaming

Process a store in chunks, keeping peak memory under a target:

```python
MEMORY_LIMIT_BYTES = 1 * 1024**3   # 1 GB

chunk_bytes   = store.vertex_count(level=0) / len(list(store.iter_chunks(level=0))) * 12
chunks_at_once = int(MEMORY_LIMIT_BYTES / chunk_bytes)

batch = []
for i, (coord, data) in enumerate(store.iter_chunks(level=0)):
    batch.append(data["positions"])
    if len(batch) >= chunks_at_once:
        process_batch(np.concatenate(batch))
        batch.clear()

if batch:
    process_batch(np.concatenate(batch))
```
