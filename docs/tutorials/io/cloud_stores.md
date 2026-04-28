# Cloud stores

ZVF stores on Amazon S3, Google Cloud Storage, and Azure Blob Storage are
accessed using `zarr-vectors[cloud]`, which brings in `s3fs` and `gcsfs`
as dependencies. The read/write API is identical to local stores — only
the path argument changes.

---

## Installation

```bash
pip install "zarr-vectors[cloud]"
```

This installs `s3fs`, `gcsfs`, and the `adlfs` Azure driver alongside the
core package.

---

## Amazon S3

### Anonymous (public) read access

Many open neuroscience datasets on S3 allow anonymous access:

```python
import s3fs
from zarr_vectors.types.points import read_points

fs    = s3fs.S3FileSystem(anon=True)
store = fs.get_mapper("s3://open-neuro-data/datasets/synchrotron.zarrvectors")

result = read_points(store, level=2)   # reads coarse level — fast
print(result["vertex_count"])
```

### Authenticated access

```python
import s3fs

# Uses ~/.aws/credentials or IAM role automatically
fs = s3fs.S3FileSystem(anon=False)

# Or pass credentials explicitly (prefer environment variables instead)
fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ.get("AWS_SESSION_TOKEN"),
)
```

### Writing to S3

```python
import s3fs
import numpy as np
from zarr_vectors.types.points import write_points

fs    = s3fs.S3FileSystem(anon=False)
store = fs.get_mapper("s3://my-bucket/datasets/scan.zarrvectors")

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

write_points(
    store,
    positions,
    chunk_shape=(500.0, 500.0, 500.0),   # larger chunks = fewer S3 objects
    bin_shape=(100.0, 100.0, 100.0),
)
```

**Chunk size guidance for S3.** Each ZVF spatial chunk becomes one S3
object. S3 charges per PUT (write) and GET (read) request. To minimise
cost and request count, use `chunk_shape` values that produce chunks of
at least 100 KB compressed. For typical synchrotron point clouds at 100
000 vertices per chunk (float32, Blosc-compressed), this is roughly
200–500 µm per axis.

### S3 bucket configuration for Neuroglancer serving

To serve a ZVF store from S3 to `zv-ngtools` or the Neuroglancer web
app, configure CORS on the bucket:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders":  ["ETag", "Content-Length"],
    "MaxAgeSeconds":  3600
  }
]
```

Apply with the AWS CLI:

```bash
aws s3api put-bucket-cors \
    --bucket my-bucket \
    --cors-configuration file://cors.json
```

---

## Google Cloud Storage

### Authenticated access

```python
import gcsfs
from zarr_vectors.types.polylines import read_polylines

# Uses Application Default Credentials (gcloud auth application-default login)
fs    = gcsfs.GCSFileSystem(project="my-gcp-project")
store = fs.get_mapper("gs://my-bucket/tracts.zarrvectors")

result = read_polylines(store, level=1)
print(result["polyline_count"])
```

### Writing to GCS

```python
import gcsfs
from zarr_vectors.types.polylines import write_polylines

fs    = gcsfs.GCSFileSystem(project="my-gcp-project")
store = fs.get_mapper("gs://my-bucket/tracts.zarrvectors")

write_polylines(
    store,
    streamlines,
    chunk_shape=(100., 100., 100.),
    bin_shape=(25., 25., 25.),
    geometry_type="streamline",
)
```

### GCS CORS configuration

```bash
gsutil cors set cors.json gs://my-bucket
```

`cors.json` uses the same structure as the S3 example above.

---

## Building a pyramid on a remote store

```python
import s3fs
from zarr_vectors.multiresolution.coarsen import build_pyramid

fs    = s3fs.S3FileSystem(anon=False)
store = fs.get_mapper("s3://my-bucket/scan.zarrvectors")

build_pyramid(
    store,
    level_configs=[
        {"bin_ratio": (2, 2, 2)},
        {"bin_ratio": (4, 4, 4)},
    ],
    n_workers=8,    # parallel chunk reads/writes
)
```

For very large datasets on cloud, the pyramid build is I/O bound. Use
a cloud VM in the same region as the bucket to minimise network latency:
building a pyramid from within AWS `us-east-1` against a bucket in the
same region is ~10× faster than from a laptop.

---

## Consolidated metadata

On stores with many resolution levels and attribute arrays, opening the
store requires one metadata request per Zarr group and array. For a store
with 3 levels × 6 arrays = 18 metadata requests, this adds ~1 s of latency
on S3 (at ~50 ms per request).

Consolidated metadata packs all `.zattrs` and `zarr.json` files into a
single `.zmetadata` key, reducing store-open latency to one request:

```python
import zarr
from zarr_vectors.core.store import open_store

# Write the consolidated metadata file
root = open_store("scan.zarrvectors", mode="r+")
zarr.consolidate_metadata(root.store)
```

After consolidation, subsequent opens are dramatically faster:

```python
# This now takes ~1 request instead of ~18
root = open_store("s3://my-bucket/scan.zarrvectors", mode="r")
```

Regenerate consolidated metadata after adding a resolution level or
modifying `.zattrs`:

```python
root = open_store(store, mode="r+")
zarr.consolidate_metadata(root.store)
```

**Automate at write time.** All `write_*` functions accept
`consolidate=True` to consolidate immediately after writing:

```python
write_points(
    store,
    positions,
    chunk_shape=(500., 500., 500.),
    consolidate=True,   # writes .zmetadata after write completes
)
```

---

## Parallel writes from HPC

For large datasets written by multiple processes (e.g. one process per
HPC node), assign non-overlapping chunk ranges to each process:

```python
from mpi4py import MPI
from zarr_vectors.types.points import write_points_partition

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()

# Partition vertices across ranks by spatial region
x_min = rank       * (total_x / nrank)
x_max = (rank + 1) * (total_x / nrank)
local_positions = positions[(positions[:, 0] >= x_min) &
                             (positions[:, 0] < x_max)]

write_points_partition(
    store,
    local_positions,
    chunk_shape=(200., 200., 200.),
    rank=rank,
    nranks=nrank,
)

# Rank 0 finalises metadata after all ranks finish
comm.Barrier()
if rank == 0:
    import zarr
    from zarr_vectors.core.store import open_store
    root = open_store(store, mode="r+")
    zarr.consolidate_metadata(root.store)
```

`write_points_partition` writes only the chunks whose spatial extent
falls within the partition's x range, avoiding write contention.

---

## Reading from cloud with the lazy API

```python
import s3fs
from zarr_vectors.lazy import ZarrVectorStore

fs = s3fs.S3FileSystem(anon=True)

with ZarrVectorStore(
    fs.get_mapper("s3://open-neuro/scan.zarrvectors"),
    cache_size=128,
    n_workers=4,
) as store:

    # Metadata — no data I/O
    print(store.vertex_count(level=0))   # from .zmetadata

    # Coarse overview — 1 chunk request
    overview = store.read(level=3)

    # Detail in a small region — N chunk requests
    detail = store.read(
        level=0,
        bbox=(np.array([500., 500., 500.]),
              np.array([700., 700., 700.])),
    )
```

---

## Estimating cloud storage cost

A quick estimate for an S3-hosted point cloud store:

```python
from zarr_vectors.core.store import open_store

root = open_store(store, mode="r")
info = zarr.open(root.store).info_complete()

# Total compressed bytes across all arrays and levels
total_bytes = sum(arr.nbytes_stored for arr in zarr.open(root.store).arrays(recurse=True))
print(f"Total compressed size: {total_bytes / 1e9:.2f} GB")

# Number of S3 objects (chunks)
total_chunks = sum(arr.nchunks_initialized for arr in zarr.open(root.store).arrays(recurse=True))
print(f"Total S3 objects: {total_chunks}")
print(f"Monthly S3 storage: ~${total_bytes / 1e9 * 0.023:.2f} (us-east-1 standard)")
print(f"Cost per 1M reads: ~${total_chunks / 1e6 * 0.40:.4f}")
```
