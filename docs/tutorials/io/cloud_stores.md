# Cloud stores

ZV stores on Amazon S3, Google Cloud Storage, Azure Blob Storage, and
public HTTP are accessed through the **backend layer** described in the
[store types spec page](../../spec/foundations/store_types.md). The
read/write API is identical to local stores — only the URL changes.

---

## Installation

The library ships with two cloud-capable backends; install whichever
matches your needs. `obstore` is preferred (Rust-based, faster parallel
reads), and the library auto-prefers it when both are installed.

```bash
pip install "zarr-vectors[obstore]"    # preferred
# OR
pip install "zarr-vectors[cloud]"      # fsspec + s3fs/gcsfs/adlfs (fallback)
```

You can also install both — the library will pick `obstore` and fall back
to `fsspec` for any URL scheme it can't handle.

---

## Backend resolution at a glance

When you pass a cloud URL to any `read_*` / `write_*` / `open_store` /
`open_zvr` call, the backend is chosen in this order:

1. **Explicit `backend=` kwarg** — e.g. `backend="fsspec"` forces fsspec
   even if obstore is installed.
2. **`ZARR_VECTORS_BACKEND` environment variable** — e.g.
   `export ZARR_VECTORS_BACKEND=obstore`.
3. **URL-scheme auto-detect** — `s3://`, `gs://`, `gcs://`, `az://`,
   `azure://`, `abfs://`, `http(s)://` → `obstore` if installed else
   `fsspec`.

If neither cloud backend is installed for a cloud URL, the call raises a
`StoreError` with an install hint.

See [`zarr_vectors/core/backends/__init__.py`](../../../zarr_vectors/core/backends/__init__.py)
for the canonical scheme table and
[`tests/test_backends.py`](../../../tests/test_backends.py) for the
test matrix.

---

## Amazon S3

### Anonymous (public) read access

Many open neuroscience datasets on S3 allow anonymous access. The
backend layer handles it transparently — there's no `anon=True` kwarg
to pass:

```python
from zarr_vectors.types.points import read_points

result = read_points(
    "s3://open-neuro-data/datasets/synchrotron.zarrvectors",
    level=2,                                # coarse level — fast
)
print(result["vertex_count"])
```

Authentication is opt-in: if the bucket allows anonymous reads, the
default backend config will use it. If you need to force anonymous
access in a credentialed environment, pass it through:

```python
read_points(
    "s3://open-neuro-data/scan.zarrvectors",
    backend="obstore",
    skip_signature=True,                    # obstore-specific kwarg
)
```

### Authenticated access

Ambient credentials work without configuration — `obstore` and `fsspec`
both read `~/.aws/credentials`, environment variables, and IAM roles:

```python
result = read_points("s3://my-bucket/scan.zarrvectors")
```

To pass credentials explicitly, forward them through `**backend_kwargs`:

```python
import os
from zarr_vectors.types.points import read_points

result = read_points(
    "s3://my-bucket/scan.zarrvectors",
    backend="obstore",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region="us-east-1",
)
```

Keyword names match the active backend (`obstore` uses `aws_*`;
`fsspec`/`s3fs` uses `key` / `secret`). Prefer ambient credentials when
possible.

### Writing to S3

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

write_points(
    "s3://my-bucket/datasets/scan.zarrvectors",
    positions,
    chunk_shape=(500., 500., 500.),       # larger chunks = fewer S3 objects
    bin_shape=(100., 100., 100.),
    backend="obstore",
    region="us-east-1",
)
```

**Chunk size guidance for S3.** Each ZV spatial chunk becomes one S3
object. S3 charges per PUT (write) and GET (read) request. To minimise
cost and request count, use `chunk_shape` values that produce chunks of
at least 100 KB compressed. For typical synchrotron point clouds at
~100 000 vertices per chunk (float32, Blosc-compressed), this is
roughly 200–500 µm per axis.

### S3 bucket configuration for Neuroglancer serving

To serve a ZV store from S3 to `zv-ngtools` or the Neuroglancer web
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

```python
from zarr_vectors.types.polylines import read_polylines, write_polylines

# Read — uses Application Default Credentials
result = read_polylines("gs://my-bucket/tracts.zarrvectors", level=1)
print(result["polyline_count"])

# Write
write_polylines(
    "gs://my-bucket/tracts.zarrvectors",
    streamlines,
    chunk_shape=(100., 100., 100.),
    bin_shape=(25., 25., 25.),
    geometry_type="streamline",
)
```

To pass GCS credentials explicitly:

```python
read_polylines(
    "gs://my-bucket/tracts.zarrvectors",
    backend="fsspec",                       # gcsfs route
    token="/path/to/service-account.json",
)
```

GCS CORS:

```bash
gsutil cors set cors.json gs://my-bucket
```

---

## Azure Blob Storage

```python
from zarr_vectors.types.points import read_points

result = read_points("az://account/container/scan.zarrvectors")
# or:   "abfs://container@account.dfs.core.windows.net/scan.zarrvectors"
```

Ambient credentials follow the standard `DefaultAzureCredential` chain
(env vars, managed identity, Azure CLI session).

---

## Building a pyramid on a remote store

The pyramid builder takes a path or URL the same way the writers do:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "s3://my-bucket/scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],     # coarsen 2× per level, no sparsity
    cross_level_depth=1,                  # ±1 cross-level edges per pair
    cross_level_storage="explicit",       # write both +1 and -1
)
```

For very large datasets on cloud, the pyramid build is I/O bound. Run
on a cloud VM in the same region as the bucket — building a pyramid
from within AWS `us-east-1` against a bucket in the same region is
~10× faster than from a laptop.

See [`docs/tutorials/multiscale/building_pyramids.md`](../multiscale/building_pyramids.md)
for the full pyramid API, and [`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for the cross-level link layout that `build_pyramid` produces.

---

## Consolidated metadata

On stores with many resolution levels and attribute arrays, opening
the store requires one metadata request per Zarr group and array. On
S3 with ~50 ms per request, this adds noticeable latency.

Consolidated metadata packs all `.zattrs` and `zarr.json` files into a
single `.zmetadata` key, reducing store-open latency to one request:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("s3://my-bucket/scan.zarrvectors", mode="r+")
zarr.consolidate_metadata(root.zarr_group.store)
```

After consolidation, subsequent opens are dramatically faster.
Regenerate consolidated metadata after any structural change (adding
a resolution level, writing new attributes).

---

## Reading from cloud with the lazy API

```python
import numpy as np
from zarr_vectors.lazy import open_zvr

store = open_zvr("s3://open-neuro/scan.zarrvectors")

print(store.levels)                          # metadata only — no chunk I/O
print(store[2].vertex_count)                 # one metadata request

# Coarse overview — a handful of chunk requests
coarse = store[store.levels[-1]].vertices.compute()

# Detail in a small region — N chunk requests
from zarr_vectors.types.points import read_points
detail = read_points(
    "s3://open-neuro/scan.zarrvectors",
    bbox=(np.array([500., 500., 500.]),
          np.array([700., 700., 700.])),
)
```

`open_zvr` accepts the same `backend=` / `**backend_kwargs` as
`open_store`.

---

## Forcing a specific backend

Pass `backend="obstore"` or `backend="fsspec"` to override
auto-detection:

```python
# Force fsspec even though obstore is installed
read_points("s3://my-bucket/scan.zarrvectors", backend="fsspec")

# Use fsspec for a non-cloud URL (e.g. SFTP)
read_points("sftp://host/path/scan.zarrvectors", backend="fsspec")
```

Or set it globally for the process:

```bash
export ZARR_VECTORS_BACKEND=fsspec
```

---

## Estimating cloud storage cost

A quick estimate for an S3-hosted point cloud store:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("s3://my-bucket/scan.zarrvectors", mode="r")
zg   = root.zarr_group

# Walk every array and sum stored bytes / chunk counts.
total_bytes  = sum(a.nbytes_stored        for _, a in zg.arrays(recurse=True))
total_chunks = sum(a.nchunks_initialized  for _, a in zg.arrays(recurse=True))

print(f"Total compressed size: {total_bytes / 1e9:.2f} GB")
print(f"Total S3 objects:      {total_chunks:,}")
print(f"Monthly S3 storage:    ~${total_bytes / 1e9 * 0.023:.2f}"
      f"  (us-east-1 standard)")
print(f"Cost per 1M GETs:      ~${total_chunks / 1e6 * 0.40:.4f}")
```

This counts every chunk across all resolution levels and every array
family — including the new `links/<delta>/`, `cross_chunk_links/<delta>/`,
and `cross_chunk_link_attributes/<name>/<delta>/` arrays produced by
the multiscale-links layout.
