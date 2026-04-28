# Store types

## Terms

**Local store**
: A Zarr store backed by the local file system. Each chunk is a file at a
  path derived from the chunk key. This is the default store type when a
  string path ending in `.zarrvectors` is passed to a `zarr-vectors`
  write function.

**ZIP store**
: A Zarr store that packages all chunk files into a single ZIP archive.
  Supports read access and append writes, but not random in-place chunk
  updates. Useful for archiving or transferring a complete store as a single
  file.

**In-memory store**
: A Zarr store backed by a Python dictionary. No data is persisted to disk.
  Useful for testing and for intermediate processing pipelines that do not
  need to materialise data.

**Object store**
: A cloud key–value service such as Amazon S3, Google Cloud Storage, or
  Azure Blob Storage. Zarr accesses these via `fsspec`-backed stores
  (`s3fs`, `gcsfs`, `adlfs`). Requires `zarr-vectors[cloud]`.

**FSSPEC**
: A Python library that provides a uniform file-system interface over many
  storage backends (local, S3, GCS, HTTP, SFTP, …). Zarr v3 uses `fsspec`
  as its primary mechanism for cloud store access.

**Consolidated metadata**
: An optional `.zmetadata` file at the store root that contains a cached
  copy of all `zarr.json` and `.zattrs` files in the hierarchy. Avoids
  many small metadata reads on high-latency stores.

---

## Introduction

ZVF stores are backend-agnostic: the same write and read API works whether
the data lives on a local SSD, a network file system, a ZIP archive, or a
cloud object store. The backing store type affects performance characteristics
(latency, throughput, cost per request) but not the data model or the
semantics of any operation.

For most users the relevant choice is between a **local store** (development,
analysis on HPC) and an **object store** (sharing, cloud-scale processing,
Neuroglancer serving). This page documents each store type, how to open it,
and its performance implications for ZVF workloads.

---

## Technical reference

### Local file system store

The default. Pass any local path as a string:

```python
from zarr_vectors.types.points import write_points, read_points

write_points("scan.zarrvectors", positions, chunk_shape=(200., 200., 200.))
result = read_points("scan.zarrvectors")
```

Internally, `zarr-vectors` calls `zarr.storage.LocalStore(path)`. Each
chunk is stored as a file; directories correspond to group and array
paths. The directory tree looks like:

```
scan.zarrvectors/
├── zarr.json
├── .zattrs
├── resolution_0/
│   ├── zarr.json
│   ├── vertices/
│   │   ├── zarr.json
│   │   └── c/
│   │       ├── 0/
│   │       │   ├── 0/
│   │       │   │   └── 0     ← chunk file (compressed bytes)
│   │       │   └── 1/
│   │       │       └── 0
│   │       └── …
│   └── …
└── …
```

**Performance notes:**
- File creation overhead dominates write performance on file systems with
  high inode metadata costs (many NFS mounts, HDD-backed systems). Use
  larger `chunk_shape` values to reduce file count.
- Linux `ext4` and `xfs` handle large directories well; `FAT32` and some
  network shares have per-directory file count limits. If you expect more
  than ~100 000 chunks per array, consider the sharding codec.
- On HPC systems with Lustre or GPFS, stripe the `.zarrvectors` directory
  for write parallelism: `lfs setstripe -c 8 scan.zarrvectors`.

### ZIP store

```python
import zarr
from zarr_vectors.core.store import open_store
from zarr_vectors.types.points import write_points

# Write to a local store, then archive
write_points("scan.zarrvectors", positions, chunk_shape=(200., 200., 200.))

# Open for reading from a ZIP
store = zarr.storage.ZipStore("scan.zarrvectors.zip", mode="r")
root  = zarr.open_group(store, mode="r")
```

ZIP stores are primarily useful for distribution and archiving. They are
not suitable as the primary write target for large datasets because the ZIP
format requires sequential append writes and does not support chunk-level
random updates.

**Limitation:** The `zarr-vectors` write functions do not currently accept a
ZIP store as output. Write to a local store first, then compress with
standard tools:

```bash
zip -r scan.zarrvectors.zip scan.zarrvectors/
```

### In-memory store

```python
import zarr
from zarr_vectors.types.points import write_points, read_points

# Pass a zarr.storage.MemoryStore to bypass disk I/O
store = zarr.storage.MemoryStore()
write_points(store, positions, chunk_shape=(200., 200., 200.))
result = read_points(store)
```

In-memory stores are discarded when the Python process exits. They are
useful for:

- Unit tests that must not touch the file system.
- Intermediate pipeline stages where data is written and immediately
  consumed without persistence.
- Profiling write/read performance without I/O bottlenecks.

### Cloud object stores (S3, GCS)

Requires `zarr-vectors[cloud]`.

#### Amazon S3

```python
import s3fs
from zarr_vectors.types.points import write_points, read_points

fs    = s3fs.S3FileSystem(anon=False)   # uses ~/.aws/credentials
store = fs.get_mapper("s3://my-bucket/datasets/scan.zarrvectors")

write_points(store, positions, chunk_shape=(500., 500., 500.))
result = read_points(store)
```

For public (anonymous) access:

```python
fs    = s3fs.S3FileSystem(anon=True)
store = fs.get_mapper("s3://open-neuro-data/scan.zarrvectors")
result = read_points(store)
```

#### Google Cloud Storage

```python
import gcsfs
from zarr_vectors.types.points import read_points

fs    = gcsfs.GCSFileSystem(project="my-gcp-project")
store = fs.get_mapper("gs://my-bucket/scan.zarrvectors")
result = read_points(store)
```

**Performance notes for object stores:**
- Each chunk read is one HTTP request. Minimise requests by increasing
  `chunk_shape` relative to your typical query bbox.
- Enable **consolidated metadata** to avoid one metadata request per group
  node during store opening. Call `zarr.consolidate_metadata(store)` after
  writing.
- Request coalescing (reading multiple chunks in a single HTTP range
  request) is not yet implemented in `zarr-vectors-py`. For Neuroglancer
  serving, use `zv-ngtools`, which implements its own request batching.
- S3 charges per PUT (write) and GET (read) request. For write-heavy
  pipelines, reduce chunk count with larger `chunk_shape` or use the
  sharding codec to pack multiple logical chunks into one object.

### Consolidated metadata

On stores with many groups (large pyramids, many attribute arrays), the
overhead of reading individual `zarr.json` and `.zattrs` files during store
opening can be significant. Consolidate metadata after writing:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
zarr.consolidate_metadata(root.store)
```

This writes a single `.zmetadata` file at the store root. Subsequent
`open_store()` calls will read from `.zmetadata` instead of issuing
individual metadata requests. Consolidated metadata must be regenerated
after any write that modifies the store structure (e.g. adding a resolution
level).

### Choosing a store type

| Scenario | Recommended store type |
|----------|----------------------|
| Local analysis / development | Local file system |
| HPC batch processing | Local file system (Lustre/GPFS with striping) |
| Archiving a completed dataset | ZIP (after local write) |
| Cloud sharing / Neuroglancer serving | S3 or GCS with consolidated metadata |
| Unit tests | In-memory |
| Very large datasets with many small chunks | Local or cloud with sharding codec (see [Sharding](../chunking/sharding.md)) |
