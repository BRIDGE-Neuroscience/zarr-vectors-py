# Memory-efficient writes

`zarr-vectors-py` write functions buffer each chunk in memory before
writing it to disk. For large datasets, all vertices in a chunk must be
in memory simultaneously. This guide describes strategies for writing
datasets larger than available RAM.

---

## Why writes are memory-bound

The VG index requires all vertices in a chunk to be sorted by bin
coordinate before writing. This means a full chunk worth of vertices
(positions + attributes) must be in memory at once.

For a chunk with 100 000 vertices and 5 float32 attributes:
`100 000 × (3 + 5) × 4 bytes ≈ 3.2 MB per chunk`. For 125 chunks this
is 400 MB — manageable. For 10 000 chunks with 1M vertices each, this
approach requires partitioning.

---

## Strategy 1: Write in spatial partitions

The most memory-efficient approach for very large datasets. Divide the
data into spatial partitions, each covering one or more full chunks, and
write one partition at a time using `write_points_partition`:

```python
import numpy as np
from zarr_vectors.types.points import write_points_partition

# Large dataset: 500M vertices in a 10000³ µm volume
# Partition by z-slabs, each covering 500 µm (one chunk deep in z)

chunk_shape = (500., 500., 500.)
slab_depth  = 500.0   # one chunk deep in z per partition

# Process one z-slab at a time
for z_idx in range(20):   # 20 slabs × 500 µm = 10 000 µm
    z_min = z_idx * slab_depth
    z_max = z_min + slab_depth

    # Load only this z-slab from disk/database
    slab_positions  = load_positions_in_range(z_min, z_max)
    slab_intensity  = load_intensity_in_range(z_min, z_max)

    write_points_partition(
        "large_scan.zarrvectors",
        slab_positions,
        chunk_shape=chunk_shape,
        bin_shape=(125., 125., 125.),
        attributes={"intensity": slab_intensity},
        z_range=(z_min, z_max),   # restricts which chunks are written
    )
    del slab_positions, slab_intensity   # free memory

# Finalise metadata and build object index
from zarr_vectors.core.store import open_store
from zarr_vectors.core.multiscale import write_multiscale_metadata
root = open_store("large_scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)
```

### Chunk-aligned partitions

For maximum efficiency, align partitions to chunk boundaries so that each
partition writes complete chunks without partial-chunk merging:

```python
# For chunk_shape=(500, 500, 500), write one x-y tile at a time
for x_idx in range(n_x_chunks):
    for y_idx in range(n_y_chunks):
        x_lo, x_hi = x_idx * 500, (x_idx + 1) * 500
        y_lo, y_hi = y_idx * 500, (y_idx + 1) * 500

        tile = load_tile(x_lo, x_hi, y_lo, y_hi)
        write_points_partition(
            "scan.zarrvectors", tile.positions,
            chunk_shape=(500., 500., 500.),
            attributes=tile.attributes,
            xy_tile=(x_lo, y_lo),
        )
```

---

## Strategy 2: Use the streaming writer

The `StreamingPointWriter` class accumulates vertices in a rolling
per-chunk buffer, flushing completed chunks to disk as they fill up.
Peak memory usage is bounded by the number of chunks in the internal
buffer multiplied by the expected vertices per chunk:

```python
from zarr_vectors.types.points import StreamingPointWriter

# Peak memory ≈ buffer_chunks × avg_vertices/chunk × bytes_per_vertex
# At buffer_chunks=8, 50k vertices/chunk, 32 bytes/vertex: ~13 MB
writer = StreamingPointWriter(
    "scan.zarrvectors",
    chunk_shape=(500., 500., 500.),
    bin_shape=(125., 125., 125.),
    attribute_names=["intensity", "label"],
    buffer_chunks=8,   # keep at most 8 in-flight chunks in memory
)

# Feed data in arbitrary-sized batches
for batch in data_source:   # any iterable yielding (positions, attributes)
    writer.write_batch(batch["positions"], batch["attributes"])

# Flush all remaining buffered chunks and write metadata
writer.close()
```

The streaming writer is slower than the bulk writer (because it cannot
exploit batch-level sorting optimisations) but has bounded memory usage
regardless of dataset size.

---

## Strategy 3: Generator-based ingest

For formats read by external libraries (LAS, TRX, etc.), use the
generator-based ingest functions that yield vertex batches without
loading the full file:

```python
from zarr_vectors.ingest.las import ingest_las_streaming

# Reads the LAS file in 1M-point blocks; never holds more than one block
ingest_las_streaming(
    "huge_survey.laz",
    "survey.zarrvectors",
    chunk_shape=(200., 200., 100.),
    block_size=1_000_000,   # read 1M points at a time
)
```

---

## Strategy 4: Rechunk after initial write

If you already have data in a format that can be read block-by-block
(e.g. a Zarr array, HDF5, or numpy memmap), write it with large
chunk_shape first (entire dataset in one chunk per axis if needed), then
rechunk to the target shape using the streaming rechunker:

```python
from zarr_vectors.core.rechunk import rechunk_store

# Initial write: no spatial chunking (entire volume is one big chunk)
write_points("scan_temp.zarrvectors", huge_array, chunk_shape=(10000., 10000., 10000.))

# Rechunk to target: streaming, bounded memory
rechunk_store(
    "scan_temp.zarrvectors",
    "scan.zarrvectors",
    chunk_shape=(500., 500., 500.),
    bin_shape=(125., 125., 125.),
    streaming=True,
    buffer_chunks=16,
)
```

This is the least memory-efficient approach but requires the least
modification to existing pipelines.

---

## Monitoring memory usage

Use `tracemalloc` or `memory_profiler` to identify peak usage:

```python
import tracemalloc
tracemalloc.start()

write_points("scan.zarrvectors", positions, chunk_shape=(500., 500., 500.))

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1e6:.1f} MB")
tracemalloc.stop()
```
