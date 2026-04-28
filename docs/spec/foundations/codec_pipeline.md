# Codec pipeline

## Terms

**Codec**
: A single transformation applied to chunk data. A codec takes bytes (or an
  array) as input and produces bytes as output. Codecs are composable;
  an ordered sequence of codecs forms a pipeline.

**Array-to-bytes codec**
: The first codec in a Zarr v3 pipeline, which converts a typed N-dimensional
  array into a flat byte sequence. Zarr v3 requires exactly one such codec.
  The standard array-to-bytes codec is `bytes`, which serialises the array
  in C-order with a declared byte order.

**Byte-to-byte codec**
: A codec that transforms one byte sequence into another — typically
  compression or encryption. Multiple byte-to-byte codecs may be stacked.
  Common codecs: `blosc`, `zstd`, `gzip`, `lz4`.

**Blosc**
: A high-performance meta-compressor that internally uses one of several
  compression algorithms (zstd, lz4, blosclz, zlib). Blosc also performs
  byte-shuffling before compression, which improves compression ratios on
  typed numeric arrays. It is the default byte-to-byte codec in
  `zarr-vectors-py`.

**Sharding codec** (`sharding_indexed`)
: A Zarr v3 codec that packs multiple logical chunks into a single storage
  key. Reduces file count and cloud storage request overhead. Discussed in
  detail in [Sharding](../chunking/sharding.md).

**Draco**
: Google's geometry compression library, optimised for 3-D mesh data.
  `zarr-vectors-py` registers a custom Zarr codec (`draco`) that applies
  Draco encoding to the `links/faces` and `vertices/` arrays of mesh-type
  stores. Requires `zarr-vectors[draco]`.

**Fill value**
: The value written to a chunk position that is not explicitly stored. In
  Zarr v3, the fill value is declared per array in `zarr.json`. ZVF uses
  `0.0` for floating-point position arrays and `-1` for integer index
  arrays.

---

## Introduction

A Zarr v3 codec pipeline controls how chunk data is serialised to bytes and
compressed before storage. ZVF does not mandate a specific codec; any Zarr
v3-compatible pipeline is valid. However, `zarr-vectors-py` ships with
sensible defaults and provides helper functions to configure specialised
pipelines for specific geometry types (e.g. Draco for meshes).

Understanding the codec pipeline matters for contributors adding new
geometry types and for users who want to tune compression ratios or
compatibility with downstream tools.

---

## Technical reference

### Default pipeline

The default codec pipeline used by `zarr-vectors-py` for all position and
attribute arrays is:

```json
[
  { "name": "bytes",  "configuration": { "endian": "little" } },
  { "name": "blosc",  "configuration": { "cname": "zstd", "clevel": 5,
                                          "shuffle": "bitshuffle",
                                          "blocksize": 0, "nthreads": 1 } }
]
```

- `bytes` (array-to-bytes): serialises the array in C-order, little-endian.
- `blosc` with `zstd` at level 5: good compression ratio with fast decode.
  `bitshuffle` is used instead of `byteshuffle` for floating-point arrays;
  it typically achieves better compression on 32-bit floats.

To override the default pipeline, pass `codec_config` to the write function:

```python
import numcodecs
from zarr_vectors.types.points import write_points

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200., 200., 200.),
    codec_config={
        "positions": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd",  "configuration": {"level": 3}},
        ]
    },
)
```

### Per-array codec configuration

Different arrays in a ZVF store may use different codec pipelines. The
`zarr-vectors-py` defaults by array type are:

| Array | Default codec | Rationale |
|-------|---------------|-----------|
| `vertices/` | `bytes → blosc(zstd, bitshuffle, l5)` | Float32 positions compress well with bitshuffle |
| `vertex_group_offsets/` | `bytes → blosc(zstd, byteshuffle, l3)` | Int64 offsets; lighter compression |
| `links/edges` | `bytes → blosc(zstd, byteshuffle, l5)` | Int32/Int64 index pairs |
| `links/faces` | `bytes → blosc(zstd, byteshuffle, l5)` | Or Draco if `[draco]` installed |
| `attributes/*` | `bytes → blosc(zstd, bitshuffle, l5)` | Varies by dtype |
| `object_index/` | `bytes → blosc(zstd, byteshuffle, l3)` | Int64 index pairs |
| `cross_chunk_links/` | `bytes → blosc(zstd, byteshuffle, l3)` | Typically small array |

### Draco codec (mesh geometry)

When `zarr-vectors[draco]` is installed, the `links/faces` and `vertices/`
arrays of mesh-type stores can use Draco compression:

```python
from zarr_vectors.ingest.obj import ingest_obj

ingest_obj(
    "brain.obj",
    "brain.zarrvectors",
    chunk_shape=(100., 100., 100.),
    use_draco=True,         # requires zarr-vectors[draco]
    draco_quantization=11,  # quantisation bits for vertex positions
)
```

The custom codec is registered with Zarr as:

```json
{
  "name": "draco",
  "configuration": {
    "quantization_bits": 11,
    "compression_level": 7
  }
}
```

Draco-compressed stores are **not readable** by tools that do not have the
`draco` codec registered. `zarr-vectors-py` will raise `ValueError` when
attempting to read a Draco-compressed store without the extra installed.

#### Quantisation and precision

Draco uses integer quantisation of vertex positions. The quantisation level
controls the trade-off between compression ratio and positional precision:

| `quantization_bits` | Precision (relative to bounding box) | Typical ratio |
|--------------------|------------------------------------|---------------|
| 8 | 1 / 256 of bbox per axis | 10–15× |
| 11 | 1 / 2048 of bbox per axis | 6–10× |
| 14 | 1 / 16384 of bbox per axis | 3–6× |

For neuroscience mesh data (brain surfaces, cell boundaries) quantised to
nanometre scale, 11 bits is a common choice that preserves sub-micrometre
precision while achieving 6–10× compression over float32.

### Choosing a codec

| Scenario | Recommended codec |
|----------|------------------|
| Default / general purpose | `blosc(zstd, bitshuffle, level 5)` |
| Fastest possible read (RAM-limited) | `blosc(lz4, byteshuffle, level 1)` |
| Maximum compression (archival) | `blosc(zstd, bitshuffle, level 9)` |
| Mesh geometry with size budget | `draco(quantization=11)` |
| Compatibility with non-Zarr tools | `bytes` (no compression) |
| Cloud with many small chunks | Consider sharding codec first |

### Codec availability and compatibility

The following codecs are available without extra dependencies:

- `bytes` (array-to-bytes)
- `gzip` (standard library)
- `zstd` (via `numcodecs`)
- `blosc` (via `numcodecs`)
- `lz4` (via `numcodecs`)

Codecs requiring extras:

- `draco` — requires `zarr-vectors[draco]`
- `sharding_indexed` — built into Zarr v3 ≥ 2.18; no extra required

When distributing ZVF stores, prefer the default Blosc pipeline for
maximum compatibility. Document any use of non-default codecs prominently
(e.g. in the store's `.zattrs` `"notes"` field) so consumers know what
is required.
