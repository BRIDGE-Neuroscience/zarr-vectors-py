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
  Draco encoding to the `links/<delta>` and `vertices/` arrays of mesh-type
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
| `vertex_fragments/` | none (opaque `uint8` blob) | Pre-packed fragment-index format; see note below |
| `link_fragments/` | none (opaque `uint8` blob) | Parallel to `vertex_fragments/`; same rationale |
| `links/<delta>/` | `bytes → blosc(zstd, byteshuffle, l5)` | Int32/Int64 index pairs; same codec for `link_width=2` (graph/poly) and `link_width=3` (mesh faces) — or Draco for mesh `delta=0` if `[draco]` installed |
| `link_attributes/<name>/<delta>/` | `bytes → blosc(zstd, bitshuffle, l5)` | Varies by dtype; parallel to `links/<delta>/` |
| `cross_chunk_links/<delta>/` | `bytes → blosc(zstd, byteshuffle, l3)` | Typically small flat blob |
| `cross_chunk_link_attributes/<name>/<delta>/` | `bytes → blosc(zstd, bitshuffle, l5)` | Parallel to `cross_chunk_links/<delta>/data` (new in 0.4) |
| `attributes/*` | `bytes → blosc(zstd, bitshuffle, l5)` | Varies by dtype |
| `object_index/manifests` | `vlen-bytes` | One chunk per ~16K objects; random-access read fetches only the chunk holding the requested OID |

```{note}
**`vertex_fragments/` and `link_fragments/` bypass the Zarr codec
pipeline.** Each chunk is written as opaque `uint8` bytes via the
FsGroup `write_bytes` path. The fragment-index format is already a
packed binary layout (header + bitmap + dense range table +
uint32/int64 CSR) with little redundancy for a general-purpose
compressor to exploit, and adding a compression step introduces
latency on hot decode paths. The arrays' group metadata carries
`encoding: "fragment_index_v1"` to identify the on-disk format
independently of any outer compression a backend might apply to the
bytes themselves.

**`object_index/manifests` uses the `vlen-bytes` codec.** Each entry
is one object's encoded manifest blob; per-object random access reads
only the zarr chunk containing the requested OID (~16K objects per
chunk), not the whole index. The manifest-blob wire format itself is
unchanged from earlier ZVF versions — only the on-disk container
changed from a `data` + `offsets` byte-blob pair to a single ragged
zarr array.

The Zarr V3 specification for variable-length byte arrays is still
in development (tracked at
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types));
ZVF 0.x stores written with `vlen-bytes` may need to be re-encoded if
the eventual spec lands incompatibly.

Legacy stores written before this layout change have
`object_index/data` and `object_index/offsets` byte-blob children
instead of `object_index/manifests`; readers in `zarr_vectors`
auto-detect and handle both layouts.

See [Fragment-index arrays](../layout/vg_index_arrays.md) for the
fragment-index byte layout and
[Object manifest](../object_model/object_manifest.md) for the
manifest-blob format.
```

### Draco codec (mesh geometry)

When `zarr-vectors[draco]` is installed, the `links/<delta>` and `vertices/`
arrays of mesh-type stores can use Draco compression. Enable it by
passing `use_draco=True` and `draco_quantization=<bits>` to `write_mesh()`
(or to the OBJ/STL/PLY converters in `zarr-vectors-tools`).

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
