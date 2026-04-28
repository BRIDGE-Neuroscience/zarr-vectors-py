# Frequently asked questions

## General

### What is the relationship between zarr-vectors and Zarr v3?

`zarr-vectors` is built *on top of* Zarr v3. A ZVF store is a valid Zarr v3
store: all arrays inside it can be opened with the standard `zarr` Python
library. `zarr-vectors` adds conventions on top of Zarr — the directory
layout, the `.zattrs` schema, the VG index arrays, and the OME-Zarr-
compatible multiscale metadata. You could read the raw position arrays
directly with `zarr.open()`, but you would not get the spatial indexing,
object model, or multi-resolution support that `zarr-vectors` provides.

### What is the relationship between zarr-vectors and OME-Zarr?

ZVF borrows the `multiscales` JSON block from the OME-Zarr NGFF
specification so that resolution pyramids are discoverable by any OME-Zarr-
aware viewer. ZVF is not a strict subset of OME-Zarr: the two formats target
different data (OME-Zarr is primarily for dense image volumes; ZVF is for
sparse vector geometry). The comparison page
[ZVF and OME-Zarr](../spec/comparisons/ome_zarr.md) documents exactly which
fields are shared and which are ZVF-specific extensions.

### Why does the store have a `.zarrvectors` extension? Is it required?

The `.zarrvectors` extension is a convention, not a requirement enforced by
the file system or the library. You can name your store anything. The
extension helps tools (and humans) identify ZVF stores at a glance and is
used by `zv-ngtools` to auto-detect the store type when loading layers.

### Can I open a ZVF store with plain `zarr.open()`?

Yes. The underlying arrays are standard Zarr v3. However, `zarr.open()` will
give you raw group and array objects; you will need to interpret the ZVF
conventions manually (VG offsets, object index, multiscale metadata).
For most use cases, the `zarr_vectors.types.*` read functions are more
convenient.

---

## Installation

### Which Python versions are supported?

Python 3.10, 3.11, and 3.12. Python 3.9 and earlier are not supported
because `zarr-vectors` relies on `zarr>=3.0`, which requires 3.10+.

### Does zarr-vectors work on Windows?

Yes. All core functionality is cross-platform. Some of the optional ingest
extras (`laspy`, `nibabel`) have their own platform requirements; consult
their documentation if you encounter issues.

### I get a `ModuleNotFoundError` for `DracoPy` — what do I do?

Install the Draco extra: `pip install "zarr-vectors[draco]"`. DracoPy
has binary wheels for most platforms; if a wheel is not available for your
platform, you may need to build it from source.

---

## Chunks and bins

### How do I choose `chunk_shape`?

The primary consideration is I/O pattern. For interactive spatial queries
(e.g. viewport-driven fetching in a visualiser), smaller chunks reduce the
amount of data loaded per request. For batch processing (e.g. reading an
entire dataset sequentially), larger chunks reduce overhead from opening
many files. A common starting point for 3-D biological data is
200–500 physical units per axis.

See [Choosing chunk and bin size](../how_to/choose_chunk_and_bin.md) for
worked examples and heuristics.

### What is the relationship between `chunk_shape` and `bin_shape`?

`bin_shape` must evenly divide `chunk_shape` in every dimension.
`chunk_shape` controls the on-disk file layout and is constant across all
resolution levels. `bin_shape` controls spatial query granularity and scales
with `bin_ratio` at each coarser level. When `bin_shape` is omitted it
defaults to `chunk_shape` (one bin per chunk).

### What happens if I omit `bin_shape`?

The store is written with one bin per chunk (i.e. `bin_shape = chunk_shape`).
Spatial queries will return data at full-chunk granularity rather than bin
granularity. This is the backward-compatible mode for stores that do not
require sub-chunk spatial indexing.

---

## Multi-resolution

### How many resolution levels should I build?

A common choice is to build enough levels so that the coarsest level fits
comfortably in memory for overview rendering. For a dataset with 10 million
streamlines, three levels with `bin_ratio=(2,2,2)` and `object_sparsity=0.5`
would give roughly 10M → 1.25M → 156K → 20K objects across levels 0–3.

### Does `bin_ratio` have to be isotropic (same in all dimensions)?

No. Anisotropic bin ratios (e.g. `(1, 2, 2)` for data with higher
resolution in the z-axis) are supported. The ratio tuple must have the same
number of elements as the spatial dimensionality of the store.

### Can I add a resolution level after writing the base level?

Yes. Use `add_resolution_level()` from `zarr_vectors.core.store`, or call
`coarsen_level()` / `build_pyramid()` on an existing store. Existing levels
are not modified.

---

## Formats and interoperability

### Can I convert a ZVF store back to TRK / SWC / OBJ?

Yes, using the export functions or the CLI:

```bash
zarr-vectors export trk  tracts.zarrvectors  tracts_out.trk
zarr-vectors export swc  neuron.zarrvectors  neuron_out.swc
zarr-vectors export obj  brain.zarrvectors   brain_out.obj
```

Export requires `zarr-vectors[ingest]`.

### How do I visualise a ZVF store in Neuroglancer?

Use [`zv-ngtools`](https://github.com/BRIDGE-Neuroscience/zv-ngtools), a
fork of `ngtools` that adds a ZVF layer type. It can serve a local
`.zarrvectors` store to a Neuroglancer instance running in your browser.
See [Neuroglancer integration](../tutorials/neuroglancer/overview.md).

### Is ZVF compatible with the Neuroglancer precomputed format?

ZVF and Neuroglancer precomputed are distinct formats that share some
design goals (spatial chunking, multiscale support). `zv-ngtools` includes
a precomputed export tool that converts a ZVF store to the Neuroglancer
precomputed annotation or skeleton format for static hosting.
See [Format comparisons](../spec/comparisons/neuroglancer_precomputed.md)
for a detailed comparison.

---

## Performance

### My writes are slow — what should I check?

- Ensure `bin_shape` evenly divides `chunk_shape`. Misaligned bin shapes
  cause extra chunk reads during writes.
- For large datasets, consider setting `object_sparsity=1.0` for all levels
  and building the pyramid as a post-processing step rather than inline.
- On networked file systems (NFS, SMB), increase `chunk_shape` to reduce the
  number of file creates.
- For cloud writes, use `zarr-vectors[cloud]` and pass an `s3fs.S3FileSystem`
  or `gcsfs.GCSFileSystem` instance as the store argument.

### Validation is slow on large stores. Can I run only level 1?

Yes. `validate("scan.zarrvectors", level=1)` runs only the structural check
(file/array presence). Each level adds progressively more work; level 5
is the most thorough but also the slowest.

---

## Development and contribution

### How do I report a bug?

Open an issue on the
[GitHub repository](https://github.com/BRIDGE-Neuroscience/zarr-vectors-py/issues).
Include the output of `zarr-vectors info <store>` and `zarr-vectors validate
<store> --level 3` if the issue is store-related.

### How do I propose a change to the specification?

See [Spec change process](../spec/contributing/spec_change_process.md).
Spec changes require an RFC-style discussion issue before a pull request.

### Is zarr-vectors stable?

The package is under active development and the API may change before the
1.0 release. Stores written with the current version will be readable by
future versions; no store format migration is planned for the 0.x series.
