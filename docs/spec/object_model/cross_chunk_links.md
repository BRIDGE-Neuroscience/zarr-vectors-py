# Links and cross-chunk links

## Terms

**Intra-chunk link**
: An edge between two vertices that live in the **same** spatial chunk
  at the **same** resolution level. Stored as a pair of local vertex
  indices in `links/<delta>/<chunk_key>` with `delta=0`.

**Cross-chunk link**
: An edge whose two endpoints live in **different** spatial chunks
  (possibly at different resolution levels). Stored as
  `((chunk_a, local_a), (chunk_b, local_b))` in
  `cross_chunk_links/<delta>/data`.

**Level delta** (`<delta>`)
: A signed integer path segment that says how many pyramid levels the
  edges span. `0` = both endpoints at the owning level (the only kind
  written pre-0.4); `+N` = endpoint B is `N` levels coarser; `-N` =
  endpoint B is `N` levels finer. Filesystem-safe literal segments:
  `"0"`, `"+1"`, `"-1"`, `"+2"`, …

**Link attribute**
: Per-edge scalar or vector data parallel to a `links/<delta>/` array.
  Lives at `link_attributes/<name>/<delta>/<chunk_key>` for intra-chunk
  edges and at `cross_chunk_link_attributes/<name>/<delta>/data` for
  cross-chunk edges (new in 0.4).

---

## Introduction

The link layout under the 0.4 schema is a single family of four
arrays, each parameterised by a level delta:

```
/N/links/<delta>/<chunk_key>
/N/cross_chunk_links/<delta>/data
/N/link_attributes/<name>/<delta>/<chunk_key>
/N/cross_chunk_link_attributes/<name>/<delta>/data
```

When an edge's two endpoints share a chunk_key (after re-evaluation
against the target level's chunk grid), the edge goes into the per-
chunk `links/<delta>/<chunk_key>` array. Otherwise it goes into the
global `cross_chunk_links/<delta>/data` blob. Either way, the level
delta is encoded in the path — readers never need to inspect the
edge to know which level its target side lives at.

This page documents:

- the on-disk encoding of both arrays at every `<delta>`,
- when each kind is generated and how chunk-alignment is decided,
- the parallel attribute arrays,
- the path helpers and listing helpers callers should use,
- and the validation rules.

For a worked end-to-end example, see
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

---

## Technical reference

### Level-delta convention

| Segment | Meaning |
|---------|---------|
| `0`     | intra-level edges (the only kind written pre-0.4) |
| `+N`    | edges from this level to `this_level + N` (coarser) |
| `-N`    | edges from this level to `this_level - N` (finer) |

Compose paths with the helpers in
[`zarr_vectors/core/paths.py`](../../../zarr_vectors/core/paths.py) —
never hand-roll the `<delta>` segment:

```python
from zarr_vectors.core.paths import (
    format_delta,           # 0 -> "0";  1 -> "+1";  -2 -> "-2"
    parse_delta,            # inverse
    links_path,                          # links/<delta>
    cross_chunk_links_path,              # cross_chunk_links/<delta>
    link_attributes_path,                # link_attributes/<name>/<delta>
    cross_chunk_link_attributes_path,    # cross_chunk_link_attributes/<name>/<delta>
)
```

To enumerate which deltas exist under a level group, use:

```python
from zarr_vectors.core.arrays import (
    list_link_deltas,                    # [0, +1, -1, ...]
    list_cross_link_deltas,
    list_link_attribute_deltas,
    list_cross_chunk_link_attribute_deltas,
)
```

### `links/<delta>/<chunk_key>` — per-chunk array

Each chunk file is a contiguous int64 byte blob holding one or more
`(M_k, link_width)` row groups. `link_width` is declared on the
array's `.zattrs`:

| Geometry | `link_width` | Row meaning |
|----------|--------------|-------------|
| Graph, polyline, streamline, skeleton (branches) | 2 | `(src_local, dst_local)` |
| Triangle mesh | 3 | `(v0_local, v1_local, v2_local)` |
| Quad mesh | 4 | `(v0, v1, v2, v3)` |

**`.zattrs` schema** (see
[`zarr_vectors/core/arrays.py:create_links_array`](../../../zarr_vectors/core/arrays.py)):

```jsonc
{
  "zv_array":   "links",
  "dtype":      "int64",
  "link_width": 2,
  "level_delta": 0     // signed integer; 0 for intra-level
}
```

**Endpoint convention for non-zero deltas:** for a row in
`links/+N/<chunk_key>`, column 0 is a local vertex index in the source
chunk at the **owning level**, and column 1 is a local vertex index in
the **same chunk key** at level `owning_level + N`. The reader doesn't
need any cross-chunk-coords information — both sides share `<chunk_key>`.

**Self-describing blob.** Each `links/<delta>/<chunk_key>` file is a
self-describing ragged blob: an int64 header with `K` followed by the
`K` per-group byte offsets, then the concatenated link bytes. Readers
recover the per-vertex-group partition without consulting any sibling
table.

### `cross_chunk_links/<delta>/data` — global flat blob

Each record is `link_width * (sid_ndim + 1)` int64s laid out as
`link_width` back-to-back `(chunk_coords, vertex_idx)` endpoints:

```
[chunk_0_0, ..., chunk_0_{ndim-1}, vi_0,
 chunk_1_0, ..., chunk_1_{ndim-1}, vi_1,
 ...
 chunk_{L-1}_0, ..., vi_{L-1}]
```

`link_width=2` (the default) encodes a classic cross-chunk edge;
`link_width=3` encodes a triangle face spanning chunks (used by mesh
writers); `link_width=1` encodes a single parent→child reference for
pyramid metanode drill-down. Endpoint 0 lives at the **owning level**;
endpoints 1..L-1 live at the **target level** (`owning_level +
level_delta`).

**`.zattrs` schema** (see
[`zarr_vectors/core/arrays.py:write_cross_chunk_links`](../../../zarr_vectors/core/arrays.py)):

```jsonc
{
  "zv_array":    "cross_chunk_links",
  "num_links":   12,
  "sid_ndim":    3,
  "level_delta": 1,
  "link_width":  2
}
```

**Sid-ndim assumption.** Source and target levels share `sid_ndim`
(uniform per store). The writer asserts both endpoints' chunk-coord
arities match `sid_ndim`; mismatched callers fail loudly with an
`ArrayError`. Chunk *spacing* may differ between levels (coarser
chunks are larger in physical units), but the chunk-key arity does
not.

**Why two arrays?** The writer routes a fine→coarse edge into
`links/<delta>/<chunk_key>` when the source chunk_key equals the
chunk_key in the coarser level that contains the target vertex —
i.e. the two endpoints share a chunk-key string after re-evaluating
against the coarser chunk grid. Otherwise the edge goes into
`cross_chunk_links/<delta>/data`. The split keeps per-chunk reads
cheap (no global scan needed for the common chunk-aligned case)
while still expressing arbitrary cross-grid edges.

### `link_attributes/<name>/<delta>/<chunk_key>` — intra-chunk attrs

Parallel to `links/<delta>/<chunk_key>`. One ragged group per chunk
matching the link group layout exactly; rows are in the same order
as the link rows.

**`.zattrs` schema:**

```jsonc
{
  "zv_array":   "link_attribute",
  "name":       "weight",
  "dtype":      "float32",
  "level_delta": 0
}
```

### `cross_chunk_link_attributes/<name>/<delta>/data` — global attrs

**New in 0.4.** Parallel to `cross_chunk_links/<delta>/data`; one flat
row per cross-chunk link in path order.

**`.zattrs` schema:**

```jsonc
{
  "zv_array":    "cross_chunk_link_attribute",
  "name":        "weight",
  "dtype":       "float32",
  "level_delta": 1,
  "num_links":   7,
  "shape":       [7]          // or [7, C] for multi-channel
}
```

**Length invariant.** The writer
[`write_cross_chunk_link_attributes`](../../../zarr_vectors/core/arrays.py)
enforces `len(values) == num_links` at runtime. A desynchronised write
fails loudly with an `ArrayError` rather than silently producing a
parallel array of the wrong size.

### Generation algorithm

**Intra-level (`delta == 0`).** Each geometry's writer
(`write_graph`, `write_polyline`, `write_mesh`, …) calls
[`partition_edges`](../../../zarr_vectors/spatial/boundary.py): for
each edge it compares the chunk indices of the two endpoints. Same
chunk → bucket into per-chunk `(M_local, link_width)` rows for
`links/0/<chunk_key>`. Different chunks → emit
`((chunk_a, local_a), (chunk_b, local_b))` for
`cross_chunk_links/0/data`.

**Cross-level (`delta != 0`).** Emitted by
[`_write_cross_level_edges`](../../../zarr_vectors/multiresolution/coarsen.py)
during pyramid construction. For each adjacent (fine, coarse) pair,
every fine vertex has exactly one trivial edge to its coarse parent
metanode (the parent map is recovered from the coarse level's own
`cross_chunk_links/<delta=-1>/` records). The edges are then
partitioned via
[`partition_cross_level_edges`](../../../zarr_vectors/spatial/boundary.py):
chunk-aligned edges (source chunk_key == target chunk_key when
re-evaluated against the coarser grid) become rows in
`links/+1/<chunk_key>`; the rest become entries in
`cross_chunk_links/+1/data`.

When `cross_level_storage="explicit"`, the same edges are also
mirrored at the coarse level under `<-delta>` with endpoint roles
swapped — `links/-1/<chunk_key>` and `cross_chunk_links/-1/data`.
When `cross_level_storage="implicit"`, only the `+delta` side is
materialised; readers reconstruct the `-delta` direction by walking
the `+delta` array at the target level.

See [Pyramid construction](../multiscale/pyramid_construction.md) for
the `cross_level_depth` / `cross_level_storage` API and
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a full walkthrough.

### Reading

```python
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.core.arrays import (
    read_chunk_links,
    read_cross_chunk_links,
    read_cross_chunk_link_attributes,
    list_link_deltas,
    list_cross_link_deltas,
)

root = open_store("graph.zarrvectors")
lg   = get_resolution_level(root, 0)

# Intra-level (default)
intra = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=0)

# Cross-level — drill up one pyramid step
plus1 = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=1)

# Global cross-chunk arrays
ccl0  = read_cross_chunk_links(lg, delta=0)
ccl1  = read_cross_chunk_links(lg, delta=1)

# Parallel CCL attributes (new in 0.4)
weights = read_cross_chunk_link_attributes(lg, "weight", delta=1)

# Enumerate available deltas
print(list_link_deltas(lg))         # e.g. [0, +1]   at the bottom level
print(list_cross_link_deltas(lg))   # e.g. [0, +1]
```

`read_cross_chunk_links` tolerates empty/placeholder arrays — when
`<delta>/data` is absent or `num_links == 0`, it returns `[]`. This
matters for fine levels with no cross-chunk parents: the writer skips
creating the directory at all.

### Validation

Walks every `<delta>` subdir under both `links/` and
`cross_chunk_links/` via the listing helpers above.

**L1 (structural):** `links/0/` exists for every geometry type that
declares it in its `arrays_present` capability list (graph, polyline,
streamline, skeleton, mesh). Any `links/<delta != 0>/` or
`cross_chunk_links/<delta != 0>/` triggers the
`CAP_MULTISCALE_LINKS` capability check on root metadata.

**L3 (consistency)** — see
[`zarr_vectors/validate/consistency.py`](../../../zarr_vectors/validate/consistency.py):

- For every `<delta>` walked, all endpoints' chunk coords must be
  decodable (arity = `sid_ndim`).
- For `delta == 0`: both endpoints' chunks must exist in the level's
  chunk grid (i.e. be present in `vertex_group_offsets/`).
- For `delta != 0`: only side A is constrained at the source level —
  side B is validated when the validator reaches the target level
  (`source_level + delta`), where its chunk must exist.
- For `cross_chunk_link_attributes/<name>/<delta>/`: meta `num_links`
  matches the parallel `cross_chunk_links/<delta>/` meta.

**L4 (semantic, opt-in):** for each `delta > 0`, the union of source-
side endpoints in `links/+delta/*` and `cross_chunk_links/+delta/data`
must cover every vertex at the source level — i.e. every fine vertex
has at least one parent at level `source_level + delta`. Useful as an
ID-preservation cross-check for stores written with the per-object
pyramid regime; off by default because it requires a full scan.
