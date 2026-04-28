# L3: Consistency validation

## Terms

**Consistency check**
: A validation check that reads array data and verifies that the values
  in one array are logically consistent with values in another. L3 checks
  read all chunks of all arrays at each level.

**VG arithmetic check**
: Verification that the offsets and counts in `vertex_group_offsets/` are
  self-consistent: no overlap between VG slices, no gap between consecutive
  VG slices, total vertex count matches `vertices/` array length.

**Manifest integrity**
: Verification that every entry in `object_index/` references a valid,
  non-empty VG, and that following `cross_chunk_links/` from each primary
  VG reaches all vertices belonging to the object.

**Attribute alignment**
: Verification that the length of each `attributes/<name>/` chunk slice
  equals the length of the corresponding `vertices/` chunk slice. Misaligned
  attributes indicate a bug in the writer's vertex-reordering logic.

---

## Introduction

L3 validation reads array data and checks that the store's internal
structure is logically consistent. It is the first level that can detect
bugs introduced by incorrect writer implementations, rechunking errors, or
manual store modifications.

L3 is substantially more expensive than L1–L2 because it reads all chunks.
For a 100 GB store, L3 may take 5–30 minutes depending on storage bandwidth.
For development and CI, run L3 on small synthetic stores; run L1–L2 on
full-size production stores unless a specific consistency issue is suspected.

---

## Technical reference

### VG offset arithmetic checks

For every non-empty chunk in `vertices/` and `vertex_group_offsets/`:

| Check | Rule | Failure type |
|-------|------|--------------|
| `vg_offsets_non_negative` | All `count` values ≥ -1 (−1 = fill, 0 = empty, >0 = populated) | Error |
| `vg_offsets_no_overlap` | No two VG slices `[offset, offset+count)` overlap | Error |
| `vg_offsets_no_gap` | VG slices are contiguous: VG k starts where VG k-1 ends (or begins at 0 for the first non-empty VG) | Error |
| `vg_offsets_total_count` | `sum(max(0, count) for all bins)` equals the number of vertices in this chunk | Error |
| `vg_offsets_in_bounds` | `offset + count ≤ N_chunk` for all VGs | Error |
| `vg_order_correct` | Vertices at indices `[offset, offset+count)` all have bin flat index equal to the VG's bin flat index | Error |

The VG order check is the most expensive: it requires computing the bin
flat index of every vertex and comparing to the VG index. It runs by
default at L3 but can be disabled with `skip_vg_order_check=True` for
large stores where only offset arithmetic is needed:

```python
result = validate("scan.zarrvectors", level=3, skip_vg_order_check=True)
```

### Attribute alignment checks

For every chunk at every level:

| Check | Rule | Failure type |
|-------|------|--------------|
| `attr_length_matches` | `len(attributes[name][chunk])` equals `len(vertices[chunk])` for all named attributes | Error |
| `attr_no_nan_default` | No NaN values in float attributes unless `fill_value = NaN` is declared | Warning |
| `obj_attr_length` | `len(object_attributes[name])` equals `object_index.shape[0]` | Error |

### Object index checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `obj_index_valid_chunks` | Every `chunk_flat` in `object_index/` is in `[0, product(chunk_grid_shape))` | Error |
| `obj_index_valid_vg` | Every `vg_flat` in `object_index/` is in `[0, B_per_chunk)` | Error |
| `obj_index_nonempty_vg` | The VG at each primary address has `count > 0` | Error |
| `obj_index_unique` | No two objects share the same primary VG address | Error |

### Cross-chunk link checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `ccl_global_id_valid` | Every global vertex ID in `cross_chunk_links/` is in `[0, chunk_flat_max × N_max + N_chunk_max)` | Error |
| `ccl_different_chunks` | `src // N_max != dst // N_max` for all links | Error |
| `ccl_vertices_exist` | The vertices referenced by all global IDs actually exist (count > 0 in the VG containing them) | Error |
| `ccl_no_polyline_cycles` | The directed graph formed by `cross_chunk_links/` for polyline/streamline stores contains no cycles | Error |
| `ccl_no_duplicate_undirected` | For undirected graph stores: no link `[a, b]` co-exists with `[b, a]` | Error |

### Edge index checks

For polyline, streamline, graph, and skeleton types:

| Check | Rule | Failure type |
|-------|------|--------------|
| `edge_indices_in_bounds` | All local vertex indices in `links/edges/` are in `[0, N_chunk)` or equal `−1` | Error |
| `no_self_loops` | No edge has `src == dst` | Error |
| `polyline_continuity` | For polyline/streamline: every non-terminal vertex has exactly one outgoing intra-chunk edge (or a cross-chunk link as continuation) | Error |

### Mesh face checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `face_indices_valid` | All positive face indices in `[0, N_chunk)` | Error |
| `boundary_indices_decodable` | All negative face indices decode to valid global vertex IDs | Error |
| `no_degenerate_faces` | No face has two or more identical vertex indices | Error |

### Pyramid consistency checks (L3 component)

| Check | Rule | Failure type |
|-------|------|--------------|
| `vertex_count_non_increasing` | Total vertex count at level N ≤ total vertex count at level N-1 | Error |
| `attribute_names_match` | Attribute name sets are identical across all levels | Warning |

### Example L3 report (abbreviated)

```
Level 3 validation of scan.zarrvectors
========================================
Checking resolution_0 (125 chunks)…
PASS  vg_offsets_non_negative [0]   all counts ≥ -1
PASS  vg_offsets_no_overlap [0]     no VG slice overlaps
PASS  vg_offsets_total_count [0]    vertex totals match
PASS  vg_order_correct [0]          all vertices in correct VG order
PASS  attr_length_matches [0]       intensity/: 125/125 chunks aligned
ERROR ccl_different_chunks [0]      2 links found where src chunk == dst chunk
                                    (links at rows 14502, 87331)
PASS  edge_indices_in_bounds [0]    all intra-chunk edges valid
PASS  vertex_count_non_increasing   level 1 (82453) ≤ level 0 (100000) ✓

Level 3 validation: FAIL — 47 passed, 0 warnings, 1 error
```

The error above (`ccl_different_chunks`) indicates that the writer
erroneously generated cross-chunk links for same-chunk vertex pairs — the
most common correctness bug in cross-chunk link generation. See
[Cross-chunk links](../object_model/cross_chunk_links.md) for the
correct generation algorithm.
