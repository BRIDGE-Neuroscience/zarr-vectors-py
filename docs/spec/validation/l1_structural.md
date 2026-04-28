# L1: Structural validation

## Terms

**Structural check**
: A validation check that examines only the presence and node type of
  paths in the Zarr store, without reading array data or interpreting
  metadata values.

**Required path**
: A store path that must exist for a valid ZVF store of a given geometry
  type. Missing required paths are L1 errors.

**Node type**
: Whether a path is a Zarr group or a Zarr array, as declared by its
  `zarr.json`. A path that exists with the wrong node type (e.g. an array
  where a group is expected) is an L1 error.

---

## Introduction

L1 validation answers the question: "does this store have the right shape?"
It checks that every required file and array is present, that paths that
should be groups are groups, and that paths that should be arrays are arrays.
It does not read any array data and does not interpret metadata values.

L1 is the fastest validation level and is appropriate as a first triage
step when opening an unfamiliar store. An L1 failure means the store is
structurally incomplete and cannot be read by any ZVF reader.

---

## Technical reference

### Checks performed

#### Root level

| Check | Description | Failure type |
|-------|-------------|--------------|
| `root_zarr_json` | `zarr.json` exists at store root and declares `node_type: group` | Error |
| `root_zattrs` | `.zattrs` exists at store root and is valid JSON | Error |
| `root_metadata_json` | `metadata.json` exists at store root | Warning (recommended) |
| `resolution_0_exists` | `resolution_0/` group exists | Error |

#### Per resolution level (repeated for each declared level)

| Check | Description | Failure type |
|-------|-------------|--------------|
| `level_zarr_json` | `resolution_N/zarr.json` exists and declares `node_type: group` | Error |
| `level_zattrs` | `resolution_N/.zattrs` exists and is valid JSON | Error |
| `vertices_array` | `resolution_N/vertices/zarr.json` declares `node_type: array` | Error |
| `vg_offsets_array` | `resolution_N/vertex_group_offsets/zarr.json` declares `node_type: array` | Error |

#### Type-specific array presence (based on `geometry_type` in root `.zattrs`)

| Geometry type | Required arrays | Warning if absent |
|---------------|----------------|-------------------|
| `point_cloud` | `vertices/`, `vertex_group_offsets/` | `attributes/` if attributes were written at other levels |
| `line` | + `links/edges/` | — |
| `polyline` | + `links/edges/`, `object_index/` | `cross_chunk_links/` if objects may span chunks |
| `streamline` | + `links/edges/`, `object_index/` | `cross_chunk_links/` |
| `graph` | + `links/edges/`, `object_index/` | `cross_chunk_links/` |
| `skeleton` | + `links/edges/`, `object_index/` | `cross_chunk_links/` |
| `mesh` | + `links/faces/`, `object_index/` | — |

#### Attribute sub-groups

| Check | Description | Failure type |
|-------|-------------|--------------|
| `attributes_is_group` | `resolution_N/attributes/` is a Zarr group if present | Error |
| `attributes_sub_arrays` | Each `resolution_N/attributes/<name>/` declares `node_type: array` | Error |
| `object_attrs_is_group` | `resolution_N/object_attributes/` is a Zarr group if present | Error |
| `groupings_array` | `resolution_N/groupings/zarr.json` declares array if present | Error |

#### Attribute consistency across levels

| Check | Description | Failure type |
|-------|-------------|--------------|
| `attribute_names_consistent` | The set of per-vertex attribute names is the same at every resolution level | Warning |
| `object_attr_names_consistent` | The set of per-object attribute names is the same at every resolution level | Warning |

### Example L1 report

```
Level 1 validation of scan.zarrvectors
=======================================
PASS  root_zarr_json               zarr.json exists at store root
PASS  root_zattrs                  .zattrs exists and is valid JSON
WARN  root_metadata_json           metadata.json not found (recommended)
PASS  resolution_0_exists          resolution_0/ group exists
PASS  level_zarr_json [level=0]    resolution_0/zarr.json exists
PASS  level_zattrs [level=0]       resolution_0/.zattrs exists
PASS  vertices_array [level=0]     resolution_0/vertices/ is an array
PASS  vg_offsets_array [level=0]   resolution_0/vertex_group_offsets/ is an array
PASS  object_index_array [level=0] resolution_0/object_index/ is an array
PASS  edges_array [level=0]        resolution_0/links/edges/ is an array
ERROR cross_chunk_links [level=0]  resolution_0/cross_chunk_links/ missing;
                                   required for streamline type

Level 1 validation: FAIL — 9 passed, 1 warning, 1 error
```

### Implementation notes for contributors

L1 checks work by listing the keys in the store and checking for the
presence of required paths. In `zarr-vectors-py`, L1 is implemented in
`zarr_vectors.validate.structure`:

```python
from zarr_vectors.validate.structure import check_l1

result = check_l1(store_root, geometry_type)
```

To add a new required path for a new geometry type (see
[Adding geometry types](../contributing/adding_geometry_types.md)), add an
entry to the `REQUIRED_ARRAYS` dict in `zarr_vectors/validate/structure.py`
and add a corresponding test fixture.
