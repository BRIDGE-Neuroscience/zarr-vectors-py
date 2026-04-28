# Validation and repair

The `zarr-vectors` validator checks ZVF stores for conformance at five
levels of increasing thoroughness. This tutorial covers running validation,
interpreting results, and repairing the most common failure modes.

For the complete check catalogue by level, see
[Validation overview](../../spec/validation/overview.md),
[L1 structural](../../spec/validation/l1_structural.md),
[L2 metadata](../../spec/validation/l2_metadata.md), and
[L3 consistency](../../spec/validation/l3_consistency.md).

---

## Running validation

### CLI

```bash
# Full validation (level 5) — recommended after writing any store
zarr-vectors validate scan.zarrvectors

# Specific level — use level 2 for fast CI checks
zarr-vectors validate scan.zarrvectors --level 2

# JSON output — pipe to jq for scripting
zarr-vectors validate scan.zarrvectors --format json | jq '.is_valid'

# Multiple stores
zarr-vectors validate scan.zarrvectors tracts.zarrvectors neurons.zarrvectors

# Validate a remote store
zarr-vectors validate s3://my-bucket/scan.zarrvectors --level 2
```

### Python API

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)

# One-line status
print(result.summary())
# Level 5 validation: PASS — 54 passed, 0 warnings, 0 errors

# Full report (all checks listed)
print(result.report())

# Programmatic access
print(result.is_valid)          # bool
print(len(result.errors))       # int
print(len(result.warnings))     # int

for err in result.errors:
    print(f"[L{err.level}] {err.check}: {err.message}")
    print(f"  at: {err.path}")
```

---

## Choosing a validation level

| Situation | Recommended level |
|-----------|-----------------|
| Quick structural check (CI, file open) | 1 |
| After writing a new store | 3 |
| After ingest from external format | 3 |
| After rechunking | 3 |
| Before publishing / sharing a dataset | 5 |
| Nightly CI on reference fixtures | 5 |
| Large store (> 100 GB), quick sanity check | 2 with `sample_fraction=0.05` |

Level 3 reads all array data and is the minimum recommended for any store
that will be shared or used in analysis. Level 5 additionally checks
multi-resolution pyramid correctness.

---

## Interpreting common errors

### L1 errors

**`cross_chunk_links missing`**

```
ERROR [L1] cross_chunk_links  resolution_0/cross_chunk_links/ missing;
                               required for streamline type
```

The store's geometry type requires `cross_chunk_links/` but the array
is absent. This typically means the store was written with an older version
of `zarr-vectors-py` that did not generate cross-chunk links, or was
written by a third-party tool that omitted the array.

*Repair:* regenerate cross-chunk links from the existing vertex data:

```python
from zarr_vectors.repair import rebuild_cross_chunk_links

rebuild_cross_chunk_links("tracts.zarrvectors", level=0)
```

---

**`object_index missing`**

```
ERROR [L1] object_index  resolution_0/object_index/ missing
```

*Repair:* rebuild the object index from vertices and edges:

```python
from zarr_vectors.repair import rebuild_object_index

rebuild_object_index("tracts.zarrvectors", level=0)
```

---

### L2 errors

**`divisibility`**

```
ERROR [L2] divisibility [d=2]  chunk_shape[2]=200.0, bin_shape[2]=60.0
                                200.0 % 60.0 = 20.0 ≠ 0
```

The `bin_shape` does not evenly divide `chunk_shape`. This cannot be
repaired in-place — the store must be rechunked with a valid `bin_shape`:

```python
from zarr_vectors.core.rechunk import rebin_store

# Change bin_shape to something that divides chunk_shape
rebin_store("scan.zarrvectors", new_bin_shape=(50., 50., 50.))
```

---

**`bin_shape_inconsistent`**

```
ERROR [L2] bin_shape_inconsistent [level=1]
           bin_shape [100,100,80] ≠ base [50,50,50] × ratio [2,2,2] = [100,100,100]
```

The `bin_shape` declared in the per-level `.zattrs` does not match
`base_bin_shape × bin_ratio`. Usually caused by a manual edit to `.zattrs`.

*Repair:* recompute and overwrite the per-level `bin_shape`:

```python
from zarr_vectors.core.store import open_store
import numpy as np

root = open_store("scan.zarrvectors", mode="r+")
base = np.array(root.attrs["base_bin_shape"])
for level_group in root.values():
    if hasattr(level_group, "attrs") and "bin_ratio" in level_group.attrs:
        ratio = np.array(level_group.attrs["bin_ratio"])
        level_group.attrs["bin_shape"] = (base * ratio).tolist()
```

---

**`levels_match_groups` / `level_0_present`**

```
ERROR [L2] levels_match_groups  multiscales entry for resolution_2
                                  references non-existent group
```

The `multiscales` metadata references a level group that does not exist.
Regenerate multiscale metadata:

```python
from zarr_vectors.core.multiscale import write_multiscale_metadata
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)
```

---

### L3 errors

**`vg_offsets_total_count`**

```
ERROR [L3] vg_offsets_total_count [chunk (2,3,1)]
           sum(vg_counts) = 4092 ≠ vertex_count = 4200
```

The total vertex count from the VG index does not match the actual vertex
count in the chunk. This indicates a bug in the writer's VG offset
computation — the most common cause is reordering vertices without
updating the VG index.

*Repair:* rebuild the VG index by re-sorting vertices and recomputing offsets:

```python
from zarr_vectors.repair import rebuild_vg_index

rebuild_vg_index("scan.zarrvectors", level=0)
# Reads vertices, re-sorts into bin order, rewrites vertex_group_offsets
```

---

**`ccl_different_chunks`**

```
ERROR [L3] ccl_different_chunks  2 cross-chunk links found where
           src chunk == dst chunk (rows 14502, 87331)
```

Cross-chunk links where both endpoints are in the same chunk — these
should be intra-chunk edges in `links/edges/`. Caused by incorrect link
generation logic that triggers on bin boundaries instead of chunk
boundaries.

*Repair:* regenerate all cross-chunk links from scratch:

```python
from zarr_vectors.repair import rebuild_cross_chunk_links

rebuild_cross_chunk_links("tracts.zarrvectors", level=0)
```

---

**`attr_length_matches`**

```
ERROR [L3] attr_length_matches [chunk (1,0,2), attr "intensity"]
           attr_length=3800 ≠ vertex_count=4200
```

A per-vertex attribute array has the wrong length in a specific chunk.
This means the attribute was not reordered when vertices were sorted into
VG order — a writer bug.

*Repair:* re-ingest the data from the original source, or use the repair
function if vertex order can be recovered:

```python
from zarr_vectors.repair import realign_attribute

# Re-sort the attribute array to match the current vertex VG order
realign_attribute("scan.zarrvectors", attribute_name="intensity", level=0)
# WARNING: This assumes vertices are already in correct VG order.
# If vertex order is also wrong, rebuild_vg_index must run first.
```

---

**`obj_index_nonempty_vg`**

```
ERROR [L3] obj_index_nonempty_vg  object 1042 primary VG at
           (chunk=8843, bin=12) has count=0 (empty VG)
```

The object index points to an empty VG. This usually means the object's
vertices were moved by a rechunking operation that did not update the
object index.

*Repair:* rebuild the object index:

```python
from zarr_vectors.repair import rebuild_object_index

rebuild_object_index("tracts.zarrvectors", level=0)
```

---

## Validation after repair

Always re-run the validator at the same or higher level after any repair:

```python
result = validate("tracts.zarrvectors", level=3)
assert result.is_valid, result.report()
print("Store is valid after repair.")
```

---

## Sampled validation for large stores

Full L3 validation on stores > 100 GB can take tens of minutes. For routine
health checks, sample a fraction of chunks:

```python
result = validate(
    "large_scan.zarrvectors",
    level=3,
    sample_fraction=0.05,   # validate 5% of chunks, chosen randomly
    seed=42,
)
print(result.summary())
# Level 3 validation (sampled 5%): PASS — 38 passed, 0 warnings, 0 errors
# NOTE: sampled validation may miss errors in unsampled chunks
```

Sampled validation is never a substitute for full validation before
publishing a dataset. Use it for fast incremental checks during development.

---

## Automated validation in CI

```yaml
# .github/workflows/validate.yml
- name: Validate reference fixtures
  run: |
    zarr-vectors validate tests/fixtures/point_cloud_3d/store.zarrvectors --level 5
    zarr-vectors validate tests/fixtures/streamline_3d_multiscale/store.zarrvectors --level 5
```

Or in a pytest fixture:

```python
import pytest
from zarr_vectors.validate import validate
from pathlib import Path

FIXTURES = list((Path("tests") / "fixtures").glob("*/store.zarrvectors"))

@pytest.mark.parametrize("store_path", FIXTURES, ids=lambda p: p.parent.name)
@pytest.mark.slow
def test_fixture_passes_l5(store_path):
    result = validate(str(store_path), level=5)
    assert result.is_valid, result.report()
```
