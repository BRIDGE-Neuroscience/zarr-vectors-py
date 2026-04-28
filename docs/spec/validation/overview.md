# Validation overview

## Terms

**Conformance level**
: An integer from 1 to 5 declaring the thoroughness of a validation run.
  Each level is a strict superset of the previous: a store that passes
  level `N` also passes all levels below `N`. Running level 5 runs all
  checks.

**`ValidationResult`**
: The object returned by `validate()`. Carries a list of passed checks,
  warnings, and errors, plus a `summary()` method and an `is_valid`
  property.

**Check**
: A single assertion evaluated during validation. A check either passes,
  emits a warning (non-fatal), or raises an error (fatal). Errors
  accumulate; validation does not stop at the first failure.

**Warning**
: A non-fatal issue that indicates the store may behave unexpectedly in
  some tools but is not technically invalid. Example: a store without
  consolidated metadata will be slow to open on object stores.

**Error**
: A fatal issue that indicates the store does not conform to the ZVF spec
  at the declared conformance level. Example: a `bin_shape` value that
  does not evenly divide `chunk_shape`.

---

## Introduction

`zarr-vectors-py` ships a multi-level validator that checks ZVF stores for
correctness and conformance. Validation is organised into five progressively
deeper levels. Shallow levels (1–2) are fast and check structural and
metadata properties. Deeper levels (3–5) are more expensive because they
require reading array data, but they catch logical inconsistencies that
metadata checks alone cannot detect.

The validator is designed to be useful in several contexts:

- **After writing:** confirm that a newly written store is valid before
  sharing it.
- **After ingest:** confirm that a third-party format was correctly
  translated.
- **After rechunking:** confirm that the rechunked store is consistent.
- **CI pipelines:** run level 1–2 checks quickly; reserve level 5 for
  nightly runs.
- **Debugging:** identify the specific check that fails to pinpoint bugs
  in writer or converter code.

---

## Technical reference

### Conformance levels

| Level | Name | What it checks | Typical runtime |
|-------|------|----------------|-----------------|
| 1 | Structural | Required files/groups/arrays exist; correct Zarr node types | < 1 s |
| 2 | Metadata | `.zattrs` schema validity; dtype/shape declarations; divisibility constraints; all keys present and correctly typed | 1–5 s |
| 3 | Consistency | VG offset arithmetic; manifest integrity; cross-chunk link validity; attribute–vertex alignment | 10 s – 10 min (reads all chunks) |
| 4 | Geometry | Type-specific constraints: tree topology for `skeleton`/`graph(is_tree)`; watertightness for `mesh(closed_surface)`; polyline gap detection | varies |
| 5 | Pyramid | Multi-resolution correctness: monotonically non-increasing vertex and object counts; `bin_ratio` / `bin_shape` / `object_sparsity` self-consistency across levels | adds per-level cost |

### Python API

```python
from zarr_vectors.validate import validate

# Run full validation (level 5)
result = validate("scan.zarrvectors", level=5)

# Print summary
print(result.summary())
# Level 5 validation: PASS
#   54 passed, 2 warnings, 0 errors

# Check programmatically
if not result.is_valid:
    for err in result.errors:
        print(f"ERROR [{err.level}] {err.check}: {err.message}")

# Run only fast checks (CI use)
result = validate("scan.zarrvectors", level=2)

# Validate a specific level group only
result = validate("scan.zarrvectors", level=3, resolution_levels=[0])
```

### `ValidationResult` API

```python
result.is_valid           # bool: True if no errors at any level
result.passed             # list[Check]: checks that passed
result.warnings           # list[Check]: non-fatal warnings
result.errors             # list[Check]: fatal errors
result.summary()          # str: one-line human-readable summary
result.report()           # str: full multi-line report with all checks
result.as_dict()          # dict: machine-readable representation

# Iterate over all checks
for check in result.all_checks:
    print(check.level, check.name, check.status, check.message)
```

### `Check` object

```python
check.level      # int: conformance level this check belongs to
check.name       # str: human-readable check name (e.g. "bin_divisibility")
check.status     # str: "pass", "warning", or "error"
check.message    # str: description of what was checked / what failed
check.path       # str: store path relevant to this check (e.g. "resolution_0/.zattrs")
```

### CLI usage

```bash
# Default: level 5
zarr-vectors validate scan.zarrvectors

# Specific level
zarr-vectors validate scan.zarrvectors --level 2

# JSON output for CI
zarr-vectors validate scan.zarrvectors --format json | jq '.is_valid'

# Validate multiple stores
zarr-vectors validate scan.zarrvectors tracts.zarrvectors --level 3
```

### Validation and writers

The write functions in `zarr-vectors-py` perform inline validation of
arguments before writing. However, this does not substitute for post-write
validation: inline checks guard against obviously invalid parameters but do
not verify the correctness of the written data (e.g. VG offset arithmetic,
cross-chunk link completeness). Always run at least level 3 after writing
a new store.

### Validation and performance

Levels 1 and 2 touch only metadata files. Level 3 and above require reading
all array chunks. For a large store (> 100 GB), level 3 validation may take
several minutes. To validate a sample of chunks rather than all:

```python
result = validate("scan.zarrvectors", level=3, sample_fraction=0.1, seed=42)
```

Sampled validation is non-deterministic (random chunk selection) and may
miss errors confined to specific chunks. It is suitable for quick sanity
checks, not for publication-quality conformance certification.
