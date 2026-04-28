# Pyramid construction

## Terms

**Metanode**
: A synthetic vertex that represents all vertices within a supervoxel bin
  at a coarser resolution level. Its position is the centroid (or another
  representative position) of the original vertices in the bin. Metanodes
  are the fundamental unit of spatial coarsening in ZVF.

**Centroid**
: The mean position of all vertices within a bin, weighted equally. The
  default position strategy for metanodes. Other strategies (first vertex,
  median, weighted centroid) are supported.

**Vertex reduction**
: The factor by which the number of stored vertices decreases from one
  level to the next. Determined by `bin_ratio`: a ratio of `(r, r, r)`
  gives at most `r^D × ` vertex reduction (exact reduction depends on
  actual vertex distribution).

**Object reduction**
: The factor by which the number of stored objects decreases from one
  level to the next. Determined by `object_sparsity`. Independent of
  `bin_ratio`.

**Total reduction**
: The product of vertex reduction and object reduction. Determines the
  total data volume at a coarser level relative to the base level.

**Attribute aggregation**
: The strategy used to compute the attribute value of a metanode from the
  attributes of the original vertices in its bin. Options: `mean`
  (default), `sum`, `max`, `min`, `median`, `first`.

---

## Introduction

Building a resolution pyramid means constructing one or more coarser
representations of the base-level data. In ZVF, coarsening operates
independently on two axes: *spatial coarsening* (merging vertices into
metanodes via binning) and *object thinning* (retaining only a subset of
discrete objects).

This page documents the coarsening algorithms for each geometry type,
the API for controlling pyramid construction, and the design choices made
in `zarr-vectors-py`.

---

## Technical reference

### Coarsening by geometry type

#### Point cloud

Point cloud coarsening is purely spatial. For each bin at the coarser level,
all vertices in the corresponding bin at the base level (or the previous
level) are merged into a single metanode:

```
metanode_position = mean(positions in bin)   (default: centroid)
metanode_attribute = aggregation(attributes in bin)
```

There are no discrete objects in a point cloud; `object_sparsity` is
ignored (always `1.0`).

**Example:**

```
Base level bin contains:   [100.2, 200.5, 300.1]
                           [101.3, 199.8, 301.0]
                           [100.9, 200.1, 299.7]

Metanode at level 1:       [100.8, 200.1, 300.3]  ← centroid
```

#### Polyline and streamline

Two-stage coarsening:

1. **Spatial coarsening (always):** for each object, the vertex sequence is
   downsampled by replacing each contiguous run of vertices within the same
   bin with one metanode at their centroid. Edge connectivity is preserved:
   consecutive metanodes are connected.

2. **Object thinning (when `object_sparsity < 1.0`):** a subset of objects
   is selected for the coarser level using the declared `sparsity_strategy`.
   Non-selected objects are not written to the coarser level.

After both stages, cross-chunk links are recomputed for the coarser level
(because metanode positions may fall in different chunks than the original
vertices).

**Illustration (single streamline, bin_ratio = 2):**

```
Base level vertices:     A  B  C  D | E  F  G  H      (| = chunk boundary)
Bin assignment:          0  0  0  1 | 0  0  1  1

Level 1 coarsening:
  Chunk 0: bin 0 → centroid(A,B,C), bin 1 → D
  Chunk 1: bin 0 → centroid(E,F),   bin 1 → centroid(G,H)

Level 1 streamline:      ABC' D  | EF'  GH'            (primed = metanode)
Cross-chunk link:        D → EF' (same as base level, different vertex IDs)
```

#### Graph and skeleton

Graph coarsening merges spatially close nodes into metanodes. For each bin,
all vertices in the bin are merged into one metanode. Edges are
updated: an edge `(u, v)` at the base level becomes an edge between the
metanodes of `u` and `v` at the coarser level. Self-loops (both endpoints
in the same bin) are removed. Duplicate edges (multiple base-level edges
mapping to the same pair of metanodes) are deduplicated.

For skeletons, the tree structure is preserved: the metanode with the
smallest path distance to the root inherits the root designation.

Object thinning for multi-skeleton stores is supported: when a store
contains many independent skeletons (e.g. a connectome store with many
neurons), `object_sparsity` controls how many are retained.

#### Mesh

Mesh coarsening is more complex than other types because mesh connectivity
(face topology) must be maintained. `zarr-vectors-py` uses a
**vertex-merging** approach:

1. Identify all vertices in each bin and replace them with one metanode.
2. Update face vertex indices to reference metanodes.
3. Remove degenerate faces (triangles with two or more identical vertices
   after merging).
4. Remove duplicate faces.

This approach is conservative (it preserves topology at the cost of some
geometric quality) but does not require a full mesh simplification algorithm.
For high-quality coarsening of mesh data, consider pre-processing with a
dedicated mesh decimation tool before writing to ZVF.

### `build_pyramid` API

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2), "object_sparsity": 1.0},
        {"bin_ratio": (4, 4, 4), "object_sparsity": 0.5},
    ],
    attribute_aggregation="mean",   # how to aggregate per-vertex attributes
    position_strategy="centroid",   # metanode position (centroid | first | median)
)
```

`level_configs` is applied sequentially: level 1 is coarsened from level 0,
level 2 is coarsened from level 1, and so on. Each level's `bin_ratio` is
relative to `base_bin_shape`, not to the previous level.

### `coarsen_level` API

For finer control, use `coarsen_level` directly:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "scan.zarrvectors",
    source_level=0,
    target_level=1,
    bin_ratio=(2, 2, 2),
    object_sparsity=0.5,
    sparsity_strategy="spatial_coverage",
    attribute_aggregation="mean",
    position_strategy="centroid",
)
```

`source_level` does not have to be 0. You can coarsen from level 1 to
level 2 to build a three-level pyramid incrementally.

### Attribute aggregation strategies

| Strategy | Description | Use case |
|----------|-------------|----------|
| `mean` | Mean of values in bin | Continuous scalars (FA, intensity, concentration) |
| `sum` | Sum of values in bin | Counts, densities |
| `max` | Maximum in bin | Peak activations, distance-to-surface |
| `min` | Minimum in bin | Minimum stress, threshold fields |
| `median` | Median of values in bin | Robust aggregation, ignores outliers |
| `first` | First vertex value | Categorical labels (takes majority instead; see below) |
| `majority` | Most frequent value | Categorical/label attributes |

Different attributes within the same store can use different aggregation
strategies:

```python
coarsen_level(
    "scan.zarrvectors",
    source_level=0, target_level=1,
    bin_ratio=(2, 2, 2),
    attribute_aggregation={
        "intensity":  "mean",
        "label":      "majority",
        "confidence": "min",
    },
)
```

### Metanode position strategies

| Strategy | Formula | Notes |
|----------|---------|-------|
| `centroid` | `mean(positions in bin)` | Default. Best spatial accuracy. |
| `first` | First vertex (by VG order) | Fast; biased toward lower-coordinate vertices. |
| `median` | Component-wise median | Robust to outliers in the bin. |
| `weighted_centroid` | Weighted mean using a named attribute | Useful when one attribute encodes importance. |

```python
coarsen_level(
    "scan.zarrvectors", 0, 1, bin_ratio=(2, 2, 2),
    position_strategy="weighted_centroid",
    position_weight_attribute="intensity",
)
```

### Legacy automatic pyramid

`build_pyramid` with no `level_configs` argument uses a legacy heuristic:
levels are added until the total vertex count falls below 10 000, using
`bin_ratio = (2, 2, 2)` and `object_sparsity = 1.0` at each level.

```python
build_pyramid("scan.zarrvectors")   # legacy: auto-select levels
```

This is convenient for quick exploratory use but not recommended for
production pipelines, where explicit `level_configs` are preferred.
