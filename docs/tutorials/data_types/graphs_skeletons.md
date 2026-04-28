# Graphs and skeletons

ZVF provides two graph-structured geometry types. Use `skeleton` for
neuronal morphologies, vascular trees, and any other branching structure
that must align to the SWC convention. Use `graph` for arbitrary
connectivity — vascular networks with anastomoses, synaptic connectivity
graphs embedded in 3-D space, or any structure where cycles are valid.

Both types use the same on-disk array schema; the distinction is the
`is_tree` flag and the additional SWC-compatible attributes that `skeleton`
stores.

All examples require only `zarr-vectors` base install except the SWC and
GraphML ingest sections, which require `zarr-vectors[ingest]`.

---

## Skeletons (SWC-aligned)

### Ingest from SWC

The most common way to create a ZVF skeleton store is by ingesting an
existing SWC file:

```python
from zarr_vectors.ingest.swc import ingest_swc
from zarr_vectors.types.graphs import read_graph

ingest_swc(
    "neuron.swc",
    "neuron.zarrvectors",
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
)

result = read_graph("neuron.zarrvectors")
print(result["node_count"])          # number of SWC rows
print(result["edge_count"])          # number of parent-child edges
print(result["attributes"]["radius"].shape)    # (n_nodes,)
print(result["attributes"]["swc_type"].shape)  # (n_nodes,)
```

After ingest, every SWC column is preserved:

| SWC column | ZVF storage |
|-----------|-------------|
| `x, y, z` | `vertices/` positions |
| `radius` | `attributes/radius/` (float32) |
| `type` | `attributes/swc_type/` (int32, SWC compartment taxonomy) |
| `id` | `attributes/swc_id/` (int64, original row ID) |
| `parent_id` | Encoded in `links/edges/` as `[child, parent]` |

### Export to SWC

Round-trip back to SWC is lossless when `swc_compatible = true` (the
default after `ingest_swc`):

```python
from zarr_vectors.export.swc import export_swc

export_swc("neuron.zarrvectors", "neuron_out.swc")
# Produces an SWC identical to the input (possibly with re-ordered rows)
```

### Write a skeleton programmatically

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes = 800

# Simulate a branching skeleton: soma at origin, random walk branches
positions = np.zeros((n_nodes, 3), dtype=np.float32)
positions[1:] = rng.normal(0, 5, (n_nodes - 1, 3)).cumsum(0)

# Parent array: node 0 is root (parent = -1), others have sequential parents
# In a real skeleton, parent relationships encode the tree topology
parents = np.arange(-1, n_nodes - 1, dtype=np.int32)
edges   = np.column_stack([
    np.arange(1, n_nodes, dtype=np.int32),   # child indices
    parents[1:],                              # parent indices
])

radii    = rng.uniform(0.2, 2.0, n_nodes).astype(np.float32)
swc_type = np.ones(n_nodes, dtype=np.int32)
swc_type[0] = 1   # soma

write_graph(
    "neuron.zarrvectors",
    positions=positions,
    edges=edges,            # [child, parent] pairs; root has parent = -1
    chunk_shape=(200.0, 200.0, 200.0),
    geometry_type="skeleton",
    is_tree=True,           # enables tree topology validation at write time
    vertex_attributes={
        "radius":   radii,
        "swc_type": swc_type,
    },
)
```

### Spatial query on a skeleton

```python
from zarr_vectors.types.graphs import read_graph

# All nodes within a 100³ µm region
result = read_graph(
    "neuron.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
)
print(result["node_count"])
print(result["edge_count"])   # edges where both endpoints are in bbox
```

To include edges that cross the bbox boundary:

```python
result = read_graph(
    "neuron.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
    include_boundary_edges=True,
)
```

---

## Multi-skeleton stores

For connectome-scale datasets with thousands of neurons, a single ZVF
store is far more efficient than per-neuron files:

### Ingest a directory of SWC files

```python
from zarr_vectors.ingest.swc import ingest_swc_directory

ingest_swc_directory(
    "morphologies/",              # directory of .swc files
    "connectome.zarrvectors",
    chunk_shape=(500.0, 500.0, 500.0),
    bin_shape=(100.0, 100.0, 100.0),
)
```

Each SWC file becomes one object (skeleton) in the store. The object ID
is the index of the SWC file in sorted filename order. To get the mapping:

```python
from zarr_vectors.ingest.swc import get_swc_object_id_map

id_map = get_swc_object_id_map("connectome.zarrvectors")
# {"neuron_001.swc": 0, "neuron_002.swc": 1, …}
```

### Read a specific neuron

```python
from zarr_vectors.types.graphs import read_graph

result = read_graph("connectome.zarrvectors", object_ids=[42])
print(result["node_count"])         # nodes in neuron 42
print(result["edge_count"])
print(result["attributes"]["radius"])

# Reconstruct as SWC-formatted dict
nodes = result["positions"]           # (N, 3)
radii = result["attributes"]["radius"]
types = result["attributes"]["swc_type"]
edges = result["edges"]               # (E, 2) [child, parent]
```

### Spatial query in a connectome store

```python
# Which neurons have processes in a given region?
result = read_graph(
    "connectome.zarrvectors",
    bbox=(np.array([2000., 3000., 1500.]),
          np.array([2500., 3500., 2000.])),
    return_object_ids=True,
)
print(result["object_ids"])       # IDs of neurons with nodes in region
print(result["node_count"])       # total nodes in the bbox across all neurons
```

---

## General graphs

### Writing a graph

```python
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes   = 2000
positions = rng.uniform(0, 500, (n_nodes, 3)).astype(np.float32)

# Simulate a vascular network: ~3 edges per node
src = rng.integers(0, n_nodes, 3000)
dst = rng.integers(0, n_nodes, 3000)
# Remove self-loops
mask  = src != dst
edges = np.column_stack([src[mask], dst[mask]]).astype(np.int32)

write_graph(
    "vessels.zarrvectors",
    positions=positions,
    edges=edges,
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    is_directed=False,
    is_tree=False,
    vertex_attributes={
        "diameter": rng.uniform(1, 20, n_nodes).astype(np.float32),
        "flow":     rng.uniform(0, 1,  n_nodes).astype(np.float32),
    },
)
```

### Writing a directed graph

```python
write_graph(
    "directed.zarrvectors",
    positions=positions,
    edges=edges,          # [source, destination] — direction is significant
    chunk_shape=(100., 100., 100.),
    is_directed=True,
)
```

### Reading a graph

```python
from zarr_vectors.types.graphs import read_graph

result = read_graph("vessels.zarrvectors")
print(result["node_count"])              # 2000
print(result["edge_count"])
print(result["positions"].shape)         # (2000, 3)
print(result["edges"].shape)             # (E, 2)
print(result["attributes"]["diameter"].shape)   # (2000,)
```

---

## GraphML ingest

GraphML files that include node coordinate attributes can be ingested
directly (requires `zarr-vectors[ingest]`):

```python
from zarr_vectors.ingest.graphml import ingest_graphml

ingest_graphml(
    "network.graphml",
    "network.zarrvectors",
    chunk_shape=(100.0, 100.0, 100.0),
    coordinate_attributes=("x", "y", "z"),  # GraphML attribute names for coordinates
    edge_attribute_columns=["weight"],
)
```

For GraphML files without explicit spatial coordinates, pass a coordinate
array from a separate source:

```python
ingest_graphml(
    "network.graphml",
    "network.zarrvectors",
    chunk_shape=(100., 100., 100.),
    external_positions=positions_array,   # (n_nodes, 3) float32
)
```

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("neuron.zarrvectors", level=4)
print(result.summary())
# Level 4 validation: PASS
#   38 passed, 0 warnings, 0 errors
```

Level 4 validation includes tree topology checks (for `is_tree = true`):
connected, acyclic, exactly one root vertex per component.

---

## Multi-resolution pyramids

Graph pyramids coarsen vertex positions and deduplicate edges:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "vessels.zarrvectors",
    level_configs=[
        {"bin_ratio": (2, 2, 2), "object_sparsity": 1.0},
    ],
    attribute_aggregation={
        "diameter": "max",     # keep maximum diameter in each bin
        "flow":     "mean",
    },
)
```

For skeleton stores with `object_sparsity < 1.0`, individual neurons are
thinned at coarser levels using the declared sparsity strategy.

---

## Common pitfalls

**`is_tree=True` raises an error at write time.**
The writer validates tree topology when `is_tree=True`. If your data has
cycles (even due to floating-point duplicate positions that produce
coincident vertices), the write will fail. Check for cycles with:

```python
import networkx as nx
G = nx.from_edgelist(edges.tolist())
print(nx.is_forest(G))     # should be True for a valid tree
```

**Cross-chunk edges in a graph are not in `links/edges/`.**
Edges between vertices in different chunks are stored in
`cross_chunk_links/`, not in the per-chunk `links/edges/` array. When
reading a specific region, pass `include_boundary_edges=True` to include
these inter-chunk connections.

**SWC parent ID −1 vs 0.**
Some SWC tools use parent ID `0` (1-indexed) for the root; others use
`-1` (ZVF convention). `ingest_swc` detects the convention automatically
from the first data row. If your SWC uses a non-standard root convention,
pass `root_parent_id=<value>` to override.

**Object IDs change after rechunking.**
Object IDs are assigned at write time and are stable across reads on the
same store. However, rechunking rebuilds the `object_index/` and may
reassign IDs. If you need stable long-term IDs (e.g. for a connectome
database), store the canonical ID as a per-object attribute:

```python
write_graph(..., object_attributes={"neuron_id": canonical_ids})
```
