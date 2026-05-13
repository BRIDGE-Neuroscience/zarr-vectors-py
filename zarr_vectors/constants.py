"""Format constants for the Zarr Vectors (ZV) format.

These are the canonical names, prefixes, and default values used
throughout the specification.  Changing a value here changes it
everywhere in the package.
"""

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

FORMAT_VERSION: str = "0.4"
"""Current ZV specification version."""

# Capability tokens stored in RootMetadata.format_capabilities.  Readers
# inspect these to know which optional 0.3+ features the store uses, and
# degrade gracefully when a capability is absent.
CAP_CROSS_CHUNK_FACES: str = "cross_chunk_faces"
"""Store contains the cross_chunk_faces array (face-identity preservation)."""

CAP_VERTEX_COUNT_CACHE: str = "vertex_count_cache"
"""Per-chunk vertex_counts/<key> sidecars are present."""

CAP_OBJECT_INDEX_PENDING: str = "object_index_pending"
"""Store has uncompacted object_index pending sidecars."""

CAP_PRESERVED_OBJECT_IDS: str = "preserved_object_ids"
"""At least one resolution level was written with ID-preserving
sparsification (``preserves_object_ids=True`` on the level metadata).
Dropped objects appear as empty manifest slots and zero
``present_mask`` bytes; ``parent_level`` carries semantic weight."""

CAP_SHARED_VERTEX_GROUPS: str = "shared_vertex_groups"
"""At least one resolution level stores per-chunk vertex groups that
may be referenced by multiple objects' manifests (shared metavertices
in the per-object pyramid regime)."""

CAP_MULTISCALE_LINKS: str = "multiscale_links"
"""Store uses the 0.4 multiscale links layout (``links/<delta>/``,
``cross_chunk_links/<delta>/``, ``link_attributes/<name>/<delta>/`` and
``cross_chunk_link_attributes/<name>/<delta>/``) and may contain
cross-pyramid-level edges (``delta != 0``)."""

DEFAULT_AXES_NAMES: tuple[str, ...] = ("x", "y", "z", "w")
"""Default axis names used when first-write inference creates root
metadata for a store that was warmed without an explicit ``axes`` kwarg.
Indexed by ``sid_ndim`` (1 → ``("x",)``, 2 → ``("x", "y")``, ...).
Stops at 4 dims; higher-dim stores must pass axes explicitly."""

# ---------------------------------------------------------------------------
# Store layout names
# ---------------------------------------------------------------------------

RESOLUTION_PREFIX: str = "resolution_"
"""Prefix for resolution level groups: resolution_0, resolution_1, ..."""

PARAMETRIC_GROUP: str = "parametric"
"""Name of the root-level parametric objects group."""

# ---------------------------------------------------------------------------
# Per-level array names
# ---------------------------------------------------------------------------

VERTICES: str = "vertices"
VERTEX_GROUP_OFFSETS: str = "vertex_group_offsets"
LINKS: str = "links"
ATTRIBUTES: str = "attributes"
OBJECT_INDEX: str = "object_index"
OBJECT_ATTRIBUTES: str = "object_attributes"
GROUPINGS: str = "groupings"
GROUPINGS_ATTRIBUTES: str = "groupings_attributes"
CROSS_CHUNK_LINKS: str = "cross_chunk_links"
CROSS_CHUNK_FACES: str = "cross_chunk_faces"
LINK_ATTRIBUTES: str = "link_attributes"
CROSS_CHUNK_LINK_ATTRIBUTES: str = "cross_chunk_link_attributes"
METANODE_CHILDREN: str = "metanode_children"
VERTEX_COUNTS: str = "vertex_counts"

# Parametric sub-arrays
PARAMETRIC_OBJECTS: str = "objects"
PARAMETRIC_OBJECT_ATTRIBUTES: str = "object_attributes"
PARAMETRIC_GROUPINGS: str = "groupings"
PARAMETRIC_GROUPINGS_ATTRIBUTES: str = "groupings_attributes"

ALL_ARRAY_NAMES: frozenset[str] = frozenset({
    VERTICES,
    VERTEX_GROUP_OFFSETS,
    LINKS,
    ATTRIBUTES,
    OBJECT_INDEX,
    OBJECT_ATTRIBUTES,
    GROUPINGS,
    GROUPINGS_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    CROSS_CHUNK_FACES,
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
    METANODE_CHILDREN,
    VERTEX_COUNTS,
})

# Array names whose on-disk layout includes a ``<level_delta>`` segment
# between the array prefix and the chunk-key / data subpath (multiscale
# links, 0.4+).  Use ``zarr_vectors.core.paths`` to compose paths.
MULTISCALE_LINK_ARRAY_NAMES: frozenset[str] = frozenset({
    LINKS,
    CROSS_CHUNK_LINKS,
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
})

# ---------------------------------------------------------------------------
# Convention values
# ---------------------------------------------------------------------------

# links_convention
LINKS_EXPLICIT: str = "explicit"
LINKS_IMPLICIT_SEQUENTIAL: str = "implicit_sequential"
LINKS_IMPLICIT_BRANCHES: str = "implicit_sequential_with_branches"

VALID_LINKS_CONVENTIONS: frozenset[str] = frozenset({
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_SEQUENTIAL,
    LINKS_IMPLICIT_BRANCHES,
})

# object_index_convention
OBJIDX_STANDARD: str = "standard"
OBJIDX_IDENTITY: str = "identity"

VALID_OBJIDX_CONVENTIONS: frozenset[str] = frozenset({
    OBJIDX_STANDARD,
    OBJIDX_IDENTITY,
})

# cross_chunk_strategy
CROSS_CHUNK_DEDUP: str = "boundary_deduplication"
CROSS_CHUNK_EXPLICIT: str = "explicit_links"
CROSS_CHUNK_BOTH: str = "both"

VALID_CROSS_CHUNK_STRATEGIES: frozenset[str] = frozenset({
    CROSS_CHUNK_DEDUP,
    CROSS_CHUNK_EXPLICIT,
    CROSS_CHUNK_BOTH,
})

# cross_level_storage (0.4 multiscale links).  Controls whether
# ``cross_level_depth`` materializes both directions or only positive
# deltas.  See ``schema/zarr_vectors.linkml.yaml`` ``CrossLevelStorage``.
XLEVEL_NONE: str = "none"
XLEVEL_IMPLICIT: str = "implicit"
XLEVEL_EXPLICIT: str = "explicit"

VALID_XLEVEL_STORAGE: frozenset[str] = frozenset({
    XLEVEL_NONE,
    XLEVEL_IMPLICIT,
    XLEVEL_EXPLICIT,
})

# Defaults applied by ``build_pyramid`` when the caller does not override.
DEFAULT_CROSS_LEVEL_DEPTH: int = 1
"""Default ``cross_level_depth``: emit ``±1`` cross-level link arrays at
every adjacent level pair.  ``0`` disables, ``-1`` means all available
pyramid levels."""

DEFAULT_CROSS_LEVEL_STORAGE: str = XLEVEL_EXPLICIT
"""Default ``cross_level_storage``: materialize both ``+N`` (at the
finer level) and ``-N`` (at the coarser level)."""

# ---------------------------------------------------------------------------
# Geometry types
# ---------------------------------------------------------------------------

GEOM_POINT_CLOUD: str = "point_cloud"
GEOM_LINE: str = "line"
GEOM_POLYLINE: str = "polyline"
GEOM_STREAMLINE: str = "streamline"
GEOM_SKELETON: str = "skeleton"
GEOM_GRAPH: str = "graph"
GEOM_MESH: str = "mesh"

VALID_GEOMETRY_TYPES: frozenset[str] = frozenset({
    GEOM_POINT_CLOUD,
    GEOM_LINE,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    GEOM_SKELETON,
    GEOM_GRAPH,
    GEOM_MESH,
})

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

ENCODING_RAW: str = "raw"
ENCODING_DRACO: str = "draco"

VALID_ENCODINGS: frozenset[str] = frozenset({
    ENCODING_RAW,
    ENCODING_DRACO,
})

# ---------------------------------------------------------------------------
# Default compression
# ---------------------------------------------------------------------------

DEFAULT_COMPRESSOR: str = "blosc"
DEFAULT_COMPRESSOR_OPTS: dict[str, object] = {
    "cname": "zstd",
    "clevel": 5,
    "shuffle": 1,  # SHUFFLE_BYTE
}

# ---------------------------------------------------------------------------
# Multi-resolution defaults
# ---------------------------------------------------------------------------

DEFAULT_REDUCTION_FACTOR: int = 8
"""Default: only emit a new resolution level when vertex count drops by 8×."""

DEFAULT_BIN_RATIO: tuple[int, ...] = (1, 1, 1)
"""Bin ratio at level 0 (no downsampling)."""

DEFAULT_COARSENING_METHOD: str = "grid_metanode"

# Valid values for LevelMetadata.coarsening_method.  Open-set: future
# strategies (e.g. mesh edge-collapse decimation) may add tokens here.
COARSEN_PER_OBJECT: str = "per_object"
"""Per-object pyramid: each surviving object's vertices are aggregated
into bin centroids (metavertices).  Metavertices may be shared between
objects; OIDs are preserved across levels."""

COARSEN_CROSS_OBJECT_METANODE: str = "cross_object_metanode"
"""Legacy aggregation that merges vertices across objects, producing a
fresh OID space at each level.  No provenance back to the source
objects."""

COARSEN_GRID_METANODE: str = "grid_metanode"
"""Alias for the legacy cross-object metanode aggregation; kept for
historical level metadata read-back."""

COARSEN_MANUAL: str = "manual"
COARSEN_NONE: str = "none"

VALID_COARSENING_METHODS: frozenset[str] = frozenset({
    COARSEN_PER_OBJECT,
    COARSEN_CROSS_OBJECT_METANODE,
    COARSEN_GRID_METANODE,
    COARSEN_MANUAL,
    COARSEN_NONE,
})

# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

AGG_MEAN: str = "mean"
AGG_SUM: str = "sum"
AGG_MODE: str = "mode"
AGG_COUNT: str = "count"
AGG_MIN: str = "min"
AGG_MAX: str = "max"

VALID_AGGREGATIONS: frozenset[str] = frozenset({
    AGG_MEAN, AGG_SUM, AGG_MODE, AGG_COUNT, AGG_MIN, AGG_MAX,
})
