"""Multi-resolution pyramid construction orchestrator.

Supports two modes:

1. **Automatic**: ``build_pyramid(store)`` auto-plans levels using
   target volume reduction and sparsity weight.
2. **Manual**: ``coarsen_level(store, source, target, bin_ratio, sparsity)``
   creates a single coarsened level with explicit control.

Both modes handle vertex coarsening (via metanodes) and object
sparsity (via object selection strategies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_VERTEX_GROUPS,
    COARSEN_CROSS_OBJECT_METANODE,
    COARSEN_GRID_METANODE,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    OBJECT_ATTRIBUTES,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_IMPLICIT,
    XLEVEL_NONE,
    VALID_XLEVEL_STORAGE,
)
from zarr_vectors.core.arrays import (
    create_cross_chunk_links_array,
    create_links_array,
    create_metanode_children_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_object_attributes,
    read_vertex_group,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_metanode_children,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    compute_bin_shape,
    validate_bin_shape_divides_chunk,
)
from zarr_vectors.core.store import (
    add_resolution_level,
    create_resolution_level,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.multiresolution.layers import (
    LevelReductionSpec,
    compute_level_specs,
    plan_pyramid_with_sparsity,
)
from zarr_vectors.multiresolution.metanodes import generate_metanodes
from zarr_vectors.multiresolution.object_selection import (
    apply_sparsity,
    compute_polyline_lengths,
    compute_representative_points,
)
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_cross_level_edges,
)
from zarr_vectors.spatial.chunking import assign_bins, assign_chunks
from zarr_vectors.typing import ChunkCoords, CrossChunkLink


# ===================================================================
# Single-level coarsening
# ===================================================================

def coarsen_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    bin_ratio: tuple[int, ...] | None = None,
    *,
    coarsen_factor: float | None = None,
    sparsity_factor: float | None = None,
    method: str = COARSEN_PER_OBJECT,
    object_sparsity: float = 1.0,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Coarsen a single level and write it to the store.

    Two interfaces are supported (use one):

    * **Factor-based** (preferred): pass ``coarsen_factor`` and/or
      ``sparsity_factor``.  Defaults to ``method="per_object"`` —
      per-object vertex aggregation with stable OIDs across levels.
      A metavertex's source vertices may come from multiple source
      objects; the resulting metavertex appears in each of those
      objects' manifests at the coarser level.
    * **Legacy `bin_ratio`**: passing ``bin_ratio`` (with optional
      ``object_sparsity``) routes through the original cross-object
      ``grid_metanode`` path which produces a fresh OID space at the
      coarser level.  Kept for back-compat.

    Args:
        store_path: Path to the zarr vectors store.
        source_level: Level to read from.
        target_level: Level to write to (must not exist).
        bin_ratio: Legacy interface — per-axis fold change.  Implies
            ``method="cross_object_metanode"`` unless ``method`` is
            given explicitly.
        coarsen_factor: Per-object vertex aggregation factor (≥ 1).
            ``1.0`` is the identity (no aggregation).
        sparsity_factor: Object-dropping factor (≥ 1).  Survivors keep
            their OIDs; dropped objects leave empty manifest slots.
            ``1.0`` is the identity (no drop).
        method: ``"per_object"`` (default for the factor interface) or
            ``"cross_object_metanode"`` / ``"grid_metanode"`` for the
            legacy aggregation.
        object_sparsity: Legacy keep-fraction.  Mapped to
            ``sparsity_factor = 1.0 / object_sparsity`` when present.
        sparsity_strategy: Object selection strategy.
        sparsity_seed: Random seed.
        agg_mode: Metanode attribute aggregation.

    Returns:
        Summary dict.  Always includes ``method``,
        ``preserves_object_ids``, ``vertex_count``.
    """
    # Reconcile the two interfaces.  ``bin_ratio`` is the legacy entry
    # and implies the cross-object metanode path unless the caller
    # explicitly opted into per-object via ``method``.
    legacy_used = bin_ratio is not None
    factor_used = coarsen_factor is not None or sparsity_factor is not None
    if legacy_used and not factor_used and method == COARSEN_PER_OBJECT:
        method = COARSEN_CROSS_OBJECT_METANODE
    if coarsen_factor is None:
        coarsen_factor = 1.0
    if sparsity_factor is None:
        # Map legacy object_sparsity → sparsity_factor.
        sparsity_factor = (
            1.0 / object_sparsity if 0.0 < object_sparsity < 1.0 else 1.0
        )

    if method in (COARSEN_CROSS_OBJECT_METANODE, COARSEN_GRID_METANODE):
        return _cross_object_metanode_coarsen(
            store_path=store_path,
            source_level=source_level,
            target_level=target_level,
            bin_ratio=bin_ratio,
            coarsen_factor=coarsen_factor,
            object_sparsity=(1.0 / sparsity_factor),
            sparsity_strategy=sparsity_strategy,
            sparsity_seed=sparsity_seed,
            agg_mode=agg_mode,
        )
    if method == COARSEN_PER_OBJECT:
        return _per_object_coarsen(
            store_path=store_path,
            source_level=source_level,
            target_level=target_level,
            coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            sparsity_strategy=sparsity_strategy,
            sparsity_seed=sparsity_seed,
        )
    raise ValueError(f"Unknown coarsen method: {method!r}")


def _cross_object_metanode_coarsen(
    *,
    store_path: str | Path,
    source_level: int,
    target_level: int,
    bin_ratio: tuple[int, ...] | None,
    coarsen_factor: float,
    object_sparsity: float,
    sparsity_strategy: str,
    sparsity_seed: int | None,
    agg_mode: str,
) -> dict[str, Any]:
    """Legacy cross-object aggregation (produces fresh OIDs)."""
    root = open_store(str(store_path), mode="r+")
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    chunk_shape = meta.chunk_shape
    base_bin = meta.effective_bin_shape

    if bin_ratio is None:
        # Derive isotropic ratio from coarsen_factor.
        bin_ratio = tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim))

    # Compute target bin shape
    bin_shape = compute_bin_shape(base_bin, bin_ratio)
    validate_bin_shape_divides_chunk(chunk_shape, bin_shape)

    # Read source level vertices
    source_group = get_resolution_level(root, source_level)
    positions = _read_all_vertices(source_group, ndim)

    if len(positions) == 0:
        return {
            "vertex_count": 0,
            "object_count": 0,
            "reduction_ratio": 0,
            "method": COARSEN_CROSS_OBJECT_METANODE,
            "preserves_object_ids": False,
        }

    n_source = len(positions)

    # Generate metanodes
    meta_result = generate_metanodes(positions, bin_shape, agg_mode=agg_mode)
    meta_positions = meta_result["metanode_positions"]
    children = meta_result["children"]
    n_metanodes = len(meta_positions)

    # Apply object sparsity (on metanodes)
    n_objects = n_metanodes
    if object_sparsity < 1.0 and n_metanodes > 1:
        kept = apply_sparsity(
            n_metanodes, object_sparsity, sparsity_strategy,
            seed=sparsity_seed,
            representative_points=meta_positions,
            bin_shape=bin_shape,
        )
        meta_positions = meta_positions[kept]
        children = [children[i] for i in kept]
        n_objects = len(meta_positions)

    if n_objects == 0:
        return {
            "vertex_count": 0,
            "object_count": 0,
            "reduction_ratio": 0,
            "method": COARSEN_CROSS_OBJECT_METANODE,
            "preserves_object_ids": False,
        }

    # Create the level
    level_group = add_resolution_level(
        root, target_level, bin_ratio,
        object_sparsity=object_sparsity,
        coarsening_method=COARSEN_GRID_METANODE,
        parent_level=source_level,
    )

    # Update vertex count in metadata
    level_group.attrs.update({
        "zarr_vectors_level": {
            **level_group.attrs.to_dict().get("zarr_vectors_level", {}),
            "vertex_count": n_objects,
        }
    })

    create_vertices_array(level_group, dtype="float32")

    # Assign to chunks and write
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)
    for chunk_coords, global_indices in sorted(chunk_assignments.items()):
        write_chunk_vertices(
            level_group, chunk_coords, [meta_positions[global_indices]],
            dtype=np.float32,
        )

    # Write metanode_children
    try:
        create_metanode_children_array(level_group)
        write_metanode_children(level_group, children)
    except Exception:
        pass

    return {
        "vertex_count": n_objects,
        "source_count": n_source,
        "reduction_ratio": n_source / max(n_objects, 1),
        "object_sparsity": object_sparsity,
        "method": COARSEN_CROSS_OBJECT_METANODE,
        "preserves_object_ids": False,
    }


def _per_object_coarsen(
    *,
    store_path: str | Path,
    source_level: int,
    target_level: int,
    coarsen_factor: float,
    sparsity_factor: float,
    sparsity_strategy: str,
    sparsity_seed: int | None,
) -> dict[str, Any]:
    """Per-object pyramid: aggregate within-bin source vertices into
    shared metavertices, preserving each surviving object's OID and
    its trajectory through the new metavertices.

    See the 12-step implementation sketch in the plan file
    ``Provenance-preserving pyramid: shared metavertices + ID-stable
    objects`` (`schema/zarr_vectors.linkml.yaml` schema captures the
    persistent metadata side).
    """
    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = root_meta.sid_ndim
    chunk_shape = root_meta.chunk_shape
    base_bin = root_meta.effective_bin_shape

    src_group = get_resolution_level(root, source_level)

    # --- Step 0: read source manifests + vertex positions ----------------
    src_manifests = read_all_object_manifests(src_group)
    n_src_objects = len(src_manifests)
    if n_src_objects == 0:
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": 0,
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
        }

    # Read source vertex positions, indexed by (chunk_coords, vg_idx).
    src_vg_positions: dict[tuple[ChunkCoords, int], npt.NDArray] = {}
    for cc in list_chunk_keys(src_group, VERTICES):
        try:
            vgs = read_chunk_vertices(src_group, cc, dtype=np.float32, ndim=ndim)
        except Exception:
            continue
        for vg_idx, vg in enumerate(vgs):
            src_vg_positions[(cc, vg_idx)] = vg

    # --- Step 1: drop a fraction of source objects ----------------------
    keep_oids: list[int]
    if sparsity_factor > 1.0 and n_src_objects > 1:
        keep_frac = 1.0 / sparsity_factor
        kept = apply_sparsity(
            n_src_objects, keep_frac, sparsity_strategy,
            seed=sparsity_seed,
            representative_points=None,
            bin_shape=base_bin,
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        keep_oids = list(range(n_src_objects))
    keep_set = set(keep_oids)

    # --- Step 2-3: build (source vertex → bin → metavertex) map ---------
    # Per-object ordered source-vertex positions (with their global index
    # in the flat source-vertex array).
    per_object_positions: dict[int, np.ndarray] = {}
    flat_positions: list[np.ndarray] = []
    flat_oid_of_v: list[int] = []
    next_global = 0
    for oid in keep_oids:
        manifest = src_manifests[oid]
        parts: list[np.ndarray] = []
        for cc, vg_idx in manifest:
            vg = src_vg_positions.get((cc, vg_idx))
            if vg is None or len(vg) == 0:
                continue
            parts.append(np.asarray(vg, dtype=np.float32))
        if not parts:
            per_object_positions[oid] = np.zeros((0, ndim), dtype=np.float32)
            continue
        obj_positions = np.concatenate(parts, axis=0)
        per_object_positions[oid] = obj_positions
        flat_positions.append(obj_positions)
        flat_oid_of_v.extend([oid] * obj_positions.shape[0])
        next_global += obj_positions.shape[0]

    if not flat_positions:
        # Surviving objects had no vertices.  Write an empty level.
        _write_empty_preserve_level(
            root, source_level, target_level,
            base_bin=base_bin,
            coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            inherited_num_objects=n_src_objects,
        )
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": len(keep_oids),
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
            "shared_vertex_groups": True,
        }

    all_pos = np.concatenate(flat_positions, axis=0)

    # Target bin shape: source bin_shape × coarsen_factor.
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)

    # Compute per-vertex bin coords: (N, ndim) int64.
    bin_shape_arr = np.asarray(target_bin_shape, dtype=np.float64)
    bin_coords = np.floor(all_pos / bin_shape_arr).astype(np.int64)
    # Combine each bin coord tuple into a single sort-key for np.unique.
    bin_keys = np.ascontiguousarray(bin_coords).view(
        np.dtype((np.void, bin_coords.dtype.itemsize * bin_coords.shape[1]))
    ).ravel()
    _, inverse = np.unique(bin_keys, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_metavertices = int(inverse.max()) + 1 if inverse.size > 0 else 0

    # --- Step 3 (continued): centroid per bin --------------------------
    meta_positions = np.zeros((n_metavertices, ndim), dtype=np.float32)
    bin_counts = np.zeros(n_metavertices, dtype=np.int64)
    np.add.at(meta_positions, inverse, all_pos)
    np.add.at(bin_counts, inverse, 1)
    meta_positions /= bin_counts[:, None]

    # --- Step 4: chunk-assign metavertices ------------------------------
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)

    # --- Step 5: per-chunk vg layout (one vg per metavertex) ------------
    metavertex_to_ref: dict[int, tuple[ChunkCoords, int]] = {}
    per_chunk_groups: dict[ChunkCoords, list[np.ndarray]] = {}
    for cc, indices in sorted(chunk_assignments.items()):
        # ``indices`` are metavertex indices that fell in this chunk.
        for vg_idx, mv_idx in enumerate(indices.tolist()):
            metavertex_to_ref[int(mv_idx)] = (cc, vg_idx)
            per_chunk_groups.setdefault(cc, []).append(
                meta_positions[mv_idx:mv_idx + 1]
            )

    # --- Step 6: write per-chunk vertex groups --------------------------
    level_meta_initial = LevelMetadata(
        level=target_level,
        vertex_count=int(n_metavertices),
        arrays_present=[VERTICES, "object_index"],
        bin_shape=target_bin_shape,
        bin_ratio=tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim)),
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=n_src_objects,
        shared_vertex_groups=True,
    )
    level_group = create_resolution_level(root, target_level, level_meta_initial)
    create_vertices_array(level_group, dtype="float32")
    create_object_index_array(level_group)

    for cc, groups in sorted(per_chunk_groups.items()):
        write_chunk_vertices(level_group, cc, groups, dtype=np.float32)

    # --- Step 7: emit per-object manifests ------------------------------
    # We need to map each source vertex back to its metavertex_index.
    # Walk per-object slices of the flat ``inverse`` array.
    cursor = 0
    new_manifests: dict[int, list[tuple[ChunkCoords, int]]] = {}
    for oid in keep_oids:
        n = per_object_positions[oid].shape[0]
        if n == 0:
            cursor += 0
            new_manifests[oid] = []
            continue
        mv_seq = inverse[cursor:cursor + n].tolist()
        cursor += n
        # Deduplicate consecutive duplicates while preserving order.
        manifest: list[tuple[ChunkCoords, int]] = []
        prev = -1
        for mv_idx in mv_seq:
            if mv_idx == prev:
                continue
            prev = mv_idx
            manifest.append(metavertex_to_ref[int(mv_idx)])
        new_manifests[oid] = manifest

    # --- Step 9: emit object_index (gap-fill for dropped OIDs) ----------
    write_object_index(
        level_group, new_manifests, sid_ndim=ndim,
        total_objects=n_src_objects,
    )

    # --- Step 10: per-object attributes with present_mask ---------------
    src_obj_attr_group_name = f"{OBJECT_ATTRIBUTES}"
    try:
        src_obj_attr_group = src_group[src_obj_attr_group_name]
        attr_names = [n for n in src_obj_attr_group]
    except Exception:
        attr_names = []
    for attr_name in attr_names:
        try:
            src_data = read_object_attributes(src_group, attr_name)
        except Exception:
            continue
        # Dense (O, C) or (O,) padded to the inherited OID space, with
        # rows for survivors copied over.  Layout matches the source's
        # OID space (which already equals n_src_objects).
        out_data = np.zeros_like(src_data)
        for oid in keep_oids:
            if oid < len(src_data):
                out_data[oid] = src_data[oid]
        mask = np.zeros(n_src_objects, dtype=np.uint8)
        for oid in keep_oids:
            mask[oid] = 1
        create_object_attributes_array(level_group, attr_name)
        write_object_attributes(level_group, attr_name, out_data, present_mask=mask)

    # --- Step 12: stamp root capability tokens --------------------------
    _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)
    _stamp_root_capability(root, CAP_SHARED_VERTEX_GROUPS)

    return {
        "vertex_count": int(n_metavertices),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "source_objects": n_src_objects,
        "method": COARSEN_PER_OBJECT,
        "preserves_object_ids": True,
        "shared_vertex_groups": True,
    }


def _write_empty_preserve_level(
    root,
    source_level: int,
    target_level: int,
    *,
    base_bin: tuple[float, ...],
    coarsen_factor: float,
    sparsity_factor: float,
    inherited_num_objects: int,
) -> None:
    """Write an empty ID-preserving level when no surviving object has vertices."""
    ndim = len(base_bin)
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        arrays_present=[VERTICES, "object_index"],
        bin_shape=target_bin_shape,
        bin_ratio=tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim)),
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=inherited_num_objects,
        shared_vertex_groups=True,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    create_vertices_array(level_group, dtype="float32")
    create_object_index_array(level_group)
    # Empty object_index with the inherited size — all manifests are [].
    write_object_index(
        level_group, {}, sid_ndim=ndim,
        total_objects=inherited_num_objects,
    )
    _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)


def _stamp_root_capability(root_group, cap: str) -> None:
    """Add ``cap`` to root metadata's ``format_capabilities`` (idempotent)."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    caps = list(zv.get("format_capabilities", []))
    if cap not in caps:
        caps.append(cap)
        zv["format_capabilities"] = caps
        root_group.attrs.update({"zarr_vectors": zv})


def _stamp_root_cross_level(
    root_group, *, depth: int, storage: str,
) -> None:
    """Persist cross_level_depth/cross_level_storage on root metadata."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    zv["cross_level_depth"] = int(depth)
    zv["cross_level_storage"] = storage
    root_group.attrs.update({"zarr_vectors": zv})


def _reconstruct_chunk_assignments(
    level_group, ndim: int,
) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]:
    """Rebuild ``{chunk_coords: vertex_indices}`` from on-disk vertex chunks.

    The "vertex index" assigned to each vertex is the position it would
    occupy in a flat enumeration that walks chunks in
    ``list_chunk_keys`` order and concatenates each chunk's vertex
    groups in order.  This matches the convention used by
    ``build_vertex_chunk_mapping`` for in-memory edge partitioning.

    Returns the assignments dict and the total vertex count.
    """
    chunk_keys = list_chunk_keys(level_group, VERTICES)
    assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    cursor = 0
    for cc in chunk_keys:
        try:
            vgs = read_chunk_vertices(level_group, cc, dtype=np.float32, ndim=ndim)
        except Exception:
            continue
        n = sum(int(vg.shape[0]) for vg in vgs)
        if n == 0:
            continue
        assignments[cc] = np.arange(cursor, cursor + n, dtype=np.int64)
        cursor += n
    return assignments, cursor


def _reconstruct_parent_from_metanode_children(
    coarse_level_group, n_fine: int,
) -> npt.NDArray[np.int64] | None:
    """Build a fine→coarse ``parent`` array from the ``metanode_children`` sidecar.

    For each metanode ``m`` at the coarse level, the sidecar records
    the list of source-level ``(chunk_coords, vertex_index)`` refs
    that became part of ``m``.  We invert that to a per-fine-vertex
    parent array.

    Returns ``None`` when the sidecar is missing.  Fine vertices not
    referenced by any metanode (e.g. dropped by sparsification) are
    marked with ``-1``.
    """
    from zarr_vectors.core.arrays import read_metanode_children
    try:
        children = read_metanode_children(coarse_level_group)
    except Exception:
        return None
    parent = np.full(n_fine, -1, dtype=np.int64)
    if isinstance(children, dict):
        items = children.items()
    else:
        items = enumerate(children)
    # The sidecar's "vertex index" is the per-chunk vg_idx (see
    # write_metanode_children); for pyramids written by the legacy
    # cross-object path each metanode's children are *flat* source
    # vertex indices, not (chunk, vg_idx) tuples.  Try both.
    for m_id, refs in items:
        for ref in refs:
            if isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[0], tuple):
                # (chunk_coords, local_idx) form — reader returns this
                # shape for object-index-style sidecars.  We don't have
                # the source chunk_assignments here to resolve it back
                # to a global index, so this branch is skipped: callers
                # that need cross-level edges in per-object mode must
                # call the in-line helper instead of post-hoc finalize.
                continue
            fi = int(ref)
            if 0 <= fi < n_fine:
                parent[fi] = int(m_id)
    return parent if (parent != -1).any() else None


def _finalize_cross_level_for_store(
    store_path: str | Path,
    *,
    cross_level_depth: int,
    cross_level_storage: str,
) -> None:
    """Persist root cross-level metadata and emit cross-level link arrays.

    Driven post-hoc from on-disk state: enumerates every adjacent
    (fine, coarse) level pair, reconstructs the fine→parent map from
    the coarse level's ``metanode_children`` sidecar, and writes
    ``±delta`` link arrays up to ``cross_level_depth``.

    ``cross_level_depth=-1`` means "walk all available level pairs".
    """
    root = open_store(str(store_path), mode="r+")
    _stamp_root_cross_level(
        root, depth=cross_level_depth, storage=cross_level_storage,
    )
    if cross_level_storage == XLEVEL_NONE or cross_level_depth == 0:
        return

    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    levels = sorted(list_resolution_levels(root))
    if len(levels) < 2:
        return

    _stamp_root_capability(root, CAP_MULTISCALE_LINKS)

    # Build per-level chunk_assignments + total counts once.
    per_level: dict[int, tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]] = {}
    for lvl in levels:
        lg = get_resolution_level(root, lvl)
        per_level[lvl] = _reconstruct_chunk_assignments(lg, ndim)

    max_delta = (
        max(levels) - min(levels)
        if cross_level_depth == -1
        else int(cross_level_depth)
    )

    for fine_idx, fine_level in enumerate(levels[:-1]):
        fine_assn, n_fine = per_level[fine_level]
        if n_fine == 0:
            continue
        # parent_step[k] holds fine→level(fine+k+1) parent at each step.
        parent = None
        prev_n = n_fine
        prev_assn = fine_assn
        for step in range(1, max_delta + 1):
            coarse_level = fine_level + step
            if coarse_level not in per_level:
                break
            coarse_lg = get_resolution_level(root, coarse_level)
            coarse_assn, n_coarse = per_level[coarse_level]
            if n_coarse == 0:
                break

            if step == 1:
                parent = _reconstruct_parent_from_metanode_children(
                    coarse_lg, n_fine=n_fine,
                )
            else:
                # Compose: parent_at_step = parent_at_(step-1)_from_(coarse-1)
                # → grandparent via that coarser level's metanode_children.
                inter_lg = get_resolution_level(root, coarse_level - 1)
                inter_n = per_level[coarse_level - 1][1]
                inter_parent = _reconstruct_parent_from_metanode_children(
                    coarse_lg, n_fine=inter_n,
                )
                if inter_parent is None or parent is None:
                    parent = None
                else:
                    composed = np.full(n_fine, -1, dtype=np.int64)
                    valid = parent >= 0
                    composed[valid] = inter_parent[parent[valid]]
                    parent = composed
            if parent is None:
                # No metanode_children info — skip this and all larger
                # deltas for this fine level.
                break

            _write_cross_level_edges(
                root,
                fine_level=fine_level,
                delta=step,
                fine_chunk_assignments=fine_assn,
                coarse_chunk_assignments=coarse_assn,
                n_fine=n_fine,
                n_coarse=n_coarse,
                parent=parent,
                sid_ndim=ndim,
                storage=cross_level_storage,
            )
            prev_n, prev_assn = n_coarse, coarse_assn


def _write_cross_level_edges(
    root_group,
    *,
    fine_level: int,
    delta: int,
    fine_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
    n_coarse: int,
    parent: npt.NDArray[np.int64],
    sid_ndim: int,
    storage: str,
) -> None:
    """Materialize ``delta``-step cross-level edges between two adjacent levels.

    ``parent[i]`` is the metanode index in the coarser level that fine
    vertex ``i`` belongs to.  The cross-level edges are trivially
    ``(i, parent[i])`` for each fine vertex.

    Writes the ``+delta`` arrays under the fine level.  When
    ``storage='explicit'`` also writes the matching ``-delta`` arrays
    under the coarse level by swapping endpoint roles.
    """
    if storage == XLEVEL_NONE or delta == 0:
        return
    coarse_level = fine_level + delta

    # Build chunk-mapping tables for both levels.
    fine_chunk_list = sorted(fine_chunk_assignments.keys())
    fine_vchunks, fine_vlocal, fine_chunk_list = build_vertex_chunk_mapping(
        fine_chunk_assignments, n_fine, fine_chunk_list,
    )
    coarse_chunk_list = sorted(coarse_chunk_assignments.keys())
    coarse_vchunks, coarse_vlocal, coarse_chunk_list = build_vertex_chunk_mapping(
        coarse_chunk_assignments, n_coarse, coarse_chunk_list,
    )

    # Trivial fine→parent edge list.
    edges = np.stack(
        [np.arange(n_fine, dtype=np.int64), parent.astype(np.int64)],
        axis=1,
    )
    aligned, cross = partition_cross_level_edges(
        edges,
        fine_vchunks, fine_vlocal, fine_chunk_list,
        coarse_vchunks, coarse_vlocal, coarse_chunk_list,
    )

    fine_lg = get_resolution_level(root_group, fine_level)
    if aligned:
        create_links_array(fine_lg, link_width=2, delta=delta)
        for cc, rows in aligned.items():
            write_chunk_links(fine_lg, cc, [rows], delta=delta)
    if cross:
        create_cross_chunk_links_array(fine_lg, delta=delta)
        write_cross_chunk_links(fine_lg, cross, sid_ndim=sid_ndim, delta=delta)

    if storage == XLEVEL_EXPLICIT:
        # Mirror at the coarse level under -delta: swap endpoint roles.
        coarse_lg = get_resolution_level(root_group, coarse_level)
        # Re-partition from the coarse side so chunk-alignment is
        # evaluated against the coarse chunk grid (intra/cross split
        # may differ from the fine-side view when grids don't align).
        rev_edges = np.stack(
            [parent.astype(np.int64), np.arange(n_fine, dtype=np.int64)],
            axis=1,
        )
        rev_aligned, rev_cross = partition_cross_level_edges(
            rev_edges,
            coarse_vchunks, coarse_vlocal, coarse_chunk_list,
            fine_vchunks, fine_vlocal, fine_chunk_list,
        )
        if rev_aligned:
            create_links_array(coarse_lg, link_width=2, delta=-delta)
            for cc, rows in rev_aligned.items():
                write_chunk_links(coarse_lg, cc, [rows], delta=-delta)
        if rev_cross:
            create_cross_chunk_links_array(coarse_lg, delta=-delta)
            write_cross_chunk_links(
                coarse_lg, rev_cross, sid_ndim=sid_ndim, delta=-delta,
            )


# ===================================================================
# Full pyramid builder
# ===================================================================

def build_pyramid(
    store_path: str | Path,
    *,
    factors: list[tuple[float, float]] | None = None,
    method: str = COARSEN_PER_OBJECT,
    level_configs: list[dict] | None = None,
    target_volume_reduction: float = 8.0,
    sparsity_weight: float = 0.0,
    reduction_factor: int = 8,
    max_levels: int = 10,
    min_vertices: int = 8,
    agg_mode: str = "mean",
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_depth: int = DEFAULT_CROSS_LEVEL_DEPTH,
    cross_level_storage: str = DEFAULT_CROSS_LEVEL_STORAGE,
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Preferred interface — pass ``factors=[(coarsen_2, sparsity_3),
    ...]`` where ``factors[i]`` is applied to produce level ``i+1``
    from level ``i``.  Either factor at ``1.0`` opts out of that axis.
    Method defaults to ``"per_object"`` (per-object pyramid with stable
    OIDs; metavertices may be shared between objects).  Pass
    ``method="cross_object_metanode"`` (or the legacy
    ``"grid_metanode"``) to fall back to the original aggregation that
    produces a fresh OID space per level.

    Legacy interface (kept for back-compat):

    1. **Explicit**: provide ``level_configs`` — a list of dicts, each
       with ``"bin_ratio"`` and optionally ``"object_sparsity"``.
    2. **Auto**: auto-plan using ``target_volume_reduction`` and
       ``sparsity_weight``.

    When ``factors`` is None and ``level_configs`` is None and
    ``sparsity_weight`` is 0.0 (default), behaviour matches the
    original pyramid builder (backward compatible).

    Args:
        store_path: Path to the store with level 0.
        level_configs: Explicit per-level configs.
        target_volume_reduction: Per-level target for auto mode.
        sparsity_weight: 0.0=all binning, 1.0=all sparsity.
        reduction_factor: Legacy threshold for old auto mode.
        max_levels: Maximum levels.
        min_vertices: Stop below this.
        agg_mode: Metanode aggregation.
        sparsity_strategy: Object selection strategy.
        sparsity_seed: Random seed.
        cross_level_depth: Maximum absolute level delta for materialized
            cross-pyramid-level link arrays (0.4 multiscale links).
            ``0`` = none, ``N`` = up to ``±N`` per pair (or ``+N`` only
            when ``cross_level_storage='implicit'``), ``-1`` = walk all
            available level pairs.  Default ``1``.
        cross_level_storage: ``"none"`` / ``"implicit"`` / ``"explicit"``.
            ``"explicit"`` materializes both ``+N`` (at the finer level)
            and ``-N`` (at the coarser level); ``"implicit"`` writes
            only ``+N``.  Default ``"explicit"``.

    Returns:
        Summary dict.
    """
    if cross_level_storage not in VALID_XLEVEL_STORAGE:
        raise ValueError(
            f"cross_level_storage={cross_level_storage!r} not in "
            f"{sorted(VALID_XLEVEL_STORAGE)}"
        )
    if cross_level_depth < -1:
        raise ValueError(
            f"cross_level_depth must be ≥ -1 (got {cross_level_depth})"
        )

    # Factor-based interface (preferred).  Each entry produces one
    # coarser level; routed through coarsen_level which knows both
    # methods.  Returns early; the legacy paths below remain available
    # for callers that pass ``level_configs`` or ``sparsity_weight``.
    if factors is not None:
        summaries: list[dict[str, Any]] = []
        for i, fac in enumerate(factors):
            if isinstance(fac, (tuple, list)) and len(fac) == 2:
                cf, sf = float(fac[0]), float(fac[1])
            else:
                raise ValueError(
                    f"factors[{i}] must be a (coarsen_factor, sparsity_factor) "
                    f"tuple; got {fac!r}"
                )
            summaries.append(coarsen_level(
                store_path,
                source_level=i,
                target_level=i + 1,
                coarsen_factor=cf,
                sparsity_factor=sf,
                method=method,
                sparsity_strategy=sparsity_strategy,
                sparsity_seed=sparsity_seed,
                agg_mode=agg_mode,
            ))
        # Persist + emit cross-level edges (writes use the on-disk
        # vertex chunk assignments — no re-derivation needed here).
        _finalize_cross_level_for_store(
            store_path,
            cross_level_depth=cross_level_depth,
            cross_level_storage=cross_level_storage,
        )
        return {
            "levels_created": len(summaries),
            "level_specs": summaries,
            "method": method,
            "cross_level_depth": cross_level_depth,
            "cross_level_storage": cross_level_storage,
        }

    root = open_store(str(store_path), mode="r+")
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    chunk_shape = meta.chunk_shape
    base_bin = meta.effective_bin_shape

    # Read all level-0 vertices
    level0 = get_resolution_level(root, 0)
    positions = _read_all_vertices(level0, ndim)

    if len(positions) == 0:
        return {"levels_created": 0, "level_specs": []}

    n_full = len(positions)

    # Count objects at level 0 (approximate: try reading object_index)
    try:
        manifests = read_all_object_manifests(level0)
        n_objects = len(manifests)
    except Exception:
        n_objects = 0

    # Plan levels
    if level_configs is not None:
        # Explicit configs → use plan_pyramid_with_sparsity
        specs = plan_pyramid_with_sparsity(
            n_full, max(n_objects, 1), base_bin, chunk_shape,
            level_configs=level_configs,
        )
    elif sparsity_weight > 0.0:
        # Auto with sparsity
        specs = plan_pyramid_with_sparsity(
            n_full, max(n_objects, 1), base_bin, chunk_shape,
            target_volume_reduction=target_volume_reduction,
            sparsity_weight=sparsity_weight,
            max_levels=max_levels,
            min_vertices=min_vertices,
        )
    else:
        # Legacy auto mode (backward compatible)
        specs = _legacy_plan(
            n_full, ndim, base_bin, chunk_shape,
            reduction_factor, max_levels, min_vertices,
        )

    if not specs:
        return {"levels_created": 0, "level_specs": []}

    # Build each level
    current_positions = positions
    levels_created = 0

    for spec in specs:
        if isinstance(spec, LevelReductionSpec):
            bin_ratio = spec.bin_ratio
            bin_shape = spec.bin_shape or compute_bin_shape(base_bin, bin_ratio)
            object_sparsity = spec.object_sparsity
        else:
            # Legacy LevelSpec
            bin_shape = tuple(spec.bin_size for _ in range(ndim))
            bin_ratio = None
            object_sparsity = 1.0

        # Generate metanodes
        result = generate_metanodes(
            current_positions, bin_shape, agg_mode=agg_mode,
        )
        meta_positions = result["metanode_positions"]
        children = result["children"]
        n_metanodes = len(meta_positions)

        if n_metanodes == 0:
            break

        # Check reduction (skip if too small, except on first level)
        actual_ratio = len(current_positions) / max(n_metanodes, 1)
        if actual_ratio < 2 and levels_created > 0:
            continue

        # Apply object sparsity
        if object_sparsity < 1.0 and n_metanodes > 1:
            kept = apply_sparsity(
                n_metanodes, object_sparsity, sparsity_strategy,
                seed=sparsity_seed,
                representative_points=meta_positions,
                bin_shape=bin_shape,
            )
            meta_positions = meta_positions[kept]
            children = [children[i] for i in kept]
            n_metanodes = len(meta_positions)

        if n_metanodes == 0:
            break

        # Create level
        actual_level = levels_created + 1
        level_meta = LevelMetadata(
            level=actual_level,
            vertex_count=n_metanodes,
            arrays_present=[VERTICES],
            bin_shape=bin_shape,
            bin_ratio=bin_ratio,
            object_sparsity=object_sparsity,
            coarsening_method="grid_metanode",
            parent_level=actual_level - 1,
        )
        level_group = create_resolution_level(root, actual_level, level_meta)
        create_vertices_array(level_group, dtype="float32")

        # Write
        chunk_assignments = assign_chunks(meta_positions, chunk_shape)
        for chunk_coords, global_indices in sorted(chunk_assignments.items()):
            write_chunk_vertices(
                level_group, chunk_coords,
                [meta_positions[global_indices]],
                dtype=np.float32,
            )

        try:
            create_metanode_children_array(level_group)
            write_metanode_children(level_group, children)
        except Exception:
            pass

        levels_created += 1
        current_positions = meta_positions

    spec_summaries = []
    for i, spec in enumerate(specs[:levels_created]):
        if isinstance(spec, LevelReductionSpec):
            spec_summaries.append({
                "level": i + 1,
                "bin_ratio": list(spec.bin_ratio),
                "object_sparsity": spec.object_sparsity,
                "expected_volume_reduction": spec.expected_volume_reduction,
            })
        else:
            spec_summaries.append({
                "level": i + 1,
                "bin_size": spec.bin_size,
                "expected_vertices": spec.expected_vertex_count,
            })

    # Persist root cross-level metadata and emit cross-level link
    # arrays for every adjacent pair we just built.
    _finalize_cross_level_for_store(
        store_path,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
    )

    return {
        "levels_created": levels_created,
        "level_specs": spec_summaries,
        "cross_level_depth": cross_level_depth,
        "cross_level_storage": cross_level_storage,
    }


# ===================================================================
# Helpers
# ===================================================================

def _read_all_vertices(
    level_group: Any, ndim: int,
) -> npt.NDArray[np.float32]:
    """Read all vertices from a level, concatenated."""
    chunk_keys = list_chunk_keys(level_group)
    parts: list[npt.NDArray] = []
    for ck in chunk_keys:
        try:
            groups = read_chunk_vertices(level_group, ck, dtype=np.float32, ndim=ndim)
            for vg in groups:
                if len(vg) > 0:
                    parts.append(vg)
        except Exception:
            continue
    if not parts:
        return np.zeros((0, ndim), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _legacy_plan(
    n_full: int,
    ndim: int,
    base_bin: tuple[float, ...],
    chunk_shape: tuple[float, ...],
    reduction_factor: int,
    max_levels: int,
    min_vertices: int,
) -> list:
    """Plan using the old LevelSpec-based approach (backward compat)."""
    base_bin_scalar = min(base_bin)
    return compute_level_specs(
        n_full, base_bin_scalar,
        reduction_factor=reduction_factor,
        max_levels=max_levels,
        min_vertices=min_vertices,
    )
