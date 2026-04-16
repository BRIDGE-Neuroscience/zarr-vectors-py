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

from zarr_vectors.constants import VERTICES
from zarr_vectors.core.arrays import (
    create_metanode_children_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    write_chunk_vertices,
    write_metanode_children,
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
from zarr_vectors.spatial.chunking import assign_chunks


# ===================================================================
# Single-level coarsening
# ===================================================================

def coarsen_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    bin_ratio: tuple[int, ...],
    *,
    object_sparsity: float = 1.0,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Coarsen a single level and write it to the store.

    Reads vertex data from ``source_level``, generates metanodes at
    the bin size implied by ``bin_ratio``, optionally applies object
    sparsity, and writes the result as ``target_level``.

    Args:
        store_path: Path to the zarr vectors store.
        source_level: Level to read from.
        target_level: Level to write to (must not exist).
        bin_ratio: Integer fold-change per axis relative to level 0.
        object_sparsity: Fraction of objects to retain (0, 1].
        sparsity_strategy: Selection strategy for object thinning.
        sparsity_seed: Random seed for ``"random"`` strategy.
        agg_mode: Attribute aggregation for metanodes.

    Returns:
        Summary dict with ``vertex_count``, ``object_count``,
        ``reduction_ratio``.
    """
    root = open_store(str(store_path), mode="r+")
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    chunk_shape = meta.chunk_shape
    base_bin = meta.effective_bin_shape

    # Compute target bin shape
    bin_shape = compute_bin_shape(base_bin, bin_ratio)
    validate_bin_shape_divides_chunk(chunk_shape, bin_shape)

    # Read source level vertices
    source_group = get_resolution_level(root, source_level)
    positions = _read_all_vertices(source_group, ndim)

    if len(positions) == 0:
        return {"vertex_count": 0, "object_count": 0, "reduction_ratio": 0}

    n_source = len(positions)

    # Compute scalar bin size for metanode generation (use mean of bin_shape)
    bin_size_scalar = float(np.mean(bin_shape))

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
        return {"vertex_count": 0, "object_count": 0, "reduction_ratio": 0}

    # Create the level
    level_group = add_resolution_level(
        root, target_level, bin_ratio,
        object_sparsity=object_sparsity,
        coarsening_method="grid_metanode",
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
    }


# ===================================================================
# Full pyramid builder
# ===================================================================

def build_pyramid(
    store_path: str | Path,
    *,
    level_configs: list[dict] | None = None,
    target_volume_reduction: float = 8.0,
    sparsity_weight: float = 0.0,
    reduction_factor: int = 8,
    max_levels: int = 10,
    min_vertices: int = 8,
    agg_mode: str = "mean",
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Two modes:

    1. **Explicit**: provide ``level_configs`` — a list of dicts, each
       with ``"bin_ratio"`` and optionally ``"object_sparsity"``.
    2. **Auto**: auto-plan using ``target_volume_reduction`` and
       ``sparsity_weight``.

    When ``level_configs`` is None and ``sparsity_weight`` is 0.0
    (default), behaviour matches the original pyramid builder
    (backward compatible).

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

    Returns:
        Summary dict.
    """
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

    return {
        "levels_created": levels_created,
        "level_specs": spec_summaries,
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
