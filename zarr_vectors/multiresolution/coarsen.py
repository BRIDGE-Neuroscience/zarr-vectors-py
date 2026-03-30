"""Multi-resolution pyramid construction orchestrator.

Reads level 0 from an existing store, generates metanode levels using
adaptive bin sizing, and writes each coarser level back into the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import VERTICES
from zarr_vectors.core.arrays import (
    create_metanode_children_array,
    create_vertices_array,
    list_chunk_keys,
    read_chunk_vertices,
    write_chunk_vertices,
    write_metanode_children,
    write_object_index,
    create_object_index_array,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.multiresolution.layers import compute_level_specs
from zarr_vectors.multiresolution.metanodes import generate_metanodes
from zarr_vectors.spatial.chunking import assign_chunks


def build_pyramid(
    store_path: str | Path,
    *,
    reduction_factor: int = 8,
    max_levels: int = 10,
    min_vertices: int = 8,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Reads all level-0 vertices, computes adaptive coarsening levels,
    and writes each level into the store as ``resolution_1``,
    ``resolution_2``, etc.

    Args:
        store_path: Path to an existing zarr vectors store with level 0.
        reduction_factor: Minimum fold-reduction to emit a level (default 8).
        max_levels: Maximum pyramid depth.
        min_vertices: Stop below this vertex count.
        agg_mode: Attribute aggregation mode for metanodes.

    Returns:
        Summary dict with ``levels_created``, ``level_specs``.
    """
    root = open_store(str(store_path), mode="r+")
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    chunk_shape = meta.chunk_shape

    # Read all level-0 vertices
    level0 = get_resolution_level(root, 0)
    chunk_keys = list_chunk_keys(level0)

    all_positions: list[npt.NDArray] = []
    for ck in chunk_keys:
        try:
            groups = read_chunk_vertices(level0, ck, dtype=np.float32, ndim=ndim)
            for vg in groups:
                all_positions.append(vg)
        except Exception:
            continue

    if not all_positions:
        return {"levels_created": 0, "level_specs": []}

    positions = np.concatenate(all_positions, axis=0)
    n_full = len(positions)

    # Compute which levels to generate
    base_bin = min(chunk_shape)
    specs = compute_level_specs(
        n_full, base_bin,
        reduction_factor=reduction_factor,
        max_levels=max_levels,
        min_vertices=min_vertices,
    )

    if not specs:
        return {"levels_created": 0, "level_specs": []}

    # Build each level
    current_positions = positions
    levels_created = 0

    for spec in specs:
        # Generate metanodes
        result = generate_metanodes(
            current_positions, spec.bin_size,
            agg_mode=agg_mode,
        )

        meta_positions = result["metanode_positions"]
        meta_counts = result["metanode_counts"]
        children = result["children"]
        n_metanodes = len(meta_positions)

        if n_metanodes == 0:
            break

        # Check actual reduction
        actual_ratio = len(current_positions) / max(n_metanodes, 1)
        if actual_ratio < reduction_factor and levels_created > 0:
            # Skip this level — not enough reduction
            continue

        # Create the resolution level (sequential numbering)
        actual_level = levels_created + 1
        bin_tuple = tuple(spec.bin_size for _ in range(ndim))
        level_meta = LevelMetadata(
            level=actual_level,
            vertex_count=n_metanodes,
            arrays_present=[VERTICES],
            bin_size=bin_tuple,
            coarsening_method="grid_metanode",
            parent_level=actual_level - 1,
        )
        level_group = create_resolution_level(root, actual_level, level_meta)
        create_vertices_array(level_group, dtype="float32")

        # Assign metanodes to chunks and write
        meta_chunk_assignments = assign_chunks(meta_positions, chunk_shape)
        for chunk_coords, global_indices in sorted(meta_chunk_assignments.items()):
            chunk_verts = meta_positions[global_indices]
            write_chunk_vertices(
                level_group, chunk_coords, [chunk_verts],
                dtype=np.float32,
            )

        # Write metanode_children mapping
        try:
            create_metanode_children_array(level_group)
            write_metanode_children(level_group, children)
        except Exception:
            pass  # metanode_children is optional

        levels_created += 1
        current_positions = meta_positions

    return {
        "levels_created": levels_created,
        "level_specs": [
            {
                "level": s.level_index,
                "bin_size": s.bin_size,
                "expected_vertices": s.expected_vertex_count,
            }
            for s in specs[:levels_created]
        ],
    }
