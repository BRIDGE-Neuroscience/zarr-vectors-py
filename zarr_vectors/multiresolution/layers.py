"""Adaptive resolution level selection.

Determines which coarsening levels to emit based on the adaptive
threshold rule: bin size doubles each candidate level, but a new
resolution level is only stored when total vertex count drops below
``N_previous / reduction_factor``.

Also provides helpers for computing bin ratios from target reductions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class LevelSpec:
    """Specification for one resolution level in the pyramid."""

    level_index: int
    bin_size: float
    expected_vertex_count: int
    reduction_ratio: float


def compute_level_specs(
    full_resolution_count: int,
    base_bin_size: float,
    *,
    reduction_factor: int = 8,
    max_levels: int = 20,
    min_vertices: int = 8,
) -> list[LevelSpec]:
    """Determine which resolution levels to create.

    Starting from the full resolution (level 0), candidate levels use
    bin sizes that double at each step: ``base * 2^k``.  A level is
    only emitted if its estimated vertex count is at most
    ``previous_count / reduction_factor``.

    Args:
        full_resolution_count: Number of vertices at level 0.
        base_bin_size: The chunk_shape component (or minimum chunk size)
            used as the starting bin size for coarsening.
        reduction_factor: Minimum fold-reduction required to emit a level
            (default 8).
        max_levels: Hard cap on number of resolution levels.
        min_vertices: Stop generating levels below this count.

    Returns:
        List of LevelSpec (excluding level 0 which is always present).
        Empty if no coarsening is warranted.
    """
    specs: list[LevelSpec] = []
    prev_count = full_resolution_count
    level_idx = 1

    for k in range(1, max_levels + 10):
        bin_size = base_bin_size * (2 ** k)

        est_count = max(1, int(full_resolution_count / (2 ** k) ** 3))

        if est_count >= prev_count:
            continue

        ratio = prev_count / max(est_count, 1)
        if ratio < reduction_factor:
            continue

        if est_count < min_vertices:
            break

        specs.append(LevelSpec(
            level_index=level_idx,
            bin_size=bin_size,
            expected_vertex_count=est_count,
            reduction_ratio=ratio,
        ))

        prev_count = est_count
        level_idx += 1

        if level_idx > max_levels:
            break

    return specs


def select_bin_sizes(
    chunk_shape: tuple[float, ...],
    full_vertex_count: int,
    *,
    reduction_factor: int = 8,
) -> list[float]:
    """Convenience: return just the bin sizes for each pyramid level.

    Uses the minimum chunk dimension as the base bin size.

    Args:
        chunk_shape: Spatial chunk dimensions.
        full_vertex_count: Vertex count at level 0.
        reduction_factor: Fold-reduction threshold.

    Returns:
        List of bin sizes for levels 1, 2, ... (level 0 is full res).
    """
    base = min(chunk_shape)
    specs = compute_level_specs(
        full_vertex_count, base,
        reduction_factor=reduction_factor,
    )
    return [s.bin_size for s in specs]


# ===================================================================
# Bin ratio helpers
# ===================================================================

def select_bin_ratio_for_reduction(
    target_reduction: float,
    ndim: int = 3,
) -> tuple[int, ...]:
    """Find the integer bin ratio whose volume is closest to a target reduction.

    For 3D with target 8: returns ``(2, 2, 2)`` since ``2³ = 8``.
    For 3D with target 4: returns ``(2, 2, 1)`` since ``2×2×1 = 4``.
    For 3D with target 27: returns ``(3, 3, 3)`` since ``3³ = 27``.

    The ratio is isotropic when possible (all dims the same). When
    ``target_reduction`` is not a perfect power, the algorithm finds
    the per-dim factor ``r`` such that ``r^ndim`` is closest, then
    adjusts individual dimensions to match the target more precisely.

    Args:
        target_reduction: Desired volume reduction factor (>= 1).
        ndim: Number of spatial dimensions.

    Returns:
        Integer ratio per dimension.
    """
    if target_reduction < 1:
        return tuple(1 for _ in range(ndim))

    # Try isotropic first
    r_float = target_reduction ** (1.0 / ndim)
    r_floor = max(1, int(math.floor(r_float)))
    r_ceil = r_floor + 1

    vol_floor = r_floor ** ndim
    vol_ceil = r_ceil ** ndim

    # Pick whichever is closer
    if abs(vol_floor - target_reduction) <= abs(vol_ceil - target_reduction):
        r_base = r_floor
    else:
        r_base = r_ceil

    # Check if isotropic is close enough
    vol_iso = r_base ** ndim
    if abs(vol_iso - target_reduction) / target_reduction < 0.2:
        return tuple(r_base for _ in range(ndim))

    # Non-isotropic: start with r_base per dim and bump individual dims
    ratios = [r_base] * ndim
    current_vol = r_base ** ndim

    if current_vol < target_reduction:
        # Need to increase some dimensions
        for d in range(ndim):
            if current_vol >= target_reduction:
                break
            ratios[d] += 1
            current_vol = 1
            for r in ratios:
                current_vol *= r
    elif current_vol > target_reduction:
        # Need to decrease some dimensions (but not below 1)
        for d in range(ndim - 1, -1, -1):
            if current_vol <= target_reduction:
                break
            if ratios[d] > 1:
                ratios[d] -= 1
                current_vol = 1
                for r in ratios:
                    current_vol *= r

    # Sort descending for convention
    ratios.sort(reverse=True)
    return tuple(ratios)


def compute_level_ratios(
    base_vertex_count: int,
    ndim: int = 3,
    *,
    target_reductions: list[float] | None = None,
    reduction_factor: int = 8,
    max_levels: int = 10,
    min_vertices: int = 8,
) -> list[tuple[int, ...]]:
    """Compute bin ratios for a multi-resolution pyramid.

    If ``target_reductions`` is given (e.g. ``[8, 64, 512]``), returns
    a ratio per level that achieves approximately that cumulative
    vertex reduction.

    If ``target_reductions`` is None, auto-generates levels using
    ``reduction_factor`` as the per-level multiplier.

    Args:
        base_vertex_count: Vertex count at level 0.
        ndim: Number of spatial dimensions.
        target_reductions: List of cumulative reduction factors.
        reduction_factor: Per-level target when auto-generating.
        max_levels: Maximum levels to generate.
        min_vertices: Stop below this count.

    Returns:
        List of bin ratio tuples, one per coarsened level.
    """
    if target_reductions is not None:
        return [
            select_bin_ratio_for_reduction(r, ndim)
            for r in target_reductions
        ]

    # Auto-generate: each level doubles the ratio
    ratios: list[tuple[int, ...]] = []
    cumulative = 1
    for _ in range(max_levels):
        cumulative *= reduction_factor
        est_count = max(1, int(base_vertex_count / cumulative))
        if est_count < min_vertices:
            break
        ratio = select_bin_ratio_for_reduction(cumulative, ndim)
        ratios.append(ratio)

    return ratios


# ===================================================================
# Sparsity-aware level planning
# ===================================================================

@dataclass
class LevelReductionSpec:
    """Full specification for a resolution level including object sparsity.

    Attributes:
        level_index: Target level number (1, 2, ...).
        bin_ratio: Integer fold-change per axis relative to level 0.
        bin_shape: Supervoxel edge lengths at this level (computed).
        object_sparsity: Fraction of objects to retain at this level.
        expected_vertex_reduction: From binning alone (product of bin_ratio).
        expected_object_reduction: From sparsity alone (1 / sparsity).
        expected_volume_reduction: vertex × object reduction.
    """

    level_index: int
    bin_ratio: tuple[int, ...]
    bin_shape: tuple[float, ...] | None = None
    object_sparsity: float = 1.0
    expected_vertex_reduction: float = 1.0
    expected_object_reduction: float = 1.0
    expected_volume_reduction: float = 1.0


def auto_plan_sparsity(
    target_volume_reduction: float,
    bin_ratio: tuple[int, ...],
    ndim: int = 3,
) -> float:
    """Compute the object sparsity needed to hit a target volume reduction.

    Total volume reduction = vertex_reduction × object_reduction.
    Vertex reduction = product(bin_ratio).
    Object reduction = 1 / sparsity.

    So sparsity = vertex_reduction / target_volume_reduction.

    Args:
        target_volume_reduction: Desired total reduction (e.g. 16).
        bin_ratio: Bin ratio giving the vertex reduction.
        ndim: Not used directly but kept for API consistency.

    Returns:
        Object sparsity in (0, 1]. Clamped to 1.0 if binning alone
        already exceeds the target.
    """
    vertex_reduction = 1.0
    for r in bin_ratio:
        vertex_reduction *= r

    if vertex_reduction >= target_volume_reduction:
        return 1.0  # binning alone is enough

    sparsity = vertex_reduction / target_volume_reduction
    return max(sparsity, 1e-6)  # never exactly zero


def plan_pyramid_with_sparsity(
    base_vertex_count: int,
    base_object_count: int,
    base_bin_shape: tuple[float, ...],
    chunk_shape: tuple[float, ...],
    *,
    level_configs: list[dict] | None = None,
    target_volume_reduction: float = 8.0,
    sparsity_weight: float = 0.0,
    max_levels: int = 10,
    min_vertices: int = 8,
) -> list[LevelReductionSpec]:
    """Plan a multi-resolution pyramid with object sparsity.

    Two modes:

    1. **Explicit configs**: ``level_configs`` is a list of dicts, each
       with ``bin_ratio`` and optionally ``object_sparsity``.
    2. **Auto-plan**: generates levels using ``target_volume_reduction``
       per level, splitting the reduction between binning and sparsity
       according to ``sparsity_weight``.

    Args:
        base_vertex_count: Vertices at level 0.
        base_object_count: Objects at level 0.
        base_bin_shape: Supervoxel edge lengths at level 0.
        chunk_shape: Chunk dimensions (constant across levels).
        level_configs: Explicit per-level configs. Each dict has:
            - ``"bin_ratio"``: tuple of ints (required)
            - ``"object_sparsity"``: float in (0,1] (default 1.0)
        target_volume_reduction: Per-level target when auto-planning.
        sparsity_weight: 0.0 = all binning, 1.0 = all sparsity,
            0.5 = balanced split. Only used in auto mode.
        max_levels: Maximum levels to generate.
        min_vertices: Stop below this count.

    Returns:
        List of LevelReductionSpec, one per coarsened level.
    """
    ndim = len(base_bin_shape)

    if level_configs is not None:
        return _plan_from_configs(
            level_configs, base_bin_shape, chunk_shape, ndim,
        )

    return _auto_plan(
        base_vertex_count, base_object_count,
        base_bin_shape, chunk_shape, ndim,
        target_volume_reduction, sparsity_weight,
        max_levels, min_vertices,
    )


def _plan_from_configs(
    configs: list[dict],
    base_bin_shape: tuple[float, ...],
    chunk_shape: tuple[float, ...],
    ndim: int,
) -> list[LevelReductionSpec]:
    """Build specs from explicit level configs."""
    from zarr_vectors.core.metadata import compute_bin_shape

    specs: list[LevelReductionSpec] = []
    for i, cfg in enumerate(configs):
        bin_ratio = tuple(cfg["bin_ratio"])
        sparsity = cfg.get("object_sparsity", 1.0)
        bin_shape = compute_bin_shape(base_bin_shape, bin_ratio)

        vertex_red = 1.0
        for r in bin_ratio:
            vertex_red *= r
        object_red = 1.0 / max(sparsity, 1e-9)
        volume_red = vertex_red * object_red

        specs.append(LevelReductionSpec(
            level_index=i + 1,
            bin_ratio=bin_ratio,
            bin_shape=bin_shape,
            object_sparsity=sparsity,
            expected_vertex_reduction=vertex_red,
            expected_object_reduction=object_red,
            expected_volume_reduction=volume_red,
        ))

    return specs


def _auto_plan(
    base_vertex_count: int,
    base_object_count: int,
    base_bin_shape: tuple[float, ...],
    chunk_shape: tuple[float, ...],
    ndim: int,
    target_volume_reduction: float,
    sparsity_weight: float,
    max_levels: int,
    min_vertices: int,
) -> list[LevelReductionSpec]:
    """Auto-plan levels using target reduction and sparsity weight."""
    from zarr_vectors.core.metadata import (
        compute_bin_shape,
        validate_bin_shape_divides_chunk,
    )

    specs: list[LevelReductionSpec] = []
    cumulative_reduction = 1.0
    current_verts = base_vertex_count
    current_objs = base_object_count

    for level_idx in range(1, max_levels + 1):
        cumulative_reduction *= target_volume_reduction

        # Split reduction between binning and sparsity
        # sparsity_weight=0: all binning. sparsity_weight=1: all sparsity.
        # The binning portion of the per-level reduction
        per_level = target_volume_reduction
        binning_portion = per_level ** (1.0 - sparsity_weight)
        sparsity_portion = per_level ** sparsity_weight

        # Find bin ratio for the binning portion
        bin_ratio = select_bin_ratio_for_reduction(
            binning_portion ** level_idx, ndim,
        )
        bin_shape = compute_bin_shape(base_bin_shape, bin_ratio)

        # Check chunk divisibility — skip if invalid
        try:
            validate_bin_shape_divides_chunk(chunk_shape, bin_shape)
        except Exception:
            # Try next power-of-2 ratio that divides
            found = False
            for k in range(1, 10):
                candidate = tuple(2 ** k for _ in range(ndim))
                candidate_bs = compute_bin_shape(base_bin_shape, candidate)
                try:
                    validate_bin_shape_divides_chunk(chunk_shape, candidate_bs)
                    bin_ratio = candidate
                    bin_shape = candidate_bs
                    found = True
                    break
                except Exception:
                    continue
            if not found:
                break

        vertex_red = 1.0
        for r in bin_ratio:
            vertex_red *= r

        # Compute sparsity to hit the cumulative target
        if vertex_red >= cumulative_reduction:
            sparsity = 1.0
        else:
            sparsity = vertex_red / cumulative_reduction
            sparsity = max(min(sparsity, 1.0), 1e-6)

        est_verts = max(1, int(base_vertex_count / vertex_red))
        est_objs = max(1, int(base_object_count * sparsity))

        if est_verts < min_vertices:
            break

        object_red = 1.0 / max(sparsity, 1e-9)

        specs.append(LevelReductionSpec(
            level_index=level_idx,
            bin_ratio=bin_ratio,
            bin_shape=bin_shape,
            object_sparsity=sparsity,
            expected_vertex_reduction=vertex_red,
            expected_object_reduction=object_red,
            expected_volume_reduction=vertex_red * object_red,
        ))

        current_verts = est_verts
        current_objs = est_objs

    return specs
