"""Bin-ratio helpers for multi-resolution pyramids.

The level-emission logic lives in
:mod:`zarr_vectors.multiresolution.coarsen` (factor-based interface);
this module is purely a small helpers shelf for computing bin ratios
from target volume reductions and for describing one resolution level
in the sparsity-aware data class form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
        for d in range(ndim):
            if current_vol >= target_reduction:
                break
            ratios[d] += 1
            current_vol = 1
            for r in ratios:
                current_vol *= r
    elif current_vol > target_reduction:
        for d in range(ndim - 1, -1, -1):
            if current_vol <= target_reduction:
                break
            if ratios[d] > 1:
                ratios[d] -= 1
                current_vol = 1
                for r in ratios:
                    current_vol *= r

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
# Level specification dataclass
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
        return 1.0

    sparsity = vertex_reduction / target_volume_reduction
    return max(sparsity, 1e-6)
