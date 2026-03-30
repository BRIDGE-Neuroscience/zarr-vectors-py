"""Adaptive resolution level selection.

Determines which coarsening levels to emit based on the adaptive
threshold rule: bin size doubles each candidate level, but a new
resolution level is only stored when total vertex count drops below
``N_previous / reduction_factor``.
"""

from __future__ import annotations

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

        # Rough estimate: vertex count scales as 1 / bin_volume
        # relative to the previous level's bin volume
        # More precisely: count ≈ N_0 / (bin_size / base_bin_size)^ndim
        # We use a simple halving heuristic (cube of 2)
        est_count = max(1, int(full_resolution_count / (2 ** k) ** 3))

        if est_count >= prev_count:
            continue  # not enough reduction, skip

        ratio = prev_count / max(est_count, 1)
        if ratio < reduction_factor:
            continue  # not enough reduction, skip

        if est_count < min_vertices:
            break  # too few vertices, stop

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
