"""Spatial chunk assignment and query utilities.

All functions are pure numpy — no store or encoding dependencies.
Chunk assignment is vectorised: ``assign_chunks`` processes millions
of vertices in one pass using ``np.floor`` and structured-array
``np.unique``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ChunkingError
from zarr_vectors.typing import BoundingBox, ChunkCoords, ChunkShape


def assign_chunks(
    positions: npt.NDArray[np.floating],
    chunk_shape: ChunkShape,
) -> dict[ChunkCoords, npt.NDArray[np.intp]]:
    """Assign each vertex to a spatial chunk.

    Args:
        positions: ``(N, D)`` array of vertex positions.
        chunk_shape: Physical size per dimension.  Length must equal D.

    Returns:
        Dict mapping ``chunk_coords`` → ``(N_k,)`` array of vertex
        indices belonging to that chunk.  Indices are into the
        original *positions* array.

    Raises:
        ChunkingError: If dimensions are inconsistent.
    """
    if positions.ndim != 2:
        raise ChunkingError(
            f"positions must be 2-D, got shape {positions.shape}"
        )
    ndim = positions.shape[1]
    if len(chunk_shape) != ndim:
        raise ChunkingError(
            f"chunk_shape length {len(chunk_shape)} != "
            f"positions dimensionality {ndim}"
        )
    if any(c <= 0 for c in chunk_shape):
        raise ChunkingError("All chunk_shape values must be > 0")

    n = len(positions)
    if n == 0:
        return {}

    cs = np.array(chunk_shape, dtype=np.float64)
    # Vectorised chunk coordinate computation
    chunk_ints = np.floor(positions / cs).astype(np.int64)  # (N, D)

    # Group indices by unique chunk coordinate rows
    # Convert to structured array for np.unique
    result: dict[ChunkCoords, npt.NDArray[np.intp]] = {}

    # Fast path for few dimensions: use tuple hashing
    if ndim <= 4:
        # Build a dict by iterating unique rows — but avoid Python loops
        # over all N rows.  Instead, use lexsort + diff to find group
        # boundaries.
        keys = chunk_ints.T  # (D, N)
        sort_idx = np.lexsort(keys[::-1])  # sort by dim0, then dim1, ...
        sorted_chunks = chunk_ints[sort_idx]  # (N, D) sorted

        # Find boundaries where chunk coords change
        diffs = np.any(sorted_chunks[1:] != sorted_chunks[:-1], axis=1)
        boundaries = np.flatnonzero(diffs) + 1
        # Split sort_idx at boundaries
        groups = np.split(sort_idx, boundaries)

        for grp in groups:
            coord = tuple(int(x) for x in chunk_ints[grp[0]])
            result[coord] = grp
    else:
        # Fallback for high-D: structured array approach
        dt = np.dtype([(f"d{i}", np.int64) for i in range(ndim)])
        structured = np.empty(n, dtype=dt)
        for i in range(ndim):
            structured[f"d{i}"] = chunk_ints[:, i]

        unique_coords, inverse = np.unique(structured, return_inverse=True)
        for idx, uc in enumerate(unique_coords):
            coord = tuple(int(uc[f"d{i}"]) for i in range(ndim))
            mask = inverse == idx
            result[coord] = np.flatnonzero(mask)

    return result


def compute_chunk_coords(
    position: npt.NDArray[np.floating],
    chunk_shape: ChunkShape,
) -> ChunkCoords:
    """Compute chunk coordinates for a single position.

    Args:
        position: ``(D,)`` array.
        chunk_shape: Physical size per dimension.

    Returns:
        Chunk coordinate tuple, e.g. ``(0, 1, 2)``.
    """
    cs = np.array(chunk_shape, dtype=np.float64)
    return tuple(int(x) for x in np.floor(position / cs))


def compute_bounds(
    positions: npt.NDArray[np.floating],
) -> BoundingBox:
    """Compute axis-aligned bounding box.

    Args:
        positions: ``(N, D)`` array.

    Returns:
        ``(min_corner, max_corner)`` — each a ``(D,)`` float64 array.

    Raises:
        ChunkingError: If positions is empty.
    """
    if len(positions) == 0:
        raise ChunkingError("Cannot compute bounds of empty positions")
    return (
        np.min(positions, axis=0).astype(np.float64),
        np.max(positions, axis=0).astype(np.float64),
    )


def compute_grid_shape(
    bounds: BoundingBox,
    chunk_shape: ChunkShape,
) -> tuple[int, ...]:
    """Compute number of chunks per dimension.

    Args:
        bounds: ``(min_corner, max_corner)``.
        chunk_shape: Physical chunk size per dimension.

    Returns:
        Tuple of chunk counts per dimension.  Each value is at least 1.
    """
    cs = np.array(chunk_shape, dtype=np.float64)
    min_corner, max_corner = bounds
    extent = np.asarray(max_corner, dtype=np.float64) - np.asarray(min_corner, dtype=np.float64)
    grid = np.ceil(extent / cs).astype(int)
    # Ensure at least 1 chunk per dimension
    grid = np.maximum(grid, 1)
    return tuple(int(x) for x in grid)


def chunks_intersecting_bbox(
    bbox_min: npt.NDArray[np.floating],
    bbox_max: npt.NDArray[np.floating],
    chunk_shape: ChunkShape,
) -> list[ChunkCoords]:
    """Return all chunk coordinates that intersect a bounding box.

    Args:
        bbox_min: ``(D,)`` minimum corner of query box.
        bbox_max: ``(D,)`` maximum corner of query box.
        chunk_shape: Physical chunk size per dimension.

    Returns:
        Sorted list of chunk coordinate tuples.
    """
    cs = np.array(chunk_shape, dtype=np.float64)
    lo = np.floor(np.asarray(bbox_min, dtype=np.float64) / cs).astype(int)
    hi = np.floor(np.asarray(bbox_max, dtype=np.float64) / cs).astype(int)

    ndim = len(cs)
    # Build cartesian product of chunk ranges
    ranges = [range(int(lo[d]), int(hi[d]) + 1) for d in range(ndim)]
    result: list[ChunkCoords] = []
    _cartesian_product(ranges, 0, (), result)
    return sorted(result)


def _cartesian_product(
    ranges: list[range],
    depth: int,
    current: tuple[int, ...],
    out: list[ChunkCoords],
) -> None:
    """Recursive cartesian product of ranges."""
    if depth == len(ranges):
        out.append(current)
        return
    for val in ranges[depth]:
        _cartesian_product(ranges, depth + 1, current + (val,), out)


def positions_in_bbox(
    positions: npt.NDArray[np.floating],
    bbox_min: npt.NDArray[np.floating],
    bbox_max: npt.NDArray[np.floating],
) -> npt.NDArray[np.intp]:
    """Return indices of positions within a bounding box (inclusive).

    Args:
        positions: ``(N, D)`` array.
        bbox_min: ``(D,)`` minimum corner.
        bbox_max: ``(D,)`` maximum corner.

    Returns:
        ``(M,)`` array of indices where all coordinates are within
        ``[bbox_min, bbox_max]``.
    """
    mask = np.all(
        (positions >= bbox_min) & (positions <= bbox_max),
        axis=1,
    )
    return np.flatnonzero(mask)
