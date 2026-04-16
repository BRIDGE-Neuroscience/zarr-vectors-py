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
from zarr_vectors.typing import BinCoords, BinShape, BoundingBox, ChunkCoords, ChunkShape


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


# ===================================================================
# Bin-level spatial assignment (supervoxel binning)
# ===================================================================

def assign_bins(
    positions: npt.NDArray[np.floating],
    bin_shape: BinShape,
) -> dict[BinCoords, npt.NDArray[np.intp]]:
    """Assign each vertex to a supervoxel bin.

    Identical to :func:`assign_chunks` but uses bin_shape instead of
    chunk_shape — produces a finer spatial grouping.

    Args:
        positions: ``(N, D)`` array of vertex positions.
        bin_shape: Supervoxel edge lengths per dimension.

    Returns:
        Dict mapping ``bin_coords`` → ``(N_k,)`` array of vertex indices.
    """
    # Delegate to assign_chunks — same logic, different granularity
    return assign_chunks(positions, bin_shape)


def bin_to_chunk(
    bin_coords: BinCoords,
    bins_per_chunk: tuple[int, ...],
) -> ChunkCoords:
    """Map a bin coordinate to its parent chunk coordinate.

    Args:
        bin_coords: N-dimensional bin grid coordinate.
        bins_per_chunk: Number of bins per chunk in each dimension.

    Returns:
        Chunk coordinate tuple.
    """
    return tuple(
        b // bpc if b >= 0 else -(-b // bpc) - (1 if (-b) % bpc != 0 else 0)
        for b, bpc in zip(bin_coords, bins_per_chunk)
    )


def chunk_to_bin_range(
    chunk_coords: ChunkCoords,
    bins_per_chunk: tuple[int, ...],
) -> tuple[BinCoords, BinCoords]:
    """Return the range of bin coordinates within a chunk (inclusive).

    Args:
        chunk_coords: Chunk grid coordinate.
        bins_per_chunk: Number of bins per chunk in each dimension.

    Returns:
        ``(min_bin_coords, max_bin_coords)`` — both inclusive.
    """
    lo = tuple(c * bpc for c, bpc in zip(chunk_coords, bins_per_chunk))
    hi = tuple(c * bpc + bpc - 1 for c, bpc in zip(chunk_coords, bins_per_chunk))
    return lo, hi


def bin_to_vg_index(
    bin_coords: BinCoords,
    chunk_coords: ChunkCoords,
    bins_per_chunk: tuple[int, ...],
) -> int:
    """Linearise an intra-chunk bin coordinate to a vertex group index.

    Uses row-major (C-order) linearisation within the chunk's bin grid.

    Args:
        bin_coords: Global bin coordinate.
        chunk_coords: Parent chunk coordinate.
        bins_per_chunk: Bins per chunk per dimension.

    Returns:
        Integer vertex group index within the chunk.
    """
    ndim = len(bins_per_chunk)
    # Compute local bin offset within the chunk
    local = tuple(
        b - c * bpc for b, c, bpc in zip(bin_coords, chunk_coords, bins_per_chunk)
    )
    # Row-major linearisation
    idx = 0
    stride = 1
    for d in range(ndim - 1, -1, -1):
        idx += local[d] * stride
        stride *= bins_per_chunk[d]
    return idx


def vg_index_to_bin(
    vg_index: int,
    chunk_coords: ChunkCoords,
    bins_per_chunk: tuple[int, ...],
) -> BinCoords:
    """Convert a vertex group index back to a global bin coordinate.

    Inverse of :func:`bin_to_vg_index`.

    Args:
        vg_index: Linearised vertex group index within the chunk.
        chunk_coords: Parent chunk coordinate.
        bins_per_chunk: Bins per chunk per dimension.

    Returns:
        Global bin coordinate tuple.
    """
    ndim = len(bins_per_chunk)
    local: list[int] = [0] * ndim
    remaining = vg_index
    for d in range(ndim - 1, -1, -1):
        local[d] = remaining % bins_per_chunk[d]
        remaining //= bins_per_chunk[d]
    return tuple(
        l + c * bpc for l, c, bpc in zip(local, chunk_coords, bins_per_chunk)
    )


def bins_intersecting_bbox(
    bbox_min: npt.NDArray[np.floating],
    bbox_max: npt.NDArray[np.floating],
    bin_shape: BinShape,
) -> list[BinCoords]:
    """Return all bin coordinates that intersect a bounding box.

    Finer-grained version of :func:`chunks_intersecting_bbox`.

    Args:
        bbox_min: ``(D,)`` minimum corner.
        bbox_max: ``(D,)`` maximum corner.
        bin_shape: Supervoxel edge lengths.

    Returns:
        Sorted list of bin coordinate tuples.
    """
    return chunks_intersecting_bbox(bbox_min, bbox_max, bin_shape)


def group_bins_by_chunk(
    bin_assignments: dict[BinCoords, npt.NDArray[np.intp]],
    bins_per_chunk: tuple[int, ...],
) -> dict[ChunkCoords, dict[int, npt.NDArray[np.intp]]]:
    """Group bin assignments into chunks with linearised vertex group indices.

    Takes the output of :func:`assign_bins` and organises it by chunk.
    Each entry maps a vertex group index (linearised bin position within
    the chunk) to the array of global vertex indices in that bin.

    Args:
        bin_assignments: Output from ``assign_bins``.
        bins_per_chunk: Bins per chunk per dimension.

    Returns:
        ``{chunk_coords: {vg_index: global_vertex_indices}}``.
    """
    result: dict[ChunkCoords, dict[int, npt.NDArray[np.intp]]] = {}

    for bc, indices in bin_assignments.items():
        cc = bin_to_chunk(bc, bins_per_chunk)
        vg_idx = bin_to_vg_index(bc, cc, bins_per_chunk)

        if cc not in result:
            result[cc] = {}
        result[cc][vg_idx] = indices

    return result
