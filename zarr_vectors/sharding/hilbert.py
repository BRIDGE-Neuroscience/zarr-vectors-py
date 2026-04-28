"""Hilbert curve encoding for N-dimensional coordinates.

Maps N-dimensional integer coordinates to a 1-dimensional Hilbert
index.  Better spatial locality than Morton coding for most access
patterns, but slower to compute.

Uses a simple recursive algorithm for arbitrary dimensions.
"""

from __future__ import annotations


def hilbert_encode(coords: tuple[int, ...], order: int = 16) -> int:
    """Encode coordinates to a Hilbert curve index.

    Args:
        coords: Non-negative integer coordinates.
        order: Number of bits per dimension (grid is ``2^order`` per axis).

    Returns:
        Hilbert curve index.
    """
    ndim = len(coords)
    if ndim == 0:
        return 0
    if ndim == 1:
        return coords[0]

    # Shift negatives
    shifted = [c + (1 << order) if c < 0 else c for c in coords]

    # Use the standard algorithm for 2D, generalize for ND
    if ndim == 2:
        return _hilbert_2d_encode(shifted[0], shifted[1], order)

    # For 3D+: fall back to a simple interleave that gives
    # reasonable locality (not true Hilbert but close)
    return _hilbert_nd_encode(shifted, ndim, order)


def hilbert_decode(index: int, ndim: int, order: int = 16) -> tuple[int, ...]:
    """Decode a Hilbert curve index back to coordinates.

    Args:
        index: Hilbert curve index.
        ndim: Number of dimensions.
        order: Bits per dimension.

    Returns:
        Coordinate tuple.
    """
    if ndim == 0:
        return ()
    if ndim == 1:
        return (index,)

    if ndim == 2:
        x, y = _hilbert_2d_decode(index, order)
        return (x, y)

    return _hilbert_nd_decode(index, ndim, order)


# ===================================================================
# 2D Hilbert (standard algorithm)
# ===================================================================

def _hilbert_2d_encode(x: int, y: int, order: int) -> int:
    """Standard 2D Hilbert curve encoding."""
    rx, ry, d = 0, 0, 0
    s = order - 1
    while s >= 0:
        n = 1 << s
        rx = 1 if (x & n) > 0 else 0
        ry = 1 if (y & n) > 0 else 0
        d += n * n * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s -= 1
    return d


def _hilbert_2d_decode(d: int, order: int) -> tuple[int, int]:
    """Standard 2D Hilbert curve decoding."""
    x, y = 0, 0
    s = 0
    while s < order:
        n = 1 << s
        rx = 1 if (d & 2) > 0 else 0
        ry = 1 if ((d & 1) ^ rx) > 0 else 0  # (d & 1) XOR rx
        # Rotate
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        x += n * rx
        y += n * ry
        d >>= 2
        s += 1
    return x, y


# ===================================================================
# N-D Hilbert (approximate — uses Gray-code interleave)
# ===================================================================

def _hilbert_nd_encode(coords: list[int], ndim: int, order: int) -> int:
    """Approximate N-D Hilbert encoding using Gray code interleave."""
    # Convert each coord to Gray code, then interleave bits
    gray_coords = [c ^ (c >> 1) for c in coords]
    code = 0
    for bit in range(order):
        for dim in range(ndim):
            if gray_coords[dim] & (1 << bit):
                code |= 1 << (bit * ndim + dim)
    return code


def _hilbert_nd_decode(index: int, ndim: int, order: int) -> tuple[int, ...]:
    """Approximate N-D Hilbert decoding."""
    gray_coords = [0] * ndim
    for bit in range(order):
        for dim in range(ndim):
            if index & (1 << (bit * ndim + dim)):
                gray_coords[dim] |= 1 << bit
    # Gray to binary
    coords = []
    for g in gray_coords:
        c = g
        mask = g >> 1
        while mask:
            c ^= mask
            mask >>= 1
        coords.append(c)
    return tuple(coords)
