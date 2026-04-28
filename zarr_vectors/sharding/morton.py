"""Morton Z-curve encoding for N-dimensional coordinates.

Interleaves bits of each coordinate dimension to produce a single
integer sort key.  Spatially nearby coordinates produce nearby
Morton codes, giving good locality for spatial queries.
"""

from __future__ import annotations


def morton_encode(coords: tuple[int, ...]) -> int:
    """Encode N-dimensional coordinates to a Morton Z-curve code.

    Interleaves bits of each coordinate.  For 3D ``(x, y, z)``,
    the output bits are ``... z2 y2 x2 z1 y1 x1 z0 y0 x0``.

    Args:
        coords: Non-negative integer coordinates.

    Returns:
        Morton code (non-negative integer).
    """
    ndim = len(coords)
    if ndim == 0:
        return 0

    # Handle negative coordinates by shifting
    shifted = []
    for c in coords:
        if c < 0:
            shifted.append(c + (1 << 20))  # shift into positive range
        else:
            shifted.append(c)

    # Find max bits needed
    max_val = max(shifted) if shifted else 0
    max_bits = max_val.bit_length() if max_val > 0 else 1

    code = 0
    for bit in range(max_bits):
        for dim in range(ndim):
            if shifted[dim] & (1 << bit):
                code |= 1 << (bit * ndim + dim)

    return code


def morton_decode(code: int, ndim: int) -> tuple[int, ...]:
    """Decode a Morton code back to N-dimensional coordinates.

    Args:
        code: Morton code.
        ndim: Number of dimensions.

    Returns:
        Coordinate tuple.
    """
    if ndim == 0:
        return ()

    coords = [0] * ndim
    max_bits = (code.bit_length() + ndim - 1) // ndim if code > 0 else 1

    for bit in range(max_bits):
        for dim in range(ndim):
            if code & (1 << (bit * ndim + dim)):
                coords[dim] |= 1 << bit

    return tuple(coords)
