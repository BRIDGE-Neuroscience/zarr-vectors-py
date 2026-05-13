"""On-disk path helpers for the 0.4 multiscale links layout.

Pre-0.4 the link-family arrays (``links``, ``cross_chunk_links``,
``link_attributes``) lived directly under the resolution-level group.
The 0.4 layout interposes a ``<level_delta>`` path segment so cross-
pyramid-level edges become first-class:

    /resolution_N/links/<delta>/<chunk_key>
    /resolution_N/cross_chunk_links/<delta>/data
    /resolution_N/link_attributes/<name>/<delta>/<chunk_key>
    /resolution_N/cross_chunk_link_attributes/<name>/<delta>/data

The delta segment is signed: ``"0"``, ``"+1"``, ``"-1"``, ``"+2"``, ...
Endpoints in a ``links/<delta>/...`` array have source side at the
owning level and target side at ``this_level + delta``.

This module is intentionally values-only: callers compose paths via
the helpers below and never assemble the raw f-strings inline, so the
delta convention has exactly one definition.
"""

from __future__ import annotations

from zarr_vectors.constants import (
    CROSS_CHUNK_LINK_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    LINK_ATTRIBUTES,
    LINKS,
)


def format_delta(delta: int) -> str:
    """Format a level delta as its on-disk path segment.

    ``0 -> "0"``, ``+N -> "+N"``, ``-N -> "-N"``.  The leading ``+``
    is preserved so a directory listing distinguishes positive deltas
    from the unsigned ``0`` at a glance.
    """
    if delta == 0:
        return "0"
    return f"+{delta}" if delta > 0 else str(delta)


def parse_delta(segment: str) -> int:
    """Inverse of :func:`format_delta`.

    Accepts ``"0"``, ``"+N"``, ``"-N"``.  Raises ``ValueError`` for
    anything else (including stray whitespace or empty input) so a
    malformed on-disk listing fails fast.
    """
    if segment == "0":
        return 0
    if not segment or segment[0] not in "+-":
        raise ValueError(f"invalid level-delta segment: {segment!r}")
    return int(segment)


def links_path(delta: int = 0) -> str:
    """Path of a ``links/<delta>/`` array within a resolution level."""
    return f"{LINKS}/{format_delta(delta)}"


def cross_chunk_links_path(delta: int = 0) -> str:
    """Path of a ``cross_chunk_links/<delta>/`` array within a level."""
    return f"{CROSS_CHUNK_LINKS}/{format_delta(delta)}"


def link_attributes_path(name: str, delta: int = 0) -> str:
    """Path of a ``link_attributes/<name>/<delta>/`` array within a level."""
    return f"{LINK_ATTRIBUTES}/{name}/{format_delta(delta)}"


def cross_chunk_link_attributes_path(name: str, delta: int = 0) -> str:
    """Path of a ``cross_chunk_link_attributes/<name>/<delta>/`` array."""
    return f"{CROSS_CHUNK_LINK_ATTRIBUTES}/{name}/{format_delta(delta)}"
