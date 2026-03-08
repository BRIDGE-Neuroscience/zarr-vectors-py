"""Compression codec configuration for ZVF arrays.

Provides sensible default codec pipelines for each array type in the
Zarr Vector Format.  All returned configurations are dicts compatible
with Zarr v3's codec pipeline specification.
"""

from __future__ import annotations

from zarr_vectors.constants import (
    ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    DEFAULT_COMPRESSOR_OPTS,
    GROUPINGS,
    GROUPINGS_ATTRIBUTES,
    LINK_ATTRIBUTES,
    LINKS,
    METANODE_CHILDREN,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_GROUP_OFFSETS,
    VERTICES,
)


def get_default_compressor(array_type: str) -> dict[str, object]:
    """Return default compressor configuration for a given array type.

    Args:
        array_type: One of the canonical array name constants (e.g.
            ``"vertices"``, ``"links"``, ``"attributes"``).

    Returns:
        Dict with compressor settings suitable for ``numcodecs`` or
        Zarr v3 codec configuration.
    """
    # Links with sequential parents benefit from delta encoding
    if array_type == LINKS:
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 2,  # SHUFFLE_BITSHUFFLE — good for correlated ints
        }

    # Offsets are monotonically increasing integers — delta + compress
    if array_type in (VERTEX_GROUP_OFFSETS, OBJECT_INDEX, METANODE_CHILDREN):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,  # SHUFFLE_BYTE
        }

    # Vertex positions and attributes — byte shuffle works well on floats
    if array_type in (VERTICES, ATTRIBUTES):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }

    # Dense arrays (object attributes, grouping attributes)
    if array_type in (OBJECT_ATTRIBUTES, GROUPINGS_ATTRIBUTES):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }

    # Everything else — safe default
    return dict(DEFAULT_COMPRESSOR_OPTS)


def get_codec_pipeline(
    array_type: str,
    encoding: str = "raw",
    compression: str | None = None,
    compression_opts: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build a Zarr v3 codec pipeline for an array.

    The pipeline is a list of codec specifications applied in order.
    For raw encoding, this is typically just a compressor.  For Draco
    encoding, the raw bytes are already compressed so we use no
    additional compression (or minimal).

    Args:
        array_type: Canonical array name constant.
        encoding: ``"raw"`` (default) or ``"draco"``.  When draco, the
            data is already compressed and minimal additional compression
            is applied.
        compression: Override compressor name (``"blosc"``, ``"zstd"``,
            ``"gzip"``, ``None`` for no compression).
        compression_opts: Override compressor options dict.

    Returns:
        List of codec config dicts for the Zarr v3 codec pipeline.
    """
    pipeline: list[dict[str, object]] = []

    if encoding == "draco":
        # Draco output is already compressed — store as raw bytes
        # Optionally wrap in a light pass-through codec
        if compression is not None and compression != "none":
            pipeline.append(
                _build_compressor(compression, compression_opts or {"clevel": 1})
            )
        return pipeline

    # Raw encoding — apply compressor
    if compression == "none" or compression is None and encoding == "raw":
        # Use defaults
        if compression is None:
            compressor = get_default_compressor(array_type)
            pipeline.append(compressor)
        # compression == "none" → no compression
        return pipeline

    if compression is not None:
        pipeline.append(
            _build_compressor(compression, compression_opts or {})
        )
        return pipeline

    return pipeline


def _build_compressor(
    name: str,
    opts: dict[str, object],
) -> dict[str, object]:
    """Build a compressor config dict.

    Args:
        name: Compressor name (``"blosc"``, ``"zstd"``, ``"gzip"``).
        opts: Additional options.

    Returns:
        Compressor configuration dict.
    """
    config: dict[str, object] = {"id": name}
    config.update(opts)
    return config
