"""Shard layout enum and chunk-to-shard mapping.

The actual on-disk shard format is owned by :mod:`zarr_vectors.sharding.io`,
which packs each shard's chunks into a single 1D uint8 Zarr array named
``__shard_<id>`` under the per-array group, with the
``{chunk_key: [offset, nbytes]}`` index stored on the shard array's attrs.

This module only declares the enum of supported layouts and the function
that maps a chunk coordinate to a shard id under each layout (Morton for
``OCTREE``, Hilbert for ``SNAKE``, Morton for ``INDEX_TABLE``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from zarr_vectors.sharding.hilbert import hilbert_encode
from zarr_vectors.sharding.morton import morton_encode
from zarr_vectors.typing import ChunkCoords


class ShardLayout(str, Enum):
    """Supported shard layouts."""

    FLAT = "flat"
    """One file per chunk (default, current behaviour)."""

    OCTREE = "octree"
    """Chunks ordered by Morton Z-curve within each shard."""

    SNAKE = "snake"
    """Chunks ordered by Hilbert curve within each shard."""

    INDEX_TABLE = "index_table"
    """Arbitrary order with binary index."""


class ShardCodec:
    """Maps chunk coordinates to shard ids under a given layout.

    The on-disk packing is done by :mod:`zarr_vectors.sharding.io`; this
    class only encapsulates the layout choice and the coord→shard-id
    mapping. ``shard_size`` is the maximum number of chunks per shard.

    Args:
        layout: Shard layout strategy.
        shard_size: Maximum chunks per shard file.
        ndim: Number of spatial dimensions (kept for serialisation /
            external readers; not used by ``chunk_to_shard_id``).
    """

    def __init__(
        self,
        layout: ShardLayout,
        shard_size: int = 64,
        ndim: int = 3,
    ) -> None:
        self.layout = layout
        self.shard_size = shard_size
        self.ndim = ndim

    def chunk_to_shard_id(self, chunk_coords: ChunkCoords) -> int:
        """Map a chunk coordinate to its shard number."""
        if self.layout == ShardLayout.FLAT:
            return hash(chunk_coords)

        if self.layout == ShardLayout.OCTREE:
            code = morton_encode(chunk_coords)
        elif self.layout == ShardLayout.SNAKE:
            code = hilbert_encode(chunk_coords, order=16)
        elif self.layout == ShardLayout.INDEX_TABLE:
            code = morton_encode(chunk_coords)
        else:
            code = hash(chunk_coords)

        return code // max(self.shard_size, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.value,
            "shard_size": self.shard_size,
            "ndim": self.ndim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShardCodec:
        return cls(
            layout=ShardLayout(d["layout"]),
            shard_size=d.get("shard_size", 64),
            ndim=d.get("ndim", 3),
        )

    def __repr__(self) -> str:
        return (
            f"ShardCodec(layout={self.layout.value}, "
            f"shard_size={self.shard_size}, ndim={self.ndim})"
        )
