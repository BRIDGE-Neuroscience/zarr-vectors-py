"""Shard layout definitions and codec for reading/writing sharded stores.

A shard file contains multiple chunks packed sequentially, with a
JSON index mapping chunk keys to ``(offset, nbytes)`` byte ranges
within the shard file.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from zarr_vectors.sharding.morton import morton_encode
from zarr_vectors.sharding.hilbert import hilbert_encode
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
    """Zarr v3 shard index — arbitrary order with binary index."""


class ShardCodec:
    """Manages reading and writing chunks within shard files.

    Each shard file packs up to ``shard_size`` chunks.  The shard
    assignment is determined by the layout's space-filling curve.

    Args:
        layout: Shard layout strategy.
        shard_size: Maximum chunks per shard file.
        ndim: Number of spatial dimensions (for encoding).
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
        """Map a chunk coordinate to its shard number.

        Args:
            chunk_coords: Spatial chunk coordinate tuple.

        Returns:
            Integer shard ID.
        """
        if self.layout == ShardLayout.FLAT:
            # Flat: each chunk is its own "shard" — use a unique hash
            return hash(chunk_coords)

        if self.layout == ShardLayout.OCTREE:
            code = morton_encode(chunk_coords)
        elif self.layout == ShardLayout.SNAKE:
            code = hilbert_encode(chunk_coords, order=16)
        elif self.layout == ShardLayout.INDEX_TABLE:
            code = morton_encode(chunk_coords)  # default ordering
        else:
            code = hash(chunk_coords)

        return code // max(self.shard_size, 1)

    def shard_filename(self, shard_id: int) -> str:
        """Generate shard file name."""
        return f"shard_{shard_id:06d}.bin"

    def index_filename(self, shard_id: int) -> str:
        """Generate shard index file name."""
        return f"shard_{shard_id:06d}.idx"

    # ---------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------

    def write_shard(
        self,
        base_dir: Path,
        shard_id: int,
        chunk_data: dict[str, bytes],
    ) -> None:
        """Write multiple chunks into a single shard file.

        For INDEX_TABLE layout, the binary index is appended to the
        shard file (Zarr v3 sharding codec format).  For other layouts,
        a separate JSON index file is written.

        Args:
            base_dir: Directory for the array.
            shard_id: Shard number.
            chunk_data: ``{chunk_key: raw_bytes}`` for chunks in this shard.
        """
        base_dir.mkdir(parents=True, exist_ok=True)

        shard_path = base_dir / self.shard_filename(shard_id)

        if self.layout == ShardLayout.INDEX_TABLE:
            self._write_shard_v3(base_dir, shard_id, chunk_data)
        else:
            self._write_shard_json(base_dir, shard_id, chunk_data)

    def _write_shard_json(
        self, base_dir: Path, shard_id: int, chunk_data: dict[str, bytes],
    ) -> None:
        """Write shard with separate JSON index."""
        shard_path = base_dir / self.shard_filename(shard_id)
        index: dict[str, list[int]] = {}

        offset = 0
        parts: list[bytes] = []
        for chunk_key in sorted(chunk_data.keys()):
            data = chunk_data[chunk_key]
            nbytes = len(data)
            index[chunk_key] = [offset, nbytes]
            parts.append(data)
            offset += nbytes

        shard_path.write_bytes(b"".join(parts))

        idx_path = base_dir / self.index_filename(shard_id)
        idx_path.write_text(json.dumps(index))

    def _write_shard_v3(
        self, base_dir: Path, shard_id: int, chunk_data: dict[str, bytes],
    ) -> None:
        """Write shard with Zarr v3 binary index appended to the file.

        The binary index is ``N × 16`` bytes at the end of the shard
        file. Each entry is ``(uint64 offset, uint64 nbytes)``.
        Empty slots have ``offset = 0xFFFFFFFFFFFFFFFF``.
        The chunk's position in the index is determined by a hash
        of its key modulo shard_size.

        A companion JSON index is also written for fast key lookup.
        """
        import struct

        n_slots = max(self.shard_size, len(chunk_data))
        EMPTY = 0xFFFFFFFFFFFFFFFF

        # Build slot assignments (open addressing by hash)
        slot_map: dict[int, str] = {}  # slot → chunk_key
        for chunk_key in chunk_data:
            slot = hash(chunk_key) % n_slots
            while slot in slot_map:
                slot = (slot + 1) % n_slots
            slot_map[slot] = chunk_key

        # Write chunk data sequentially
        offset = 0
        parts: list[bytes] = []
        index_entries: list[tuple[int, int]] = [(EMPTY, 0)] * n_slots
        json_index: dict[str, list[int]] = {}

        for slot in sorted(slot_map.keys()):
            chunk_key = slot_map[slot]
            data = chunk_data[chunk_key]
            nbytes = len(data)
            index_entries[slot] = (offset, nbytes)
            json_index[chunk_key] = [offset, nbytes]
            parts.append(data)
            offset += nbytes

        # Append binary index
        index_bytes = b""
        for off, nb in index_entries:
            index_bytes += struct.pack("<QQ", off, nb)

        shard_path = base_dir / self.shard_filename(shard_id)
        shard_path.write_bytes(b"".join(parts) + index_bytes)

        # Also write JSON index for compatibility
        idx_path = base_dir / self.index_filename(shard_id)
        idx_path.write_text(json.dumps(json_index))

    def read_chunk_from_shard(
        self,
        base_dir: Path,
        chunk_key: str,
        chunk_coords: ChunkCoords,
    ) -> bytes:
        """Read a single chunk from its shard file.

        Args:
            base_dir: Array directory.
            chunk_key: Chunk key string (e.g. ``"0.0.0"``).
            chunk_coords: Chunk coordinate tuple.

        Returns:
            Raw chunk bytes.

        Raises:
            FileNotFoundError: If the shard or chunk is not found.
        """
        shard_id = self.chunk_to_shard_id(chunk_coords)
        idx_path = base_dir / self.index_filename(shard_id)
        shard_path = base_dir / self.shard_filename(shard_id)

        if not idx_path.exists():
            raise FileNotFoundError(
                f"Shard index not found: {idx_path}"
            )

        with open(idx_path) as f:
            index = json.load(f)

        if chunk_key not in index:
            raise FileNotFoundError(
                f"Chunk '{chunk_key}' not in shard {shard_id}"
            )

        offset, nbytes = index[chunk_key]
        with open(shard_path, "rb") as f:
            f.seek(offset)
            return f.read(nbytes)

    def list_chunks_in_shard(
        self,
        base_dir: Path,
        shard_id: int,
    ) -> list[str]:
        """List chunk keys in a shard."""
        idx_path = base_dir / self.index_filename(shard_id)
        if not idx_path.exists():
            return []
        with open(idx_path) as f:
            index = json.load(f)
        return sorted(index.keys())

    def list_all_shards(self, base_dir: Path) -> list[int]:
        """List all shard IDs in a directory."""
        if not base_dir.is_dir():
            return []
        shard_ids: list[int] = []
        for f in base_dir.iterdir():
            if f.name.endswith(".idx"):
                try:
                    sid = int(f.stem.split("_")[1])
                    shard_ids.append(sid)
                except (ValueError, IndexError):
                    continue
        return sorted(shard_ids)

    def list_all_chunk_keys(self, base_dir: Path) -> list[str]:
        """List all chunk keys across all shards."""
        keys: list[str] = []
        for sid in self.list_all_shards(base_dir):
            keys.extend(self.list_chunks_in_shard(base_dir, sid))
        return sorted(keys)

    # ---------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------

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
