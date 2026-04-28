"""Configurable shard layouts for zarr vectors stores.

Supports packing multiple chunks into shard files with spatial
ordering (octree/Morton, Hilbert snake) or flat index-table layout.

Usage::

    from zarr_vectors.sharding import ShardLayout, ShardCodec

    codec = ShardCodec(ShardLayout.OCTREE, shard_size=64, ndim=3)
    shard_id = codec.chunk_to_shard_id((2, 3, 1))
"""

from zarr_vectors.sharding.layout import ShardLayout, ShardCodec

__all__ = ["ShardLayout", "ShardCodec"]
