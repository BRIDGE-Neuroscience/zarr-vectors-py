"""Sharded I/O integration for FsGroup.

Provides ``ShardedWriter`` and ``ShardedReader`` that batch chunk
writes into shard files and read individual chunks from shards.
These are used by the type-module writers when ``shard_layout``
is specified.

For transparent integration, ``write_sharded_store`` wraps an
existing flat store into sharded format after writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zarr_vectors.core.store import FsGroup, open_store
from zarr_vectors.sharding.layout import ShardCodec, ShardLayout
from zarr_vectors.typing import ChunkCoords


def shard_store(
    store_path: str | Path,
    layout: ShardLayout,
    shard_size: int = 64,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a flat store to sharded layout.

    Reads all chunk files for each array, packs them into shard files
    using the specified layout, writes shard indices, and removes the
    original flat chunk files.

    Args:
        store_path: Path to the store.
        layout: Target shard layout.
        shard_size: Chunks per shard.
        arrays: List of array names to shard. If None, shards all
            arrays found under each resolution level.

    Returns:
        Summary dict with ``shards_created``, ``chunks_packed``.
    """
    store_path = Path(store_path)
    root = open_store(str(store_path), mode="r+")

    from zarr_vectors.core.store import read_root_metadata, list_resolution_levels
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    codec = ShardCodec(layout, shard_size, ndim)

    levels = list_resolution_levels(root)
    total_shards = 0
    total_chunks = 0

    for level_idx in levels:
        level_dir = store_path / f"resolution_{level_idx}"
        if not level_dir.is_dir():
            continue

        # Find array directories
        if arrays is not None:
            array_names = arrays
        else:
            array_names = [
                d.name for d in level_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

        for array_name in array_names:
            arr_dir = level_dir / array_name
            if not arr_dir.is_dir():
                continue

            # Collect flat chunk files (not shard files, not .zattrs)
            chunk_files = [
                f for f in arr_dir.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and not f.name.endswith(".idx")
                and not f.name.startswith("shard_")
            ]

            if not chunk_files:
                continue

            # Group chunks by shard ID
            shard_groups: dict[int, dict[str, bytes]] = {}
            for cf in chunk_files:
                chunk_key = cf.name
                # Parse chunk coords from key
                try:
                    coords = tuple(int(x) for x in chunk_key.split("."))
                except ValueError:
                    continue

                shard_id = codec.chunk_to_shard_id(coords)
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = {}
                shard_groups[shard_id][chunk_key] = cf.read_bytes()

            # Write shards
            for shard_id, chunk_data in shard_groups.items():
                codec.write_shard(arr_dir, shard_id, chunk_data)
                total_shards += 1
                total_chunks += len(chunk_data)

            # Remove original flat files
            for cf in chunk_files:
                cf.unlink()

    # Store shard metadata in root attrs
    attrs = root.attrs.to_dict()
    attrs["shard_layout"] = layout.value
    attrs["shard_size"] = shard_size
    root.attrs.update(attrs)

    return {
        "shards_created": total_shards,
        "chunks_packed": total_chunks,
        "layout": layout.value,
        "shard_size": shard_size,
    }


def unshard_store(store_path: str | Path) -> dict[str, Any]:
    """Convert a sharded store back to flat layout.

    Reads all shard files, extracts individual chunks as flat files,
    and removes the shard files and indices.

    Args:
        store_path: Path to the store.

    Returns:
        Summary dict.
    """
    store_path = Path(store_path)
    root = open_store(str(store_path), mode="r+")

    from zarr_vectors.core.store import read_root_metadata, list_resolution_levels
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim

    # Read shard config from attrs
    attrs = root.attrs.to_dict()
    layout_str = attrs.get("shard_layout", "flat")
    shard_size = attrs.get("shard_size", 64)

    if layout_str == "flat":
        return {"chunks_extracted": 0, "message": "already flat"}

    codec = ShardCodec(ShardLayout(layout_str), shard_size, ndim)
    levels = list_resolution_levels(root)
    total_chunks = 0

    for level_idx in levels:
        level_dir = store_path / f"resolution_{level_idx}"
        if not level_dir.is_dir():
            continue

        array_names = [
            d.name for d in level_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        for array_name in array_names:
            arr_dir = level_dir / array_name

            for shard_id in codec.list_all_shards(arr_dir):
                idx_path = arr_dir / codec.index_filename(shard_id)
                shard_path = arr_dir / codec.shard_filename(shard_id)

                if not idx_path.exists():
                    continue

                with open(idx_path) as f:
                    index = json.load(f)

                shard_data = shard_path.read_bytes()
                for chunk_key, (offset, nbytes) in index.items():
                    chunk_bytes = shard_data[offset:offset + nbytes]
                    (arr_dir / chunk_key).write_bytes(chunk_bytes)
                    total_chunks += 1

                # Remove shard files
                idx_path.unlink()
                shard_path.unlink()

    # Update metadata — remove shard keys
    attrs = root.attrs.to_dict()
    cleaned = {k: v for k, v in attrs.items() if k not in ("shard_layout", "shard_size")}
    # Overwrite attrs file completely
    import json as _json
    attrs_path = Path(store_path) / ".zattrs"
    attrs_path.write_text(_json.dumps(cleaned, indent=2))

    return {"chunks_extracted": total_chunks}


def is_sharded(store_path: str | Path) -> bool:
    """Check if a store uses sharded layout."""
    store_path = Path(store_path)
    try:
        root = open_store(str(store_path))
        attrs = root.attrs.to_dict()
        return attrs.get("shard_layout", "flat") != "flat"
    except Exception:
        return False


def get_shard_info(store_path: str | Path) -> dict[str, Any]:
    """Get sharding information for a store."""
    store_path = Path(store_path)
    root = open_store(str(store_path))
    attrs = root.attrs.to_dict()

    layout = attrs.get("shard_layout", "flat")
    shard_size = attrs.get("shard_size", 0)

    if layout == "flat":
        return {"layout": "flat", "sharded": False}

    from zarr_vectors.core.store import read_root_metadata, list_resolution_levels
    meta = read_root_metadata(root)
    codec = ShardCodec(ShardLayout(layout), shard_size, meta.sid_ndim)

    # Count shards in level 0
    level_dir = store_path / "resolution_0"
    shard_count = 0
    if level_dir.is_dir():
        for arr_dir in level_dir.iterdir():
            if arr_dir.is_dir():
                shard_count += len(codec.list_all_shards(arr_dir))
                break  # just count vertices shards

    return {
        "layout": layout,
        "sharded": True,
        "shard_size": shard_size,
        "shard_count": shard_count,
    }


def reshard(
    store_path: str | Path,
    target_layout: ShardLayout,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Convert a store between shard layouts.

    Handles all transitions: flat→sharded, sharded→sharded
    (different layout or shard_size), and sharded→flat.

    Args:
        store_path: Path to the store.
        target_layout: Desired shard layout.
        shard_size: Chunks per shard in the target layout.

    Returns:
        Summary dict with ``action`` and details.
    """
    store_path = Path(store_path)

    if target_layout == ShardLayout.FLAT:
        if is_sharded(str(store_path)):
            result = unshard_store(store_path)
            return {
                "action": "unshard",
                "source_layout": "sharded",
                "target_layout": "flat",
                **result,
            }
        return {"action": "noop", "message": "already flat"}

    # Target is sharded
    if is_sharded(str(store_path)):
        current_info = get_shard_info(str(store_path))
        if (current_info["layout"] == target_layout.value
                and current_info.get("shard_size") == shard_size):
            return {"action": "noop", "message": "already in target layout"}
        # Unshard first, then re-shard with new layout
        unshard_store(store_path)

    result = shard_store(store_path, target_layout, shard_size)
    return {
        "action": "reshard",
        "target_layout": target_layout.value,
        **result,
    }
