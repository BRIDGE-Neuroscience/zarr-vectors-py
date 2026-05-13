"""Sharded I/O over the Zarr-native chunk layout.

Each per-spatial-chunk blob is a single-chunk 1D ``uint8`` Zarr array
under its parent per-array group (Option G of the migration plan).
Sharding packs a *set* of those chunks into a single packed Zarr 1D
``uint8`` array named ``__shard_<id>`` with a JSON ``shard_index`` on
the shard array's attrs mapping each packed chunk key to ``[offset,
nbytes]`` in the packed buffer.

The shard layouts (octree / snake / index_table) determine which chunk
goes into which shard via :class:`ShardCodec.chunk_to_shard_id` — the
on-disk shape stays uniform regardless of layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.constants import (
    CROSS_CHUNK_LINK_ATTRIBUTES,
    LINK_ATTRIBUTES,
)
from zarr_vectors.core.store import (
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.sharding.layout import ShardCodec, ShardLayout

_SHARD_PREFIX = "__shard_"


def _shard_array_name(shard_id: int) -> str:
    return f"{_SHARD_PREFIX}{shard_id:06d}"


def _is_shard_name(name: str) -> bool:
    return name.startswith(_SHARD_PREFIX)


def _parse_chunk_coords(chunk_key: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(x) for x in chunk_key.split("."))
    except ValueError:
        return None


_DOUBLE_DESCENT_PREFIXES = frozenset({
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
})


def _list_array_names(level_group, requested: list[str] | None) -> list[str]:
    """Enumerate per-array-group names under a resolution level.

    A "per-array group" is a sub-group whose direct children are Zarr
    arrays named by spatial chunk keys (``0.1.2`` etc.) and/or shard
    arrays (``__shard_<id>``).  Top-level sub-groups whose children are
    *themselves* sub-groups (e.g. ``attributes``) get one level of
    descent — yielding ``attributes/<name>`` entries.  Multiscale link
    attribute groups (``link_attributes``, ``cross_chunk_link_attributes``)
    nest two levels (``<name>/<delta>``) and get a second descent.
    """
    if requested is not None:
        return requested

    names: list[str] = []
    for top in level_group:
        try:
            sub_zarr = level_group.zarr_group[top]
        except KeyError:
            continue
        if _looks_like_per_chunk_group(sub_zarr):
            names.append(top)
            continue
        # First descent: attributes/<name>, links/<delta>,
        # cross_chunk_links/<delta>.
        for child in sub_zarr.group_keys():
            child_zarr = sub_zarr[child]
            if _looks_like_per_chunk_group(child_zarr):
                names.append(f"{top}/{child}")
                continue
            # Second descent for link_attributes/<name>/<delta> and
            # cross_chunk_link_attributes/<name>/<delta>.
            if top not in _DOUBLE_DESCENT_PREFIXES:
                continue
            for grand in child_zarr.group_keys():
                grand_zarr = child_zarr[grand]
                if _looks_like_per_chunk_group(grand_zarr):
                    names.append(f"{top}/{child}/{grand}")
    return names


def _looks_like_per_chunk_group(zarr_group) -> bool:
    """Return True if ``zarr_group`` has children matching the chunk-key
    pattern or the shard-name pattern.
    """
    for name in zarr_group.array_keys():
        if _is_shard_name(name) or _parse_chunk_coords(name) is not None:
            return True
    return False


def shard_store(
    store_path: str | Path,
    layout: ShardLayout,
    shard_size: int = 64,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a flat store to a sharded layout.

    For each per-array group, read every chunk-key array, pack the
    bytes into shard arrays ``__shard_<id>``, write the per-shard
    index to the shard array's attrs, and delete the original chunk
    arrays.
    """
    store_path = Path(store_path)
    root = open_store(str(store_path), mode="r+")
    meta = read_root_metadata(root)
    codec = ShardCodec(layout, shard_size, meta.sid_ndim)

    total_shards = 0
    total_chunks = 0

    for level_idx in list_resolution_levels(root):
        level = root[f"resolution_{level_idx}"]
        for array_name in _list_array_names(level, arrays):
            if not level.array_exists(array_name):
                continue
            keys = [k for k in level.list_chunks(array_name) if not _is_shard_name(k)]
            chunk_keys = [k for k in keys if _parse_chunk_coords(k) is not None]
            if not chunk_keys:
                continue

            # Bucket chunks by shard id.
            shard_groups: dict[int, dict[str, bytes]] = {}
            for chunk_key in chunk_keys:
                coords = _parse_chunk_coords(chunk_key)
                shard_id = codec.chunk_to_shard_id(coords)
                shard_groups.setdefault(shard_id, {})[chunk_key] = (
                    level.read_bytes(array_name, chunk_key)
                )

            # Pack each shard into a single Zarr array + attrs index.
            for shard_id, chunk_data in shard_groups.items():
                packed, index = _pack_shard(chunk_data)
                shard_name = _shard_array_name(shard_id)
                level.write_bytes(array_name, shard_name, packed)
                shard_zarr = level.zarr_group[f"{array_name}/{shard_name}"]
                shard_zarr.attrs["shard_index"] = {
                    k: list(v) for k, v in index.items()
                }
                total_shards += 1
                total_chunks += len(chunk_data)

            # Delete original flat chunks (they've been packed).
            arr_group_zarr = level.zarr_group[array_name]
            for chunk_key in chunk_keys:
                if chunk_key in arr_group_zarr:
                    del arr_group_zarr[chunk_key]

    # Record sharding metadata on root.
    root_zg = root.zarr_group
    root_zg.attrs["shard_layout"] = layout.value
    root_zg.attrs["shard_size"] = shard_size

    return {
        "shards_created": total_shards,
        "chunks_packed": total_chunks,
        "layout": layout.value,
        "shard_size": shard_size,
    }


def unshard_store(store_path: str | Path) -> dict[str, Any]:
    """Convert a sharded store back to flat per-chunk Zarr arrays."""
    store_path = Path(store_path)
    root = open_store(str(store_path), mode="r+")
    root_zg = root.zarr_group

    layout_str = root_zg.attrs.get("shard_layout", "flat") if "shard_layout" in root_zg.attrs else "flat"
    if layout_str == "flat":
        return {"chunks_extracted": 0, "message": "already flat"}

    total_chunks = 0

    for level_idx in list_resolution_levels(root):
        level = root[f"resolution_{level_idx}"]
        for array_name in _list_array_names(level, None):
            if not level.array_exists(array_name):
                continue
            arr_group_zarr = level.zarr_group[array_name]
            # Use raw zarr array_keys() to find shard names — the
            # public list_chunks() flattens shards into chunk-key
            # listings for transparent reads.
            shard_names = [
                k for k in arr_group_zarr.array_keys() if _is_shard_name(k)
            ]
            for shard_name in shard_names:
                shard_zarr = arr_group_zarr[shard_name]
                index = dict(shard_zarr.attrs.get("shard_index", {}))
                packed_arr = shard_zarr[:]
                packed = bytes(packed_arr.tobytes())
                for chunk_key, (offset, nbytes) in index.items():
                    chunk_bytes = packed[offset:offset + nbytes]
                    level.write_bytes(array_name, chunk_key, chunk_bytes)
                    total_chunks += 1
                if shard_name in arr_group_zarr:
                    del arr_group_zarr[shard_name]

    # Drop the sharding metadata.
    if "shard_layout" in root_zg.attrs:
        del root_zg.attrs["shard_layout"]
    if "shard_size" in root_zg.attrs:
        del root_zg.attrs["shard_size"]

    return {"chunks_extracted": total_chunks}


def is_sharded(store_path: str | Path) -> bool:
    """Check if a store uses sharded layout."""
    try:
        root = open_store(str(store_path))
        attrs = root.attrs.to_dict()
        return attrs.get("shard_layout", "flat") != "flat"
    except Exception:
        return False


def get_shard_info(store_path: str | Path) -> dict[str, Any]:
    """Get sharding information for a store."""
    root = open_store(str(store_path))
    attrs = root.attrs.to_dict()
    layout = attrs.get("shard_layout", "flat")
    shard_size = attrs.get("shard_size", 0)
    if layout == "flat":
        return {"layout": "flat", "sharded": False}

    meta = read_root_metadata(root)
    _ = ShardCodec(ShardLayout(layout), shard_size, meta.sid_ndim)  # validate

    shard_count = 0
    for level_idx in list_resolution_levels(root):
        level = root[f"resolution_{level_idx}"]
        for array_name in _list_array_names(level, None):
            if not level.array_exists(array_name):
                continue
            shard_count += sum(
                1 for k in level.list_chunks(array_name) if _is_shard_name(k)
            )
            break  # one array's count is representative; matches legacy behaviour
        if shard_count:
            break

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
    """Convert a store between shard layouts."""
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

    if is_sharded(str(store_path)):
        current_info = get_shard_info(str(store_path))
        if (current_info["layout"] == target_layout.value
                and current_info.get("shard_size") == shard_size):
            return {"action": "noop", "message": "already in target layout"}
        unshard_store(store_path)

    result = shard_store(store_path, target_layout, shard_size)
    return {
        "action": "reshard",
        "target_layout": target_layout.value,
        **result,
    }


# ===================================================================
# Packing helpers
# ===================================================================


def _pack_shard(chunk_data: dict[str, bytes]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Concatenate ``chunk_data`` values; return (packed_bytes, index).

    Index maps chunk_key → (offset, nbytes) in the packed buffer.
    Keys are written in sorted order so the layout is deterministic.
    """
    index: dict[str, tuple[int, int]] = {}
    parts: list[bytes] = []
    offset = 0
    for chunk_key in sorted(chunk_data.keys()):
        data = chunk_data[chunk_key]
        nbytes = len(data)
        index[chunk_key] = (offset, nbytes)
        parts.append(data)
        offset += nbytes
    return b"".join(parts), index
