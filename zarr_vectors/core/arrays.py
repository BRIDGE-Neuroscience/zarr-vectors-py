"""Create, write, and read ZV per-level arrays.

Each "array" is a subdirectory within a resolution-level group.  Chunk
data is stored as raw binary files named by their spatial chunk
coordinates (e.g. ``vertices/0.0.0``).  Array metadata is in
``<array>/.zattrs``.

This module is the single point of contact for all array I/O — type
modules (``types/*.py``) call these functions rather than touching
the store or encoding modules directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    ATTRIBUTES,
    CROSS_CHUNK_FACES,
    CROSS_CHUNK_LINKS,
    GROUPINGS,
    GROUPINGS_ATTRIBUTES,
    LINK_ATTRIBUTES,
    LINKS,
    METANODE_CHILDREN,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_COUNTS,
    VERTEX_GROUP_OFFSETS,
    VERTICES,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.encoding.ragged import (
    decode_object_index,
    decode_paired_offsets,
    decode_ragged_ints,
    decode_vertex_groups,
    encode_object_index,
    encode_paired_offsets,
    encode_ragged_ints,
    encode_vertex_groups,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.typing import (
    ChunkCoords,
    CrossChunkLink,
    ObjectManifest,
    VertexGroupRef,
)


# ===================================================================
# Helpers
# ===================================================================

def _chunk_key(coords: ChunkCoords) -> str:
    """Convert chunk coordinates to a dot-separated key string.

    ``(0, 1, 2)`` → ``"0.1.2"``
    """
    return ".".join(str(c) for c in coords)


def _parse_chunk_key(key: str) -> ChunkCoords:
    """Parse a dot-separated chunk key back to coordinates.

    ``"0.1.2"`` → ``(0, 1, 2)``
    """
    return tuple(int(x) for x in key.split("."))


def _ensure_array_dir(level_group: FsGroup, array_name: str) -> None:
    """Ensure an array subdirectory exists within a level group."""
    level_group.require_group(array_name)


def read_zv_array_tag(meta: dict) -> str | None:
    """Return the discriminator string from an array's ``.zattrs`` dict.

    Looks under the new key ``zv_array`` first and falls back to the
    legacy ``zvf_array`` key for stores written before the rename.  Use
    this helper instead of indexing the dict directly so that conformance
    / validation code keeps working against existing-on-disk stores.
    """
    return meta.get("zv_array", meta.get("zvf_array"))


# ===================================================================
# Array creation (set up directory + metadata)
# ===================================================================

def create_vertices_array(
    level_group: FsGroup,
    dtype: str = "float32",
    encoding: str = "raw",
) -> None:
    """Create the ``vertices/`` array within a resolution level.

    Args:
        level_group: The resolution level FsGroup.
        dtype: Numpy dtype string for vertex positions.
        encoding: ``"raw"`` or ``"draco"``.
    """
    _ensure_array_dir(level_group, VERTICES)
    _ensure_array_dir(level_group, VERTEX_GROUP_OFFSETS)
    level_group.write_array_meta(VERTICES, {
        "zv_array": "vertices",
        "dtype": dtype,
        "encoding": encoding,
    })


def create_links_array(
    level_group: FsGroup,
    link_width: int,
    dtype: str = "int64",
) -> None:
    """Create the ``links/`` array.

    Args:
        level_group: The resolution level FsGroup.
        link_width: Number of vertex indices per link entry (L).
            1 for skeleton parents, 2 for edges, 3 for triangle faces.
        dtype: Integer dtype.
    """
    _ensure_array_dir(level_group, LINKS)
    level_group.write_array_meta(LINKS, {
        "zv_array": "links",
        "dtype": dtype,
        "link_width": link_width,
    })


def create_attribute_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    channel_names: list[str] | None = None,
) -> None:
    """Create a vertex attribute array ``attributes/<name>/``.

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name (e.g. ``"radius"``, ``"gene_expression"``).
        dtype: Numpy dtype string.
        channel_names: Optional list of channel names.
    """
    full_name = f"{ATTRIBUTES}/{name}"
    _ensure_array_dir(level_group, full_name)
    meta: dict[str, Any] = {
        "zv_array": "attribute",
        "name": name,
        "dtype": dtype,
    }
    if channel_names is not None:
        meta["channel_names"] = channel_names
    level_group.write_array_meta(full_name, meta)


def create_object_index_array(level_group: FsGroup) -> None:
    """Create the ``object_index/`` array."""
    _ensure_array_dir(level_group, OBJECT_INDEX)
    level_group.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
    })


def create_object_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    num_channels: int = 1,
) -> None:
    """Create an object attribute array ``object_attributes/<name>/``.

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name.
        dtype: Numpy dtype string.
        num_channels: Number of channels (C dimension).
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "object_attribute",
        "name": name,
        "dtype": dtype,
        "num_channels": num_channels,
    })


def create_groupings_array(level_group: FsGroup) -> None:
    """Create the ``groupings/`` array."""
    _ensure_array_dir(level_group, GROUPINGS)
    level_group.write_array_meta(GROUPINGS, {
        "zv_array": "groupings",
    })


def create_groupings_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    num_channels: int = 1,
) -> None:
    """Create a groupings attribute array ``groupings_attributes/<name>/``."""
    full_name = f"{GROUPINGS_ATTRIBUTES}/{name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "groupings_attribute",
        "name": name,
        "dtype": dtype,
        "num_channels": num_channels,
    })


def create_cross_chunk_links_array(level_group: FsGroup) -> None:
    """Create the ``cross_chunk_links/`` array."""
    _ensure_array_dir(level_group, CROSS_CHUNK_LINKS)
    level_group.write_array_meta(CROSS_CHUNK_LINKS, {
        "zv_array": "cross_chunk_links",
    })


def create_link_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
) -> None:
    """Create a link attribute array ``link_attributes/<name>/``."""
    full_name = f"{LINK_ATTRIBUTES}/{name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "link_attribute",
        "name": name,
        "dtype": dtype,
    })


def create_metanode_children_array(level_group: FsGroup) -> None:
    """Create the ``metanode_children/`` array (for levels > 0)."""
    _ensure_array_dir(level_group, METANODE_CHILDREN)
    level_group.write_array_meta(METANODE_CHILDREN, {
        "zv_array": "metanode_children",
    })


# ===================================================================
# Writing data
# ===================================================================

def write_chunk_vertices(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    groups: list[npt.NDArray[np.floating]],
    dtype: np.dtype | str = np.float32,
) -> npt.NDArray[np.int64]:
    """Write vertex groups to a spatial chunk.

    Encodes the groups as a contiguous byte buffer in ``vertices/``,
    and writes the K×2 byte offsets to ``vertex_group_offsets/``.
    The link_offset column is set to -1 (no links); callers that also
    write links should update via :func:`write_chunk_links`.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        groups: List of arrays, each ``(N_k, D)``.
        dtype: Numpy dtype for serialisation.

    Returns:
        ``(K,)`` int64 array of vertex byte offsets (for external use).
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)

    raw_bytes, vertex_offsets = encode_vertex_groups(groups, dtype)
    level_group.write_bytes(VERTICES, key, raw_bytes)

    # Build paired offsets: vertex offsets + placeholder link offsets (-1)
    link_offsets = np.full_like(vertex_offsets, -1)
    paired_bytes = encode_paired_offsets(vertex_offsets, link_offsets)
    level_group.write_bytes(VERTEX_GROUP_OFFSETS, key, paired_bytes)

    # Sidecar: total vertex count for this chunk (one int64).  Lets
    # ``chunk_local_to_global_offsets`` build the per-chunk → global
    # mapping in O(chunks) bytes of I/O instead of streaming every
    # vertex blob to discover its length.
    n_verts = int(sum(len(g) for g in groups))
    level_group.write_bytes(
        VERTEX_COUNTS, key, np.int64(n_verts).tobytes(),
    )

    return vertex_offsets


def read_chunk_vertex_count(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> int | None:
    """Read the per-chunk vertex count sidecar.

    Returns ``None`` when the sidecar is absent (legacy 0.2 stores).
    Callers should fall back to summing vertex_group_offsets in that
    case.
    """
    key = _chunk_key(chunk_coords)
    if not level_group.chunk_exists(VERTEX_COUNTS, key):
        return None
    raw = level_group.read_bytes(VERTEX_COUNTS, key)
    return int(np.frombuffer(raw, dtype=np.int64)[0])


def write_chunk_links(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    link_groups: list[npt.NDArray[np.integer]],
    dtype: np.dtype | str = np.int64,
) -> npt.NDArray[np.int64]:
    """Write link groups to a spatial chunk and update paired offsets.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        link_groups: List of arrays, each ``(M_k, L)``.
        dtype: Integer dtype.

    Returns:
        ``(K,)`` int64 array of link byte offsets.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)

    raw_bytes, link_offsets = encode_ragged_ints(link_groups, dtype)
    level_group.write_bytes(LINKS, key, raw_bytes)

    # Update the paired offsets to include link offsets
    if level_group.chunk_exists(VERTEX_GROUP_OFFSETS, key):
        existing = level_group.read_bytes(VERTEX_GROUP_OFFSETS, key)
        vertex_offsets, _ = decode_paired_offsets(existing)
        if len(vertex_offsets) != len(link_offsets):
            raise ArrayError(
                f"Link group count ({len(link_offsets)}) != "
                f"vertex group count ({len(vertex_offsets)}) in chunk {key}"
            )
        paired_bytes = encode_paired_offsets(vertex_offsets, link_offsets)
        level_group.write_bytes(VERTEX_GROUP_OFFSETS, key, paired_bytes)

    return link_offsets


def write_chunk_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype: np.dtype | str = np.float32,
) -> None:
    """Write vertex attribute data for groups in a spatial chunk.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"radius"``).
        chunk_coords: Spatial chunk coordinates.
        attr_groups: List of arrays aligned with vertex groups.
            Each array is ``(N_k,)`` for scalar or ``(N_k, C)`` for
            multi-channel attributes.
        dtype: Numpy dtype.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{ATTRIBUTES}/{attr_name}"
    raw_bytes, offsets = encode_vertex_groups(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)
    level_group.write_bytes(full_name, key + "_offsets", offsets.tobytes())


def write_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype: np.dtype | str = np.float32,
) -> None:
    """Write per-edge attribute data parallel to the links array.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        chunk_coords: Spatial chunk coordinates.
        attr_groups: List of arrays, each ``(M_k,)`` or ``(M_k, C)``,
            aligned with link groups.
        dtype: Numpy dtype.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{LINK_ATTRIBUTES}/{attr_name}"
    raw_bytes, _ = encode_vertex_groups(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)


def write_object_index(
    level_group: FsGroup,
    manifests: dict[int, ObjectManifest],
    sid_ndim: int,
) -> None:
    """Write object index: object_id → ordered vertex group references.

    Args:
        level_group: Resolution level group.
        manifests: ``{object_id: [(chunk_coords, vg_index), ...], ...}``.
            Object IDs must be contiguous starting from 0.
        sid_ndim: Number of spatial index dimensions.
    """
    if not manifests:
        return

    max_id = max(manifests.keys())
    # Build a dense list, filling gaps with empty manifests
    manifest_list: list[list[tuple[tuple[int, ...], int]]] = []
    for oid in range(max_id + 1):
        manifest_list.append(manifests.get(oid, []))

    raw_bytes, offsets = encode_object_index(manifest_list, sid_ndim)
    level_group.write_bytes(OBJECT_INDEX, "data", raw_bytes)
    level_group.write_bytes(OBJECT_INDEX, "offsets", offsets.tobytes())
    level_group.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
        "num_objects": max_id + 1,
        "sid_ndim": sid_ndim,
    })


def write_object_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
) -> None:
    """Write dense O×C object attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(O,)`` or ``(O, C)`` array.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_bytes(full_name, "data", data.tobytes())
    level_group.write_array_meta(full_name, {
        "zv_array": "object_attribute",
        "name": attr_name,
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    })


def write_groupings(
    level_group: FsGroup,
    groups: dict[int, list[int]],
) -> None:
    """Write group memberships: group_id → list of object_ids.

    Args:
        level_group: Resolution level group.
        groups: ``{group_id: [object_id, ...], ...}``.
            Group IDs must be contiguous starting from 0.
    """
    if not groups:
        return

    max_gid = max(groups.keys())
    group_list: list[npt.NDArray] = []
    for gid in range(max_gid + 1):
        members = groups.get(gid, [])
        group_list.append(np.array(members, dtype=np.int64))

    raw_bytes, offsets = encode_ragged_ints(group_list, dtype=np.dtype(np.int64))
    level_group.write_bytes(GROUPINGS, "data", raw_bytes)
    level_group.write_bytes(GROUPINGS, "offsets", offsets.tobytes())
    level_group.write_array_meta(GROUPINGS, {
        "zv_array": "groupings",
        "num_groups": max_gid + 1,
    })


def write_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
) -> None:
    """Write dense G×C groupings attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(G,)`` or ``(G, C)`` array.
    """
    full_name = f"{GROUPINGS_ATTRIBUTES}/{attr_name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_bytes(full_name, "data", data.tobytes())
    level_group.write_array_meta(full_name, {
        "zv_array": "groupings_attribute",
        "name": attr_name,
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    })


def write_cross_chunk_links(
    level_group: FsGroup,
    links: list[CrossChunkLink],
    sid_ndim: int,
) -> None:
    """Write cross-chunk link pairs.

    Each link is ``((chunk_A, vertex_A), (chunk_B, vertex_B))``.

    Args:
        level_group: Resolution level group.
        links: List of CrossChunkLink tuples.
        sid_ndim: Number of spatial index dimensions.
    """
    if not links:
        return

    # Each link → 2 * (sid_ndim + 1) ints
    entry_len = 2 * (sid_ndim + 1)
    flat: list[int] = []
    for (chunk_a, vi_a), (chunk_b, vi_b) in links:
        flat.extend(chunk_a)
        flat.append(vi_a)
        flat.extend(chunk_b)
        flat.append(vi_b)

    arr = np.array(flat, dtype=np.int64)
    level_group.write_bytes(CROSS_CHUNK_LINKS, "data", arr.tobytes())
    level_group.write_array_meta(CROSS_CHUNK_LINKS, {
        "zv_array": "cross_chunk_links",
        "num_links": len(links),
        "sid_ndim": sid_ndim,
    })


def write_metanode_children(
    level_group: FsGroup,
    children: dict[int, list[VertexGroupRef]],
    sid_ndim: int,
) -> None:
    """Write metanode → child vertex references for drill-down.

    Args:
        level_group: Resolution level group.
        children: ``{metanode_id: [(chunk_coords, vertex_index), ...], ...}``.
        sid_ndim: Number of spatial index dimensions.
    """
    if not children:
        return

    max_id = max(children.keys())
    child_list: list[list[tuple[tuple[int, ...], int]]] = []
    for mid in range(max_id + 1):
        child_list.append(children.get(mid, []))

    raw_bytes, offsets = encode_object_index(child_list, sid_ndim)
    level_group.write_bytes(METANODE_CHILDREN, "data", raw_bytes)
    level_group.write_bytes(METANODE_CHILDREN, "offsets", offsets.tobytes())
    level_group.write_array_meta(METANODE_CHILDREN, {
        "zv_array": "metanode_children",
        "num_metanodes": max_id + 1,
        "sid_ndim": sid_ndim,
    })


# ===================================================================
# Reading data
# ===================================================================

def read_chunk_vertices(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
) -> list[npt.NDArray[np.floating]]:
    """Read all vertex groups from a spatial chunk.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions (D).

    Returns:
        List of arrays, each ``(N_k, D)``.

    Raises:
        ArrayError: If the chunk does not exist or data is malformed.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    try:
        raw = level_group.read_bytes(VERTICES, key)
    except Exception as e:
        raise ArrayError(f"Cannot read vertices chunk {key}: {e}") from e

    offsets = _read_vertex_offsets(level_group, chunk_coords)
    return decode_vertex_groups(raw, offsets, dtype, ndim)


def read_vertex_group(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    vg_index: int,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
) -> npt.NDArray[np.floating]:
    """Read a single vertex group using byte offsets for efficient access.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        vg_index: Index of the vertex group within the chunk.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions.

    Returns:
        Array of shape ``(N, D)``.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    raw = level_group.read_bytes(VERTICES, key)
    offsets = _read_vertex_offsets(level_group, chunk_coords)

    if vg_index < 0 or vg_index >= len(offsets):
        raise ArrayError(
            f"Vertex group index {vg_index} out of range "
            f"(chunk {key} has {len(offsets)} groups)"
        )

    start = int(offsets[vg_index])
    end = int(offsets[vg_index + 1]) if vg_index + 1 < len(offsets) else len(raw)
    segment = raw[start:end]

    arr = np.frombuffer(segment, dtype=dtype)
    if ndim > 1:
        arr = arr.reshape(-1, ndim)
    return arr


def read_chunk_links(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.int64,
    link_width: int | None = None,
) -> list[npt.NDArray[np.integer]]:
    """Read all link groups from a spatial chunk.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        dtype: Integer dtype.
        link_width: Number of columns per link (L). If None, read from
            array metadata.

    Returns:
        List of arrays, each ``(M_k, L)``.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    if link_width is None:
        meta = level_group.read_array_meta(LINKS)
        link_width = meta.get("link_width", 2)

    try:
        raw = level_group.read_bytes(LINKS, key)
    except Exception as e:
        raise ArrayError(f"Cannot read links chunk {key}: {e}") from e

    link_offsets = _read_link_offsets(level_group, chunk_coords)
    return decode_ragged_ints(raw, link_offsets, dtype, ncols=link_width)


def read_chunk_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ncols: int = 1,
) -> list[npt.NDArray]:
    """Read vertex attribute data for a chunk.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype.
        ncols: Number of columns (channels). Use 1 for scalars.

    Returns:
        List of arrays aligned with vertex groups.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = f"{ATTRIBUTES}/{attr_name}"

    try:
        raw = level_group.read_bytes(full_name, key)
    except Exception as e:
        raise ArrayError(
            f"Cannot read attribute '{attr_name}' chunk {key}: {e}"
        ) from e

    try:
        raw_offsets = level_group.read_bytes(full_name, key + "_offsets")
        attr_offsets = np.frombuffer(raw_offsets, dtype=np.int64)
    except Exception:
        attr_offsets = np.array([0], dtype=np.int64)

    return decode_vertex_groups(raw, attr_offsets, dtype, ncols)


def read_object_manifest(
    level_group: FsGroup,
    object_id: int,
) -> ObjectManifest:
    """Read the ordered vertex group reference list for one object.

    Folds pending sidecars on read.

    Args:
        level_group: Resolution level group.
        object_id: Object ID.

    Returns:
        List of ``(chunk_coords, vg_index)`` tuples.
    """
    # Cheap path when no pending sidecars exist — preserve the original
    # bounds check / error semantics.
    pending_batches = _list_pending_batches(level_group)
    meta = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = meta["sid_ndim"]
    num_objects = meta["num_objects"]

    if not pending_batches:
        if object_id < 0 or object_id >= num_objects:
            raise ArrayError(
                f"Object ID {object_id} out of range [0, {num_objects})"
            )
        raw = level_group.read_bytes(OBJECT_INDEX, "data")
        offsets = np.frombuffer(
            level_group.read_bytes(OBJECT_INDEX, "offsets"),
            dtype=np.int64,
        )
        all_manifests = decode_object_index(raw, offsets, sid_ndim)
        return all_manifests[object_id]

    # With pending: union and look up.
    merged = read_all_object_manifests(level_group)
    if object_id < 0 or object_id >= len(merged):
        raise ArrayError(
            f"Object ID {object_id} out of range [0, {len(merged)})"
        )
    return merged[object_id]


def read_all_object_manifests(
    level_group: FsGroup,
) -> list[ObjectManifest]:
    """Read all object manifests at once, folding any pending sidecars.

    Returns:
        List indexed by object_id, each a list of ``(chunk_coords, vg_index)``.
    """
    meta = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = meta["sid_ndim"]

    raw = level_group.read_bytes(OBJECT_INDEX, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(OBJECT_INDEX, "offsets"),
        dtype=np.int64,
    )
    main = list(decode_object_index(raw, offsets, sid_ndim))

    # Fold pending sidecars in batch-id order (later batches overwrite
    # earlier ones for the same oid).  Capability:
    # ``CAP_OBJECT_INDEX_PENDING``; absence means no sidecars to merge.
    pending = read_object_index_pending(level_group)
    for oid, manifest in pending:
        while oid >= len(main):
            main.append([])
        main[oid] = manifest
    return main


# ---------------- pending sidecars (incremental append) -------------------


_PENDING_PREFIX = f"{OBJECT_INDEX}/pending"


def _list_pending_batches(level_group: FsGroup) -> list[int]:
    """List the batch IDs of pending object_index sidecars in order."""
    if not level_group.array_exists(_PENDING_PREFIX):
        return []
    try:
        pending_grp = level_group[_PENDING_PREFIX]
    except Exception:
        return []
    batches: list[int] = []
    for name in pending_grp:
        try:
            batches.append(int(name))
        except ValueError:
            continue
    return sorted(batches)


def next_pending_batch_id(level_group: FsGroup) -> int:
    """Return the next free batch ID for a pending object_index sidecar."""
    existing = _list_pending_batches(level_group)
    return (existing[-1] + 1) if existing else 0


def write_object_index_pending(
    level_group: FsGroup,
    manifests: dict[int, ObjectManifest],
    sid_ndim: int,
    *,
    batch_id: int | None = None,
) -> int:
    """Write a pending object_index sidecar batch.

    Pending sidecars are union-folded by :func:`read_all_object_manifests`
    and collapsed into the main array by :func:`compact_object_index`.

    Args:
        level_group: Resolution level group.
        manifests: ``{object_id: [(chunk_coords, vg_index), ...]}`` —
            sparse; only the OIDs in the dict are written.
        sid_ndim: Number of spatial index dimensions (matches the
            main index's ``sid_ndim``).
        batch_id: Force a specific batch id.  ``None`` picks the next
            unused id.

    Returns:
        The batch id written.
    """
    if not manifests:
        return -1
    if batch_id is None:
        batch_id = next_pending_batch_id(level_group)

    oids = sorted(manifests.keys())
    sparse_list = [manifests[oid] for oid in oids]
    raw_bytes, offsets = encode_object_index(sparse_list, sid_ndim)

    base = f"{_PENDING_PREFIX}/{batch_id}"
    level_group.write_bytes(base, "oids", np.asarray(oids, dtype=np.int64).tobytes())
    level_group.write_bytes(base, "data", raw_bytes)
    level_group.write_bytes(base, "offsets", offsets.tobytes())
    level_group.write_array_meta(base, {
        "zv_array": "object_index_pending",
        "batch_id": batch_id,
        "num_objects": len(oids),
        "sid_ndim": sid_ndim,
    })
    return batch_id


def read_object_index_pending(
    level_group: FsGroup,
) -> list[tuple[int, ObjectManifest]]:
    """Read every pending sidecar in ascending batch order.

    Returns:
        Flat list of ``(object_id, manifest)`` pairs.  Same ``oid`` may
        appear multiple times when the user committed several batches
        for the same object — caller decides resolution policy (the
        standard reader uses last-write-wins).
    """
    batches = _list_pending_batches(level_group)
    out: list[tuple[int, ObjectManifest]] = []
    for batch_id in batches:
        base = f"{_PENDING_PREFIX}/{batch_id}"
        try:
            meta = level_group.read_array_meta(base)
        except Exception:
            continue
        sid_ndim = int(meta["sid_ndim"])
        oids = np.frombuffer(
            level_group.read_bytes(base, "oids"), dtype=np.int64,
        )
        raw = level_group.read_bytes(base, "data")
        offsets = np.frombuffer(
            level_group.read_bytes(base, "offsets"), dtype=np.int64,
        )
        decoded = decode_object_index(raw, offsets, sid_ndim)
        for oid, manifest in zip(oids.tolist(), decoded):
            out.append((int(oid), manifest))
    return out


def write_cross_chunk_faces(
    level_group: FsGroup,
    cross_faces: list[list[tuple[ChunkCoords, int]]],
    sid_ndim: int,
) -> None:
    """Persist face-identity for faces that span multiple chunks.

    Each face is a list of ``L`` ``(chunk_coords, local_vertex_index)``
    records — ``L = 3`` for triangles, ``L = 4`` for quads, etc.  The
    on-disk record packs ``ndim + 2`` int64 values: the chunk
    coordinates, a ``vg_idx`` slot (always 0 today for mesh writers
    that emit one vertex-group per chunk), and the in-group local
    vertex index.

    Writers that don't care about face identity can leave the
    edge-pair decomposition in :data:`CROSS_CHUNK_LINKS` and skip this
    array entirely — readers that ignore the new array still get
    connectivity through the existing edges.

    Capability token: :data:`CAP_CROSS_CHUNK_FACES`.
    """
    if not cross_faces:
        return
    record_size = sid_ndim + 2
    offsets_list: list[int] = [0]
    flat: list[int] = []
    for face in cross_faces:
        for cc, local_idx in face:
            if len(cc) != sid_ndim:
                raise ArrayError(
                    f"chunk_coords length {len(cc)} != sid_ndim {sid_ndim}"
                )
            flat.extend(int(c) for c in cc)
            flat.append(0)  # vg_idx (forward-compat slot)
            flat.append(int(local_idx))
        offsets_list.append(len(flat) // record_size)
    data = np.asarray(flat, dtype=np.int64)
    offsets = np.asarray(offsets_list, dtype=np.int64)
    level_group.write_bytes(CROSS_CHUNK_FACES, "data", data.tobytes())
    level_group.write_bytes(CROSS_CHUNK_FACES, "offsets", offsets.tobytes())
    level_group.write_array_meta(CROSS_CHUNK_FACES, {
        "zv_array": "cross_chunk_faces",
        "num_faces": len(cross_faces),
        "sid_ndim": sid_ndim,
        "record_size": record_size,
    })


def read_cross_chunk_faces(
    level_group: FsGroup,
) -> list[list[tuple[ChunkCoords, int]]]:
    """Read face-identity records for cross-chunk faces.

    Returns ``[]`` when the array is absent (older 0.2 stores or 0.3
    stores without the :data:`CAP_CROSS_CHUNK_FACES` capability).
    """
    if not level_group.array_exists(CROSS_CHUNK_FACES):
        return []
    try:
        meta = level_group.read_array_meta(CROSS_CHUNK_FACES)
    except Exception:
        return []
    sid_ndim = int(meta["sid_ndim"])
    record_size = int(meta.get("record_size", sid_ndim + 2))
    data = np.frombuffer(
        level_group.read_bytes(CROSS_CHUNK_FACES, "data"), dtype=np.int64,
    )
    offsets = np.frombuffer(
        level_group.read_bytes(CROSS_CHUNK_FACES, "offsets"), dtype=np.int64,
    )
    n_faces = len(offsets) - 1
    out: list[list[tuple[ChunkCoords, int]]] = []
    for i in range(n_faces):
        start, end = int(offsets[i]), int(offsets[i + 1])
        face: list[tuple[ChunkCoords, int]] = []
        for r in range(start, end):
            record = data[r * record_size:(r + 1) * record_size]
            cc = tuple(int(x) for x in record[:sid_ndim])
            local_idx = int(record[sid_ndim + 1])
            face.append((cc, local_idx))
        out.append(face)
    return out


def compact_object_index(level_group: FsGroup) -> dict[str, int]:
    """Fold every pending object_index sidecar into the main array.

    Reads the main index + all pending batches, applies last-write-wins
    on duplicate oids, rewrites :data:`OBJECT_INDEX`, and deletes the
    pending sidecars.

    Args:
        level_group: Resolution level group.

    Returns:
        Summary dict with ``batches_folded`` and ``num_objects``.
    """
    if not level_group.array_exists(_PENDING_PREFIX):
        return {"batches_folded": 0, "num_objects": 0}
    batches = _list_pending_batches(level_group)
    if not batches:
        return {"batches_folded": 0, "num_objects": 0}

    merged = read_all_object_manifests(level_group)  # already folds pending
    main_meta = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = int(main_meta["sid_ndim"])

    manifests = {oid: m for oid, m in enumerate(merged)}
    write_object_index(level_group, manifests, sid_ndim=sid_ndim)

    # Remove pending tree.
    level_group.delete_subtree(_PENDING_PREFIX)

    return {"batches_folded": len(batches), "num_objects": len(merged)}


def read_object_vertices(
    level_group: FsGroup,
    object_id: int,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
) -> list[npt.NDArray[np.floating]]:
    """Read all vertex data for an object by following its manifest.

    Args:
        level_group: Resolution level group.
        object_id: Object ID.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions.

    Returns:
        List of vertex group arrays in reconstruction order.
    """
    manifest = read_object_manifest(level_group, object_id)
    groups: list[npt.NDArray] = []
    for chunk_coords, vg_index in manifest:
        vg = read_vertex_group(level_group, chunk_coords, vg_index,
                               dtype=dtype, ndim=ndim)
        groups.append(vg)
    return groups


def read_object_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
) -> npt.NDArray:
    """Read dense O×C object attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        dtype: Override dtype. If None, read from metadata.

    Returns:
        Array of shape ``(O,)`` or ``(O, C)``.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    meta = level_group.read_array_meta(full_name)
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    shape = tuple(meta["shape"])

    raw = level_group.read_bytes(full_name, "data")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def read_group_object_ids(
    level_group: FsGroup,
    group_id: int,
) -> list[int]:
    """Read the list of object IDs belonging to a group.

    Args:
        level_group: Resolution level group.
        group_id: Group ID.

    Returns:
        List of object ID integers.
    """
    meta = level_group.read_array_meta(GROUPINGS)
    num_groups = meta["num_groups"]

    if group_id < 0 or group_id >= num_groups:
        raise ArrayError(
            f"Group ID {group_id} out of range [0, {num_groups})"
        )

    raw = level_group.read_bytes(GROUPINGS, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(GROUPINGS, "offsets"),
        dtype=np.int64,
    )

    all_groups = decode_ragged_ints(raw, offsets, dtype=np.dtype(np.int64), ncols=1)
    return all_groups[group_id].tolist()


def read_all_groupings(
    level_group: FsGroup,
) -> list[list[int]]:
    """Read all group memberships.

    Returns:
        List indexed by group_id, each a list of object_id ints.
    """
    meta = level_group.read_array_meta(GROUPINGS)

    raw = level_group.read_bytes(GROUPINGS, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(GROUPINGS, "offsets"),
        dtype=np.int64,
    )

    all_groups = decode_ragged_ints(raw, offsets, dtype=np.dtype(np.int64), ncols=1)
    return [g.tolist() for g in all_groups]


def read_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
) -> npt.NDArray:
    """Read dense G×C groupings attribute data."""
    full_name = f"{GROUPINGS_ATTRIBUTES}/{attr_name}"
    meta = level_group.read_array_meta(full_name)
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    shape = tuple(meta["shape"])

    raw = level_group.read_bytes(full_name, "data")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def read_cross_chunk_links(
    level_group: FsGroup,
) -> list[CrossChunkLink]:
    """Read all cross-chunk links.

    Returns:
        List of ``((chunk_A, vertex_A), (chunk_B, vertex_B))`` tuples.
    """
    meta = level_group.read_array_meta(CROSS_CHUNK_LINKS)
    num_links = meta["num_links"]
    sid_ndim = meta["sid_ndim"]

    raw = level_group.read_bytes(CROSS_CHUNK_LINKS, "data")
    arr = np.frombuffer(raw, dtype=np.int64)

    entry_len = 2 * (sid_ndim + 1)
    half = sid_ndim + 1
    links: list[CrossChunkLink] = []

    for i in range(0, len(arr), entry_len):
        chunk_a = tuple(int(x) for x in arr[i : i + sid_ndim])
        vi_a = int(arr[i + sid_ndim])
        chunk_b = tuple(int(x) for x in arr[i + half : i + half + sid_ndim])
        vi_b = int(arr[i + half + sid_ndim])
        links.append(((chunk_a, vi_a), (chunk_b, vi_b)))

    return links


def read_metanode_children(
    level_group: FsGroup,
    metanode_id: int | None = None,
) -> dict[int, list[VertexGroupRef]] | list[VertexGroupRef]:
    """Read metanode children references.

    Args:
        level_group: Resolution level group.
        metanode_id: If given, return children for this metanode only.
            If None, return all as a dict.

    Returns:
        If metanode_id given: list of ``(chunk_coords, vertex_index)``.
        If None: dict mapping metanode_id → list of refs.
    """
    meta = level_group.read_array_meta(METANODE_CHILDREN)
    sid_ndim = meta["sid_ndim"]

    raw = level_group.read_bytes(METANODE_CHILDREN, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(METANODE_CHILDREN, "offsets"),
        dtype=np.int64,
    )

    all_children = decode_object_index(raw, offsets, sid_ndim)

    if metanode_id is not None:
        if metanode_id < 0 or metanode_id >= len(all_children):
            raise ArrayError(
                f"Metanode ID {metanode_id} out of range [0, {len(all_children)})"
            )
        return all_children[metanode_id]

    return {i: c for i, c in enumerate(all_children)}


# ===================================================================
# Listing / introspection
# ===================================================================

def list_chunk_keys(
    level_group: FsGroup,
    array_name: str = VERTICES,
) -> list[ChunkCoords]:
    """List all chunk coordinates that have data for an array.

    Args:
        level_group: Resolution level group.
        array_name: Array name (default: ``"vertices"``).

    Returns:
        Sorted list of chunk coordinate tuples.
    """
    keys = level_group.list_chunks(array_name)
    coords: list[ChunkCoords] = []
    for k in keys:
        try:
            coords.append(_parse_chunk_key(k))
        except ValueError:
            continue  # skip non-chunk files (e.g. .zattrs)
    return sorted(coords)


def count_vertex_groups(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> int:
    """Count vertex groups in a chunk (from offsets array)."""
    offsets = _read_vertex_offsets(level_group, chunk_coords)
    return len(offsets)


# ===================================================================
# Internal helpers
# ===================================================================

def _read_vertex_offsets(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> npt.NDArray[np.int64]:
    """Read the vertex byte offsets from vertex_group_offsets for a chunk."""
    key = _chunk_key(chunk_coords)
    raw = level_group.read_bytes(VERTEX_GROUP_OFFSETS, key)
    vertex_offsets, _ = decode_paired_offsets(raw)
    return vertex_offsets


def _read_link_offsets(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> npt.NDArray[np.int64]:
    """Read the link byte offsets from vertex_group_offsets for a chunk."""
    key = _chunk_key(chunk_coords)
    raw = level_group.read_bytes(VERTEX_GROUP_OFFSETS, key)
    _, link_offsets = decode_paired_offsets(raw)
    return link_offsets


def _vertex_group_counts(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    vert_dtype: np.dtype,
) -> list[int]:
    """Compute vertex count per group from offsets and vertex data size.

    Returns list of vertex counts, one per group.
    """
    key = _chunk_key(chunk_coords)
    raw = level_group.read_bytes(VERTICES, key)
    total_bytes = len(raw)
    offsets = _read_vertex_offsets(level_group, chunk_coords)

    # Read ndim from vertex metadata
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        # We don't store ndim explicitly, so infer from first group
    except Exception:
        pass

    counts: list[int] = []
    for i in range(len(offsets)):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else total_bytes
        nbytes = end - start
        # Each vertex is vert_dtype.itemsize * ndim bytes
        # But we don't know ndim here — just count raw elements
        n_elements = nbytes // vert_dtype.itemsize
        counts.append(n_elements)

    return counts
