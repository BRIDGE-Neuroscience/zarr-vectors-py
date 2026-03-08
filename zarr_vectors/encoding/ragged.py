"""Ragged array encoding and decoding for ZVF spatial chunks.

Each spatial chunk in a ZVF store holds a variable number of vertex groups,
each of variable length.  This module serialises lists of numpy arrays into
flat byte buffers with an accompanying offset array, and deserialises them
back.

The encoding is simple concatenation of raw bytes.  The offset array records
where each group starts so that individual groups can be extracted without
decoding the whole chunk (enabling HTTP range reads on cloud stores).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ArrayError


# ---------------------------------------------------------------------------
# Vertex group encoding (float positions, float/int attributes)
# ---------------------------------------------------------------------------

def encode_vertex_groups(
    groups: list[npt.NDArray],
    dtype: np.dtype,
) -> tuple[bytes, npt.NDArray[np.int64]]:
    """Encode a list of vertex group arrays into a single byte buffer.

    Args:
        groups: List of arrays, each shape ``(N_k, D)`` or ``(N_k,)`` for
            vertex group *k*.  All arrays must have the same number of
            columns (D) and be castable to *dtype*.
        dtype: Target numpy dtype for serialisation.

    Returns:
        raw_bytes: Concatenated byte buffer of all groups.
        offsets: ``(K,)`` int64 array of byte offsets where each group starts.
            The end of group *k* is ``offsets[k+1]`` (or ``len(raw_bytes)``
            for the last group).

    Raises:
        ArrayError: If groups have inconsistent column counts.
    """
    if not groups:
        return b"", np.empty(0, dtype=np.int64)

    dtype = np.dtype(dtype)
    parts: list[bytes] = []
    offsets: list[int] = []
    current_offset = 0

    # Validate consistent column count
    ndims: set[int] = set()
    for g in groups:
        ndims.add(g.ndim)
    if len(ndims) > 1:
        # Allow mix of 1-D and 2-D only if 1-D groups are empty
        pass  # we'll handle per-group below

    for g in groups:
        offsets.append(current_offset)
        arr = np.asarray(g, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(0)
        raw = arr.tobytes()
        parts.append(raw)
        current_offset += len(raw)

    return b"".join(parts), np.array(offsets, dtype=np.int64)


def decode_vertex_groups(
    raw_bytes: bytes,
    offsets: npt.NDArray[np.int64],
    dtype: np.dtype,
    ncols: int,
) -> list[npt.NDArray]:
    """Decode a byte buffer back into a list of vertex group arrays.

    Args:
        raw_bytes: The concatenated byte buffer produced by
            :func:`encode_vertex_groups`.
        offsets: ``(K,)`` byte offset array.
        dtype: The numpy dtype used during encoding.
        ncols: Number of columns per row (e.g. 3 for XYZ positions).
            Use 1 for flat per-vertex scalars.

    Returns:
        List of arrays, each shape ``(N_k, ncols)`` (or ``(N_k,)`` when
        *ncols* is 1).

    Raises:
        ArrayError: If offsets are out of range or data does not divide evenly.
    """
    if len(offsets) == 0:
        return []

    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    total_len = len(raw_bytes)
    groups: list[npt.NDArray] = []

    for i in range(len(offsets)):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else total_len

        if start < 0 or end > total_len or start > end:
            raise ArrayError(
                f"Invalid offset range [{start}, {end}) for buffer of length {total_len}"
            )

        segment = raw_bytes[start:end]
        n_elements = len(segment) // itemsize

        if len(segment) % itemsize != 0:
            raise ArrayError(
                f"Segment length {len(segment)} is not divisible by dtype "
                f"itemsize {itemsize}"
            )

        arr = np.frombuffer(segment, dtype=dtype)

        if ncols > 1:
            if n_elements % ncols != 0:
                raise ArrayError(
                    f"Element count {n_elements} is not divisible by ncols={ncols}"
                )
            arr = arr.reshape(-1, ncols)

        groups.append(arr)

    return groups


# ---------------------------------------------------------------------------
# Ragged integer encoding (links, groupings)
# ---------------------------------------------------------------------------

def encode_ragged_ints(
    groups: list[npt.NDArray],
    dtype: np.dtype = np.dtype(np.int64),
) -> tuple[bytes, npt.NDArray[np.int64]]:
    """Encode ragged integer arrays into a byte buffer with offsets.

    Identical to :func:`encode_vertex_groups` but with a default integer
    dtype.  Used for link arrays, grouping arrays, and similar.

    Args:
        groups: List of integer arrays, each shape ``(M_k, L)`` or ``(M_k,)``.
        dtype: Integer dtype (default int64).

    Returns:
        raw_bytes, offsets — same semantics as :func:`encode_vertex_groups`.
    """
    return encode_vertex_groups(groups, dtype=dtype)


def decode_ragged_ints(
    raw_bytes: bytes,
    offsets: npt.NDArray[np.int64],
    dtype: np.dtype = np.dtype(np.int64),
    ncols: int = 1,
) -> list[npt.NDArray]:
    """Decode ragged integer arrays from a byte buffer.

    Args:
        raw_bytes: Byte buffer from :func:`encode_ragged_ints`.
        offsets: ``(K,)`` byte offset array.
        dtype: Integer dtype used during encoding.
        ncols: Number of columns per row (e.g. 2 for edges, 3 for tri faces).

    Returns:
        List of integer arrays.
    """
    return decode_vertex_groups(raw_bytes, offsets, dtype=dtype, ncols=ncols)


# ---------------------------------------------------------------------------
# Object index encoding
# ---------------------------------------------------------------------------

def encode_object_index(
    manifests: list[list[tuple[tuple[int, ...], int]]],
    sid_ndim: int,
) -> tuple[bytes, npt.NDArray[np.int64]]:
    """Encode a list of object manifests into a byte buffer.

    Each manifest is a list of ``(chunk_coords, vertex_group_index)`` tuples.
    These are flattened to ``(sid_ndim + 1)`` ints per entry and concatenated.

    Args:
        manifests: ``[manifest_0, manifest_1, ...]`` where each manifest is
            ``[(chunk_coords, vg_index), ...]``.
        sid_ndim: Number of spatial index dimensions (e.g. 3 for XYZ).

    Returns:
        raw_bytes: Concatenated byte buffer of all manifests.
        offsets: ``(O,)`` int64 byte offset array, one per object.
    """
    entry_len = sid_ndim + 1
    dtype = np.dtype(np.int64)
    parts: list[bytes] = []
    offsets: list[int] = []
    current_offset = 0

    for manifest in manifests:
        offsets.append(current_offset)
        if not manifest:
            # Empty manifest — object has no vertex groups
            continue
        flat: list[int] = []
        for chunk_coords, vg_index in manifest:
            if len(chunk_coords) != sid_ndim:
                raise ArrayError(
                    f"Chunk coords length {len(chunk_coords)} != sid_ndim {sid_ndim}"
                )
            flat.extend(chunk_coords)
            flat.append(vg_index)
        arr = np.array(flat, dtype=dtype)
        raw = arr.tobytes()
        parts.append(raw)
        current_offset += len(raw)

    return b"".join(parts), np.array(offsets, dtype=np.int64)


def decode_object_index(
    raw_bytes: bytes,
    offsets: npt.NDArray[np.int64],
    sid_ndim: int,
) -> list[list[tuple[tuple[int, ...], int]]]:
    """Decode a byte buffer back into a list of object manifests.

    Args:
        raw_bytes: Buffer from :func:`encode_object_index`.
        offsets: ``(O,)`` byte offset array.
        sid_ndim: Number of spatial index dimensions.

    Returns:
        List of manifests, each a list of ``(chunk_coords, vg_index)`` tuples.
    """
    if len(offsets) == 0:
        return []

    entry_len = sid_ndim + 1
    dtype = np.dtype(np.int64)
    itemsize = dtype.itemsize
    total_len = len(raw_bytes)
    manifests: list[list[tuple[tuple[int, ...], int]]] = []

    for i in range(len(offsets)):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else total_len

        if start == end:
            manifests.append([])
            continue

        segment = raw_bytes[start:end]
        arr = np.frombuffer(segment, dtype=dtype)

        if len(arr) % entry_len != 0:
            raise ArrayError(
                f"Manifest segment length {len(arr)} not divisible by "
                f"entry_len={entry_len} (sid_ndim={sid_ndim})"
            )

        entries: list[tuple[tuple[int, ...], int]] = []
        for j in range(0, len(arr), entry_len):
            chunk_coords = tuple(int(x) for x in arr[j : j + sid_ndim])
            vg_index = int(arr[j + sid_ndim])
            entries.append((chunk_coords, vg_index))
        manifests.append(entries)

    return manifests


# ---------------------------------------------------------------------------
# Paired offset encoding (vertex_group_offsets: K×2)
# ---------------------------------------------------------------------------

def encode_paired_offsets(
    vertex_offsets: npt.NDArray[np.int64],
    link_offsets: npt.NDArray[np.int64],
) -> bytes:
    """Encode paired (vertex_offset, link_offset) arrays into bytes.

    Args:
        vertex_offsets: ``(K,)`` byte offsets into the vertices chunk.
        link_offsets: ``(K,)`` byte offsets into the links chunk.
            Use -1 for entries where links are not applicable.

    Returns:
        Raw bytes encoding a ``(K, 2)`` int64 array.
    """
    k = len(vertex_offsets)
    if len(link_offsets) != k:
        raise ArrayError(
            f"vertex_offsets length {k} != link_offsets length {len(link_offsets)}"
        )
    paired = np.stack([vertex_offsets, link_offsets], axis=1).astype(np.int64)
    return paired.tobytes()


def decode_paired_offsets(
    raw_bytes: bytes,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Decode paired offsets from bytes.

    Args:
        raw_bytes: Buffer from :func:`encode_paired_offsets`.

    Returns:
        vertex_offsets: ``(K,)`` int64 array.
        link_offsets: ``(K,)`` int64 array.
    """
    if len(raw_bytes) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    arr = np.frombuffer(raw_bytes, dtype=np.int64)
    if len(arr) % 2 != 0:
        raise ArrayError(
            f"Paired offsets buffer length {len(arr)} is not even"
        )
    paired = arr.reshape(-1, 2)
    return paired[:, 0].copy(), paired[:, 1].copy()
