"""Async-batched chunk reader.

Symmetric to :mod:`zarr_vectors.core._batch_writer`.  Where the writer
turns ``N`` serial ``store.set`` PUTs into one :func:`asyncio.gather`,
this module does the same for ``store.get`` reads — collapsing the
round-trip cost of an ``N``-chunk read-all from ``O(N)`` to ``O(1)``
against high-latency object stores.

The single entry point is :func:`flush_prefetch`, driven by the
:meth:`zarr_vectors.core.group.Group.batched_reads` context manager.
The caller supplies a ``plan`` of ``(array_name, [chunk_keys, ...])``
pairs; this module loads every requested chunk in one async gather and
returns a ``{(array_name, chunk_key): bytes}`` cache that the Group
serves :meth:`read_bytes` calls from while the context is active.

The reads go through zarr's :class:`AsyncArray.getitem` so any codec
pipeline (BytesCodec only, BytesCodec+Zstd, future additions) is
honoured per-chunk.  Icechunk-backed stores fall back to the
synchronous :meth:`read_bytes` path on a per-chunk basis because
icechunk tracks arrays as session-managed entities and the async
gather pattern bypasses that contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import zarr
from zarr.core.sync import sync


def _is_icechunk_store(store: Any) -> bool:
    """Return True when ``store`` is an icechunk-backed Store.

    Detection is by class name to avoid importing icechunk at module
    load time.  Mirrors the check in
    :mod:`zarr_vectors.core._batch_writer`.
    """
    cls = type(store)
    return cls.__name__ == "IcechunkStore" or cls.__module__.startswith("icechunk")


async def _async_get_chunk(
    async_group: Any,
    path: str,
) -> bytes:
    """Resolve ``path`` to an inner array and return its bytes."""
    try:
        node = await async_group.getitem(path)
    except KeyError:
        return None  # signals "missing"; caller skips this entry
    if not isinstance(node, zarr.AsyncArray):
        return None
    if node.shape[0] == 0:
        return b""
    data = await node.getitem(slice(None))
    return bytes(np.asarray(data).tobytes())


async def _gather_chunks(
    async_group: Any,
    paths: list[str],
) -> list[bytes | None]:
    """``await asyncio.gather`` every ``_async_get_chunk(path)`` in one shot."""
    return await asyncio.gather(
        *(_async_get_chunk(async_group, p) for p in paths)
    )


def flush_prefetch(
    zarr_group: zarr.Group,
    plan: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], bytes]:
    """Prefetch every chunk in ``plan`` and return a flat cache.

    ``plan`` is a list of ``(array_name, [chunk_keys, ...])`` tuples.
    Each ``chunk_key`` is the standard chunk_key string (e.g.
    ``"0.1.2"``) used by :meth:`Group.read_bytes`.  Returns a dict keyed
    by ``(array_name, chunk_key)`` whose values are the decoded chunk
    bytes; missing chunks are omitted (the caller falls through to the
    sync ``read_bytes`` path on a cache miss).

    For icechunk-backed stores, falls back to serial sync reads via
    :func:`_sync_fallback` — the async-gather pattern bypasses
    icechunk's session-tracking contract.
    """
    if not plan:
        return {}

    if _is_icechunk_store(zarr_group.store):
        return _sync_fallback(zarr_group, plan)

    # Flatten plan into (array_name, chunk_key, path) triples.  Path is
    # the AsyncGroup-relative key (e.g. "vertices/0.1.2") that resolves
    # to the inner per-chunk array.
    flat: list[tuple[str, str, str]] = []
    for array_name, chunk_keys in plan:
        for chunk_key in chunk_keys:
            flat.append((array_name, chunk_key, f"{array_name}/{chunk_key}"))

    if not flat:
        return {}

    paths = [p for _, _, p in flat]
    results = sync(_gather_chunks(zarr_group._async_group, paths))

    cache: dict[tuple[str, str], bytes] = {}
    for (array_name, chunk_key, _), data in zip(flat, results):
        if data is not None:
            cache[(array_name, chunk_key)] = data
    return cache


def _sync_fallback(
    zarr_group: zarr.Group,
    plan: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], bytes]:
    """Serial-read fallback for icechunk and other stores that don't
    play well with the async-gather pattern.

    Walks the plan one entry at a time using sync zarr access, returning
    the same dict shape :func:`flush_prefetch` does.
    """
    cache: dict[tuple[str, str], bytes] = {}
    for array_name, chunk_keys in plan:
        try:
            arr_group = zarr_group[array_name]
        except KeyError:
            continue
        for chunk_key in chunk_keys:
            try:
                arr = arr_group[chunk_key]
            except KeyError:
                continue
            if not isinstance(arr, zarr.Array):
                continue
            if arr.shape[0] == 0:
                cache[(array_name, chunk_key)] = b""
                continue
            cache[(array_name, chunk_key)] = bytes(np.asarray(arr[:]).tobytes())
    return cache
