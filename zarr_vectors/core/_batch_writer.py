"""Async-batched chunk + array-metadata writer.

The default ZV write paths
(:meth:`zarr_vectors.core.group.Group.write_bytes` and
:meth:`zarr_vectors.core.group.Group.write_array_meta`) each issue
synchronous zarr operations that bottom out at one obstore PUT per
operation, serialised through zarr's sync→async bridge.  Against a
high-latency object store (GCS/S3/Azure) every PUT round-trip is paid
serially — for 200k points + 3 attributes that's ~1000 chunk PUTs +
~6 per-array metadata PUTs at ~15 ms each ≈ 15 s of wall time spent
waiting on the network.

This module provides a deferred-batch path that:

1. Collects ``(array_name, chunk_key, bytes)`` chunk triples and
   ``{array_name: meta_dict}`` array metadata while the caller iterates
   (no round-trips during collection).
2. On flush, ensures each unique parent group either gets its
   ``zarr.json`` PUT directly (when the caller queued metadata for it)
   or via a single sync ``require_group`` (when only chunk PUTs are
   queued, no metadata to merge).
3. Builds inner-array metadata + data buffers in pure Python.
4. Issues every store ``set()`` in a single :func:`asyncio.gather`, so
   the obstore-backed pipe runs at full async parallelism — the latency
   of the slowest PUT, not the sum of all PUTs.

The on-disk layout matches what ``write_bytes`` produces, with one
deliberate difference: inner arrays use a **BytesCodec-only** codec
pipeline (no zstd).  zarr's default pipeline appends ``ZstdCodec`` for
small uint8 chunks; we drop that here because (a) cloud-write latency
dominates compression CPU by 10×+ for the workloads this path targets,
and (b) keeping the metadata minimal lets us serialise it with a string
template instead of reaching into zarr's codec registry.  Existing
stores written through the sync path keep their zstd codec; the reader
honours whatever the per-chunk ``zarr.json`` says.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync


def _is_icechunk_store(store: Any) -> bool:
    """Return True when ``store`` is an icechunk-backed :class:`zarr.abc.store.Store`.

    Detection is by class name to avoid importing icechunk at module
    load time (it's an optional dep).  Icechunk tracks arrays and
    groups as first-class entities; raw ``store.set("…/zarr.json", …)``
    PUTs don't register them, so the batched direct-PUT path can't be
    used and we have to fall back to the synchronous
    ``zarr.Array.create_array`` + ``array[:] = …`` path that the
    icechunk session knows how to commit.
    """
    cls = type(store)
    return cls.__name__ == "IcechunkStore" or cls.__module__.startswith("icechunk")


# ---------------------------------------------------------------------------
# Inner-array metadata template
# ---------------------------------------------------------------------------

# zarr v3 metadata for a 1D uint8 array with shape=(N,), chunks=(N,),
# BytesCodec only.  Verified round-trip against zarr 3.2.x.  We use
# str.replace on ``__SIZE__`` instead of json.dumps because every chunk
# differs only by length and reconstructing a dict + serialising adds
# meaningful CPU when the batch is large.
_INNER_ARRAY_META_TEMPLATE = (
    '{"shape":[__SIZE__],"data_type":"uint8",'
    '"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[__SIZE__]}},'
    '"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},'
    '"fill_value":0,"codecs":[{"name":"bytes"}],"attributes":{},'
    '"zarr_format":3,"node_type":"array","storage_transformers":[]}'
)

# Empty-chunk variant: shape=(0,) with chunks=(1,) so zarr accepts it.
# Matches the legacy "n == 0" branch in ``FsGroup.write_bytes``.
_EMPTY_INNER_ARRAY_META = (
    '{"shape":[0],"data_type":"uint8",'
    '"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1]}},'
    '"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},'
    '"fill_value":0,"codecs":[{"name":"bytes"}],"attributes":{},'
    '"zarr_format":3,"node_type":"array","storage_transformers":[]}'
).encode("utf-8")


def _inner_array_meta_bytes(n: int) -> bytes:
    """Return the inner-array ``zarr.json`` bytes for an N-byte chunk."""
    if n == 0:
        return _EMPTY_INNER_ARRAY_META
    return _INNER_ARRAY_META_TEMPLATE.replace("__SIZE__", str(n)).encode("utf-8")


# ---------------------------------------------------------------------------
# Async gather
# ---------------------------------------------------------------------------


async def _async_put_chunks(
    store: Any,
    items: list[tuple[str, bytes]],
) -> None:
    """``await asyncio.gather`` every ``store.set(key, buffer)`` in ``items``."""
    proto = default_buffer_prototype()
    tasks = [
        store.set(key, proto.buffer.from_bytes(data))
        for key, data in items
    ]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


_GROUP_ZARR_JSON_PREFIX = '{"zarr_format":3,"node_type":"group","attributes":'


def _group_zarr_json_bytes(attributes: dict[str, Any]) -> bytes:
    """Build a v3 group ``zarr.json`` with the given attributes."""
    return (
        _GROUP_ZARR_JSON_PREFIX + json.dumps(attributes) + "}"
    ).encode("utf-8")


def _flush_batch_sync(
    zarr_group: zarr.Group,
    triples: list[tuple[str, str, bytes]],
    array_metas: dict[str, dict[str, Any]],
) -> None:
    """Synchronous fallback for stores that can't accept raw ``set()``
    PUTs against array paths (currently just icechunk).

    Replays each queued operation through the zarr ``Array.create_array``
    + ``array[:] = data`` path that ``Group.write_bytes`` would have
    used in unbatched mode, so a transactional backend's commit picks
    them up.
    """
    # Array metadata first so the parent group exists with the right
    # attributes before any per-chunk array creation.
    for array_name, meta in array_metas.items():
        arr_group = zarr_group.require_group(array_name)
        arr_group.attrs.update(meta)

    for array_name, chunk_key, data in triples:
        arr_group = zarr_group.require_group(array_name)
        if chunk_key in arr_group:
            del arr_group[chunk_key]
        n = len(data)
        if n == 0:
            arr_group.create_array(
                chunk_key, shape=(0,), chunks=(1,), dtype="uint8",
            )
            continue
        a = arr_group.create_array(
            chunk_key, shape=(n,), chunks=(n,), dtype="uint8",
        )
        a[:] = np.frombuffer(data, dtype="uint8")


def flush_batch(
    zarr_group: zarr.Group,
    triples: Iterable[tuple[str, str, bytes]],
    *,
    array_metas: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Flush a batch of chunk writes + array-metadata writes.

    ``triples`` is the chunk batch: ``(array_name, chunk_key, data)``.
    Each item produces an inner-array ``zarr.json`` and (for non-empty
    data) a ``c/0`` PUT.

    ``array_metas`` is the per-array metadata batch:
    ``{array_name: attributes_dict}``.  Each entry produces one PUT for
    ``{array_name}/zarr.json`` with the attributes inlined into a fresh
    v3 group node.  This collapses the legacy ``require_group(name)`` +
    ``attrs.update(meta)`` sync pair (2-3 round-trips) into a single
    PUT that joins the chunk gather.

    For array_names that appear only in ``triples`` (chunk writes
    without queued metadata), the parent group is created with a single
    sync ``require_group`` call — fast and amortised over the batch.

    All PUTs go through one :func:`asyncio.gather`, then the function
    blocks until they complete (or the first error propagates).
    Idempotent on empty inputs.
    """
    triples = list(triples)
    array_metas = dict(array_metas or {})

    if not triples and not array_metas:
        return

    # Icechunk doesn't pick up arrays added via raw ``store.set`` of
    # their ``zarr.json`` — it tracks arrays as first-class entities and
    # expects them through ``zarr.Array.create_array``.  Fall back to
    # synchronous writes for icechunk-backed stores; the icechunk
    # session batches everything at commit time anyway.
    if _is_icechunk_store(zarr_group.store):
        _flush_batch_sync(zarr_group, triples, array_metas)
        return

    # Resolve the addressing prefix once.
    base_path = zarr_group.path
    base_prefix = f"{base_path.rstrip('/')}/" if base_path else ""

    # Chunk-write array_names that have no queued metadata still need a
    # parent zarr.json.  Create those via require_group (sync, amortised
    # — typically 2-6 names per batch).  Array_names with queued meta
    # skip require_group entirely; the meta PUT below creates the group
    # directly.
    chunk_array_names = {array_name for array_name, _, _ in triples}
    needs_require_group = chunk_array_names - array_metas.keys()
    for array_name in sorted(needs_require_group):
        zarr_group.require_group(array_name)

    puts: list[tuple[str, bytes]] = []

    # Group-level zarr.json PUTs (one per queued array meta).
    for array_name, meta in sorted(array_metas.items()):
        puts.append(
            (f"{base_prefix}{array_name}/zarr.json", _group_zarr_json_bytes(meta))
        )

    # Per-chunk inner-array PUTs.
    for array_name, chunk_key, data in triples:
        inner_prefix = f"{base_prefix}{array_name}/{chunk_key}/"
        n = len(data)
        puts.append((f"{inner_prefix}zarr.json", _inner_array_meta_bytes(n)))
        if n > 0:
            puts.append((f"{inner_prefix}c/0", data))

    sync(_async_put_chunks(zarr_group.store, puts))
