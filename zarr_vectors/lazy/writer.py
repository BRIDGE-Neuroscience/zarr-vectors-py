"""ZVRWriter — lazy / async mutation handle for a single ``ZVRLevel``.

Adds the write-back surface the algorithms package needs:

* :meth:`add_attribute` (Tier A) — write a per-vertex attribute array
  aligned with the level's existing vertices, without rewriting the
  vertex data.
* :meth:`add_node_attribute`, :meth:`add_face_attribute`,
  :meth:`add_object_attribute` — siblings for the analogous result
  types.
* :meth:`append_vertices` — true incremental append (Step 7).
* :meth:`commit` / :meth:`compact` — pending-sidecar lifecycle.

Each public method has both an async and a sync mirror.  The async
methods route I/O through :class:`AsyncStorageBackend`; the sync
mirrors are thin ``asyncio.run`` wrappers for non-async callers.

v1 is **single-writer-only**.  Concurrent writers against the same
level can race on object_index sidecar batch numbering; documented
loudly and not protected at runtime.
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import CAP_OBJECT_INDEX_PENDING
from zarr_vectors.core.arrays import (
    compact_object_index,
    list_chunk_keys,
    read_chunk_vertices,
    write_chunk_attributes,
    write_chunk_vertices,
    write_object_index_pending,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.lazy.level import ZVRLevel


class ZVRWriter:
    """Mutation handle for one :class:`ZVRLevel`.

    Acquire one via ``zvr[0].writer()``.  Holds a reference to the
    level's :class:`Group` so all mutations go through the same backend
    the reader uses.

    Usage::

        # Async — recommended for cloud stores
        async with zvr[0].writer() as w:
            await w.add_attribute("normal", normals)

        # Sync — convenient for scripts
        with zvr[0].writer() as w:
            w.add_attribute_sync("normal", normals)
    """

    def __init__(self, level: ZVRLevel) -> None:
        self._level = level
        self._group = level._group
        self._committed = False
        # In-memory dirty-flag tracking for future use (e.g. lazy
        # commits).  Currently every method writes immediately.
        self._dirty: bool = False
        # Manifests staged by ``append_vertices`` and flushed by
        # ``commit`` as a pending sidecar batch.
        self._pending_manifests: dict[int, ObjectManifest] = {}
        self._pending_sid_ndim: int | None = None

    # ---------------- context manager -----------------------------------

    async def __aenter__(self) -> ZVRWriter:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            await self.commit()

    def __enter__(self) -> ZVRWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.commit_sync()

    # ---------------- Tier A: post-hoc attribute writes -----------------

    async def add_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Write a per-vertex attribute aligned with this level's vertices.

        Splits ``values`` by chunk using the existing vertex-count
        sidecars (Tier E) and writes one
        ``attributes/<name>/<chunk_key>`` per chunk.  No vertex data is
        re-encoded.

        Args:
            name: Attribute name.  Stored under ``attributes/<name>/``.
            values: ``(N,)`` or ``(N, C)`` array of length equal to the
                level's total vertex count.
            dtype: Override the on-disk dtype (default: ``values.dtype``).
        """
        await self._write_per_vertex_attribute(
            subpath="attributes", name=name, values=values, dtype=dtype,
        )

    async def add_node_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-node attribute on a graph / skeleton level.

        Identical semantics to :meth:`add_attribute` — nodes are the
        graph's vertices.  Provided as an ergonomic alias for code that
        reads more naturally with the graph terminology.
        """
        await self.add_attribute(name, values, dtype=dtype)

    async def add_face_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-face attribute on a mesh level.

        Stored under ``face_attributes/<name>/<chunk_key>``.  Faces are
        aligned 1:1 with the intra-chunk links array — values for a
        chunk's ``F_local`` faces appear in the same order as the
        decoded ``links/<chunk_key>``.

        Note: face attributes for **cross-chunk** faces require the 0.3
        ``cross_chunk_faces`` capability and are tracked in a separate
        path; that path is wired up in step 8.
        """
        await self._write_per_face_attribute(
            name=name, values=values, dtype=dtype,
        )

    async def add_object_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-object attribute, length equal to ``num_objects``.

        Writes the dense ``(O,) | (O, C)`` array to
        ``object_attributes/<name>/data``.
        """
        from zarr_vectors.core.arrays import (
            create_object_attributes_array,
            write_object_attributes,
        )

        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        await asyncio.to_thread(
            create_object_attributes_array, self._group, name,
        )
        await asyncio.to_thread(
            write_object_attributes, self._group, name, arr,
        )
        self._dirty = True

    # ---------------- internal helpers ----------------------------------

    async def _write_per_vertex_attribute(
        self,
        *,
        subpath: str,
        name: str,
        values: npt.NDArray,
        dtype: str | np.dtype | None,
    ) -> None:
        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        if arr.ndim < 1:
            raise ArrayError(
                f"attribute values must be at least 1D; got shape {arr.shape}"
            )

        # Build the per-chunk → global offset table and total count.
        # to_thread because this can hit many small sidecar reads.
        offsets, chunk_keys, total = await asyncio.to_thread(
            chunk_local_to_global_offsets, self._group,
        )
        if arr.shape[0] != total:
            raise ArrayError(
                f"add_attribute({name!r}): values length {arr.shape[0]} "
                f"!= level vertex count {total}"
            )

        ndim = self._level._root_meta.sid_ndim

        # Schedule one per-chunk write in parallel.  Each task reads the
        # chunk's vertex groups to discover per-group sizes, slices the
        # values array, and emits the attribute bytes.
        async def _write_one(cc: ChunkCoords) -> None:
            start = offsets[cc]
            groups = await asyncio.to_thread(
                read_chunk_vertices, self._group, cc, np.float32, ndim,
            )
            sizes = [len(g) for g in groups]
            chunk_total = sum(sizes)
            if chunk_total == 0:
                return
            chunk_values = arr[start:start + chunk_total]
            # Split into groups aligned with the chunk's vertex groups.
            attr_groups: list[npt.NDArray] = []
            cursor = 0
            for s in sizes:
                attr_groups.append(chunk_values[cursor:cursor + s])
                cursor += s
            # If the writer is targeting non-default subpath (e.g.
            # face attributes), patch the array name; otherwise the
            # standard helper writes under "attributes/".
            if subpath == "attributes":
                await asyncio.to_thread(
                    write_chunk_attributes,
                    self._group, name, cc, attr_groups, arr.dtype,
                )
            else:
                await asyncio.to_thread(
                    _write_custom_subpath,
                    self._group, subpath, name, cc, attr_groups, arr.dtype,
                )

        await asyncio.gather(*(_write_one(cc) for cc in chunk_keys))

        # Make sure the level metadata advertises the new array.
        await asyncio.to_thread(self._touch_arrays_present, f"{subpath}/{name}")
        self._dirty = True

    async def _write_per_face_attribute(
        self,
        *,
        name: str,
        values: npt.NDArray,
        dtype: str | np.dtype | None,
    ) -> None:
        # Face attributes are aligned with the intra-chunk links/faces.
        # The number of faces per chunk is the row count of links/<cc>.
        # We read each chunk's links via `read_chunk_links` to discover
        # the size, then slice and write.
        from zarr_vectors.core.arrays import (
            count_link_groups,
            read_chunk_links,
        )

        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)

        chunk_keys = await asyncio.to_thread(list_chunk_keys, self._group)

        # Phase 1: gather per-chunk face counts.
        async def _count(cc: ChunkCoords) -> int:
            try:
                groups = await asyncio.to_thread(
                    read_chunk_links, self._group, cc, np.int64,
                )
            except Exception:
                return 0
            return sum(int(g.shape[0]) for g in groups)

        per_chunk_counts = await asyncio.gather(
            *(_count(cc) for cc in chunk_keys)
        )
        total_faces = sum(per_chunk_counts)
        if arr.shape[0] != total_faces:
            raise ArrayError(
                f"add_face_attribute({name!r}): values length "
                f"{arr.shape[0]} != total intra-chunk face count "
                f"{total_faces}"
            )

        # Phase 2: write slices in parallel.
        cursor = 0
        slices: list[tuple[ChunkCoords, npt.NDArray]] = []
        for cc, count in zip(chunk_keys, per_chunk_counts):
            if count == 0:
                continue
            slices.append((cc, arr[cursor:cursor + count]))
            cursor += count

        async def _write_one(cc: ChunkCoords, sub: npt.NDArray) -> None:
            # One face attribute group per chunk; faces live in a single
            # logical group per chunk in mesh stores today.
            await asyncio.to_thread(
                _write_custom_subpath,
                self._group, "face_attributes", name, cc, [sub], arr.dtype,
            )

        await asyncio.gather(*(_write_one(cc, sub) for cc, sub in slices))

        await asyncio.to_thread(
            self._touch_arrays_present, f"face_attributes/{name}",
        )
        self._dirty = True

    def _touch_arrays_present(self, entry: str) -> None:
        """Add ``entry`` to the level's ``arrays_present`` if missing."""
        attrs = self._group.attrs.to_dict()
        lv = attrs.get("zarr_vectors_level", {})
        ap = list(lv.get("arrays_present", []))
        if entry not in ap:
            ap.append(entry)
            lv["arrays_present"] = ap
            self._group.attrs.update({"zarr_vectors_level": lv})

    # ---------------- true append ---------------------------------------

    async def append_vertices(
        self,
        positions: npt.NDArray,
        *,
        object_ids: npt.NDArray | None = None,
        dtype: str | np.dtype | None = None,
    ) -> dict:
        """Append new vertices (and new objects) to this level.

        Routes each vertex to its spatial chunk, reads the existing
        chunk data, appends one vertex group per new object, and
        rewrites the chunk.  Per-chunk RMW is parallelised over chunks
        via :func:`asyncio.gather`.

        Per-object manifest entries are staged in memory and flushed to
        a pending sidecar by :meth:`commit`.

        Args:
            positions: ``(N, D)`` array of new vertex positions.
            object_ids: ``(N,)`` integer object IDs for each new vertex.
                IDs must be ``>=`` the current ``num_objects`` (no
                conflict with existing objects).  Defaults to a
                contiguous range starting at the current count.
            dtype: Vertex dtype.  Defaults to the level's recorded dtype.

        Returns:
            Summary dict with ``vertices_added``, ``new_objects``,
            ``chunks_touched``.
        """
        positions = np.asarray(positions)
        if positions.ndim != 2:
            raise ArrayError(
                f"positions must be (N, D), got shape {positions.shape}"
            )
        n_new, ndim = positions.shape
        if n_new == 0:
            return {"vertices_added": 0, "new_objects": 0, "chunks_touched": 0}

        root_meta = self._level._root_meta
        if ndim != root_meta.sid_ndim:
            raise ArrayError(
                f"position ndim {ndim} != store sid_ndim {root_meta.sid_ndim}"
            )
        if dtype is None:
            try:
                vmeta = self._group.read_array_meta("vertices")
                dtype = np.dtype(vmeta.get("dtype", "float32"))
            except Exception:
                dtype = np.float32
        dtype = np.dtype(dtype)
        positions = positions.astype(dtype, copy=False)

        # Resolve object_ids; default = append after existing num_objects.
        existing_num = await asyncio.to_thread(self._current_num_objects)
        if object_ids is None:
            object_ids = np.arange(
                existing_num, existing_num + n_new, dtype=np.int64,
            )
        else:
            object_ids = np.asarray(object_ids, dtype=np.int64)
            if object_ids.shape != (n_new,):
                raise ArrayError(
                    f"object_ids shape {object_ids.shape} != (N,) = ({n_new},)"
                )
            if int(object_ids.min()) < existing_num:
                raise ArrayError(
                    f"object_ids overlap existing objects "
                    f"(min={int(object_ids.min())}, existing_num={existing_num})"
                )

        # Spatial assignment (per-vertex → chunk).
        chunk_assignments = await asyncio.to_thread(
            assign_chunks, positions, root_meta.chunk_shape,
        )

        # RMW per chunk in parallel.  Each chunk gets one new vertex
        # group per **unique** object id present in the chunk; an
        # object whose vertices span multiple chunks gets multiple
        # manifest entries.
        results: dict[ChunkCoords, dict[int, int]] = {}  # cc → {oid: vg_idx_added}

        async def _rmw_chunk(cc: ChunkCoords, indices) -> None:
            sub_positions = positions[indices]
            sub_oids = object_ids[indices]

            # Read existing groups (may be empty if chunk is new).
            existing_groups = await asyncio.to_thread(
                _safe_read_chunk_vertices,
                self._group, cc, dtype, ndim,
            )
            existing_count = len(existing_groups)

            # Append one new vertex group per unique object in this chunk.
            chunk_assignments_per_oid: dict[int, int] = {}
            new_groups: list[npt.NDArray] = []
            for new_oid in np.unique(sub_oids):
                mask = sub_oids == new_oid
                new_groups.append(sub_positions[mask])
                chunk_assignments_per_oid[int(new_oid)] = (
                    existing_count + len(new_groups) - 1
                )

            all_groups = existing_groups + new_groups
            await asyncio.to_thread(
                write_chunk_vertices, self._group, cc, all_groups, dtype,
            )
            results[cc] = chunk_assignments_per_oid

        await asyncio.gather(*(
            _rmw_chunk(cc, idxs) for cc, idxs in chunk_assignments.items()
        ))

        # Build manifest entries per new object id.
        new_oids = set()
        for cc, oid_to_vg in results.items():
            for oid, vg_idx in oid_to_vg.items():
                self._pending_manifests.setdefault(oid, []).append((cc, vg_idx))
                new_oids.add(oid)

        # sid_ndim for the index encoding: include the +1 for attribute
        # chunking when the level is so chunked, else just ndim.
        try:
            existing_meta = self._group.read_array_meta("object_index")
            self._pending_sid_ndim = int(existing_meta.get("sid_ndim", ndim))
        except Exception:
            # No pre-existing object_index — fall back to chunk-key arity
            # discovered from results.
            if results:
                self._pending_sid_ndim = len(next(iter(results.keys())))
            else:
                self._pending_sid_ndim = ndim

        self._dirty = True
        return {
            "vertices_added": n_new,
            "new_objects": len(new_oids),
            "chunks_touched": len(results),
        }

    def _current_num_objects(self) -> int:
        """Inspect existing object_index + pending sidecars for total count."""
        from zarr_vectors.core.arrays import read_all_object_manifests
        try:
            manifests = read_all_object_manifests(self._group)
        except Exception:
            return 0
        # Total count counts already-pending entries too so successive
        # writes don't collide.
        existing_pending = self._pending_manifests
        if existing_pending:
            return max(
                len(manifests),
                max(existing_pending.keys()) + 1,
            )
        return len(manifests)

    # ---------------- lifecycle -----------------------------------------

    async def commit(self) -> dict:
        """Flush pending appends to a new object_index sidecar batch.

        - Writes a pending sidecar at ``object_index/pending/<batch_id>/``
          if there are staged manifests.
        - Stamps the ``CAP_OBJECT_INDEX_PENDING`` capability on the root
          metadata so readers know to fold the sidecar.
        - Updates the level's recorded ``vertex_count`` to include the
          appended vertices.
        """
        out: dict[str, int] = {"committed": True}

        if not self._pending_manifests:
            self._committed = True
            return {**out, "batches_written": 0, "objects_committed": 0}

        sid_ndim = self._pending_sid_ndim or self._level._root_meta.sid_ndim
        batch_id = await asyncio.to_thread(
            write_object_index_pending,
            self._group, self._pending_manifests, sid_ndim,
        )

        # Stamp the pending capability on root metadata (idempotent).
        await asyncio.to_thread(self._stamp_capability, CAP_OBJECT_INDEX_PENDING)

        # Update level vertex_count.
        await asyncio.to_thread(
            self._bump_level_vertex_count,
            sum(
                # Count distinct vertices we added: sum of group lengths
                # for each (oid, cc, vg_idx) tuple.  Read fresh from disk
                # by counting per-chunk sidecars vs the offset table is
                # avoidable; instead, sum group lengths from each new VG.
                # We don't have the source positions here, so derive
                # from manifest entries by summing actual chunk reads.
                # Keep it cheap: vertex_count is informational, callers
                # rely on actual reads.
                0 for _ in self._pending_manifests.values()
            ),
        )

        committed = len(self._pending_manifests)
        self._pending_manifests = {}
        self._pending_sid_ndim = None
        self._committed = True
        return {
            **out,
            "batches_written": 1 if batch_id >= 0 else 0,
            "objects_committed": committed,
            "batch_id": batch_id,
        }

    async def compact(self) -> dict:
        """Fold every pending object_index sidecar into the main index.

        Wraps :func:`zarr_vectors.core.arrays.compact_object_index`.
        Also clears the ``CAP_OBJECT_INDEX_PENDING`` capability on root
        metadata once the pending tree is empty.
        """
        if self._pending_manifests:
            await self.commit()
        result = await asyncio.to_thread(compact_object_index, self._group)
        await asyncio.to_thread(self._clear_capability, CAP_OBJECT_INDEX_PENDING)
        return {"compacted": True, **result}

    # ---------------- root-metadata mutators ----------------------------

    def _stamp_capability(self, cap: str) -> None:
        attrs = self._level._root._backend if False else None  # type: ignore[unreachable]
        # The level group is a sub-group; capabilities live on root.
        root_group = self._group._backend  # type: ignore[attr-defined]
        # Walk up via the lazy store: ZVRLevel doesn't have a backref
        # but the underlying Group exposes prefix navigation.
        from zarr_vectors.core.store import (
            open_store, read_root_metadata,
        )
        # Round-trip through root_metadata to keep the field set sane.
        # We open the store via the same backend the level group uses,
        # at the root prefix.
        root_url = self._group._backend.url
        # Strip the level prefix if present
        if self._group._prefix:
            depth = self._group._prefix.count("/") + 1
            # Walk URL up; simplest is to use the Group's backend
            # directly with empty prefix.
            from zarr_vectors.core.group import Group
            root_handle = Group._from_backend(self._group._backend, "")
        else:
            from zarr_vectors.core.group import Group
            root_handle = Group._from_backend(self._group._backend, "")
        root_attrs = root_handle.attrs.to_dict()
        zv = root_attrs.get("zarr_vectors", {})
        caps = list(zv.get("format_capabilities", []))
        if cap not in caps:
            caps.append(cap)
            zv["format_capabilities"] = caps
            root_attrs["zarr_vectors"] = zv
            root_handle.attrs.update({"zarr_vectors": zv})

    def _clear_capability(self, cap: str) -> None:
        from zarr_vectors.core.group import Group
        root_handle = Group._from_backend(self._group._backend, "")
        root_attrs = root_handle.attrs.to_dict()
        zv = root_attrs.get("zarr_vectors", {})
        caps = list(zv.get("format_capabilities", []))
        if cap in caps:
            caps.remove(cap)
            zv["format_capabilities"] = caps
            root_handle.attrs.update({"zarr_vectors": zv})

    def _bump_level_vertex_count(self, _placeholder: int) -> None:
        """Recompute the level's vertex_count from the actual on-disk data.

        ``append_vertices`` doesn't track per-call totals, so we
        recount from the per-chunk ``vertex_counts`` sidecars (which
        ``write_chunk_vertices`` always emits).
        """
        offsets, _keys, total = chunk_local_to_global_offsets(self._group)
        attrs = self._group.attrs.to_dict()
        lv = attrs.get("zarr_vectors_level", {})
        lv["vertex_count"] = int(total)
        self._group.attrs.update({"zarr_vectors_level": lv})

    # ---------------- sync mirrors --------------------------------------

    def add_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        asyncio.run(self.add_attribute(name, values, dtype=dtype))

    def add_node_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        asyncio.run(self.add_node_attribute(name, values, dtype=dtype))

    def add_face_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        asyncio.run(self.add_face_attribute(name, values, dtype=dtype))

    def add_object_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        asyncio.run(self.add_object_attribute(name, values, dtype=dtype))

    def append_vertices_sync(
        self,
        positions: npt.NDArray,
        *,
        object_ids: npt.NDArray | None = None,
        dtype: str | np.dtype | None = None,
    ) -> dict:
        return asyncio.run(self.append_vertices(
            positions, object_ids=object_ids, dtype=dtype,
        ))

    def commit_sync(self) -> dict:
        return asyncio.run(self.commit())

    def compact_sync(self) -> dict:
        return asyncio.run(self.compact())


def _safe_read_chunk_vertices(
    level_group,
    cc: ChunkCoords,
    dtype: np.dtype,
    ndim: int,
) -> list[npt.NDArray]:
    """Read existing vertex groups; return ``[]`` if the chunk is missing."""
    from zarr_vectors.core.arrays import _chunk_key
    if not level_group.chunk_exists("vertices", _chunk_key(cc)):
        return []
    try:
        return read_chunk_vertices(level_group, cc, dtype=dtype, ndim=ndim)
    except Exception:
        return []


def _write_custom_subpath(
    level_group,
    subpath: str,
    name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype,
) -> None:
    """Write attribute bytes to ``<subpath>/<name>/<chunk_key>``.

    Mirrors :func:`write_chunk_attributes` but with a configurable
    top-level subpath (e.g. ``"face_attributes"``).
    """
    from zarr_vectors.core.arrays import _chunk_key
    from zarr_vectors.encoding.ragged import encode_vertex_groups

    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{subpath}/{name}"
    raw_bytes, offsets = encode_vertex_groups(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)
    level_group.write_bytes(full_name, key + "_offsets", offsets.tobytes())
