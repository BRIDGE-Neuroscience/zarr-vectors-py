"""High-level edit API: :class:`EditSession` and free functions.

This module provides the two surfaces approved in the plan:

1. **Free functions** (``edit_vertex``, ``add_vertex``, ``remove_vertex``,
   ``edit_link``, ``edit_attribute``, ``edit_object``, ...) — apply
   immediately, no pyramid refresh, ``atomic=True`` by default.  Each
   call commits in one transaction when the store is icechunk-backed.

2. :class:`EditSession` — a context manager that buffers edits into a
   change set keyed by ``(level, chunk_coords)``, coalesces multiple
   edits to the same chunk into a single read-modify-write, optionally
   refreshes the pyramid on exit, and commits once on flush.

Both surfaces share the same underlying ``_apply_*`` helpers so
behaviour is consistent.

Edit semantics (atomic vs minimal, source-row retention, propagate-to-
objects, refresh policy) are documented in the approved plan at
``.claude/plans/i-need-to-consider-zany-wand.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.change_set import (
    ChunkChangeBuilder,
    CrossChunkLinkOp,
    EditReport,
    ManifestOp,
)
from zarr_vectors.ops.refs import (
    AttributeRef,
    CrossChunkLinkRef,
    FragmentRef,
    LinkRef,
    ObjectRef,
    VertexRef,
)
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


RefreshPolicy = Literal[True, False, "batch"]
"""Allowed values for ``EditSession.refresh_pyramid``."""

PropagateTo = Literal["all"] | list[int]
"""Either the sentinel ``"all"`` or an explicit list of OIDs."""


class EditSession:
    """Transactional edit handle for one ZV store.

    Acquire with ``with EditSession(root, atomic=True, refresh_pyramid=
    "batch") as ed: ...``.  Inside the block every ``ed.edit_*`` /
    ``ed.add_*`` / ``ed.remove_*`` call accumulates an edit in the
    in-memory change set.  On ``__exit__`` (clean) the session:

    1. Flushes every dirty chunk via ``Group.batched_writes()``.
    2. Applies pending cross-chunk-link and object-manifest ops.
    3. Runs ``rebuild_pyramid_from_level`` if ``refresh_pyramid``
       requires it.
    4. Calls ``commit(root, message)`` once on icechunk-backed stores.

    Exceptions inside the block trigger a discard: the change set is
    dropped and ``discard_changes`` is called on the icechunk session
    so no partial state leaks to the next snapshot.
    """

    def __init__(
        self,
        root: Group,
        *,
        atomic: bool = True,
        refresh_pyramid: RefreshPolicy = False,
        propagate_to_objects: PropagateTo = "all",
        message: str = "zarr-vectors edit session",
        concurrent_writers: bool = False,
    ) -> None:
        if refresh_pyramid not in (True, False, "batch"):
            raise EditError(
                f"refresh_pyramid must be one of True / False / 'batch'; "
                f"got {refresh_pyramid!r}"
            )
        self.root = root
        self.atomic = bool(atomic)
        self.refresh_pyramid: RefreshPolicy = refresh_pyramid
        self.propagate_to_objects: PropagateTo = propagate_to_objects
        self.message = message
        self.concurrent_writers = bool(concurrent_writers)

        # Per-(level, chunk) builders, lazily filled.
        self._builders: dict[tuple[int, ChunkCoords], ChunkChangeBuilder] = {}
        # Per-level pending CCL operations.
        self._ccl_ops: list[CrossChunkLinkOp] = []
        # Per-level pending manifest ops, keyed by (level, oid).
        self._manifest_ops: dict[tuple[int, int], ManifestOp] = {}
        # Per-level manifest cache (lazy, populated on first lookup).
        self._all_manifests: dict[int, list[ObjectManifest]] = {}
        # Per-level next-available OID (atomic mode appends new OIDs here).
        self._next_oid: dict[int, int] = {}

        self._report = EditReport()
        self._touched_levels: set[int] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> EditSession:
        from zarr_vectors.core.store import session_for
        if (
            self.concurrent_writers
            and session_for(self.root) is None
        ):
            from zarr_vectors.exceptions import ConcurrentEditError
            raise ConcurrentEditError(
                "concurrent_writers=True requires an icechunk-backed "
                "store; non-transactional backends are single-writer "
                "only."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._discard()
            return
        try:
            self.flush()
        except Exception:
            self._discard()
            raise

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def report(self) -> EditReport:
        """Cumulative :class:`EditReport` for this session."""
        return self._report

    def change_set(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the pending diff.

        Useful for review / queueing pipelines that want to inspect the
        edits before flush.  Mutating the returned dict does *not*
        affect the session.
        """
        return {
            "dirty_chunks": [
                {"level": lv, "chunk": list(cc), "appended_fragments": list(
                    b.appended_fragments,
                )}
                for (lv, cc), b in self._builders.items()
                if b.is_dirty()
            ],
            "ccl_ops": [
                {"op": op.op, "delta": op.delta, "index": op.index}
                for op in self._ccl_ops
            ],
            "manifest_ops": [
                {
                    "level": op.level,
                    "object_id": op.object_id,
                    "new_oid": op.new_oid,
                    "deleted": op.new_manifest is None,
                }
                for op in self._manifest_ops.values()
            ],
            "atomic": self.atomic,
            "refresh_pyramid": self.refresh_pyramid,
        }

    # ------------------------------------------------------------------
    # Vertex edits
    # ------------------------------------------------------------------

    def edit_vertex(
        self,
        ref: VertexRef,
        *,
        new_pos: npt.ArrayLike | None = None,
        new_attrs: dict[str, npt.ArrayLike] | None = None,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        """Edit one vertex.

        Args:
            ref: Physical :class:`VertexRef` (use ``VertexRef.from_object``
                / ``VertexRef.from_position`` if you don't have one).
            new_pos: New position.  When ``None``, only attributes change.
            new_attrs: ``{attr_name: new_value}`` to update alongside
                the position.
            atomic: Override the session default for this edit.
            propagate_to_objects: Override the session default.
        """
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )

        if new_pos is None and not new_attrs:
            return  # noop

        if new_pos is None:
            # Pure attribute edit on the existing row; no fragment churn,
            # no manifest changes regardless of atomic flag.
            builder = self._builder(ref.level, ref.chunk)
            self._ensure_attrs_loaded(builder, list((new_attrs or {}).keys()))
            self._overwrite_attributes_in_place(
                builder, ref.fragment, ref.local,
                {k: np.asarray(v) for k, v in (new_attrs or {}).items()},
            )
            self._mark_edit(ref.level)
            return

        new_pos_arr = np.asarray(new_pos, dtype=np.float64).reshape(-1)
        # Detect chunk-cross.
        new_cc = _chunk_for(self.root, new_pos_arr)
        if tuple(new_cc) == tuple(ref.chunk):
            self._edit_vertex_in_chunk(
                ref, new_pos_arr, new_attrs,
                atomic=atomic_eff, propagate=prop_eff,
            )
        else:
            from zarr_vectors.ops.relocate import relocate_vertex_in_session
            relocate_vertex_in_session(
                self, ref, new_pos_arr, new_cc,
                new_attrs=new_attrs,
                atomic=atomic_eff, propagate=prop_eff,
            )
        self._mark_edit(ref.level)

    def remove_vertex(
        self,
        ref: VertexRef,
        *,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        """Remove one vertex row.

        Under ``atomic=True`` this is implemented as a soft-delete:
        the row stays but every referring object's manifest is updated
        to skip it (new OIDs allocated for the propagated objects).
        Under ``atomic=False`` the row is physically deleted; subsequent
        rows in the fragment shift down and link/manifest indices
        targeting later rows are rewritten.  Today we only support the
        atomic path; ``atomic=False`` removal is rejected with a clear
        error message to avoid silent index drift.
        """
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )
        if not atomic_eff:
            raise EditError(
                "remove_vertex(atomic=False) requires shifting downstream "
                "row indices in links and manifests; this destructive path "
                "is not implemented yet.  Use atomic=True (soft-delete) or "
                "remove the whole object via remove_object()."
            )
        # Atomic soft-delete: update manifests so propagated objects no
        # longer point at (chunk, fragment, local).  Keep the row on disk
        # so non-propagated OIDs still see it.
        affected_oids = self._oids_referencing(ref.level, ref.chunk, ref.fragment)
        targets = self._select_targets(affected_oids, prop_eff)
        for oid in targets:
            manifest = self._get_manifest(ref.level, oid)
            new_manifest = [
                (cc, fi) for (cc, fi) in manifest
                if not (tuple(cc) == tuple(ref.chunk) and fi == ref.fragment)
            ]
            self._stage_manifest(ref.level, oid, new_manifest, atomic=True)
        self._mark_edit(ref.level)

    def add_vertex(
        self,
        *,
        level: int,
        pos: npt.ArrayLike,
        attrs: dict[str, npt.ArrayLike] | None = None,
        object_id: int | None = None,
    ) -> VertexRef:
        """Add a new vertex at ``pos`` to ``level``.

        The vertex goes into the chunk that owns ``pos`` (per
        ``floor(pos / chunk_shape)``) as a new single-row fragment.

        When ``object_id`` is given, the new fragment is appended to
        that object's manifest.  Otherwise the fragment is unreferenced
        until the caller writes a manifest entry pointing at it.
        """
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        cc = _chunk_for(self.root, pos_arr)
        builder = self._builder(level, cc)
        new_frag = builder.append_fragment(
            pos_arr[np.newaxis, :],
            attrs={
                k: np.atleast_1d(np.asarray(v))
                for k, v in (attrs or {}).items()
            },
        )
        if object_id is not None:
            manifest = self._get_manifest(level, object_id)
            manifest = list(manifest) + [(tuple(cc), new_frag)]
            self._stage_manifest(level, object_id, manifest, atomic=False)
        self._mark_edit(level)
        return VertexRef(level=level, chunk=tuple(cc), fragment=new_frag, local=0)

    # ------------------------------------------------------------------
    # Internal vertex-edit helpers
    # ------------------------------------------------------------------

    def _edit_vertex_in_chunk(
        self,
        ref: VertexRef,
        new_pos: npt.NDArray[np.floating],
        new_attrs: dict[str, npt.ArrayLike] | None,
        *,
        atomic: bool,
        propagate: PropagateTo,
    ) -> None:
        builder = self._builder(ref.level, ref.chunk)
        if not atomic:
            # Minimal overwrite — row indices stay; manifests untouched.
            self._ensure_attrs_loaded(
                builder, list((new_attrs or {}).keys()),
            )
            builder.overwrite_vertex_row(
                ref.fragment, ref.local, new_pos,
                new_attrs={
                    k: np.asarray(v) for k, v in (new_attrs or {}).items()
                },
            )
            return

        # Atomic: append a new single-row fragment, update propagated
        # object manifests to point at it.
        attrs_dict: dict[str, npt.NDArray] = {}
        if new_attrs:
            for name, val in new_attrs.items():
                attrs_dict[name] = np.atleast_1d(np.asarray(val))
        new_frag = builder.append_fragment(
            new_pos[np.newaxis, :],
            attrs=attrs_dict or None,
        )

        affected_oids = self._oids_referencing(
            ref.level, ref.chunk, ref.fragment,
        )
        targets = self._select_targets(affected_oids, propagate)
        for oid in targets:
            manifest = self._get_manifest(ref.level, oid)
            new_manifest = [
                (tuple(ref.chunk), new_frag) if (
                    tuple(cc) == tuple(ref.chunk) and fi == ref.fragment
                ) else (cc, fi)
                for (cc, fi) in manifest
            ]
            self._stage_manifest(
                ref.level, oid, new_manifest, atomic=True,
            )

    def _overwrite_attributes_in_place(
        self,
        builder: ChunkChangeBuilder,
        fragment: int,
        local: int,
        attrs: dict[str, npt.NDArray],
    ) -> None:
        for name, val in attrs.items():
            attr_list = builder.attr_groups.get(name)
            if attr_list is None:
                continue
            attr_list[fragment][local] = val.astype(
                builder.attr_dtype.get(name, np.float32),
                copy=False,
            )
            builder.attrs_dirty[name] = True

    def _ensure_attrs_loaded(
        self,
        builder: ChunkChangeBuilder,
        names: Iterable[str],
    ) -> None:
        for name in names:
            builder.require_attribute(self.root, name)

    # ------------------------------------------------------------------
    # Manifest / OID bookkeeping
    # ------------------------------------------------------------------

    def _all_manifests_for(self, level: int) -> list[ObjectManifest]:
        if level in self._all_manifests:
            return self._all_manifests[level]
        from zarr_vectors.core.arrays import read_all_object_manifests
        from zarr_vectors.core.store import get_resolution_level
        level_group = get_resolution_level(self.root, level)
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            manifests = []
        self._all_manifests[level] = manifests
        self._next_oid[level] = len(manifests)
        return manifests

    def _get_manifest(self, level: int, oid: int) -> ObjectManifest:
        # Honour any pending in-session edit for this OID first.
        pending = self._manifest_ops.get((level, oid))
        if pending is not None and pending.new_manifest is not None:
            return list(pending.new_manifest)
        manifests = self._all_manifests_for(level)
        if oid < 0 or oid >= len(manifests):
            raise EditError(
                f"object_id {oid} out of range at level {level} "
                f"(have {len(manifests)} OIDs)"
            )
        return list(manifests[oid])

    def _oids_referencing(
        self,
        level: int,
        chunk: ChunkCoords,
        fragment: int,
    ) -> list[int]:
        """Scan all manifests for those that reference ``(chunk, fragment)``."""
        out: list[int] = []
        manifests = self._all_manifests_for(level)
        chunk_t = tuple(chunk)
        for oid, manifest in enumerate(manifests):
            for cc, fi in manifest:
                if tuple(cc) == chunk_t and fi == fragment:
                    out.append(oid)
                    break
        return out

    def _select_targets(
        self,
        affected_oids: list[int],
        propagate: PropagateTo,
    ) -> list[int]:
        if propagate == "all":
            return list(affected_oids)
        wanted = set(int(o) for o in propagate)
        return [oid for oid in affected_oids if oid in wanted]

    def _stage_manifest(
        self,
        level: int,
        oid: int,
        new_manifest: ObjectManifest | None,
        *,
        atomic: bool,
    ) -> None:
        """Schedule a manifest write.

        Under ``atomic=True`` allocate a new OID at the tail and leave
        the original OID's manifest untouched; record the remap.  Under
        ``atomic=False`` overwrite the manifest at ``oid``.
        """
        self._all_manifests_for(level)  # populate _next_oid
        if atomic:
            new_oid = self._next_oid[level]
            self._next_oid[level] += 1
            self._manifest_ops[(level, new_oid)] = ManifestOp(
                level=level, object_id=new_oid,
                new_manifest=new_manifest, new_oid=new_oid,
            )
            self._report.oid_remap[int(oid)] = int(new_oid)
        else:
            self._manifest_ops[(level, oid)] = ManifestOp(
                level=level, object_id=oid,
                new_manifest=new_manifest, new_oid=None,
            )

    # ------------------------------------------------------------------
    # Builders & touch tracking
    # ------------------------------------------------------------------

    def _builder(self, level: int, chunk: ChunkCoords) -> ChunkChangeBuilder:
        key = (level, tuple(chunk))
        existing = self._builders.get(key)
        if existing is not None:
            return existing
        builder = ChunkChangeBuilder.from_disk(self.root, level, key[1])
        self._builders[key] = builder
        return builder

    def _mark_edit(self, level: int) -> None:
        self._touched_levels.add(level)
        self._report.n_edits += 1
        if self.refresh_pyramid is True:
            self._refresh_now(level)

    # ------------------------------------------------------------------
    # Flush / commit / discard
    # ------------------------------------------------------------------

    def flush(self) -> EditReport:
        """Write every pending edit to disk and (if applicable) commit.

        Returns the cumulative :class:`EditReport`.  Idempotent: calling
        ``flush`` a second time is a no-op once everything is clean.
        """
        if self._closed:
            return self._report

        # 1. Apply dirty chunks via batched_writes for parallelism.
        from zarr_vectors.core.store import commit, get_resolution_level
        from zarr_vectors.core.arrays import (
            write_chunk_attributes,
            write_chunk_links,
            write_chunk_vertices,
        )

        with self.root.batched_writes():
            for (level, cc), builder in self._builders.items():
                if not builder.is_dirty():
                    continue
                level_group = get_resolution_level(self.root, level)
                if builder.vertices_dirty:
                    write_chunk_vertices(
                        level_group, cc, builder.vertex_groups,
                        dtype=builder.vertex_dtype,
                    )
                for delta, dirty in builder.links_dirty.items():
                    if not dirty:
                        continue
                    write_chunk_links(
                        level_group, cc, builder.link_groups[delta],
                        delta=delta,
                    )
                for name, dirty in builder.attrs_dirty.items():
                    if not dirty:
                        continue
                    write_chunk_attributes(
                        level_group, name, cc, builder.attr_groups[name],
                        dtype=builder.attr_dtype.get(name, np.float32),
                    )
                self._report.touched_chunks.append((level, tuple(cc)))

        # 2. Apply CCL ops grouped by (level, delta).
        self._flush_ccl_ops()

        # 3. Apply manifest ops grouped by level.
        self._flush_manifest_ops()

        # 4. Pyramid refresh.
        if self.refresh_pyramid == "batch" and self._touched_levels:
            min_level = min(self._touched_levels)
            self._refresh_now(min_level)
        elif self.refresh_pyramid is False and self._touched_levels:
            min_level = min(self._touched_levels)
            from zarr_vectors.core.store import list_resolution_levels
            all_levels = list_resolution_levels(self.root)
            self._report.dirty_pyramid_levels = [
                lv for lv in all_levels if lv > min_level
            ]

        # 5. Commit (icechunk-only).
        snapshot = commit(self.root, self.message)
        self._report.snapshot_id = snapshot

        self._closed = True
        return self._report

    def _flush_ccl_ops(self) -> None:
        if not self._ccl_ops:
            return
        from zarr_vectors.core.arrays import (
            read_cross_chunk_links,
            write_cross_chunk_links,
        )
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        meta = RootMetadata.from_dict(self.root.attrs.to_dict())
        sid_ndim = meta.sid_ndim

        # Group by (level, delta).  Only level 0 is the common edit
        # target today; we still group generically.
        from collections import defaultdict
        groups: dict[tuple[int, int], list[CrossChunkLinkOp]] = defaultdict(list)
        for op in self._ccl_ops:
            groups[(0, op.delta)].append(op)

        for (level, delta), ops in groups.items():
            level_group = get_resolution_level(self.root, level)
            current = read_cross_chunk_links(level_group, delta=delta)
            # Apply ops in submission order.
            rows: list[list[tuple[ChunkCoords, int]]] = [list(r) for r in current]
            for op in ops:
                if op.op == "append":
                    rows.append(list(op.payload or []))
                elif op.op == "delete":
                    if op.index is not None and 0 <= op.index < len(rows):
                        rows.pop(op.index)
                elif op.op == "overwrite":
                    if op.index is not None and 0 <= op.index < len(rows):
                        rows[op.index] = list(op.payload or [])
            write_cross_chunk_links(level_group, rows, sid_ndim, delta=delta)

    def _flush_manifest_ops(self) -> None:
        if not self._manifest_ops:
            return
        from zarr_vectors.core.arrays import write_object_index
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        meta = RootMetadata.from_dict(self.root.attrs.to_dict())
        sid_ndim = meta.sid_ndim

        # Group ops by level.
        from collections import defaultdict
        by_level: dict[int, list[ManifestOp]] = defaultdict(list)
        for op in self._manifest_ops.values():
            by_level[op.level].append(op)

        for level, ops in by_level.items():
            level_group = get_resolution_level(self.root, level)
            # Start from disk state then apply ops.
            manifests = list(self._all_manifests_for(level))
            max_oid = len(manifests) - 1
            for op in ops:
                oid = op.new_oid if op.new_oid is not None else op.object_id
                if oid > max_oid:
                    manifests.extend([] for _ in range(oid - max_oid))
                    max_oid = oid
                manifests[oid] = list(op.new_manifest or [])
            manifest_dict = {i: m for i, m in enumerate(manifests)}
            write_object_index(
                level_group, manifest_dict, sid_ndim,
                total_objects=len(manifests),
            )

    def _refresh_now(self, source_level: int) -> None:
        from zarr_vectors.ops.refresh import rebuild_pyramid_from_level
        rebuild_pyramid_from_level(self.root, source_level)

    def _discard(self) -> None:
        from zarr_vectors.core.store import discard_changes
        discard_changes(self.root)
        self._builders.clear()
        self._ccl_ops.clear()
        self._manifest_ops.clear()
        self._closed = True


# ======================================================================
# Free-function wrappers
# ======================================================================

def edit_vertex(
    root: Group,
    ref: VertexRef,
    *,
    new_pos: npt.ArrayLike | None = None,
    new_attrs: dict[str, npt.ArrayLike] | None = None,
    atomic: bool = True,
    propagate_to_objects: PropagateTo = "all",
    message: str = "edit_vertex",
) -> EditReport:
    """Edit one vertex; immediately flush and (if applicable) commit.

    No pyramid refresh — call :func:`zarr_vectors.ops.refresh.
    rebuild_pyramid_from_level` if you need the coarser levels to track.
    """
    with EditSession(
        root,
        atomic=atomic,
        refresh_pyramid=False,
        propagate_to_objects=propagate_to_objects,
        message=message,
    ) as ed:
        ed.edit_vertex(
            ref,
            new_pos=new_pos,
            new_attrs=new_attrs,
        )
    return ed.report


def add_vertex(
    root: Group,
    *,
    level: int,
    pos: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
    object_id: int | None = None,
    message: str = "add_vertex",
) -> tuple[VertexRef, EditReport]:
    """Append a new single-row fragment containing ``pos``."""
    with EditSession(
        root, atomic=True, refresh_pyramid=False, message=message,
    ) as ed:
        ref = ed.add_vertex(
            level=level, pos=pos, attrs=attrs, object_id=object_id,
        )
    return ref, ed.report


def remove_vertex(
    root: Group,
    ref: VertexRef,
    *,
    atomic: bool = True,
    propagate_to_objects: PropagateTo = "all",
    message: str = "remove_vertex",
) -> EditReport:
    """Remove one vertex (soft-delete under ``atomic=True``)."""
    with EditSession(
        root,
        atomic=atomic,
        refresh_pyramid=False,
        propagate_to_objects=propagate_to_objects,
        message=message,
    ) as ed:
        ed.remove_vertex(ref)
    return ed.report


# ======================================================================
# Internal helpers
# ======================================================================

def _chunk_for(root: Group, pos: npt.NDArray[np.floating]) -> ChunkCoords:
    """Return ``floor(pos / chunk_shape)`` as a tuple."""
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.spatial.chunking import compute_chunk_coords

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    chunk_shape = tuple(meta.chunk_shape)
    return compute_chunk_coords(np.asarray(pos, dtype=np.float64), chunk_shape)
