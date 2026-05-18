"""Fragment-level edits: add / edit / remove a fragment within a chunk.

A *fragment* is one row group inside a chunk's ``vertices/<chunk>``
array.  These operations are coarse — they touch every row of one
fragment at once — and are usually invoked by higher-level code rather
than end-user UIs.

Semantics:

- ``add_fragment`` always appends a new fragment to the named chunk
  (the new index is ``len(vertex_groups)`` for that chunk).  Returns
  the resulting :class:`FragmentRef`.
- ``edit_fragment(ref, new_vertices=...)`` replaces every row of the
  named fragment.  Under ``atomic=True`` a new fragment is appended
  with ``new_vertices`` and every referring object's manifest is
  rewritten to point at the new fragment under a fresh OID; under
  ``atomic=False`` the rows are replaced in place.
- ``remove_fragment(ref, atomic=True)`` is a **tombstone** — the
  fragment's row list is set to empty but the fragment index stays in
  the chunk's sidecar so downstream fragment indices don't shift.
  Vacuum's tombstone-GC pass (deferred) can later drop empty
  fragments and shift indices.  ``atomic=False`` is rejected for the
  same reason ``remove_vertex(atomic=False)`` is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import FragmentRef
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession, PropagateTo


def add_fragment_in_session(
    session: EditSession,
    *,
    level: int,
    chunk: ChunkCoords,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
) -> FragmentRef:
    """Append a new fragment to ``chunk`` at ``level``.

    Returns the new :class:`FragmentRef`.  The caller is responsible
    for wiring object manifests if the fragment should be reachable
    via an OID — ``add_fragment`` itself only touches chunk-level data.
    """
    rows = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    if rows.size == 0:
        raise EditError("add_fragment: vertices must be non-empty")
    builder = session._builder(level, chunk)
    attrs_dict: dict[str, npt.NDArray] | None = None
    if attrs:
        attrs_dict = {
            k: np.atleast_1d(np.asarray(v)) for k, v in attrs.items()
        }
    new_idx = builder.append_fragment(rows, attrs=attrs_dict)
    session._mark_edit(level)
    return FragmentRef(
        level=level, chunk=tuple(int(c) for c in chunk),
        fragment=int(new_idx),
    )


def edit_fragment_in_session(
    session: EditSession,
    ref: FragmentRef,
    *,
    new_vertices: npt.ArrayLike,
    new_attrs: dict[str, npt.ArrayLike] | None,
    atomic: bool,
    propagate: PropagateTo,
) -> None:
    """Replace every row of ``ref``'s fragment.

    Atomic: append a new fragment with ``new_vertices`` and rewrite
    referring object manifests to point at it (under fresh OIDs).

    Minimal: overwrite the fragment's row group in place (object
    manifests stay because the fragment index is unchanged).
    """
    rows = np.atleast_2d(np.asarray(new_vertices, dtype=np.float64))
    if rows.size == 0:
        raise EditError(
            "edit_fragment: new_vertices is empty; use remove_fragment "
            "to clear a fragment instead."
        )

    builder = session._builder(ref.level, ref.chunk)
    if ref.fragment < 0 or ref.fragment >= len(builder.vertex_groups):
        raise EditError(
            f"edit_fragment: fragment {ref.fragment} out of range "
            f"for chunk {ref.chunk} "
            f"(has {len(builder.vertex_groups)} fragments)"
        )

    attrs_dict: dict[str, npt.NDArray] | None = None
    if new_attrs:
        attrs_dict = {
            k: np.atleast_1d(np.asarray(v)) for k, v in new_attrs.items()
        }

    if not atomic:
        # In-place replace: overwrite the array, refresh attribute
        # alignment.  Manifest references stay intact.
        builder.vertex_groups[ref.fragment] = rows.astype(
            builder.vertex_dtype, copy=True,
        )
        builder.vertices_dirty = True
        if attrs_dict:
            for name, val in attrs_dict.items():
                attr_list = builder.attr_groups.get(name)
                if attr_list is None:
                    continue
                attr_list[ref.fragment] = val.astype(
                    builder.attr_dtype.get(name, np.float32), copy=True,
                )
                builder.attrs_dirty[name] = True
        session._mark_edit(ref.level)
        return

    # Atomic: append a new fragment with the new rows, then redirect
    # every referring object's manifest entry from ref.fragment to
    # new_idx.
    new_idx = builder.append_fragment(rows, attrs=attrs_dict)
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)
    chunk_t = tuple(ref.chunk)
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest = [
            (chunk_t, int(new_idx)) if (
                tuple(cc) == chunk_t and fi == ref.fragment
            ) else (cc, fi)
            for (cc, fi) in manifest
        ]
        session._stage_manifest(ref.level, oid, new_manifest, atomic=True)
    session._mark_edit(ref.level)


def remove_fragment_in_session(
    session: EditSession,
    ref: FragmentRef,
    *,
    atomic: bool,
    propagate: PropagateTo,
) -> None:
    """Tombstone a fragment: empty its row list, keep its index.

    Under ``atomic=True``: referring object manifests are rewritten
    under fresh OIDs that drop the ``(chunk, ref.fragment)`` block;
    the original OIDs continue to point at the (now empty) fragment
    so the read path stays consistent for old readers.

    Under ``atomic=False``: refused (consistent with
    ``remove_vertex(atomic=False)``).  Use the atomic path and run
    ``vacuum`` later to physically compact.
    """
    if not atomic:
        raise EditError(
            "remove_fragment(atomic=False) would require physical "
            "fragment-index compaction; not supported in this iteration. "
            "Use atomic=True (tombstone) and run vacuum() later."
        )
    builder = session._builder(ref.level, ref.chunk)
    if ref.fragment < 0 or ref.fragment >= len(builder.vertex_groups):
        raise EditError(
            f"remove_fragment: fragment {ref.fragment} out of range "
            f"for chunk {ref.chunk}"
        )

    # Empty the fragment's row group + its parallel link / attr groups.
    builder.vertex_groups[ref.fragment] = np.empty(
        (0, builder.vertex_ndim), dtype=builder.vertex_dtype,
    )
    builder.vertices_dirty = True
    for name, attr_list in builder.attr_groups.items():
        if ref.fragment < len(attr_list):
            shape = attr_list[ref.fragment].shape
            new_shape = (0,) + shape[1:]
            attr_list[ref.fragment] = np.empty(
                new_shape,
                dtype=builder.attr_dtype.get(name, np.float32),
            )
            builder.attrs_dirty[name] = True
    for delta, link_list in builder.link_groups.items():
        if ref.fragment < len(link_list):
            shape = link_list[ref.fragment].shape
            new_shape = (0,) + shape[1:]
            link_list[ref.fragment] = np.empty(new_shape, dtype=np.int64)
            builder.links_dirty[delta] = True

    # Drop the fragment from every referring manifest (under fresh OIDs).
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)
    chunk_t = tuple(ref.chunk)
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest = [
            (cc, fi) for (cc, fi) in manifest
            if not (tuple(cc) == chunk_t and fi == ref.fragment)
        ]
        session._stage_manifest(ref.level, oid, new_manifest, atomic=True)
    session._mark_edit(ref.level)
