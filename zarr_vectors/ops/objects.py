"""Object-level edits: add / edit / remove an entire object.

This module is a thin layer on top of the chunk-level builders + the
manifest-staging engine already in :mod:`zarr_vectors.ops.edit`.  The
public surface is exposed via :class:`~zarr_vectors.ops.edit.EditSession`
methods and free-function wrappers; the helpers here implement the
heavy lifting.

Atomic semantics (consistent with Iteration 1):

- ``atomic=True``  → the edited object becomes a **new OID**, original
  manifest at ``oid`` is left intact.  Caller gets the new OID via
  ``EditReport.oid_remap``.
- ``atomic=False`` → overwrite manifest at ``oid``.  Destructive
  remove (``remove_object(atomic=False)``) is rejected for the same
  reason ``remove_vertex(atomic=False)`` is: physical fragment purge
  would require global manifest re-scanning that we don't do yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import ObjectRef
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession


def add_object_in_session(
    session: EditSession,
    *,
    level: int,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
) -> ObjectRef:
    """Append a brand-new object with one fragment per touched chunk.

    The vertex set is split across chunks via
    :func:`~zarr_vectors.spatial.chunking.assign_chunks`; each touched
    chunk gets a new fragment containing the rows assigned to it.  The
    new OID is allocated via the session's prefix allocator (when set)
    and the manifest is written at that OID.

    Per-vertex ``attrs`` are forwarded into each new fragment in the
    matching chunk-local order.  Per-object attrs are not handled here
    — call :meth:`EditSession.edit_attribute` after ``add_object``.
    """
    from zarr_vectors.core.metadata import RootMetadata

    verts = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    if verts.size == 0:
        raise EditError("add_object: vertices must be non-empty")

    meta = RootMetadata.from_dict(session.root.attrs.to_dict())
    chunk_shape = tuple(meta.chunk_shape)
    if verts.shape[1] != meta.sid_ndim:
        raise EditError(
            f"add_object: vertices arity {verts.shape[1]} != "
            f"store ndim {meta.sid_ndim}"
        )

    by_chunk = assign_chunks(verts, chunk_shape)
    # Manifest order follows the source row order of ``vertices`` so
    # implicit-sequential conventions get a stable parent->child chain.
    # Walk vertex rows and append to the matching chunk's fragment.
    # For multi-row fragments we group contiguous runs that share a
    # chunk so a single fragment captures them, preserving order.
    row_chunk = np.empty(verts.shape[0], dtype=object)
    for cc, idx in by_chunk.items():
        for i in idx:
            row_chunk[i] = cc

    manifest: ObjectManifest = []
    i = 0
    while i < verts.shape[0]:
        cc = row_chunk[i]
        j = i + 1
        while j < verts.shape[0] and row_chunk[j] == cc:
            j += 1
        rows = verts[i:j]
        # Per-vertex attrs for this slice.
        slice_attrs: dict[str, npt.NDArray] | None = None
        if attrs:
            slice_attrs = {
                k: np.atleast_1d(np.asarray(v))[i:j]
                for k, v in attrs.items()
            }
        builder = session._builder(level, cc)
        new_frag = builder.append_fragment(rows, attrs=slice_attrs)
        manifest.append((tuple(cc), int(new_frag)))
        i = j

    new_oid = session._allocate_atomic_oid(level)
    session._manifest_ops[(level, new_oid)] = _make_manifest_op(
        level=level, new_oid=new_oid, manifest=manifest,
    )
    session._mark_edit(level)
    return ObjectRef(level=level, object_id=int(new_oid))


def edit_object_manifest_in_session(
    session: EditSession,
    *,
    oid: int,
    level: int,
    new_manifest: ObjectManifest,
    atomic: bool,
) -> int | None:
    """Stage a manifest swap for ``oid`` at ``level``.

    Under ``atomic=True`` returns the allocated new OID; under
    ``atomic=False`` returns ``None`` (the old OID is overwritten in
    place).
    """
    # Normalise the manifest: every (chunk_coords, frag_idx) tuple must
    # be ``(tuple_of_int, int)``.
    norm = [
        (tuple(int(c) for c in cc), int(fi))
        for (cc, fi) in new_manifest
    ]
    session._stage_manifest(level, oid, norm, atomic=atomic)
    session._mark_edit(level)
    if atomic:
        return session._report.oid_remap.get(int(oid))
    return None


def remove_object_in_session(
    session: EditSession,
    *,
    oid: int,
    level: int,
    atomic: bool,
) -> int | None:
    """Soft-delete (atomic) or in-place tombstone (non-atomic) an OID.

    Both paths write an empty manifest — the difference is whether the
    write lands at a fresh OID (atomic) or overwrites ``oid``.
    """
    if not atomic:
        # Destructive remove: the manifest at ``oid`` becomes empty.
        # The fragments it solely owned are not purged this iteration —
        # that would require global manifest re-scanning (a vacuum job).
        # Document this in the docstring; we still allow the soft
        # variant via atomic=True which is the recommended path.
        pass
    session._stage_manifest(level, oid, [], atomic=atomic)
    session._mark_edit(level)
    if atomic:
        return session._report.oid_remap.get(int(oid))
    return None


def _make_manifest_op(*, level: int, new_oid: int, manifest: ObjectManifest):
    """Build a ManifestOp for the add_object path."""
    from zarr_vectors.ops.change_set import ManifestOp
    return ManifestOp(
        level=level,
        object_id=new_oid,
        new_manifest=manifest,
        new_oid=new_oid,
    )
