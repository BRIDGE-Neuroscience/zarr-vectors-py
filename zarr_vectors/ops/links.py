"""Link edits: intra-chunk + cross-chunk + convention-aware.

Three storage shapes are involved:

- ``links/<delta>/<chunk>``: per-chunk ragged ints, one group per
  vertex fragment.  Indices are chunk-local.  This is where intra-chunk
  edges live; under ``implicit_sequential_with_branches`` it also holds
  branch-override rows (child → parent pairs that contradict the
  implicit ``parent = i-1`` baseline).
- ``cross_chunk_links/<delta>/data``: global flat array of
  ``(chunk_coords, vertex_idx)`` endpoint records.  This is where
  edges spanning chunks live.
- ``link_attributes/<name>/<delta>/<chunk>`` /
  ``cross_chunk_link_attributes/<name>/<delta>/data``: per-link
  attribute payloads parallel to the link rows.

Convention contract:

| ``links_convention`` | add / edit / remove behaviour |
|---|---|
| ``"explicit"`` | every link is a stored row; all edits work uniformly |
| ``"implicit_sequential"`` | no rows stored; all link edits raise (caller must promote the store via ``materialise_object_links_explicit``) |
| ``"implicit_sequential_with_branches"`` | branch-override rows only; add/edit/remove operate on the stored branch entries; structural removal of an implicit edge raises |

Atomic semantics for links are weaker than for vertices: links don't
carry OID identity directly.  Under ``atomic=True`` an edit appends a
new row and leaves the old one in place; under ``atomic=False`` the
row is overwritten.  Object manifests are unaffected by link edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
)
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import CrossChunkLinkRef, LinkRef
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession


# ---------------------------------------------------------------------
# Intra-chunk link edits
# ---------------------------------------------------------------------

def add_link_in_session(
    session: EditSession,
    *,
    level: int,
    src: int,
    dst: int,
    chunk: ChunkCoords | None = None,
    fragment: int | None = None,
    delta: int = 0,
    attrs: dict[str, npt.ArrayLike] | None = None,
) -> LinkRef | CrossChunkLinkRef:
    """Append a link.

    Args:
        src, dst: chunk-local vertex indices when ``chunk`` is given;
            otherwise interpreted as global indices and routed to
            cross-chunk storage when the endpoints fall in different
            chunks (currently raised; multi-chunk routing is a future
            iteration).
        chunk: required for intra-chunk add (``delta=0``).  The link is
            written into the link group for ``fragment`` (default: the
            first fragment in the chunk).
        delta: ``0`` for intra-level, non-zero for cross-level (not yet
            supported by this helper — call write_chunk_links directly).
        attrs: per-link attribute values to write alongside the new row.
    """
    _check_link_convention(session, write_branch_entry=True)
    if delta != 0:
        raise EditError(
            f"add_link(delta={delta}) is not supported in this iteration; "
            f"use the multi-resolution coarsen pipeline for cross-level links."
        )
    if chunk is None:
        raise EditError(
            "add_link requires chunk= to be supplied so the row can be "
            "placed in the correct per-chunk link group."
        )

    builder = session._builder(level, chunk)
    link_width = 2
    groups = builder.require_links(session.root, delta=0, link_width=link_width)
    if not groups:
        raise EditError(
            f"add_link: chunk {chunk} has no vertex fragments yet; add a "
            f"vertex first or call add_fragment()."
        )
    target_frag = 0 if fragment is None else int(fragment)
    if target_frag < 0 or target_frag >= len(groups):
        raise EditError(
            f"add_link: fragment {target_frag} out of range for chunk "
            f"{chunk} (has {len(groups)} fragments)"
        )
    row = np.asarray([int(src), int(dst)], dtype=np.int64)
    new_row_idx = builder.append_link_row(0, target_frag, row)

    if attrs:
        _write_link_attr_row(
            session, level, chunk, target_frag, new_row_idx, attrs,
        )

    session._mark_edit(level)
    return LinkRef(
        level=level,
        chunk=tuple(int(c) for c in chunk),
        fragment=target_frag,
        row=new_row_idx,
        delta=0,
    )


def edit_link_in_session(
    session: EditSession,
    ref: LinkRef,
    *,
    new_endpoints: tuple[int, int] | None,
    new_attrs: dict[str, npt.ArrayLike] | None,
    atomic: bool,
) -> None:
    _check_link_convention(session, write_branch_entry=True)
    if ref.delta != 0:
        raise EditError(
            f"edit_link(delta={ref.delta}) is not supported; "
            f"cross-level links are managed by the coarsen pipeline."
        )
    builder = session._builder(ref.level, ref.chunk)
    groups = builder.require_links(session.root, delta=0, link_width=2)
    if ref.fragment < 0 or ref.fragment >= len(groups):
        raise EditError(
            f"edit_link: fragment {ref.fragment} out of range for chunk "
            f"{ref.chunk}"
        )
    group = groups[ref.fragment]
    if ref.row < 0 or ref.row >= group.shape[0]:
        raise EditError(
            f"edit_link: row {ref.row} out of range in fragment "
            f"{ref.fragment} (size {group.shape[0]})"
        )

    if new_endpoints is not None:
        src, dst = int(new_endpoints[0]), int(new_endpoints[1])
        new_row = np.asarray([src, dst], dtype=group.dtype)
        if atomic:
            new_idx = builder.append_link_row(0, ref.fragment, new_row)
            target_row = new_idx
        else:
            builder.overwrite_link_row(0, ref.fragment, ref.row, new_row)
            target_row = ref.row
    else:
        target_row = ref.row

    if new_attrs:
        _write_link_attr_row(
            session, ref.level, ref.chunk, ref.fragment, target_row, new_attrs,
        )
    session._mark_edit(ref.level)


def remove_link_in_session(
    session: EditSession,
    ref: LinkRef,
    *,
    atomic: bool,
) -> None:
    """Drop the link row.

    Under ``atomic=True`` the row is removed from the per-fragment
    group (links have no OID identity, so atomic == minimal for the
    remove case — the kwarg is accepted for symmetry).
    """
    del atomic  # symmetric with edit_link/edit_vertex; not used today
    _check_link_convention(session, write_branch_entry=True)
    if ref.delta != 0:
        raise EditError(
            f"remove_link(delta={ref.delta}) is not supported; "
            f"cross-level links are managed by the coarsen pipeline."
        )
    builder = session._builder(ref.level, ref.chunk)
    builder.require_links(session.root, delta=0, link_width=2)
    builder.drop_link_row(0, ref.fragment, ref.row)
    session._mark_edit(ref.level)


# ---------------------------------------------------------------------
# Cross-chunk link edits
# ---------------------------------------------------------------------

def add_cross_chunk_link_in_session(
    session: EditSession,
    *,
    level: int,
    endpoints: list[tuple[ChunkCoords, int]],
    delta: int = 0,
) -> CrossChunkLinkRef:
    """Stage an append into ``cross_chunk_links/<delta>/data``."""
    _check_link_convention(session, write_branch_entry=True)
    from zarr_vectors.ops.change_set import CrossChunkLinkOp
    payload = [
        (tuple(int(c) for c in cc), int(vi)) for cc, vi in endpoints
    ]
    session._ccl_ops.append(
        CrossChunkLinkOp(op="append", delta=int(delta), payload=payload),
    )
    session._mark_edit(level)
    # The row index is only known after flush; we return a ref pointing
    # at the optimistic post-append position so callers can chain edits
    # within the same session (flush_ccl_ops applies ops in order).
    return CrossChunkLinkRef(
        level=level,
        row=_predict_ccl_row_index(session, level, delta),
        delta=int(delta),
    )


def edit_cross_chunk_link_in_session(
    session: EditSession,
    ref: CrossChunkLinkRef,
    *,
    new_endpoints: list[tuple[ChunkCoords, int]],
    atomic: bool,
) -> None:
    _check_link_convention(session, write_branch_entry=True)
    from zarr_vectors.ops.change_set import CrossChunkLinkOp
    payload = [
        (tuple(int(c) for c in cc), int(vi)) for cc, vi in new_endpoints
    ]
    if atomic:
        # Atomic = append the new row, leave the old one in place.
        session._ccl_ops.append(
            CrossChunkLinkOp(op="append", delta=ref.delta, payload=payload),
        )
    else:
        session._ccl_ops.append(
            CrossChunkLinkOp(
                op="overwrite", delta=ref.delta,
                payload=payload, index=ref.row,
            ),
        )
    session._mark_edit(ref.level)


def remove_cross_chunk_link_in_session(
    session: EditSession,
    ref: CrossChunkLinkRef,
) -> None:
    from zarr_vectors.ops.change_set import CrossChunkLinkOp
    session._ccl_ops.append(
        CrossChunkLinkOp(op="delete", delta=ref.delta, index=ref.row),
    )
    session._mark_edit(ref.level)


# ---------------------------------------------------------------------
# Materialise implicit-sequential as explicit branch table
# ---------------------------------------------------------------------

def materialise_object_links_explicit(
    root,
    level: int,
    object_id: int,
    *,
    flip_convention: bool = False,
) -> int:
    """Convert one object's implicit-sequential topology into explicit
    branch-table rows so :class:`EditSession` link edits can address
    every edge in the chain.

    Walks the object's manifest, lists every vertex in manifest order,
    and emits a ``(child, parent)`` row for every consecutive pair into
    the chunk containing the child.  The reader behaviour is unchanged
    (the same edges are produced) but downstream ``edit_link`` /
    ``remove_link`` calls can now address each edge by its row index.

    Args:
        root: ZV store group.
        level: resolution level.
        object_id: object whose chain to materialise.
        flip_convention: when True, also flip the store's
            ``links_convention`` to ``"explicit"``.  Off by default
            since the flip is global and affects every other object's
            read path.

    Returns:
        Number of branch-table rows added.
    """
    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        read_chunk_links,
        read_chunk_vertices,
        read_object_manifest,
        write_chunk_links,
    )
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.core.store import get_resolution_level

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    ndim = meta.sid_ndim
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv == LINKS_EXPLICIT:
        return 0  # already explicit; nothing to materialise

    level_group = get_resolution_level(root, level)
    manifest = read_object_manifest(level_group, object_id)
    if not manifest:
        return 0

    # Expand every manifest entry to a full per-vertex sequence of
    # ``(chunk, chunk_local_index)``.  The implicit_sequential reader
    # uses chunk-local row indices over the chunk's *flat* vertex
    # array, so we need that index too.
    chunk_cache: dict[ChunkCoords, list[int]] = {}

    def _fragment_start_in_chunk(cc: ChunkCoords, frag_idx: int) -> tuple[int, int]:
        """Return ``(start, count)`` of fragment ``frag_idx`` in the
        chunk's flat vertex array."""
        if cc not in chunk_cache:
            chunks = read_chunk_vertices(level_group, cc, ndim=ndim)
            chunk_cache[cc] = [int(g.shape[0]) for g in chunks]
        counts = chunk_cache[cc]
        if frag_idx >= len(counts):
            return (0, 0)
        return (sum(counts[:frag_idx]), counts[frag_idx])

    sequence: list[tuple[ChunkCoords, int]] = []  # (chunk, chunk_local_idx)
    for (cc, frag_idx) in manifest:
        start, count = _fragment_start_in_chunk(cc, frag_idx)
        for k in range(count):
            sequence.append((cc, start + k))
    if len(sequence) < 2:
        return 0

    # For each consecutive (parent, child) pair, append
    # ``(child_local, parent_local)`` to the child's chunk's link group.
    # When parent and child sit in different chunks we'd emit a
    # cross-chunk row; that path is deferred for this iteration and the
    # pair is skipped.
    grouped_writes: dict[ChunkCoords, list[npt.NDArray]] = {}
    n_added = 0
    for i in range(1, len(sequence)):
        parent_cc, parent_local = sequence[i - 1]
        child_cc, child_local = sequence[i]
        if parent_cc != child_cc:
            continue
        grouped_writes.setdefault(child_cc, []).append(
            np.array([child_local, parent_local], dtype=np.int64),
        )

    for cc, new_rows in grouped_writes.items():
        try:
            current_groups = read_chunk_links(level_group, cc, delta=0)
        except Exception:
            # No link array yet at this chunk — start with one empty
            # group per fragment.
            current_groups = []
        if not current_groups:
            chunks = read_chunk_vertices(level_group, cc, ndim=ndim)
            current_groups = [
                np.empty((0, 2), dtype=np.int64) for _ in chunks
            ]
        # Append every new row into the first fragment of the chunk
        # (the implicit_sequential reader doesn't care which fragment
        # holds the branch override — it walks all rows).
        appended = np.stack(new_rows, axis=0)
        current_groups[0] = np.concatenate(
            [current_groups[0], appended], axis=0,
        )
        write_chunk_links(level_group, cc, current_groups, delta=0)
        n_added += len(new_rows)

    if flip_convention:
        attrs = root.attrs.to_dict()
        zv = dict(attrs.get("zarr_vectors", {}))
        zv["links_convention"] = LINKS_EXPLICIT
        root.attrs.update({"zarr_vectors": zv})

    del list_chunk_keys  # silence unused-import warning
    return n_added


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _check_link_convention(
    session: EditSession,
    *,
    write_branch_entry: bool,
) -> None:
    """Refuse link edits on stores that can't represent them.

    ``implicit_sequential`` is the strictest: no link rows are stored
    at all.  ``implicit_sequential_with_branches`` accepts edits that
    write to the branch table.
    """
    from zarr_vectors.core.metadata import RootMetadata
    meta = RootMetadata.from_dict(session.root.attrs.to_dict())
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv == LINKS_EXPLICIT:
        return
    if conv == LINKS_IMPLICIT_BRANCHES and write_branch_entry:
        return
    raise EditError(
        f"link edits on store with links_convention={conv!r} are not "
        f"supported.  Run materialise_object_links_explicit(root, level, "
        f"object_id, flip_convention=True) first, or rewrite the store "
        f"with links_convention='explicit'."
    )


def _write_link_attr_row(
    session: EditSession,
    level: int,
    chunk: ChunkCoords,
    fragment: int,
    row: int,
    attrs: dict[str, npt.ArrayLike],
) -> None:
    """RMW a per-link attribute row.  Delegates to the attributes
    module's per-link path."""
    from zarr_vectors.ops.attributes import _edit_link_attr
    from zarr_vectors.ops.refs import AttributeRef
    for name, val in attrs.items():
        ref = AttributeRef(
            scope="link",
            name=name,
            target=LinkRef(
                level=level, chunk=chunk, fragment=fragment,
                row=row, delta=0,
            ),
        )
        _edit_link_attr(session, ref, val)


def _predict_ccl_row_index(
    session: EditSession,
    level: int,
    delta: int,
) -> int:
    """Best-effort guess of the row index a new CCL append will land at.

    Reads the current array length on disk plus any pending appends in
    this session for the same (level, delta).
    """
    from zarr_vectors.core.arrays import read_cross_chunk_links
    from zarr_vectors.core.store import get_resolution_level
    try:
        level_group = get_resolution_level(session.root, level)
        current = read_cross_chunk_links(level_group, delta=delta)
        base = len(current)
    except Exception:
        base = 0
    pending = sum(
        1 for op in session._ccl_ops
        if op.delta == delta and op.op == "append"
    )
    return base + pending - 1  # the just-appended op's row
