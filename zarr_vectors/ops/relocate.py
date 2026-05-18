"""Chunk-cross vertex relocation.

When an edited vertex's new position falls in a chunk other than its
source chunk, the edit engine routes through :func:`relocate_vertex_in_
session`.  The user is unaware of the chunk boundary — relocation is
seamless.

The source-row retention rule from the approved plan:

+--------+--------------------------+-----------------------+--------------+
| atomic | objects ref'ing src row  | propagate covers      | source row?  |
+========+==========================+=======================+==============+
| True   | any count                | any subset            | keep         |
| False  | 1 (single-object)        | the lone object       | delete       |
| False  | N > 1                    | all N objects         | delete       |
| False  | N > 1                    | a proper subset       | keep         |
+--------+--------------------------+-----------------------+--------------+

For every propagated object, the manifest's block referencing
``(src_chunk, src_frag, src_local)`` is rewritten to point at
``(dst_chunk, new_frag, 0)``.  Under ``atomic=True`` the rewritten
manifest is appended as a new OID; the original OID's manifest is left
untouched so old readers keep seeing the original position.

Link repartitioning and cross-level link rewrites are deferred to the
batched flush — the relocation helper just records the necessary
ops on the session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors.ops.refs import VertexRef
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession, PropagateTo


def relocate_vertex_in_session(
    session: EditSession,
    ref: VertexRef,
    new_pos: npt.NDArray[np.floating],
    new_cc: ChunkCoords,
    *,
    new_attrs: dict[str, npt.ArrayLike] | None = None,
    atomic: bool,
    propagate: PropagateTo,
) -> None:
    """Apply a chunk-crossing vertex move to ``session``'s change set.

    Steps:

    1. Append the moved vertex as a new single-row fragment in
       ``new_cc``.
    2. Decide whether to delete the source row per the retention rule.
    3. Stage the necessary manifest updates for every propagated object.

    Link-array repartitioning is staged but the heavy boundary-edge
    rewriting (intra → cross promotions) is handled lazily at flush.
    """
    # 1. Insert into target chunk.
    target_builder = session._builder(ref.level, new_cc)
    attrs_dict: dict[str, npt.NDArray] | None = None
    if new_attrs:
        attrs_dict = {
            k: np.atleast_1d(np.asarray(v))
            for k, v in new_attrs.items()
        }
    new_frag = target_builder.append_fragment(
        new_pos[np.newaxis, :], attrs=attrs_dict,
    )

    # 2. Find referring objects and decide retention.
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)

    delete_source = _should_delete_source(
        atomic=atomic,
        n_referring=len(affected),
        n_targeted=len(targets),
    )

    # 3. Update manifests for propagated objects.
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest = [
            (tuple(new_cc), new_frag) if (
                tuple(cc) == tuple(ref.chunk) and fi == ref.fragment
            ) else (cc, fi)
            for (cc, fi) in manifest
        ]
        session._stage_manifest(
            ref.level, oid, new_manifest, atomic=atomic,
        )

    # 4. Optionally delete the source row.
    if delete_source:
        source_builder = session._builder(ref.level, ref.chunk)
        source_builder.drop_fragment_row(ref.fragment, ref.local)
        # Manifests for *non*-targeted objects still reference
        # (source_chunk, source_frag, source_local).  Under "delete
        # source", every referring object was a target by definition
        # (single-object or all-objects-selected), so this is safe.
        # The same row index is now stale; downstream rows have shifted
        # but the only references to this fragment were the ones we
        # just rewrote.


def _should_delete_source(
    *,
    atomic: bool,
    n_referring: int,
    n_targeted: int,
) -> bool:
    """Apply the source-row retention rule.

    True iff the row should be removed from the source chunk.
    """
    if atomic:
        return False
    if n_referring == 0:
        # An unreferenced row is "free to delete" — same as the
        # single-object delete case.
        return True
    if n_referring == 1:
        # Lone object: deletable iff it was targeted.
        return n_targeted == n_referring
    # n_referring > 1: deletable only when every referring object is
    # targeted (i.e. all-of-them are selected).
    return n_targeted == n_referring
