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
    # 1. Insert into target chunk.  Carry the source row's per-vertex
    # attributes forward so the new fragment is the same logical vertex,
    # not a zero-filled stub.  Caller-supplied new_attrs override.
    source_builder = session._builder(ref.level, ref.chunk)
    carry = _carry_source_attrs(session, source_builder, ref.fragment, ref.local)
    if new_attrs:
        for k, v in new_attrs.items():
            carry[k] = np.atleast_1d(np.asarray(v))
    target_builder = session._builder(ref.level, new_cc)
    # ``ChunkChangeBuilder.append_fragment`` only forwards values for
    # attribute names already loaded on the target builder.  Load each
    # carried attribute eagerly so the new fragment's row receives the
    # correct value rather than the zero-fill default.
    for name in carry:
        target_builder.require_attribute(session.root, name)
    new_frag = target_builder.append_fragment(
        new_pos[np.newaxis, :], attrs=carry or None,
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


def _carry_source_attrs(
    session: EditSession,
    source_builder,
    fragment: int,
    local: int,
) -> dict[str, npt.NDArray]:
    """Decode the source row's per-vertex attributes for forwarding.

    Walks every attribute name that exists under
    ``<level>/vertex_attributes/`` on disk (or is already loaded in the
    builder), loads it via ``require_attribute``, and returns
    ``{name: array_of_one_row}``.  Caller-supplied ``new_attrs`` are
    applied on top by the relocate routine, so explicit overrides win
    over inherited values.
    """
    from zarr_vectors.constants import VERTEX_ATTRIBUTES
    from zarr_vectors.core.store import get_resolution_level

    out: dict[str, npt.NDArray] = {}
    # Names that were already loaded in this session — guaranteed
    # available even if the on-disk listing is racy.
    names: set[str] = set(source_builder.attr_groups.keys())
    # Plus anything on disk under <level>/vertex_attributes/.  The
    # intermediate ``vertex_attributes/`` directory is not a proper
    # zarr group (no zarr.json metadata), so ``__iter__`` / ``__contains__``
    # don't see it — list the underlying store keys with the prefix
    # instead.
    try:
        level_group = get_resolution_level(session.root, source_builder.level)
        names.update(_list_attr_names_from_store(level_group, VERTEX_ATTRIBUTES))
    except Exception:
        pass

    for name in names:
        attr_list = source_builder.require_attribute(session.root, name)
        if fragment >= len(attr_list):
            continue
        group = attr_list[fragment]
        if local >= group.shape[0]:
            continue
        out[name] = np.atleast_1d(group[local].copy())
    return out


def _list_attr_names_from_store(level_group, parent: str) -> list[str]:
    """List attribute names under ``<level>/<parent>/`` via the zarr
    store's ``list_prefix`` primitive.

    ``parent`` is e.g. ``"vertex_attributes"``.  The intermediate
    directory isn't a proper zarr group, so we walk the underlying
    store keys.  Returns the sorted unique segment names directly
    under the prefix.
    """
    import asyncio

    zg = level_group._zarr
    store = zg.store
    level_path = zg.path.strip("/")
    prefix = f"{level_path}/{parent}/" if level_path else f"{parent}/"
    n_prefix = len(prefix)

    async def collect():
        names: set[str] = set()
        async for key in store.list_prefix(prefix):
            tail = key[n_prefix:]
            if not tail:
                continue
            name = tail.split("/", 1)[0]
            if name and not name.startswith("."):
                names.add(name)
        return names

    try:
        return sorted(asyncio.run(collect()))
    except RuntimeError:
        # Already-running event loop (rare in sync test paths) — fall
        # back to the empty list rather than crashing.
        return []


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
