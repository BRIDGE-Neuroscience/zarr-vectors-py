"""Edit-time operations on ZV stores.

Public surface:

- :class:`EditSession` — transactional context manager batching many
  edits into one flush + one commit, with optional pyramid refresh.
- Free functions ``edit_vertex`` / ``add_vertex`` / ``remove_vertex``
  for one-shot edits.
- Reference dataclasses :class:`VertexRef`, :class:`LinkRef`,
  :class:`FragmentRef`, :class:`ObjectRef`, :class:`AttributeRef` —
  physical addresses passed to the edit functions.
- :class:`EditReport` — diff summary returned by every edit.
- :func:`rebuild_pyramid_from_level` — refresh coarser levels after
  edits when ``refresh_pyramid=False`` was used.

The accepted edit semantics (atomic vs minimal, source-row retention
across chunk boundaries, propagate-to-objects) are documented in the
approved plan and in :mod:`zarr_vectors.ops.edit`.
"""

from zarr_vectors.ops.change_set import EditReport
from zarr_vectors.ops.edit import (
    EditSession,
    add_vertex,
    edit_vertex,
    remove_vertex,
)
from zarr_vectors.ops.refresh import rebuild_pyramid_from_level
from zarr_vectors.ops.refs import (
    AttributeRef,
    CrossChunkLinkRef,
    FragmentRef,
    LinkRef,
    ObjectRef,
    VertexRef,
)

__all__ = [
    "AttributeRef",
    "CrossChunkLinkRef",
    "EditReport",
    "EditSession",
    "FragmentRef",
    "LinkRef",
    "ObjectRef",
    "VertexRef",
    "add_vertex",
    "edit_vertex",
    "rebuild_pyramid_from_level",
    "remove_vertex",
]
