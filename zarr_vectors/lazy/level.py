"""ZVRLevel — lazy handle to a single resolution level."""

from __future__ import annotations

from typing import Any

from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import FsGroup
from zarr_vectors.lazy.arrays import ZVRAttributeCollection, ZVRVertexCollection, ZVRObjectIndex
from zarr_vectors.typing import ChunkCoords

import numpy.typing as npt
import numpy as np


class ZVRLevel:
    """Lazy handle to one resolution level.

    Chunk listings are cached on first access.  Vertex and attribute
    data are not read until ``.compute()`` is called on the
    corresponding collection.

    Args:
        group: The level's :class:`FsGroup`.
        level_index: Integer level number (0, 1, ...).
        root_meta: Root-level metadata (for chunk/bin shapes).
        level_meta: Level-specific metadata, or None if unavailable.
    """

    def __init__(
        self,
        group: FsGroup,
        level_index: int,
        root_meta: RootMetadata,
        level_meta: LevelMetadata | None,
    ) -> None:
        self._group = group
        self._level_index = level_index
        self._root_meta = root_meta
        self._level_meta = level_meta
        self._chunk_keys_cache: list[ChunkCoords] | None = None
        self._vertices_cache: ZVRVertexCollection | None = None
        self._attributes_cache: dict[str, ZVRAttributeCollection] = {}
        self._object_index_cache: ZVRObjectIndex | None = None

    # ---------------------------------------------------------------
    # Metadata properties
    # ---------------------------------------------------------------

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def vertex_count(self) -> int:
        """Vertex count from metadata (no data I/O)."""
        if self._level_meta is not None:
            return self._level_meta.vertex_count
        # Fallback: read from attrs
        attrs = self._group.attrs.to_dict()
        zvl = attrs.get("zarr_vectors_level", attrs)
        return zvl.get("vertex_count", 0)

    @property
    def bin_shape(self) -> tuple[float, ...] | None:
        if self._level_meta is not None:
            return self._level_meta.bin_shape
        return None

    @property
    def bin_ratio(self) -> tuple[int, ...] | None:
        if self._level_meta is not None:
            return self._level_meta.bin_ratio
        return None

    @property
    def object_sparsity(self) -> float:
        if self._level_meta is not None:
            return self._level_meta.object_sparsity
        return 1.0

    @property
    def chunk_keys(self) -> list[ChunkCoords]:
        """List of chunk coordinate tuples with data at this level."""
        if self._chunk_keys_cache is None:
            self._chunk_keys_cache = list_chunk_keys(self._group)
        return self._chunk_keys_cache

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_keys)

    # ---------------------------------------------------------------
    # Lazy collections
    # ---------------------------------------------------------------

    @property
    def vertices(self) -> ZVRVertexCollection:
        """Lazy vertex collection for this level."""
        if self._vertices_cache is None:
            self._vertices_cache = ZVRVertexCollection(
                level_group=self._group,
                chunk_keys=self.chunk_keys,
                ndim=self._root_meta.sid_ndim,
                dtype="float32",
                vertex_count=self.vertex_count,
            )
        return self._vertices_cache

    @property
    def attributes(self) -> _AttributeAccessor:
        """Dict-like access to lazy attribute collections.

        Usage::

            level.attributes["intensity"].compute()
        """
        return _AttributeAccessor(self)

    def _get_attribute(self, name: str) -> ZVRAttributeCollection:
        if name not in self._attributes_cache:
            self._attributes_cache[name] = ZVRAttributeCollection(
                level_group=self._group,
                attr_name=name,
                chunk_keys=self.chunk_keys,
            )
        return self._attributes_cache[name]

    @property
    def object_index(self) -> ZVRObjectIndex:
        """Lazy object index accessor."""
        if self._object_index_cache is None:
            self._object_index_cache = ZVRObjectIndex(self._group)
        return self._object_index_cache

    # ---------------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------------

    def filter(
        self,
        *,
        bbox: tuple[npt.NDArray, npt.NDArray] | None = None,
        object_ids: list[int] | None = None,
        group_ids: list[int] | None = None,
    ) -> "ZVRView":
        """Apply filter constraints, returning a lazy filtered view.

        Filters can be chained: ``level.filter(group_ids=[0]).filter(bbox=roi)``.

        Args:
            bbox: Bounding box ``(min_corner, max_corner)``.
            object_ids: Keep only these object IDs.
            group_ids: Keep only objects in these groups.

        Returns:
            A :class:`ZVRView` with the specified constraints.
        """
        from zarr_vectors.lazy.views import ZVRView, FilterSpec
        view = ZVRView(
            self._group, self._root_meta, self._level_meta,
            self.chunk_keys, FilterSpec(),
        )
        return view.filter(
            bbox=bbox, object_ids=object_ids, group_ids=group_ids,
        )

    # ---------------------------------------------------------------
    # Geometry-specific collections
    # ---------------------------------------------------------------

    @property
    def polylines(self) -> "ZVRPolylineCollection":
        """Lazy polyline collection for streamline/polyline geometry.

        Each polyline is accessible by object ID::

            level.polylines[42].compute()  # full streamline
        """
        from zarr_vectors.lazy.views import ZVRPolylineCollection
        return ZVRPolylineCollection(self._group, ndim=self._root_meta.sid_ndim)

    def rechunk(
        self,
        by: str,
        *,
        bins: list[float] | None = None,
        output: str | None = None,
    ) -> dict:
        """Rechunk this level's store along a non-spatial dimension.

        Convenience wrapper around :func:`zarr_vectors.rechunk.rechunk`.

        Args:
            by: Dimension to rechunk by (``"group"``, ``"object_id"``,
                ``"attribute:<name>"``).
            bins: Bin edges for continuous values.
            output: Output path. If None, rechunks in-place.

        Returns:
            Summary dict from the rechunk engine.
        """
        from zarr_vectors.rechunk import rechunk as _rechunk, RechunkSpec
        store_path = str(self._group.path.parent)
        spec = RechunkSpec(by=by, bins=bins)
        return _rechunk(store_path, spec, output=output)

    # ---------------------------------------------------------------
    # Repr
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        bs = self.bin_shape or self._root_meta.effective_bin_shape
        return (
            f"ZVRLevel({self._level_index}, "
            f"vertices={self.vertex_count}, "
            f"chunks={self.chunk_count}, "
            f"bin_shape={bs})"
        )


class _AttributeAccessor:
    """Dict-like proxy for lazy attribute access."""

    def __init__(self, level: ZVRLevel) -> None:
        self._level = level

    def __getitem__(self, name: str) -> ZVRAttributeCollection:
        return self._level._get_attribute(name)

    def __contains__(self, name: str) -> bool:
        try:
            self._level._group.read_array_meta(f"attributes/{name}")
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"AttributeAccessor(level={self._level.level_index})"
