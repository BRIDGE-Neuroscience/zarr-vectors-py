"""Lazy array wrappers for vertices, attributes, and object indices.

Each collection defers I/O until ``.compute()`` or ``.to_delayed()``
is called.  When Dask is available, parallel I/O is used automatically.
When Dask is not installed, operations fall back to synchronous reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_vertices,
    read_object_vertices,
    read_vertex_group,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.typing import ChunkCoords

# Try to import dask — graceful fallback if unavailable
try:
    import dask
    from dask import delayed as dask_delayed

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dask = None  # type: ignore

    def dask_delayed(func):  # type: ignore
        """Fallback: wrap function so .compute() calls it immediately."""

        class _FakeDelayed:
            def __init__(self, *args, **kwargs):
                self._func = func
                self._args = args
                self._kwargs = kwargs

            def compute(self):
                return self._func(*self._args, **self._kwargs)

            def __repr__(self):
                return f"Delayed({self._func.__name__})"

        class _Wrapper:
            def __call__(self, *args, **kwargs):
                return _FakeDelayed(*args, **kwargs)

        return _Wrapper()


# ===================================================================
# Vertex collection
# ===================================================================


class ZVRVertexCollection:
    """Lazy collection of vertices across chunks.

    No data is read until ``.compute()`` or ``.to_delayed()`` is called.

    Args:
        level_group: Resolution level FsGroup handle.
        chunk_keys: List of chunk coordinate tuples.
        ndim: Number of spatial dimensions.
        dtype: Vertex dtype string.
        vertex_count: Total vertex count (from metadata).
    """

    def __init__(
        self,
        level_group: FsGroup,
        chunk_keys: list[ChunkCoords],
        ndim: int = 3,
        dtype: str = "float32",
        vertex_count: int = 0,
    ) -> None:
        self._group = level_group
        self._chunk_keys = chunk_keys
        self._ndim = ndim
        self._dtype = np.dtype(dtype)
        self._vertex_count = vertex_count

    def __len__(self) -> int:
        """Vertex count from metadata (no I/O)."""
        return self._vertex_count

    def __getitem__(self, *coords) -> Any:
        """Lazy access to a single chunk by coordinates.

        Usage::

            verts[0, 0, 0].compute()  # read chunk (0,0,0)
        """
        # Handle zvr[0].vertices[0, 0, 0] and zvr[0].vertices[(0, 0, 0)]
        if len(coords) == 1 and isinstance(coords[0], tuple):
            chunk_coords = coords[0]
        else:
            chunk_coords = coords

        return _delayed_read_chunk(
            self._group, chunk_coords, self._dtype, self._ndim,
        )

    def to_delayed(self) -> list:
        """Return a list of delayed objects, one per chunk.

        Each delayed object, when computed, returns an ``(M, D)``
        numpy array of vertex positions for that chunk.

        Returns:
            List of delayed objects (dask.delayed if Dask is available,
            otherwise a synchronous fallback).
        """
        return [
            _delayed_read_chunk(self._group, ck, self._dtype, self._ndim)
            for ck in self._chunk_keys
        ]

    def compute(self) -> npt.NDArray[np.floating]:
        """Materialise all vertices into a single numpy array.

        Reads all chunks and concatenates.  Uses Dask for parallel
        I/O when available.

        Returns:
            ``(N, D)`` array of vertex positions.
        """
        if not self._chunk_keys:
            return np.zeros((0, self._ndim), dtype=self._dtype)

        delayed_chunks = self.to_delayed()

        if HAS_DASK and len(delayed_chunks) > 1:
            results = dask.compute(*delayed_chunks)
        else:
            results = [d.compute() for d in delayed_chunks]

        # Filter out empties and concatenate
        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return np.zeros((0, self._ndim), dtype=self._dtype)
        return np.concatenate(non_empty, axis=0)

    def __iter__(self):
        """Iterate over chunks lazily, yielding delayed objects."""
        for ck in self._chunk_keys:
            yield _delayed_read_chunk(self._group, ck, self._dtype, self._ndim)

    def __repr__(self) -> str:
        return (
            f"ZVRVertexCollection("
            f"chunks={len(self._chunk_keys)}, "
            f"vertices={self._vertex_count}, "
            f"ndim={self._ndim}, "
            f"dtype={self._dtype})"
        )


# ===================================================================
# Attribute collection
# ===================================================================


class ZVRAttributeCollection:
    """Lazy collection of per-vertex attributes across chunks.

    Args:
        level_group: Resolution level FsGroup handle.
        attr_name: Name of the attribute (e.g. ``"intensity"``).
        chunk_keys: List of chunk coordinate tuples.
    """

    def __init__(
        self,
        level_group: FsGroup,
        attr_name: str,
        chunk_keys: list[ChunkCoords],
    ) -> None:
        self._group = level_group
        self._attr_name = attr_name
        self._chunk_keys = chunk_keys

    def to_delayed(self) -> list:
        """One delayed object per chunk."""
        return [
            _delayed_read_attribute(self._group, self._attr_name, ck)
            for ck in self._chunk_keys
        ]

    def compute(self) -> npt.NDArray:
        """Materialise all attribute values into a single array."""
        if not self._chunk_keys:
            return np.array([], dtype=np.float32)

        delayed_chunks = self.to_delayed()

        if HAS_DASK and len(delayed_chunks) > 1:
            results = dask.compute(*delayed_chunks)
        else:
            results = [d.compute() for d in delayed_chunks]

        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return np.array([], dtype=np.float32)
        return np.concatenate(non_empty, axis=0)

    def __repr__(self) -> str:
        return (
            f"ZVRAttributeCollection('{self._attr_name}', "
            f"chunks={len(self._chunk_keys)})"
        )


# ===================================================================
# Object index
# ===================================================================


class ZVRObjectIndex:
    """Lazy accessor for object manifests.

    The full manifest list is loaded on first access and cached.

    Args:
        level_group: Resolution level FsGroup handle.
    """

    def __init__(self, level_group: FsGroup) -> None:
        self._group = level_group
        self._manifests: list | None = None

    def _ensure_loaded(self) -> list:
        if self._manifests is None:
            try:
                self._manifests = read_all_object_manifests(self._group)
            except Exception:
                self._manifests = []
        return self._manifests

    @property
    def object_count(self) -> int:
        """Number of objects in the manifest."""
        return len(self._ensure_loaded())

    @property
    def object_ids(self) -> list[int]:
        """List of object IDs."""
        return list(range(self.object_count))

    def __getitem__(self, object_id: int) -> list:
        """Get the manifest for a single object.

        Returns:
            List of ``(chunk_coords, vg_index)`` tuples.
        """
        manifests = self._ensure_loaded()
        if object_id < 0 or object_id >= len(manifests):
            raise IndexError(f"Object ID {object_id} out of range [0, {len(manifests)})")
        return manifests[object_id]

    def __len__(self) -> int:
        return self.object_count

    def __repr__(self) -> str:
        return f"ZVRObjectIndex(objects={self.object_count})"


# ===================================================================
# Delayed read helpers
# ===================================================================


@dask_delayed
def _read_chunk_all_vgs(
    group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype,
    ndim: int,
) -> npt.NDArray[np.floating]:
    """Read all vertex groups from a chunk and concatenate."""
    try:
        groups = read_chunk_vertices(group, chunk_coords, dtype=dtype, ndim=ndim)
        non_empty = [g for g in groups if len(g) > 0]
        if not non_empty:
            return np.zeros((0, ndim), dtype=dtype)
        return np.concatenate(non_empty, axis=0)
    except Exception:
        return np.zeros((0, ndim), dtype=dtype)


@dask_delayed
def _read_attribute_chunk(
    group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
) -> npt.NDArray:
    """Read all attribute groups from a chunk and concatenate."""
    try:
        # Try to detect ncols from array metadata
        ncols = 1
        try:
            meta = group.read_array_meta(f"attributes/{attr_name}")
            cn = meta.get("channel_names")
            if cn:
                ncols = len(cn)
        except Exception:
            pass

        try:
            groups = read_chunk_attributes(
                group, attr_name, chunk_coords,
                dtype=np.float32, ncols=ncols,
            )
            non_empty = [g for g in groups if len(g) > 0]
            if non_empty:
                return np.concatenate(non_empty, axis=0)
        except Exception:
            pass

        # Fallback: read raw bytes and decode as flat array
        from zarr_vectors.constants import ATTRIBUTES
        key = f"{chunk_coords[0]}" + "".join(f".{c}" for c in chunk_coords[1:])
        raw = group.read_bytes(f"{ATTRIBUTES}/{attr_name}", key)
        if len(raw) == 0:
            return np.array([], dtype=np.float32)
        arr = np.frombuffer(raw, dtype=np.float32)
        if ncols > 1:
            arr = arr.reshape(-1, ncols)
        return arr
    except Exception:
        return np.array([], dtype=np.float32)


def _delayed_read_chunk(
    group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype,
    ndim: int,
) -> Any:
    """Create a delayed read for a single chunk's vertices."""
    return _read_chunk_all_vgs(group, chunk_coords, dtype, ndim)


def _delayed_read_attribute(
    group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
) -> Any:
    """Create a delayed read for a single chunk's attributes."""
    return _read_attribute_chunk(group, attr_name, chunk_coords)
