"""ZVRStore — lazy wrapper around an open zarr vectors store.

No data is read until ``.compute()`` is called on a collection.
Metadata (root attrs, level attrs, chunk listings) is loaded on
first access and cached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.core.store import (
    FsGroup,
    list_resolution_levels,
    open_store,
    read_root_metadata,
    read_level_metadata,
)
from zarr_vectors.core.metadata import RootMetadata, LevelMetadata
from zarr_vectors.lazy.level import ZVRLevel


class ZVRStore:
    """Lazy handle to a zarr vectors store.

    Attributes are read from ``.zattrs`` on first access and cached.
    Resolution levels are accessed by integer index via ``__getitem__``.

    Args:
        root: An open :class:`FsGroup` for the store root.
        meta: Parsed :class:`RootMetadata`.
    """

    def __init__(self, root: FsGroup, meta: RootMetadata) -> None:
        self._root = root
        self._meta = meta
        self._levels_cache: dict[int, ZVRLevel] = {}
        self._level_list: list[int] | None = None

    # ---------------------------------------------------------------
    # Metadata properties (no data I/O)
    # ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Filesystem path of the store."""
        return self._root.path

    @property
    def chunk_shape(self) -> tuple[float, ...]:
        return self._meta.chunk_shape

    @property
    def bin_shape(self) -> tuple[float, ...]:
        """Effective bin shape (base_bin_shape or chunk_shape)."""
        return self._meta.effective_bin_shape

    @property
    def base_bin_shape(self) -> tuple[float, ...] | None:
        return self._meta.base_bin_shape

    @property
    def bins_per_chunk(self) -> tuple[int, ...]:
        return self._meta.bins_per_chunk

    @property
    def geometry_types(self) -> list[str]:
        return self._meta.geometry_types

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        return self._meta.bounds

    @property
    def ndim(self) -> int:
        return self._meta.sid_ndim

    @property
    def levels(self) -> list[int]:
        """Sorted list of available resolution level indices."""
        if self._level_list is None:
            self._level_list = list_resolution_levels(self._root)
        return self._level_list

    @property
    def format_version(self) -> str:
        return self._meta.format_version

    @property
    def headers(self) -> dict[str, Any]:
        """Dict of stored format headers, keyed by format name.

        Returns:
            ``{format_name: Header}`` for each stored header.
            Empty dict if no headers are stored.
        """
        from zarr_vectors.headers.registry import HeaderRegistry
        try:
            reg = HeaderRegistry(self._root)
            return {fmt: reg.get(fmt) for fmt in reg.available_formats}
        except Exception:
            return {}

    # ---------------------------------------------------------------
    # Level access
    # ---------------------------------------------------------------

    def __getitem__(self, level: int) -> ZVRLevel:
        """Get a lazy handle to a resolution level."""
        if level not in self._levels_cache:
            from zarr_vectors.core.store import get_resolution_level
            level_group = get_resolution_level(self._root, level)
            try:
                level_meta = read_level_metadata(self._root, level)
            except Exception:
                level_meta = None
            self._levels_cache[level] = ZVRLevel(
                level_group, level, self._meta, level_meta,
            )
        return self._levels_cache[level]

    def level(self, index: int) -> ZVRLevel:
        """Alias for ``self[index]``."""
        return self[index]

    # ---------------------------------------------------------------
    # Repr
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        types = ", ".join(self.geometry_types)
        return (
            f"ZVRStore('{self.path}', "
            f"levels={self.levels}, "
            f"geometry=[{types}], "
            f"chunk={self.chunk_shape}, "
            f"bin={self.bin_shape})"
        )


def open_zvr(path: str | Path) -> ZVRStore:
    """Open a zarr vectors store lazily.

    Reads only root metadata (a few KB). No vertex data is loaded.

    Args:
        path: Filesystem path to the ``.zarrvectors`` store.

    Returns:
        A :class:`ZVRStore` handle for lazy access.
    """
    root = open_store(str(path))
    meta = read_root_metadata(root)
    return ZVRStore(root, meta)
