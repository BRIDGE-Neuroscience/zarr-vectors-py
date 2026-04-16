"""ZVF store creation, opening, and management.

This module isolates all storage I/O.  No other module in the package
should directly interact with the filesystem or zarr — everything goes
through the abstractions defined here.

When ``zarr >= 3.0`` is available the store uses native Zarr groups.
Otherwise a lightweight :class:`FsGroup` fallback is provided that
stores metadata as JSON files and chunk data as raw binary files in
a directory tree.  The fallback is fully functional for local
filesystems and sufficient for testing.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

from zarr_vectors.constants import (
    FORMAT_VERSION,
    PARAMETRIC_GROUP,
    RESOLUTION_PREFIX,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    ParametricTypeDef,
    serialise_parametric_types,
    deserialise_parametric_types,
)
from zarr_vectors.exceptions import MetadataError, StoreError


# ===================================================================
# Lightweight filesystem group (zarr-free fallback)
# ===================================================================

class FsGroup:
    """A minimal zarr-Group–compatible interface backed by a directory.

    Attributes are stored in ``<dir>/.zattrs`` as JSON.  Sub-groups
    are subdirectories.  Chunk data is stored as raw binary files
    within array subdirectories.

    This is *not* a full Zarr implementation — it provides just enough
    API surface for the rest of zarr-vectors to work without a zarr
    install.
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self._path = Path(path)
        if create:
            self._path.mkdir(parents=True, exist_ok=True)
        if not self._path.is_dir():
            raise StoreError(f"Store path does not exist: {self._path}")

    # --- attributes ---

    @property
    def attrs(self) -> _FsAttrs:
        """Dict-like access to ``.zattrs`` metadata."""
        return _FsAttrs(self._path / ".zattrs")

    # --- sub-groups ---

    def create_group(self, name: str, **kwargs: Any) -> FsGroup:
        """Create a sub-group (subdirectory).

        Raises:
            StoreError: If the group already exists.
        """
        child_path = self._path / name
        if child_path.exists():
            # Allow re-opening existing groups
            return FsGroup(child_path)
        return FsGroup(child_path, create=True)

    def require_group(self, name: str) -> FsGroup:
        """Get or create a sub-group."""
        child_path = self._path / name
        return FsGroup(child_path, create=True)

    def __getitem__(self, key: str) -> FsGroup:
        """Get a sub-group by name.

        Supports ``/``-separated paths: ``group["a/b/c"]``.

        Raises:
            StoreError: If the group does not exist.
        """
        parts = key.strip("/").split("/")
        current = self
        for part in parts:
            child_path = current._path / part
            if not child_path.is_dir():
                raise StoreError(f"Group '{key}' not found in {self._path}")
            current = FsGroup(child_path)
        return current

    def __contains__(self, key: str) -> bool:
        """Check if a sub-group or array exists."""
        return (self._path / key).exists()

    def __iter__(self) -> Iterator[str]:
        """Iterate over sub-group / array names."""
        if not self._path.is_dir():
            return iter([])
        return iter(
            entry.name
            for entry in sorted(self._path.iterdir())
            if entry.is_dir() and not entry.name.startswith(".")
        )

    # --- chunk I/O ---

    def write_bytes(self, array_name: str, chunk_key: str, data: bytes) -> None:
        """Write raw bytes to a chunk file.

        Args:
            array_name: Array name (e.g. ``"vertices"``).
            chunk_key: Chunk key (e.g. ``"0.0.0"``).
            data: Raw bytes.
        """
        arr_dir = self._path / array_name
        arr_dir.mkdir(parents=True, exist_ok=True)
        (arr_dir / chunk_key).write_bytes(data)

    def read_bytes(self, array_name: str, chunk_key: str) -> bytes:
        """Read raw bytes from a chunk file.

        Raises:
            StoreError: If the chunk does not exist.
        """
        chunk_path = self._path / array_name / chunk_key
        if not chunk_path.exists():
            raise StoreError(
                f"Chunk '{array_name}/{chunk_key}' not found in {self._path}"
            )
        return chunk_path.read_bytes()

    def chunk_exists(self, array_name: str, chunk_key: str) -> bool:
        """Check if a chunk file exists."""
        return (self._path / array_name / chunk_key).exists()

    def list_chunks(self, array_name: str) -> list[str]:
        """List all chunk keys for an array.

        Returns:
            Sorted list of chunk key strings (e.g. ``["0.0.0", "0.0.1"]``).
        """
        arr_dir = self._path / array_name
        if not arr_dir.is_dir():
            return []
        return sorted(
            f.name for f in arr_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )

    # --- array metadata ---

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
        """Write array metadata to ``<array>/.zattrs``."""
        arr_dir = self._path / array_name
        arr_dir.mkdir(parents=True, exist_ok=True)
        _write_json(arr_dir / ".zattrs", meta)

    def read_array_meta(self, array_name: str) -> dict[str, Any]:
        """Read array metadata from ``<array>/.zattrs``."""
        return _read_json(self._path / array_name / ".zattrs")

    def array_exists(self, array_name: str) -> bool:
        """Check if an array directory exists."""
        return (self._path / array_name).is_dir()

    # --- path ---

    @property
    def path(self) -> Path:
        """Filesystem path of this group."""
        return self._path

    def __repr__(self) -> str:
        return f"FsGroup({self._path})"


class _FsAttrs:
    """Dict-like wrapper around a ``.zattrs`` JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        return _read_json(self._path)

    def _save(self, d: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(self._path, d)

    def __getitem__(self, key: str) -> Any:
        d = self._load()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __setitem__(self, key: str, value: Any) -> None:
        d = self._load()
        d[key] = value
        self._save(d)

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def update(self, other: dict[str, Any]) -> None:
        d = self._load()
        d.update(other)
        self._save(d)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of all attributes as a plain dict."""
        return self._load()

    def __repr__(self) -> str:
        return f"_FsAttrs({self._path})"


# ===================================================================
# JSON helpers
# ===================================================================

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ===================================================================
# Store API
# ===================================================================

def create_store(
    path: str | Path,
    root_metadata: RootMetadata,
) -> FsGroup:
    """Create a new ZVF store.

    Creates the root directory, writes root metadata, and creates
    the ``resolution_0/`` and ``parametric/`` groups.

    Args:
        path: Filesystem path for the store (should end in ``.zarr``).
        root_metadata: Root metadata to write.

    Returns:
        The root :class:`FsGroup`.

    Raises:
        StoreError: If the path already exists.
        MetadataError: If root_metadata is invalid.
    """
    path = Path(path)
    if path.exists():
        raise StoreError(f"Store already exists at {path}")

    root_metadata.validate()

    root = FsGroup(path, create=True)

    # Write root metadata
    root.attrs.update(root_metadata.to_dict())

    # Create resolution_0 group
    level0 = root.create_group(f"{RESOLUTION_PREFIX}0")

    # Create parametric group
    root.create_group(PARAMETRIC_GROUP)

    return root


def open_store(
    path: str | Path,
    mode: str = "r",
) -> FsGroup:
    """Open an existing ZVF store.

    Args:
        path: Filesystem path to the store.
        mode: ``"r"`` (read-only), ``"r+"`` (read-write), ``"a"`` (append).

    Returns:
        The root :class:`FsGroup`.

    Raises:
        StoreError: If the path does not exist or is not a valid ZVF store.
        MetadataError: If root metadata cannot be parsed.
    """
    path = Path(path)
    if not path.is_dir():
        raise StoreError(f"Store not found at {path}")

    root = FsGroup(path)

    # Validate root metadata exists and is parseable
    attrs = root.attrs.to_dict()
    if "zarr_vectors" not in attrs:
        raise StoreError(
            f"Not a valid ZVF store: missing 'zarr_vectors' in root .zattrs "
            f"at {path}"
        )

    # Parse to validate (will raise MetadataError if malformed)
    RootMetadata.from_dict(attrs)

    return root


def create_resolution_level(
    root: FsGroup,
    level: int,
    level_metadata: LevelMetadata,
) -> FsGroup:
    """Create a new resolution level group within the store.

    Args:
        root: Root store group.
        level: Level index (0, 1, 2, ...).
        level_metadata: Metadata for this level.

    Returns:
        The new level :class:`FsGroup`.

    Raises:
        MetadataError: If level_metadata is invalid.
    """
    level_metadata.validate()
    group_name = f"{RESOLUTION_PREFIX}{level}"
    level_group = root.require_group(group_name)
    level_group.attrs.update(level_metadata.to_dict())
    return level_group


def get_resolution_level(root: FsGroup, level: int) -> FsGroup:
    """Get an existing resolution level group.

    Raises:
        StoreError: If the level does not exist.
    """
    group_name = f"{RESOLUTION_PREFIX}{level}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level} not found in store")
    return root[group_name]


def list_resolution_levels(root: FsGroup) -> list[int]:
    """Return sorted list of resolution level indices present in the store."""
    levels: list[int] = []
    for name in root:
        if name.startswith(RESOLUTION_PREFIX):
            try:
                idx = int(name[len(RESOLUTION_PREFIX):])
                levels.append(idx)
            except ValueError:
                continue
    return sorted(levels)


def get_parametric_group(root: FsGroup) -> FsGroup:
    """Get the ``/parametric/`` group, creating it if needed."""
    return root.require_group(PARAMETRIC_GROUP)


def read_root_metadata(root: FsGroup) -> RootMetadata:
    """Read and parse root metadata from the store.

    Raises:
        MetadataError: If metadata is missing or malformed.
    """
    attrs = root.attrs.to_dict()
    return RootMetadata.from_dict(attrs)


def read_level_metadata(root: FsGroup, level: int) -> LevelMetadata:
    """Read and parse level metadata.

    Raises:
        StoreError: If the level does not exist.
        MetadataError: If metadata is malformed.
    """
    level_group = get_resolution_level(root, level)
    attrs = level_group.attrs.to_dict()
    return LevelMetadata.from_dict(attrs)


def write_parametric_types(
    root: FsGroup,
    types: list[ParametricTypeDef],
) -> None:
    """Write parametric type registry to ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    para.attrs.update(serialise_parametric_types(types))


def read_parametric_types(root: FsGroup) -> list[ParametricTypeDef]:
    """Read parametric type registry from ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    return deserialise_parametric_types(para.attrs.to_dict())


def store_info(root: FsGroup) -> dict[str, Any]:
    """Return summary information about a ZVF store."""
    meta = read_root_metadata(root)
    levels = list_resolution_levels(root)

    level_summaries: list[dict[str, Any]] = []
    for lvl in levels:
        try:
            lm = read_level_metadata(root, lvl)
            level_summaries.append({
                "level": lm.level,
                "vertex_count": lm.vertex_count,
                "bin_shape": list(lm.bin_shape) if lm.bin_shape else None,
                "bin_ratio": list(lm.bin_ratio) if lm.bin_ratio else None,
                "object_sparsity": lm.object_sparsity,
                "coarsening_method": lm.coarsening_method,
                "arrays_present": lm.arrays_present,
            })
        except (MetadataError, StoreError):
            level_summaries.append({"level": lvl, "error": "unreadable"})

    ptypes = read_parametric_types(root)

    return {
        "format_version": meta.format_version,
        "geometry_types": meta.geometry_types,
        "spatial_index_dims": meta.spatial_index_dims,
        "chunk_shape": list(meta.chunk_shape),
        "base_bin_shape": list(meta.base_bin_shape) if meta.base_bin_shape else None,
        "bins_per_chunk": list(meta.bins_per_chunk),
        "bounds": [list(meta.bounds[0]), list(meta.bounds[1])],
        "links_convention": meta.links_convention,
        "object_index_convention": meta.object_index_convention,
        "cross_chunk_strategy": meta.cross_chunk_strategy,
        "reduction_factor": meta.reduction_factor,
        "levels": level_summaries,
        "parametric_types": [t.to_dict() for t in ptypes],
    }


# ===================================================================
# Manual level management
# ===================================================================

def add_resolution_level(
    root: FsGroup,
    level_index: int,
    bin_ratio: tuple[int, ...],
    *,
    object_sparsity: float = 1.0,
    coarsening_method: str = "manual",
    parent_level: int | None = None,
) -> FsGroup:
    """Create a new resolution level with a specified bin ratio.

    Computes bin_shape from the store's base_bin_shape and the given
    ratio, validates that chunk_shape is divisible, creates the level
    directory with metadata, and returns a handle for writing data.

    Args:
        root: Root store group (must be opened read-write).
        level_index: Level number to create (must not already exist).
        bin_ratio: Integer fold-change per dimension relative to level 0.
        object_sparsity: Fraction of objects to retain (0, 1].
        coarsening_method: Description of how this level was generated.
        parent_level: Source level index. Defaults to ``level_index - 1``.

    Returns:
        The new level :class:`FsGroup`, ready for array writes.

    Raises:
        StoreError: If the level already exists.
        MetadataError: If bin_ratio produces a bin_shape that doesn't
            divide chunk_shape, or other validation failures.
    """
    from zarr_vectors.core.metadata import (
        compute_bin_shape,
        validate_bin_shape_divides_chunk,
    )

    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    chunk_shape = meta.chunk_shape

    # Compute the new bin shape
    bin_shape = compute_bin_shape(base_bin, bin_ratio)

    # Validate divisibility
    validate_bin_shape_divides_chunk(chunk_shape, bin_shape)

    if parent_level is None:
        parent_level = max(0, level_index - 1)

    # Check level doesn't already exist
    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name in root:
        raise StoreError(f"Resolution level {level_index} already exists")

    level_meta = LevelMetadata(
        level=level_index,
        vertex_count=0,  # will be updated after writing data
        arrays_present=[],
        bin_shape=bin_shape,
        bin_ratio=bin_ratio,
        object_sparsity=object_sparsity,
        coarsening_method=coarsening_method,
        parent_level=parent_level,
    )
    level_meta.validate()

    level_group = root.require_group(group_name)
    level_group.attrs.update(level_meta.to_dict())
    return level_group


def remove_resolution_level(root: FsGroup, level_index: int) -> None:
    """Remove a resolution level from the store.

    Deletes the level directory and all its contents. Level 0 cannot
    be removed.

    Args:
        root: Root store group.
        level_index: Level to remove.

    Raises:
        StoreError: If the level doesn't exist or is level 0.
    """
    if level_index == 0:
        raise StoreError("Cannot remove level 0 (full resolution)")

    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level_index} not found")

    import shutil
    level_path = root.path / group_name
    shutil.rmtree(level_path)


def list_available_ratios(root: FsGroup) -> list[tuple[int, ...]]:
    """Return the bin ratios for all existing resolution levels.

    Level 0 always has ratio ``(1, 1, ..., 1)``. Levels without an
    explicit bin_ratio in metadata are computed from bin_shape /
    base_bin_shape.

    Args:
        root: Root store group.

    Returns:
        Sorted list of bin ratio tuples, one per level.
    """
    from zarr_vectors.core.metadata import compute_bin_ratio

    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)
    ratios: list[tuple[int, ...]] = []

    for lvl in levels:
        if lvl == 0:
            ratios.append(tuple(1 for _ in range(ndim)))
            continue
        try:
            lm = read_level_metadata(root, lvl)
            if lm.bin_ratio is not None:
                ratios.append(lm.bin_ratio)
            elif lm.bin_shape is not None:
                ratios.append(compute_bin_ratio(base_bin, lm.bin_shape))
            else:
                ratios.append(tuple(1 for _ in range(ndim)))
        except Exception:
            ratios.append(tuple(1 for _ in range(ndim)))

    return ratios
