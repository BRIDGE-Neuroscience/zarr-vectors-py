"""ZV store creation, opening, and management.

Naming: the on-disk format is referred to as **ZV** (Zarr Vectors).  The
older ``ZVF`` initialism may still appear in archived doc text but is
not used in the wire format.

All storage I/O routes through a :class:`zarr.abc.store.Store` wrapped
by the :class:`Group` abstraction in :mod:`zarr_vectors.core.group`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import zarr
from zarr.storage import LocalStore

from zarr_vectors.constants import (
    FORMAT_VERSION,  # noqa: F401  (re-exported for callers)
    PARAMETRIC_GROUP,
    RESOLUTION_PREFIX,
)
from zarr_vectors.core.group import Group, _BackendShim
from zarr_vectors.core.metadata import (
    LevelMetadata,
    ParametricTypeDef,
    RootMetadata,
    deserialise_parametric_types,
    serialise_parametric_types,
)
from zarr_vectors.exceptions import MetadataError, StoreError


# ===================================================================
# Path / URL → Zarr store
# ===================================================================


def _resolve_local_path(path: str | Path) -> Path:
    """Coerce a path-or-``file://``-URL to a local :class:`Path`."""
    if isinstance(path, Path):
        return path
    if isinstance(path, str) and path.startswith("file://"):
        parsed = urlparse(path)
        p = unquote(parsed.path)
        if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return Path(p)
    return Path(path)


def _detect_scheme(url: str | Path) -> str:
    if isinstance(url, Path):
        return ""
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if len(scheme) <= 1:
        return ""
    return scheme


def _make_zarr_store(
    path: str | Path,
    *,
    backend: str | None = None,  # ignored for now; reserved for icechunk etc.
    **_kwargs: Any,
) -> LocalStore:
    """Construct a Zarr store for ``path``.

    Phase 1-3 scope: only local (``file://`` or path) stores supported.
    Cloud schemes (``s3://`` etc.) and Icechunk routing will land in a
    follow-up phase.
    """
    scheme = _detect_scheme(path)
    if scheme not in {"", "file"}:
        raise StoreError(
            f"URL scheme {scheme!r} not yet supported by the Zarr-native "
            f"store layer (phase 1-3 covers local paths only). "
            f"Cloud/icechunk routing arrives in a follow-up phase."
        )
    return LocalStore(_resolve_local_path(path))


# ===================================================================
# FsGroup — backwards-compatible shim
# ===================================================================


class FsGroup(Group):
    """Backwards-compatible subclass of :class:`Group` rooted at a local path.

    .. deprecated::
        Direct use of ``FsGroup`` is deprecated.  New code should call
        :func:`create_store` / :func:`open_store`.

    Args:
        path: Filesystem path or :class:`pathlib.Path`.  A ``file://``
            URL is also accepted.
        create: If True, create the directory if it does not already
            exist.  If False, raise :class:`StoreError` if the directory
            is missing (matching the legacy behaviour).
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        root = _resolve_local_path(path)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise StoreError(f"Store path does not exist: {root}")
        store = LocalStore(root)
        # Use mode="a" so the store is read-write but does not clobber
        # any existing root group.
        zg = zarr.open_group(store, mode="a")
        super().__init__(zg)


# ===================================================================
# Store API
# ===================================================================


def create_store(
    path: str | Path,
    root_metadata: RootMetadata,
    *,
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Create a new ZV store.

    Creates the root group, writes ``root_metadata``, and creates the
    ``resolution_0/`` and ``parametric/`` sub-groups.

    Args:
        path: URL or filesystem path for the new store (typically ends
            in ``.zarr`` or ``.zarrvectors``).
        root_metadata: Root metadata to write.
        backend: Force a particular backend (``"local"`` / ``"obstore"`` /
            ``"fsspec"``).  ``None`` (the default) auto-detects from the
            URL scheme.
        **backend_kwargs: Forwarded to the backend constructor (e.g.
            credentials).

    Returns:
        The root :class:`Group`.

    Raises:
        StoreError: If a store already exists at ``path``.
        MetadataError: If ``root_metadata`` is invalid.
    """
    root_metadata.validate()
    # Auto-stamp the capability that every 0.3 writer emits.  Doing it
    # here means callers don't have to remember.
    from zarr_vectors.constants import CAP_VERTEX_COUNT_CACHE
    if CAP_VERTEX_COUNT_CACHE not in root_metadata.format_capabilities:
        root_metadata.format_capabilities = [
            *root_metadata.format_capabilities,
            CAP_VERTEX_COUNT_CACHE,
        ]
    local_root: Path | None = None
    scheme = _detect_scheme(path)
    if scheme in {"", "file"}:
        local_root = _resolve_local_path(path)
        if local_root.exists() and local_root.is_dir() and any(local_root.iterdir()):
            raise StoreError(f"Store already exists at {local_root}")
        local_root.mkdir(parents=True, exist_ok=True)

    store = _make_zarr_store(path, backend=backend, **backend_kwargs)
    zg = zarr.open_group(store, mode="w")
    root = FsGroup.__new__(FsGroup) if isinstance(store, LocalStore) else Group.__new__(Group)
    root._zarr = zg

    root.attrs.update(root_metadata.to_dict())
    root.create_group(f"{RESOLUTION_PREFIX}0")
    root.create_group(PARAMETRIC_GROUP)
    return root


def open_store(
    path: str | Path,
    mode: str = "r",
    *,
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Open an existing ZV store.

    Args:
        path: URL or filesystem path to the store.
        mode: ``"r"`` (read-only), ``"r+"`` (read-write), ``"a"`` (append).
            Currently informational — actual mutability is governed by
            the backend's permissions.
        backend: Force a particular backend (auto-detect by default).
        **backend_kwargs: Forwarded to the backend constructor.

    Returns:
        The root :class:`Group`.

    Raises:
        StoreError: If the store does not exist or is structurally invalid.
        MetadataError: If root metadata cannot be parsed.
    """
    scheme = _detect_scheme(path)
    if scheme in {"", "file"}:
        local_root = _resolve_local_path(path)
        if not local_root.is_dir():
            raise StoreError(f"Store not found at {local_root}")

    store = _make_zarr_store(path, backend=backend, **backend_kwargs)
    # mode is informational at the public API — the underlying zarr group
    # is always opened read-write so that ``lazy.writer`` can write
    # through the same handle returned by ``open_store(mode='r')``.
    try:
        zg = zarr.open_group(store, mode="r+")
    except zarr.errors.GroupNotFoundError as e:
        raise StoreError(f"Not a valid ZV store at {path}: {e}") from None
    root = FsGroup.__new__(FsGroup) if isinstance(store, LocalStore) else Group.__new__(Group)
    root._zarr = zg

    attrs = root.attrs.to_dict()
    if "zarr_vectors" not in attrs:
        raise StoreError(
            f"Not a valid ZV store: missing 'zarr_vectors' in root attrs "
            f"at {root.url}"
        )

    RootMetadata.from_dict(attrs)  # validates; raises MetadataError if bad
    return root


def rebind(
    group: Group,
    backend: str | Any,
    **backend_kwargs: Any,
) -> Group:
    """Re-open the underlying store with a different driver (no data move).

    Under the Zarr-native layer, ``rebind`` opens a new Zarr store at
    the same URL and swaps it in.  For phases 1-3 only the local Zarr
    store is supported, so this is effectively a no-op unless the
    caller explicitly passes a different store.
    """
    old_url = group.url
    if isinstance(backend, str):
        new_store = _make_zarr_store(old_url, backend=backend, **backend_kwargs)
        new_url = old_url
    elif hasattr(backend, "set"):  # zarr.abc.store.Store duck-type
        new_store = backend
        new_url = group.url  # accept as-is
    else:
        # Legacy StorageBackend-shaped object (LocalBackend etc.) —
        # require its declared URL match the current one so we catch
        # programming mistakes rather than silently no-op'ing.
        new_url = getattr(backend, "url", None)
        if new_url is None:
            return group
        if _canonical(new_url) != _canonical(old_url):
            raise StoreError(
                f"rebind requires matching URLs; current store is at "
                f"{old_url!r}, new backend is at {new_url!r}. Use "
                f"open_store to point at a different location."
            )
        return group

    if _canonical(new_url) != _canonical(old_url):
        raise StoreError(
            f"rebind requires matching URLs; current store is at "
            f"{old_url!r}, new backend is at {new_url!r}. Use "
            f"open_store to point at a different location."
        )
    zg = zarr.open_group(new_store, mode="r+")
    group._zarr = zg
    return group


def _canonical(url: str) -> str:
    """Normalise a URL for cross-backend equality checks."""
    return url.rstrip("/").lower()


# ===================================================================
# Resolution levels
# ===================================================================


def create_resolution_level(
    root: Group,
    level: int,
    level_metadata: LevelMetadata,
) -> Group:
    """Create a new resolution level group within the store."""
    level_metadata.validate()
    group_name = f"{RESOLUTION_PREFIX}{level}"
    level_group = root.require_group(group_name)
    level_group.attrs.update(level_metadata.to_dict())
    return level_group


def get_resolution_level(root: Group, level: int) -> Group:
    """Get an existing resolution level group.

    Raises:
        StoreError: If the level does not exist.
    """
    group_name = f"{RESOLUTION_PREFIX}{level}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level} not found in store")
    return root[group_name]


def list_resolution_levels(root: Group) -> list[int]:
    """Return sorted level indices present in the store."""
    levels: list[int] = []
    for name in root:
        if name.startswith(RESOLUTION_PREFIX):
            try:
                idx = int(name[len(RESOLUTION_PREFIX):])
                levels.append(idx)
            except ValueError:
                continue
    return sorted(levels)


def get_parametric_group(root: Group) -> Group:
    """Get the ``/parametric/`` group, creating it if needed."""
    return root.require_group(PARAMETRIC_GROUP)


def read_root_metadata(root: Group) -> RootMetadata:
    """Read and parse root metadata from the store."""
    return RootMetadata.from_dict(root.attrs.to_dict())


def read_level_metadata(root: Group, level: int) -> LevelMetadata:
    """Read and parse level metadata.

    Raises:
        StoreError: If the level does not exist.
        MetadataError: If metadata is malformed.
    """
    level_group = get_resolution_level(root, level)
    return LevelMetadata.from_dict(level_group.attrs.to_dict())


def write_parametric_types(
    root: Group,
    types: list[ParametricTypeDef],
) -> None:
    """Write parametric type registry to ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    para.attrs.update(serialise_parametric_types(types))


def read_parametric_types(root: Group) -> list[ParametricTypeDef]:
    """Read parametric type registry from ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    return deserialise_parametric_types(para.attrs.to_dict())


def store_info(root: Group) -> dict[str, Any]:
    """Return summary information about a ZV store."""
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
    root: Group,
    level_index: int,
    bin_ratio: tuple[int, ...],
    *,
    object_sparsity: float = 1.0,
    coarsening_method: str = "manual",
    parent_level: int | None = None,
) -> Group:
    """Create a new resolution level with a specified bin ratio."""
    from zarr_vectors.core.metadata import (
        compute_bin_shape,
        validate_bin_shape_divides_chunk,
    )

    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    chunk_shape = meta.chunk_shape

    bin_shape = compute_bin_shape(base_bin, bin_ratio)
    validate_bin_shape_divides_chunk(chunk_shape, bin_shape)

    if parent_level is None:
        parent_level = max(0, level_index - 1)

    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name in root:
        raise StoreError(f"Resolution level {level_index} already exists")

    level_meta = LevelMetadata(
        level=level_index,
        vertex_count=0,
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


def remove_resolution_level(root: Group, level_index: int) -> None:
    """Remove a resolution level from the store.

    Level 0 cannot be removed.

    Raises:
        StoreError: If the level does not exist or is level 0.
    """
    if level_index == 0:
        raise StoreError("Cannot remove level 0 (full resolution)")

    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level_index} not found")

    root.delete_subtree(group_name)


def list_available_ratios(root: Group) -> list[tuple[int, ...]]:
    """Return bin ratios for all existing resolution levels."""
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
