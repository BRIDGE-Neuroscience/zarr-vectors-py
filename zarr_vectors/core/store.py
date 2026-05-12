"""ZV store creation, opening, and management.

Naming: the on-disk format is referred to as **ZV** (Zarr Vectors).  The
older ``ZVF`` initialism may still appear in archived doc text but is
not used in the wire format.


This module is the public entry point for store-level operations.  All
storage I/O flows through a pluggable :class:`StorageBackend`, exposed via
the unified :class:`Group` abstraction in :mod:`zarr_vectors.core.group`.

Backwards compatibility: ``FsGroup`` is preserved as a thin subclass of
:class:`Group` that accepts the legacy ``(path, *, create=False)``
constructor and is always backed by :class:`LocalBackend`.  Existing code
that imports ``FsGroup`` from this module continues to work unchanged.

Backend selection order (when no explicit ``backend=`` is passed):

1. ``ZARR_VECTORS_BACKEND`` environment variable.
2. URL-scheme auto-detect (see :mod:`zarr_vectors.core.backends`).

Filesystem paths and ``file://`` URLs route to :class:`LocalBackend`.
Cloud URL schemes route to ``obstore`` (preferred) or ``fsspec``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.constants import (
    FORMAT_VERSION,  # noqa: F401  (re-exported for callers)
    PARAMETRIC_GROUP,
    RESOLUTION_PREFIX,
)
from zarr_vectors.core.backends import (
    LocalBackend,
    StorageBackend,
    make_backend,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.metadata import (
    LevelMetadata,
    ParametricTypeDef,
    RootMetadata,
    deserialise_parametric_types,
    serialise_parametric_types,
)
from zarr_vectors.exceptions import MetadataError, StoreError


# ===================================================================
# FsGroup — backwards-compatible shim
# ===================================================================


class FsGroup(Group):
    """Backwards-compatible subclass of :class:`Group` rooted at a local path.

    .. deprecated::
        Direct use of ``FsGroup`` is deprecated.  New code should call
        :func:`create_store` / :func:`open_store` (which now accept a
        ``backend=`` kwarg) and operate on the returned :class:`Group`.

    Args:
        path: Filesystem path or :class:`pathlib.Path`.  A ``file://``
            URL is also accepted.
        create: If True, create the directory if it does not already
            exist.  If False, raise :class:`StoreError` if the directory
            is missing (matching the legacy behaviour).
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        backend = LocalBackend(path)
        if create:
            backend.ensure_prefix("")
        elif not backend.root.is_dir():
            raise StoreError(f"Store path does not exist: {backend.root}")
        super().__init__(backend, prefix="")


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
    be = make_backend(path, backend, **backend_kwargs)
    cls = FsGroup if isinstance(be, LocalBackend) else Group
    root = cls._from_backend(be, prefix="")

    if root.attrs.to_dict():
        raise StoreError(f"Store already exists at {be.url}")
    if isinstance(be, LocalBackend) and be.root.exists() and any(be.root.iterdir()):
        raise StoreError(f"Store already exists at {be.url}")

    be.ensure_prefix("")
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
    be = make_backend(path, backend, **backend_kwargs)
    cls = FsGroup if isinstance(be, LocalBackend) else Group
    root = cls._from_backend(be, prefix="")

    if isinstance(be, LocalBackend) and not be.root.is_dir():
        raise StoreError(f"Store not found at {be.url}")

    attrs = root.attrs.to_dict()
    if "zarr_vectors" not in attrs:
        raise StoreError(
            f"Not a valid ZV store: missing 'zarr_vectors' in root .zattrs "
            f"at {be.url}"
        )

    RootMetadata.from_dict(attrs)  # validates; raises MetadataError if bad
    return root


def rebind(
    group: Group,
    backend: str | StorageBackend,
    **backend_kwargs: Any,
) -> Group:
    """Swap the storage backend of an open store **without moving data**.

    The new backend must point at the same canonical URL as the current
    one — this is a connection rebind, not a data migration.  Use it to
    switch driver implementations (e.g. ``fsspec`` → ``obstore`` for
    performance) or to change credentials on the same underlying store.

    Args:
        group: A :class:`Group` returned by :func:`create_store` or
            :func:`open_store`.  Must be the root group.
        backend: Either a backend name string (``"obstore"`` / ``"fsspec"``
            / ``"local"``) or a pre-constructed :class:`StorageBackend`
            already pointing at the same URL.
        **backend_kwargs: Forwarded to the backend constructor when
            ``backend`` is a string.

    Returns:
        The same :class:`Group` instance, with its backend swapped.

    Raises:
        StoreError: If the new backend resolves to a different URL than
            the current one.
    """
    old = group._backend
    if isinstance(backend, str):
        new = make_backend(old.url, backend, **backend_kwargs)
    else:
        new = backend

    if _canonical(new.url) != _canonical(old.url):
        raise StoreError(
            f"rebind requires matching URLs; current backend is at "
            f"{old.url!r}, new backend is at {new.url!r}.  Use open_store "
            f"to point at a different location."
        )

    group._backend = new
    try:
        old.close()
    except Exception:  # pragma: no cover - defensive
        pass
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
