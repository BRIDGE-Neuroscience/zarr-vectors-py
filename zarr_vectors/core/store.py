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
    CAP_VERTEX_COUNT_CACHE,
    DEFAULT_AXES_NAMES,
    FORMAT_VERSION,  # noqa: F401  (re-exported for callers)
    PARAMETRIC_GROUP,
    RESOLUTION_PREFIX,
    VERTICES,
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


def _make_zarr_store_with_session(
    path: str | Path,
    *,
    mode: str = "r+",
    backend: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Construct a Zarr store for ``path``.

    Returns ``(store, session)`` where ``session`` is non-``None`` only
    for transactional backends (currently just ``icechunk``).  The
    session must be kept alive for the lifetime of the store and is
    flushed via :func:`commit`.

    ``backend="icechunk"`` routes through
    :func:`zarr_vectors.core.backends.icechunk_backend.make_icechunk_session`
    and accepts ``branch``, ``snapshot_id``, ``repository_config``, and
    any storage-specific kwargs (``region``, ``endpoint_url``, ...).

    Local / file:// URLs without an explicit backend use
    :class:`zarr.storage.LocalStore`.  Cloud schemes without
    ``backend="icechunk"`` raise: byte-level cloud routing through this
    Zarr-Store layer is not wired up; use ``backend="icechunk"`` for a
    transactional cloud-backed store.
    """
    if backend == "icechunk":
        from zarr_vectors.core.backends.icechunk_backend import (
            make_icechunk_session,
        )

        return make_icechunk_session(str(path), mode=mode, **kwargs)

    scheme = _detect_scheme(path)
    if scheme not in {"", "file"}:
        raise StoreError(
            f"URL scheme {scheme!r} is not yet wired into the Zarr-native "
            f"store layer for backend={backend!r}. Pass backend='icechunk' "
            f"for a transactional cloud-backed store, or use a local path."
        )
    return LocalStore(_resolve_local_path(path)), None


def _make_zarr_store(
    path: str | Path,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Back-compat wrapper that discards the session.

    Existing call sites that don't care about transactional backends
    can keep calling this; callers that need to keep a session alive
    (``create_store``, ``open_store``) use
    :func:`_make_zarr_store_with_session` directly.
    """
    store, _ = _make_zarr_store_with_session(path, backend=backend, **kwargs)
    return store


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
    root_metadata: RootMetadata | None = None,
    *,
    chunk_shape: tuple[float, ...] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    axes: list[dict[str, str]] | None = None,
    geometry_types: list[str] | None = None,
    crs: dict[str, Any] | None = None,
    vertex_dtype: str = "float32",
    vertex_encoding: str = "raw",
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Create a new ZV store.

    The minimal form ``create_store(path)`` produces a structurally valid
    "warm" shell: root group with format markers only, a ``resolution_0/``
    sub-group, and the empty ragged-vertex pair
    ``resolution_0/vertices/`` + ``resolution_0/vertex_group_offsets/``.
    Dimensionality, bounds, and chunk shape are inferred and stamped
    into root attrs on the first write call (e.g. :func:`write_points`).

    Callers that already have a fully-populated :class:`RootMetadata`
    can pass it as the second positional arg (or ``root_metadata=``);
    its fields are written eagerly and validated up-front.

    The ``/parametric`` sub-group is **not** created here.  It is created
    lazily on first use via :func:`get_parametric_group`.

    Args:
        path: URL or filesystem path for the new store.
        root_metadata: Optional fully-populated :class:`RootMetadata`.
            When given, individual kwargs below are ignored.
        chunk_shape: Spatial chunk size per dimension.  Optional — when
            ``None``, inferred on first write.
        bounds: ``(min_corner, max_corner)``.  Optional — when ``None``,
            inferred on first write and grown across subsequent writes.
        axes: OME-Zarr-style axes list.  Optional — when ``None``,
            generated from :data:`DEFAULT_AXES_NAMES` on first write.
        geometry_types: List of geometry types the store will contain.
        crs: Coordinate reference system dict.
        vertex_dtype: dtype for the level-0 vertices array.
        vertex_encoding: ``"raw"`` or ``"draco"``.
        backend: Force a particular backend (``"local"`` / ``"icechunk"``).
        **backend_kwargs: Forwarded to the backend constructor.

    Returns:
        The root :class:`Group`.

    Raises:
        StoreError: If a store already exists at ``path``.
        MetadataError: If ``root_metadata`` is supplied and invalid.
    """
    if root_metadata is not None:
        root_metadata.validate()
        if CAP_VERTEX_COUNT_CACHE not in root_metadata.format_capabilities:
            root_metadata.format_capabilities = [
                *root_metadata.format_capabilities,
                CAP_VERTEX_COUNT_CACHE,
            ]
        full_attrs: dict[str, Any] | None = root_metadata.to_dict()
    else:
        full_attrs = None

    local_root: Path | None = None
    if backend != "icechunk":
        scheme = _detect_scheme(path)
        if scheme in {"", "file"}:
            local_root = _resolve_local_path(path)
            if local_root.exists() and local_root.is_dir() and any(local_root.iterdir()):
                raise StoreError(f"Store already exists at {local_root}")
            local_root.mkdir(parents=True, exist_ok=True)

    store, session = _make_zarr_store_with_session(
        path, mode="w", backend=backend, **backend_kwargs,
    )
    if session is not None:
        store._zv_icechunk_session = session
    zg = zarr.open_group(store, mode="w")
    root = FsGroup.__new__(FsGroup) if isinstance(store, LocalStore) else Group.__new__(Group)
    root._zarr = zg

    if full_attrs is not None:
        root.attrs.update(full_attrs)
    else:
        _write_root_attrs_partial(
            root,
            chunk_shape=chunk_shape,
            bounds=bounds,
            axes=axes,
            geometry_types=geometry_types,
            crs=crs,
        )

    # resolution_0/ + empty vertices pair — the "warm" payload.
    level0 = root.create_group(f"{RESOLUTION_PREFIX}0")
    level0.attrs.update(
        LevelMetadata(
            level=0,
            vertex_count=0,
            arrays_present=[VERTICES],
        ).to_dict()
    )
    # Defer import: arrays.py imports from store.py (FsGroup).
    from zarr_vectors.core.arrays import create_vertices_array
    create_vertices_array(level0, dtype=vertex_dtype, encoding=vertex_encoding)
    return root


def _write_root_attrs_partial(
    root: Group,
    *,
    chunk_shape: tuple[float, ...] | None,
    bounds: tuple[list[float], list[float]] | None,
    axes: list[dict[str, str]] | None,
    geometry_types: list[str] | None,
    crs: dict[str, Any] | None,
) -> None:
    """Write the ``zarr_vectors`` root-attrs block with only the fields
    the caller supplied.  Missing structural fields stay absent and are
    later filled in by :func:`_ensure_root_metadata_for_write` on the
    first write.
    """
    existing = root.attrs.to_dict().get("zarr_vectors", {})
    zv: dict[str, Any] = dict(existing)
    zv["format_version"] = FORMAT_VERSION
    caps = list(zv.get("format_capabilities") or [])
    if CAP_VERTEX_COUNT_CACHE not in caps:
        caps.append(CAP_VERTEX_COUNT_CACHE)
    zv["format_capabilities"] = caps
    if axes is not None:
        zv["spatial_index_dims"] = axes
    if chunk_shape is not None:
        zv["chunk_shape"] = list(chunk_shape)
    if bounds is not None:
        zv["bounds"] = [list(bounds[0]), list(bounds[1])]
    if geometry_types is not None:
        zv["geometry_types"] = list(geometry_types)
    if crs is not None:
        zv["crs"] = crs
    root.attrs.update({"zarr_vectors": zv})


def _ensure_root_metadata_for_write(
    root: Group,
    *,
    inferred_ndim: int,
    inferred_bounds: tuple[list[float], list[float]],
    chunk_shape_hint: tuple[float, ...] | None = None,
    geometry_type: str | None = None,
    base_bin_shape: tuple[float, ...] | None = None,
    links_convention: str | None = None,
    object_index_convention: str | None = None,
    cross_chunk_strategy: str | None = None,
) -> RootMetadata:
    """Fill in any missing structural root-attrs from the values a writer
    just inferred from its input data.

    - If structural fields are already set, they are kept and ``bounds``
      is grown by min/max union with ``inferred_bounds``.
    - If unset, defaults are derived: axes from
      :data:`DEFAULT_AXES_NAMES`, ``chunk_shape`` from a single-chunk-
      covers-bounds heuristic when no hint is supplied.

    Persists the result to root attrs and returns the now-complete
    :class:`RootMetadata`.
    """
    existing = root.attrs.to_dict().get("zarr_vectors", {})
    zv: dict[str, Any] = dict(existing)
    zv.setdefault("format_version", FORMAT_VERSION)
    caps = list(zv.get("format_capabilities") or [])
    if CAP_VERTEX_COUNT_CACHE not in caps:
        caps.append(CAP_VERTEX_COUNT_CACHE)
    zv["format_capabilities"] = caps

    axes_existing = zv.get("spatial_index_dims")
    if not axes_existing:
        if inferred_ndim > len(DEFAULT_AXES_NAMES):
            raise MetadataError(
                f"Cannot auto-generate axes for ndim={inferred_ndim}; "
                f"pass an explicit `axes` kwarg to create_store."
            )
        names = DEFAULT_AXES_NAMES[:inferred_ndim]
        zv["spatial_index_dims"] = [
            {"name": n, "type": "space", "unit": ""} for n in names
        ]
    elif len(axes_existing) != inferred_ndim:
        raise MetadataError(
            f"Inferred ndim={inferred_ndim} does not match stored "
            f"spatial_index_dims length {len(axes_existing)}"
        )

    existing_bounds = zv.get("bounds")
    inf_min, inf_max = list(inferred_bounds[0]), list(inferred_bounds[1])
    if existing_bounds and existing_bounds[0] and existing_bounds[1]:
        new_min = [min(a, b) for a, b in zip(existing_bounds[0], inf_min)]
        new_max = [max(a, b) for a, b in zip(existing_bounds[1], inf_max)]
        zv["bounds"] = [new_min, new_max]
    else:
        zv["bounds"] = [inf_min, inf_max]

    if not zv.get("chunk_shape"):
        if chunk_shape_hint is not None:
            zv["chunk_shape"] = list(chunk_shape_hint)
        else:
            min_corner = zv["bounds"][0]
            max_corner = zv["bounds"][1]
            extent = [hi - lo for hi, lo in zip(max_corner, min_corner)]
            max_extent = max(extent) if extent else 1.0
            max_abs_bound = max((abs(v) for v in max_corner), default=0.0)
            side = max_extent + max_abs_bound + 1.0
            zv["chunk_shape"] = [side] * inferred_ndim

    if geometry_type is not None:
        gts = list(zv.get("geometry_types") or [])
        if geometry_type not in gts:
            gts.append(geometry_type)
        zv["geometry_types"] = gts
    elif "geometry_types" not in zv:
        zv["geometry_types"] = []

    if base_bin_shape is not None:
        zv["base_bin_shape"] = list(base_bin_shape)

    if links_convention is not None:
        zv.setdefault("links_convention", links_convention)
    if object_index_convention is not None:
        zv.setdefault("object_index_convention", object_index_convention)
    if cross_chunk_strategy is not None:
        zv.setdefault("cross_chunk_strategy", cross_chunk_strategy)

    root.attrs.update({"zarr_vectors": zv})
    return RootMetadata.from_dict({"zarr_vectors": zv})


def _create_or_open_store(
    path: str | Path,
    *,
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Warm the store with :func:`create_store` if no store exists at
    ``path``, else open it read-write.  Writers route through this so
    callers can either ``create_store(path)`` first and then write, or
    write directly against a fresh path.
    """
    if backend == "icechunk":
        try:
            return create_store(path, backend=backend, **backend_kwargs)
        except StoreError:
            return open_store(path, mode="r+", backend=backend, **backend_kwargs)
    scheme = _detect_scheme(path)
    if scheme in {"", "file"}:
        local_root = _resolve_local_path(path)
        if local_root.exists() and local_root.is_dir() and any(local_root.iterdir()):
            return open_store(path, mode="r+", backend=backend, **backend_kwargs)
    return create_store(path, backend=backend, **backend_kwargs)


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
    # Local-FS existence check; transactional backends (icechunk) verify
    # repository existence inside their own session factory.
    if backend != "icechunk":
        scheme = _detect_scheme(path)
        if scheme in {"", "file"}:
            local_root = _resolve_local_path(path)
            if not local_root.is_dir():
                raise StoreError(f"Store not found at {local_root}")

    # Pass mode through so the icechunk backend can pick readonly_session
    # vs writable_session.  For non-icechunk backends mode is ignored.
    store, session = _make_zarr_store_with_session(
        path, mode=mode, backend=backend, **backend_kwargs,
    )
    if session is not None:
        store._zv_icechunk_session = session
    # mode is informational at the public API — the underlying zarr group
    # is always opened read-write so that ``lazy.writer`` can write
    # through the same handle returned by ``open_store(mode='r')``.
    zarr_open_mode = "r" if mode == "r" else "r+"
    try:
        zg = zarr.open_group(store, mode=zarr_open_mode)
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
    # Permissive parse: a freshly-warmed store has no dims/bounds/chunk_
    # _shape yet.  The required `format_version` key is still enforced.
    RootMetadata.from_dict(attrs, strict=False)
    return root


# ===================================================================
# Transactional helpers (icechunk)
# ===================================================================


def session_for(group: Group) -> Any | None:
    """Return the underlying ``icechunk.Session`` if any, else ``None``.

    Looks up the session stashed on the group's Zarr store at
    construction time (see :func:`create_store` / :func:`open_store`).
    Works for root groups *and* any sub-group reached via
    ``root.create_group(...)`` / ``root[name]`` — they all share the
    same underlying Zarr store.

    Useful when callers need branch / snapshot operations that aren't
    surfaced by the zarr-vectors API (creating branches, tagging,
    listing snapshots, etc.).
    """
    zg = getattr(group, "_zarr", None)
    if zg is None:
        return None
    return getattr(zg.store, "_zv_icechunk_session", None)


def commit(group: Group, message: str = "zarr-vectors write") -> str | None:
    """Commit pending changes when the store is backed by a transactional
    backend (currently ``icechunk``).

    For non-transactional backends this is a no-op and returns ``None``;
    writes are durable as soon as they hit the store.

    For icechunk-backed stores this calls ``session.commit(message)``
    and returns the new snapshot id (a hex string).  The same session
    continues to be writable after the commit — subsequent ZV writes
    accumulate uncommitted state until the next ``commit(group, ...)``
    call.

    Args:
        group: A :class:`Group` returned by :func:`create_store` or
            :func:`open_store` (or any sub-group of one).
        message: Commit message; defaults to a placeholder.  Empty
            strings are rejected by icechunk.

    Returns:
        Snapshot id (``str``) for icechunk-backed groups, else ``None``.
    """
    session = session_for(group)
    if session is None:
        return None
    return session.commit(message)


def discard_changes(group: Group) -> None:
    """Drop uncommitted changes on a transactional backend.

    No-op for non-transactional backends.
    """
    session = session_for(group)
    if session is None:
        return
    discard = getattr(session, "discard_changes", None)
    if discard is not None:
        discard()


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
