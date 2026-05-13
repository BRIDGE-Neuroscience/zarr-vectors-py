"""Icechunk-backed Zarr store factory.

Icechunk (https://icechunk.io/) is a transactional Zarr v3 store that
wraps an underlying object store (local FS, S3, GCS, Azure) and
provides commit-style versioning on top.  Unlike the byte-level
:class:`~zarr_vectors.core.backends.base.StorageBackend` protocol
implementations (``local`` / ``obstore`` / ``fsspec``), icechunk is a
**Zarr-Store-level** backend: it returns a ``zarr.abc.store.Store``
that gets handed directly to :func:`zarr.open_group` without any ZV
shim in between.

This module's only public surface is :func:`make_icechunk_session`,
which:

1. Parses the URL scheme and dispatches to the right
   ``icechunk.*_storage(...)`` factory.
2. Opens or creates an :class:`icechunk.Repository` against that
   storage.
3. Opens a writable or read-only :class:`icechunk.Session` (by
   branch name or snapshot id).
4. Returns ``(session.store, session)``.

The session must be kept alive for as long as the store is used —
the high-level :func:`zarr_vectors.core.store.create_store` /
:func:`zarr_vectors.core.store.open_store` stash it on the underlying
Zarr store as ``store._zv_icechunk_session`` (so every sub-group
reached via ``zg.store`` can locate it); the public
:func:`zarr_vectors.core.store.commit` / ``discard_changes`` /
``session_for`` helpers flush or retrieve it.

Install with::

    pip install "zarr-vectors[icechunk]"
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from zarr_vectors.exceptions import StoreError


_OpenMode = Literal["w", "r", "r+", "a"]


def _import_icechunk():
    """Lazy import of the optional ``icechunk`` package."""
    try:
        import icechunk  # type: ignore[import-not-found]
    except ImportError as e:
        raise StoreError(
            "icechunk is not installed. Install with: "
            "pip install zarr-vectors[icechunk]"
        ) from e
    return icechunk


def _make_storage(url: str, ic, **kwargs: Any):
    """Pick the right icechunk Storage factory for the URL scheme.

    ``kwargs`` are forwarded verbatim to the scheme-specific factory.
    Common cloud kwargs: ``region``, ``endpoint_url``, ``access_key_id``,
    ``secret_access_key``, ``session_token``, ``anonymous``, ``from_env``.
    Local kwargs are empty.

    The ``memory://`` URL maps to :func:`icechunk.in_memory_storage`.
    """
    parsed = urlparse(url) if isinstance(url, str) else None
    scheme = parsed.scheme.lower() if parsed else ""

    if scheme in ("", "file"):
        # Local filesystem icechunk.  Strip ``file://`` if present.
        if scheme == "file":
            from zarr_vectors.core.backends.local import _file_url_to_path

            path = str(_file_url_to_path(url))
        else:
            path = url
        return ic.local_filesystem_storage(path, **kwargs)

    if scheme == "memory":
        return ic.in_memory_storage(**kwargs)

    if scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/") or None
        return ic.s3_storage(bucket=bucket, prefix=prefix, **kwargs)

    if scheme in ("gs", "gcs"):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/") or None
        return ic.gcs_storage(bucket=bucket, prefix=prefix, **kwargs)

    if scheme in ("az", "azure", "abfs"):
        container = parsed.netloc
        prefix = parsed.path.lstrip("/") or None
        return ic.azure_storage(container=container, prefix=prefix, **kwargs)

    raise StoreError(
        f"icechunk: unsupported URL scheme {scheme!r} in {url!r}. "
        f"Supported: '' / file, memory, s3, gs/gcs, az/azure/abfs."
    )


def _repository_exists(ic, storage) -> bool:
    """Best-effort check that an icechunk repository is already present."""
    exists_fn = getattr(ic.Repository, "exists", None)
    if exists_fn is None:
        # Older releases without the helper — fall back to trying to open
        # and catching the error.  Slower but correct.
        try:
            ic.Repository.open(storage)
            return True
        except Exception:
            return False
    try:
        return bool(exists_fn(storage))
    except Exception:
        return False


def make_icechunk_session(
    url: str,
    *,
    mode: _OpenMode = "r+",
    branch: str = "main",
    snapshot_id: str | None = None,
    repository_config: Any = None,
    **storage_kwargs: Any,
) -> tuple[Any, Any]:
    """Build an icechunk session for ``url`` and return ``(store, session)``.

    Args:
        url: Store URL.  Local path or ``file://`` for the local-FS
            storage; ``memory://`` for in-memory; ``s3``/``gs``/``gcs``/
            ``az``/``azure``/``abfs`` for cloud storage.
        mode: ``"w"`` opens a writable session on a freshly-created repo
            (raises if the repo already exists).  ``"r+"`` / ``"a"`` open
            a writable session on an existing repo.  ``"r"`` opens a
            read-only session.  Read-only at a specific snapshot:
            pass ``mode="r"`` together with ``snapshot_id="..."``.
        branch: Branch name to open the session against.  Ignored when
            ``snapshot_id`` is given (which always opens read-only).
        snapshot_id: Open a read-only session at this specific snapshot.
            Implies ``mode="r"``.
        repository_config: Optional pre-built ``icechunk.RepositoryConfig``.
        **storage_kwargs: Forwarded to the scheme-specific icechunk
            storage factory (e.g. ``region="us-west-2"`` for S3).

    Returns:
        A 2-tuple ``(zarr_store, session)`` where ``zarr_store`` is a
        :class:`zarr.abc.store.Store` ready to hand to
        :func:`zarr.open_group`, and ``session`` is the
        :class:`icechunk.Session` that owns it.
    """
    ic = _import_icechunk()
    storage = _make_storage(url, ic, **storage_kwargs)

    repo_exists = _repository_exists(ic, storage)
    if mode == "w":
        if repo_exists:
            raise StoreError(
                f"icechunk repository already exists at {url!r}; "
                f"use mode='r+' / 'a' to open it instead."
            )
        repo = ic.Repository.create(storage, config=repository_config)
    else:
        if not repo_exists:
            raise StoreError(
                f"icechunk repository not found at {url!r}; "
                f"use mode='w' to create one."
            )
        repo = ic.Repository.open(storage, config=repository_config)

    if snapshot_id is not None:
        session = repo.readonly_session(snapshot_id=snapshot_id)
    elif mode == "r":
        session = repo.readonly_session(branch=branch)
    else:
        session = repo.writable_session(branch=branch)

    return session.store, session
