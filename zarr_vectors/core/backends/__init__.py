"""Pluggable storage backends.

This package contains the :class:`StorageBackend` protocol and built-in
implementations (``local``, ``obstore``, ``fsspec``), plus the URL-scheme
dispatcher used by :func:`zarr_vectors.core.store.create_store` and friends.

Resolution order for the active backend:

1. Explicit ``backend=`` kwarg on the public API.
2. ``ZARR_VECTORS_BACKEND`` environment variable.
3. URL-scheme auto-detect:
   - no scheme or ``file://`` → ``local``
   - cloud schemes (``s3``, ``gs``, ``gcs``, ``az``, ``azure``, ``abfs``,
     ``http``, ``https``) → ``obstore`` if installed, else ``fsspec``,
     else a :class:`~zarr_vectors.exceptions.StoreError` with an install
     hint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from zarr_vectors.exceptions import StoreError

from zarr_vectors.core.backends.async_base import AsyncStorageBackend
from zarr_vectors.core.backends.base import StorageBackend
from zarr_vectors.core.backends.local import LocalBackend

SCHEMES_LOCAL = frozenset({"", "file"})
SCHEMES_OBJECT_STORE = frozenset(
    {"s3", "gs", "gcs", "az", "azure", "abfs", "http", "https"}
)

_ENV_VAR = "ZARR_VECTORS_BACKEND"

__all__ = [
    "StorageBackend",
    "AsyncStorageBackend",
    "LocalBackend",
    "SCHEMES_LOCAL",
    "SCHEMES_OBJECT_STORE",
    "detect_scheme",
    "resolve_backend_name",
    "make_backend",
    "make_async_backend",
]


def detect_scheme(url: str | Path) -> str:
    """Return the URL scheme of ``url``, lowercased; empty string if none.

    A bare Windows drive letter (``C:\\foo``) is treated as local
    (returns ``""``), not as the scheme ``c``.
    """
    if isinstance(url, Path):
        return ""
    if not isinstance(url, str):
        return ""
    # urlparse misreads ``C:\foo`` as scheme=='c'.  Reject single-letter
    # schemes — no real scheme is a single character.
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if len(scheme) <= 1:
        return ""
    return scheme


def resolve_backend_name(
    url: str | Path,
    explicit: str | None = None,
    *,
    env_override: str | None = None,
) -> str:
    """Decide which backend to use for ``url``.

    Args:
        url: The store URL or path.
        explicit: User-supplied ``backend=`` kwarg.  Wins if set.
        env_override: Override for the ``ZARR_VECTORS_BACKEND`` env var
            (for testing).  Pass ``""`` to ignore the env var entirely.

    Returns:
        One of ``"local"``, ``"obstore"``, ``"fsspec"``.

    Raises:
        StoreError: If a cloud scheme is given but no compatible backend
            is installed.
    """
    if explicit:
        return explicit.lower()
    env_val = env_override if env_override is not None else os.environ.get(_ENV_VAR)
    if env_val:
        return env_val.lower()

    scheme = detect_scheme(url)
    if scheme in SCHEMES_LOCAL:
        return "local"
    if scheme in SCHEMES_OBJECT_STORE:
        if _have("obstore"):
            return "obstore"
        if _have("fsspec"):
            return "fsspec"
        raise StoreError(
            f"URL {url!r} has scheme {scheme!r} which requires a cloud "
            f"backend, but neither 'obstore' nor 'fsspec' is installed. "
            f"Install with: pip install zarr-vectors[obstore]"
        )
    # Unknown scheme — let local handle it; if it's broken, the backend
    # constructor will raise something more specific.
    return "local"


def make_backend(
    url: str | Path,
    backend: str | None = None,
    *,
    env_override: str | None = None,
    **kwargs: Any,
) -> StorageBackend:
    """Resolve and construct the appropriate backend for ``url``.

    Args:
        url: Store URL or path.
        backend: Explicit backend name (``"local"`` / ``"obstore"`` /
            ``"fsspec"``).  ``None`` means auto-detect.
        env_override: Test hook — see :func:`resolve_backend_name`.
        **kwargs: Forwarded to the backend constructor.
    """
    name = resolve_backend_name(url, backend, env_override=env_override)

    if name == "local":
        return LocalBackend(url, **kwargs)
    if name == "obstore":
        from zarr_vectors.core.backends.obstore_backend import ObstoreBackend

        return ObstoreBackend(url, **kwargs)
    if name == "fsspec":
        from zarr_vectors.core.backends.fsspec_backend import FsspecBackend

        return FsspecBackend(url, **kwargs)
    raise StoreError(f"Unknown backend: {name!r}")


def make_async_backend(
    url: str | Path,
    backend: str | None = None,
    *,
    env_override: str | None = None,
    **kwargs: Any,
) -> AsyncStorageBackend:
    """Resolve and construct an async-capable backend for ``url``.

    Each concrete backend implements both :class:`StorageBackend` and
    :class:`AsyncStorageBackend`, so the returned object can also be
    used synchronously.  This entry point exists to give callers a
    statically-typed handle when they intend to use the async surface.

    Args:
        url: Store URL or path.
        backend: Explicit backend name.  ``None`` means auto-detect.
        env_override: Test hook — see :func:`resolve_backend_name`.
        **kwargs: Forwarded to the backend constructor.
    """
    return make_backend(url, backend, env_override=env_override, **kwargs)


def _have(module: str) -> bool:
    """Return True if ``module`` is importable.

    Honours ``sys.modules`` overrides used in tests — a sentinel of
    ``None`` indicates "not available", and a stub module object
    (regardless of whether it has a real ``__spec__``) indicates
    "available".
    """
    import importlib.util
    import sys

    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
