"""AsyncStorageBackend protocol.

The async counterpart to :class:`zarr_vectors.core.backends.base.StorageBackend`.
Every concrete backend implements *both* protocols (sync + async); callers
choose one based on whether they're in async code.

Method semantics mirror the sync surface exactly — same key conventions,
same return types — but every I/O-touching call is a coroutine.

Backend-specific notes:

- ``LocalBackend`` routes via ``asyncio.to_thread`` so its async methods
  cooperate with an event loop without bringing in aiofiles.
- ``ObstoreBackend`` uses obstore's native ``*_async`` functions for true
  parallel object-store I/O without per-request thread overhead.
- ``FsspecBackend`` uses fsspec's coroutine ``_*`` methods on async
  filesystems (``s3fs``, ``gcsfs``, ``adlfs``), and falls back to
  ``asyncio.to_thread`` on sync-only filesystems.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class AsyncStorageBackend(Protocol):
    """Async byte-level key/value backend rooted at a single URL.

    Methods are coroutines.  All ``key`` arguments are forward-slash-
    separated relative paths under the store root.  Empty string refers
    to the root.
    """

    @property
    def url(self) -> str:
        """Canonical URL of the store root."""
        ...

    async def aput_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes to ``key``, overwriting any existing value."""
        ...

    async def aget_bytes(self, key: str) -> bytes:
        """Read raw bytes from ``key``.

        Raises:
            KeyError: If ``key`` does not exist.
        """
        ...

    async def aexists(self, key: str) -> bool:
        """Return True if ``key`` exists."""
        ...

    async def adelete(self, key: str) -> None:
        """Delete a single key.  Silent if absent."""
        ...

    async def adelete_prefix(self, prefix: str) -> None:
        """Delete every key whose path starts with ``prefix``."""
        ...

    def alist_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> AsyncIterator[str]:
        """Async iterator over keys under ``prefix``.

        Note: returns an :class:`AsyncIterator`, not a coroutine.  Use
        ``async for entry in backend.alist_prefix(p): ...``.

        Semantics match the sync :meth:`list_prefix`: trailing ``/`` for
        container entries in non-recursive mode, file keys only in
        recursive mode.
        """
        ...

    async def aensure_prefix(self, prefix: str) -> None:
        """Best-effort create a container at ``prefix``.  No-op on object stores."""
        ...

    async def aclose(self) -> None:
        """Release any connections / file handles."""
        ...
