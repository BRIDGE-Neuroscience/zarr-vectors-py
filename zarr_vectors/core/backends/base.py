"""StorageBackend protocol.

A :class:`StorageBackend` is a scheme-agnostic key/value store rooted at a
single URL.  All keys are forward-slash-separated relative paths under that
root.  Backends know nothing about ZV conventions (``.zattrs``, chunk-key
naming, group hierarchy) — those live in :class:`zarr_vectors.core.group.Group`.

This split lets new backends ship as ~50 lines of byte plumbing without
re-implementing format conventions.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Byte-level key/value backend rooted at a single URL.

    Implementations must provide the methods below.  All ``key`` arguments
    are forward-slash-separated relative paths under the store root.
    Empty string refers to the root itself.
    """

    @property
    def url(self) -> str:
        """Canonical URL of the store root (e.g. ``"file:///C:/x/y"``)."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes to ``key``, overwriting any existing value."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Read raw bytes from ``key``.

        Raises:
            KeyError: If ``key`` does not exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists."""
        ...

    def delete(self, key: str) -> None:
        """Delete a single key.  Silent if absent."""
        ...

    def delete_prefix(self, prefix: str) -> None:
        """Delete every key whose path starts with ``prefix``.

        Used to implement ``rmtree``-style subtree removal.
        """
        ...

    def list_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> Iterator[str]:
        """Yield keys under ``prefix``.

        - ``recursive=True``: yields every descendant *file* key.  Keys
          are full paths relative to the store root.  No trailing ``/``.
        - ``recursive=False``: yields immediate children — files and
          containers.  Container entries are flagged with a trailing
          ``/`` so callers can tell them apart from files.

        Yields nothing if ``prefix`` does not exist.
        """
        ...

    def ensure_prefix(self, prefix: str) -> None:
        """Best-effort create a container at ``prefix``.

        On filesystem backends this calls ``mkdir(parents=True,
        exist_ok=True)``.  On flat object stores this is a no-op — the
        prefix exists implicitly once any key under it is written.
        """
        ...

    def close(self) -> None:
        """Release any connections / file handles."""
        ...
