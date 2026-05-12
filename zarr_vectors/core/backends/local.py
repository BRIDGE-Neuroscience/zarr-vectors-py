"""Local filesystem backend.

A thin wrapper over :mod:`pathlib` that satisfies the
:class:`StorageBackend` protocol.  Equivalent in behaviour to the original
``FsGroup`` direct-Path I/O.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import AsyncIterator, Iterator
from urllib.parse import unquote, urlparse


class LocalBackend:
    """Local filesystem :class:`StorageBackend`.

    Args:
        url_or_path: A filesystem path, ``file://`` URL, or ``pathlib.Path``.
            Paths without a scheme are treated as local filesystem paths.
    """

    def __init__(self, url_or_path: str | Path) -> None:
        if isinstance(url_or_path, Path):
            self._root = url_or_path
        elif isinstance(url_or_path, str) and url_or_path.startswith("file://"):
            self._root = Path(_file_url_to_path(url_or_path))
        else:
            self._root = Path(url_or_path)
        # Canonical URL — resolve so it round-trips even if the dir does
        # not exist yet.  ``Path.as_uri`` only works on absolute paths.
        self._url = self._root.absolute().as_uri()

    # ---------------- properties ----------------

    @property
    def url(self) -> str:
        return self._url

    @property
    def root(self) -> Path:
        """The root :class:`Path` of this backend.  LocalBackend-only."""
        return self._root

    # ---------------- key/path mapping ----------------

    def _key_to_path(self, key: str) -> Path:
        if not key:
            return self._root
        # Reject keys that try to escape the root via ".."
        parts = [p for p in key.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ValueError(f"Key may not contain '..': {key!r}")
        return self._root.joinpath(*parts)

    # ---------------- byte I/O ----------------

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        path = self._key_to_path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def delete_prefix(self, prefix: str) -> None:
        path = self._key_to_path(prefix)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()

    def list_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> Iterator[str]:
        """Yield keys under ``prefix``.

        Recursive mode yields file keys only.  Non-recursive mode yields
        immediate children; directory entries get a trailing ``/`` so
        callers can distinguish containers from files without an extra
        round-trip.
        """
        base = self._key_to_path(prefix)
        if not base.is_dir():
            return
        root = self._root
        if recursive:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    yield p.relative_to(root).as_posix()
        else:
            for p in sorted(base.iterdir()):
                rel = p.relative_to(root).as_posix()
                if p.is_dir():
                    yield rel + "/"
                else:
                    yield rel

    def ensure_prefix(self, prefix: str) -> None:
        self._key_to_path(prefix).mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    # ---------------- async I/O (asyncio.to_thread) ----------------
    #
    # Local-fs operations are blocking syscalls.  ``asyncio.to_thread``
    # releases the event loop while a thread does the work — enough to
    # let callers ``asyncio.gather`` many per-chunk RMW operations in
    # parallel without bringing aiofiles into the dep tree.

    async def aput_bytes(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self.put_bytes, key, data)

    async def aget_bytes(self, key: str) -> bytes:
        return await asyncio.to_thread(self.get_bytes, key)

    async def aexists(self, key: str) -> bool:
        return await asyncio.to_thread(self.exists, key)

    async def adelete(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)

    async def adelete_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(self.delete_prefix, prefix)

    async def alist_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> AsyncIterator[str]:
        # Materialise in a thread, then yield from the in-memory list.
        # Cheap for typical chunk counts; avoids per-entry thread hops.
        entries = await asyncio.to_thread(
            lambda: list(self.list_prefix(prefix, recursive=recursive))
        )
        for entry in entries:
            yield entry

    async def aensure_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(self.ensure_prefix, prefix)

    async def aclose(self) -> None:
        # No connection state to release.
        return None

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"LocalBackend({self._root!s})"


def _file_url_to_path(url: str) -> str:
    """Parse a ``file://`` URL to a local filesystem path string."""
    parsed = urlparse(url)
    p = unquote(parsed.path)
    # On Windows, ``file:///C:/foo`` parses to path='/C:/foo' — strip the
    # leading slash before the drive letter.
    if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == ":":
        p = p[1:]
    return p
