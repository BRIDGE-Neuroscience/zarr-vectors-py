"""fsspec-backed :class:`StorageBackend`.

Routes operations through ``fsspec`` and its filesystem adapters
(``s3fs``, ``gcsfs``, ``adlfs``, plus the local-file driver).  Installed
adapters are picked up automatically.

``fsspec`` and the relevant adapters must be installed separately:

.. code-block:: bash

    pip install "zarr-vectors[fsspec]"
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator

from zarr_vectors.exceptions import StoreError


class FsspecBackend:
    """Generic fsspec :class:`StorageBackend`.

    Args:
        url: Any URL that ``fsspec.url_to_fs`` understands.
        **kwargs: Forwarded to ``fsspec.url_to_fs`` (and thence to the
            underlying filesystem constructor — e.g.
            ``key=``/``secret=`` for s3fs).
    """

    def __init__(self, url: str, **kwargs: Any) -> None:
        try:
            import fsspec
        except ImportError as e:  # pragma: no cover - exercised via dispatcher
            raise StoreError(
                "fsspec is not installed.  Install with: "
                "pip install zarr-vectors[fsspec]"
            ) from e

        self._url = url
        self._fs, self._root = fsspec.url_to_fs(url, **kwargs)
        # Normalise the root so list/exists keys are absolute paths the
        # underlying filesystem understands.
        self._root = self._root.rstrip("/")

    # ---------------- properties ----------------

    @property
    def url(self) -> str:
        return self._url

    # ---------------- key/path mapping ----------------

    def _full_key(self, key: str) -> str:
        key = key.strip("/")
        if not key:
            return self._root
        return f"{self._root}/{key}" if self._root else key

    # ---------------- byte I/O ----------------

    def put_bytes(self, key: str, data: bytes) -> None:
        self._fs.pipe_file(self._full_key(key), data)

    def get_bytes(self, key: str) -> bytes:
        full = self._full_key(key)
        if not self._fs.exists(full):
            raise KeyError(key)
        return self._fs.cat_file(full)

    def exists(self, key: str) -> bool:
        return self._fs.exists(self._full_key(key))

    def delete(self, key: str) -> None:
        full = self._full_key(key)
        try:
            if self._fs.isdir(full):
                self._fs.rm(full, recursive=True)
            else:
                self._fs.rm_file(full)
        except FileNotFoundError:
            pass

    def delete_prefix(self, prefix: str) -> None:
        full = self._full_key(prefix)
        if self._fs.exists(full):
            self._fs.rm(full, recursive=True)

    def list_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> Iterator[str]:
        full = self._full_key(prefix)
        if not self._fs.exists(full):
            return

        root_len = len(self._root) + 1 if self._root else 0

        if recursive:
            # find() returns flat list of files only.
            for path in sorted(self._fs.find(full)):
                yield path[root_len:] if root_len else path
        else:
            # ls returns immediate children with type info.
            try:
                entries = self._fs.ls(full, detail=True)
            except FileNotFoundError:
                return
            entries = sorted(entries, key=lambda e: e["name"])
            for entry in entries:
                path = entry["name"]
                rel = path[root_len:] if root_len else path
                if entry.get("type") == "directory":
                    yield rel if rel.endswith("/") else rel + "/"
                else:
                    yield rel

    def ensure_prefix(self, prefix: str) -> None:
        full = self._full_key(prefix)
        # On local fs this is meaningful; on object stores it's a no-op.
        try:
            self._fs.makedirs(full, exist_ok=True)
        except (NotImplementedError, AttributeError):
            pass

    def close(self) -> None:
        return None

    # ---------------- async I/O (fsspec coroutines or thread fallback) ----
    #
    # On async-capable filesystems (s3fs, gcsfs, adlfs) the underscore
    # methods (``_pipe_file``, ``_cat_file``, …) are coroutines.  When
    # ``fs.async_impl`` is False we fall back to ``asyncio.to_thread``
    # so the protocol still works against local-fsspec, memory, http
    # etc.  The probe is per-call so a mixed setup (one async fs,
    # another sync) Just Works.

    @property
    def _is_async_fs(self) -> bool:
        return bool(getattr(self._fs, "async_impl", False))

    async def aput_bytes(self, key: str, data: bytes) -> None:
        if self._is_async_fs and hasattr(self._fs, "_pipe_file"):
            await self._fs._pipe_file(self._full_key(key), data)
        else:
            await asyncio.to_thread(self.put_bytes, key, data)

    async def aget_bytes(self, key: str) -> bytes:
        full = self._full_key(key)
        if self._is_async_fs and hasattr(self._fs, "_cat_file"):
            exists_async = getattr(self._fs, "_exists", None)
            if exists_async is not None:
                ok = await exists_async(full)
                if not ok:
                    raise KeyError(key)
            return await self._fs._cat_file(full)
        return await asyncio.to_thread(self.get_bytes, key)

    async def aexists(self, key: str) -> bool:
        if self._is_async_fs and hasattr(self._fs, "_exists"):
            return await self._fs._exists(self._full_key(key))
        return await asyncio.to_thread(self.exists, key)

    async def adelete(self, key: str) -> None:
        full = self._full_key(key)
        if self._is_async_fs and hasattr(self._fs, "_rm"):
            try:
                await self._fs._rm(full)
            except FileNotFoundError:
                pass
            return
        await asyncio.to_thread(self.delete, key)

    async def adelete_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(self.delete_prefix, prefix)

    async def alist_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> AsyncIterator[str]:
        # fsspec's async `_find`/`_ls` are available on async filesystems.
        full = self._full_key(prefix)
        root_len = len(self._root) + 1 if self._root else 0

        if self._is_async_fs:
            try:
                if recursive and hasattr(self._fs, "_find"):
                    paths = await self._fs._find(full)
                    for path in sorted(paths):
                        yield path[root_len:] if root_len else path
                    return
                if not recursive and hasattr(self._fs, "_ls"):
                    entries = await self._fs._ls(full, detail=True)
                    for entry in sorted(entries, key=lambda e: e["name"]):
                        path = entry["name"]
                        rel = path[root_len:] if root_len else path
                        if entry.get("type") == "directory":
                            yield rel if rel.endswith("/") else rel + "/"
                        else:
                            yield rel
                    return
            except FileNotFoundError:
                return

        entries = await asyncio.to_thread(
            lambda: list(self.list_prefix(prefix, recursive=recursive))
        )
        for entry in entries:
            yield entry

    async def aensure_prefix(self, prefix: str) -> None:
        full = self._full_key(prefix)
        if self._is_async_fs and hasattr(self._fs, "_makedirs"):
            try:
                await self._fs._makedirs(full, exist_ok=True)
            except (NotImplementedError, AttributeError):
                pass
            return
        await asyncio.to_thread(self.ensure_prefix, prefix)

    async def aclose(self) -> None:
        return None

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"FsspecBackend({self._url!r})"
