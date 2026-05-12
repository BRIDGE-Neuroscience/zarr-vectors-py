"""obstore-backed :class:`StorageBackend`.

Uses the `obstore <https://developmentseed.org/obstore/>`_ Rust-based
object-store client.  Supports S3, GCS, Azure, HTTP, and local-file
URLs through a single uniform API, with parallel uploads/downloads and
native delimited listing.

``obstore`` must be installed separately:

.. code-block:: bash

    pip install "zarr-vectors[obstore]"
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator
from urllib.parse import urlparse

from zarr_vectors.exceptions import StoreError


class ObstoreBackend:
    """Object-store backend using the obstore client.

    The constructor parses the URL and selects the appropriate obstore
    store class (S3Store, GCSStore, AzureStore, HTTPStore, LocalStore).
    Credentials and other connection options are passed through
    ``**kwargs``.

    Args:
        url: A store URL.  Supported schemes: ``s3``, ``gs`` / ``gcs``,
            ``az`` / ``azure`` / ``abfs``, ``http`` / ``https``, ``file``,
            or a plain filesystem path.
        **kwargs: Forwarded to the selected obstore store constructor
            (e.g. ``access_key_id``, ``secret_access_key``, ``region``).
    """

    def __init__(self, url: str, **kwargs: Any) -> None:
        try:
            import obstore  # noqa: F401
            from obstore.store import (
                AzureStore,
                GCSStore,
                HTTPStore,
                LocalStore,
                S3Store,
            )
        except ImportError as e:  # pragma: no cover - exercised via dispatcher
            raise StoreError(
                "obstore is not installed.  Install with: "
                "pip install zarr-vectors[obstore]"
            ) from e

        self._obstore = obstore
        self._url = url

        parsed = urlparse(url) if isinstance(url, str) else None
        scheme = parsed.scheme.lower() if parsed else ""

        if scheme == "s3":
            bucket = parsed.netloc
            self._prefix = parsed.path.lstrip("/")
            self._store = S3Store(bucket, **kwargs)
        elif scheme in ("gs", "gcs"):
            bucket = parsed.netloc
            self._prefix = parsed.path.lstrip("/")
            self._store = GCSStore(bucket, **kwargs)
        elif scheme in ("az", "azure", "abfs"):
            container = parsed.netloc
            self._prefix = parsed.path.lstrip("/")
            self._store = AzureStore(container, **kwargs)
        elif scheme in ("http", "https"):
            self._prefix = ""
            self._store = HTTPStore.from_url(url, **kwargs)
        elif scheme in ("file", ""):
            from zarr_vectors.core.backends.local import _file_url_to_path

            local_path = _file_url_to_path(url) if scheme == "file" else url
            self._prefix = ""
            self._store = LocalStore(local_path, **kwargs)
        else:
            raise StoreError(
                f"obstore: unsupported URL scheme {scheme!r} in {url!r}"
            )

    # ---------------- properties ----------------

    @property
    def url(self) -> str:
        return self._url

    # ---------------- key/path mapping ----------------

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{key}" if key else self._prefix
        return key

    # ---------------- byte I/O ----------------

    def put_bytes(self, key: str, data: bytes) -> None:
        self._obstore.put(self._store, self._full_key(key), data)

    def get_bytes(self, key: str) -> bytes:
        try:
            result = self._obstore.get(self._store, self._full_key(key))
        except Exception as e:
            # obstore raises a NotFoundError or generic exception; surface
            # as KeyError to match the protocol contract.
            if "not found" in str(e).lower() or "NoSuchKey" in str(e):
                raise KeyError(key) from None
            raise
        return bytes(result.bytes())

    def exists(self, key: str) -> bool:
        try:
            self._obstore.head(self._store, self._full_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        try:
            self._obstore.delete(self._store, self._full_key(key))
        except Exception:
            pass

    def delete_prefix(self, prefix: str) -> None:
        full_prefix = self._full_key(prefix)
        for entry in self._obstore.list(self._store, prefix=full_prefix):
            for obj in entry:
                self._obstore.delete(self._store, obj["path"])

    def list_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> Iterator[str]:
        full_prefix = self._full_key(prefix)
        # Object stores treat prefixes as opaque strings; ensure a
        # trailing slash so we don't get accidental matches like
        # "foo/bar" matching prefix "foo/ba".
        listing_prefix = full_prefix + "/" if full_prefix else ""
        plen = len(self._prefix) + 1 if self._prefix else 0

        if recursive:
            for entry in self._obstore.list(self._store, prefix=listing_prefix):
                for obj in entry:
                    full = obj["path"]
                    yield full[plen:] if plen else full
        else:
            # Delimited listing yields immediate children only.
            result = self._obstore.list_with_delimiter(
                self._store, prefix=listing_prefix
            )
            # Sub-prefixes (containers)
            for sub in sorted(result.get("common_prefixes", [])):
                rel = sub[plen:] if plen else sub
                # Always trail with /; obstore already includes one.
                yield rel if rel.endswith("/") else rel + "/"
            # Objects (files)
            for obj in sorted(result.get("objects", []), key=lambda o: o["path"]):
                full = obj["path"]
                yield full[plen:] if plen else full

    def ensure_prefix(self, prefix: str) -> None:
        # No-op: object stores create prefixes implicitly on first write.
        return None

    def close(self) -> None:
        # obstore stores hold a small amount of connection state; no
        # explicit close is required.
        return None

    # ---------------- async I/O (native obstore coroutines) ----------------
    #
    # obstore ships ``*_async`` peers for the I/O verbs we use; they are
    # real coroutines that run on the underlying tokio runtime, so
    # ``asyncio.gather`` over them parallelises object-store requests
    # without per-request thread overhead.  We probe for each function
    # and fall back to ``asyncio.to_thread`` when an older obstore
    # release is in use.

    def _coro_or_thread(self, async_name: str, sync_callable, *args):
        """Pick the obstore async function if present, else thread-wrap."""
        fn = getattr(self._obstore, async_name, None)
        if fn is not None:
            return fn(self._store, *args)
        return asyncio.to_thread(sync_callable, *args)

    async def aput_bytes(self, key: str, data: bytes) -> None:
        full = self._full_key(key)
        fn = getattr(self._obstore, "put_async", None)
        if fn is not None:
            await fn(self._store, full, data)
        else:
            await asyncio.to_thread(self.put_bytes, key, data)

    async def aget_bytes(self, key: str) -> bytes:
        full = self._full_key(key)
        fn = getattr(self._obstore, "get_async", None)
        if fn is None:
            return await asyncio.to_thread(self.get_bytes, key)
        try:
            result = await fn(self._store, full)
        except Exception as e:
            if "not found" in str(e).lower() or "NoSuchKey" in str(e):
                raise KeyError(key) from None
            raise
        # ``result.bytes()`` may itself be a coroutine on the async path.
        body = result.bytes()
        if asyncio.iscoroutine(body):
            body = await body
        return bytes(body)

    async def aexists(self, key: str) -> bool:
        full = self._full_key(key)
        fn = getattr(self._obstore, "head_async", None)
        if fn is None:
            return await asyncio.to_thread(self.exists, key)
        try:
            await fn(self._store, full)
            return True
        except Exception:
            return False

    async def adelete(self, key: str) -> None:
        full = self._full_key(key)
        fn = getattr(self._obstore, "delete_async", None)
        if fn is None:
            await asyncio.to_thread(self.delete, key)
            return
        try:
            await fn(self._store, full)
        except Exception:
            pass

    async def adelete_prefix(self, prefix: str) -> None:
        # Delegate to the sync path inside a thread; ``delete_prefix``
        # is already a list-then-delete loop that benefits little from
        # interleaving with other work.
        await asyncio.to_thread(self.delete_prefix, prefix)

    async def alist_prefix(
        self, prefix: str, *, recursive: bool = False
    ) -> AsyncIterator[str]:
        # Materialise via the sync iterator in a thread; the per-entry
        # work is trivial after the listing request returns.
        entries = await asyncio.to_thread(
            lambda: list(self.list_prefix(prefix, recursive=recursive))
        )
        for entry in entries:
            yield entry

    async def aensure_prefix(self, prefix: str) -> None:
        return None

    async def aclose(self) -> None:
        return None

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"ObstoreBackend({self._url!r})"
