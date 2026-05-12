"""Backend-agnostic group abstraction.

A :class:`Group` is a backend reference plus a key prefix.  All ZV
conventions (``.zattrs`` JSON files, chunk-key-as-filename, sub-group as
prefix) live in this class — backends know only about bytes and keys.

Public surface is identical to the legacy ``FsGroup``:

* ``attrs`` — dict-like access to ``.zattrs`` JSON
* ``create_group`` / ``require_group`` / ``__getitem__`` / ``__contains__``
  / ``__iter__`` — hierarchy navigation
* ``write_bytes`` / ``read_bytes`` / ``chunk_exists`` / ``list_chunks`` —
  chunk-file I/O within arrays
* ``write_array_meta`` / ``read_array_meta`` / ``array_exists`` —
  per-array metadata
* ``path`` — :class:`pathlib.Path` when the backend is local, raises
  otherwise (use ``url`` instead)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from zarr_vectors.exceptions import StoreError

from zarr_vectors.core.backends.base import StorageBackend
from zarr_vectors.core.backends.local import LocalBackend


_ZATTRS = ".zattrs"


class Group:
    """A ZV group rooted at ``prefix`` within ``backend``.

    Args:
        backend: A :class:`StorageBackend` instance.
        prefix: Forward-slash-separated key prefix within the backend.
            Empty string means the root.
    """

    def __init__(self, backend: StorageBackend, prefix: str = "") -> None:
        self._backend = backend
        self._prefix = prefix.strip("/")

    @classmethod
    def _from_backend(cls, backend: StorageBackend, prefix: str = "") -> Group:
        """Build an instance of ``cls`` from an existing backend reference.

        Bypasses subclass ``__init__`` so child-group accessors can return
        the same concrete type as ``self`` without re-validating paths.
        """
        instance = cls.__new__(cls)
        instance._backend = backend
        instance._prefix = prefix.strip("/")
        return instance

    # ---------------- internal helpers ----------------

    def _join(self, *parts: str) -> str:
        """Join ``self._prefix`` with one or more sub-parts."""
        bits = [self._prefix] if self._prefix else []
        for p in parts:
            p = p.strip("/")
            if p:
                bits.append(p)
        return "/".join(bits)

    def _is_container(self, key: str) -> bool:
        """Return True if ``key`` looks like a ZV group container."""
        if self._backend.exists(self._zattrs_key(key)):
            return True
        for _ in self._backend.list_prefix(key, recursive=False):
            return True
        # Empty directory on local fs (no .zattrs yet, no children).
        return self._backend.exists(key)

    @staticmethod
    def _zattrs_key(prefix: str) -> str:
        return f"{prefix}/{_ZATTRS}" if prefix else _ZATTRS

    # ---------------- attributes ----------------

    @property
    def attrs(self) -> _Attrs:
        """Dict-like access to this group's ``.zattrs`` JSON."""
        return _Attrs(self._backend, self._zattrs_key(self._prefix))

    # ---------------- sub-groups ----------------

    def create_group(self, name: str, **_kwargs: Any) -> Group:
        """Create a sub-group, or return an existing one with the same name.

        On filesystem backends this creates an empty directory.  On flat
        object stores it is a no-op until the first byte is written.
        """
        child = self._join(name)
        self._backend.ensure_prefix(child)
        return type(self)._from_backend(self._backend, child)

    def require_group(self, name: str) -> Group:
        """Get or create a sub-group.  Always succeeds."""
        return self.create_group(name)

    def __getitem__(self, key: str) -> Group:
        """Navigate to a sub-group by name.  Slash-separated paths supported.

        Raises:
            StoreError: If the target sub-group does not exist.
        """
        target = self._join(key)
        if not self._is_container(target):
            raise StoreError(
                f"Group {key!r} not found under {self._prefix or '<root>'} "
                f"at {self._backend.url}"
            )
        return type(self)._from_backend(self._backend, target)

    def __contains__(self, key: str) -> bool:
        """Return True if ``key`` (file or sub-group) exists under this group."""
        full = self._join(key)
        if self._backend.exists(full):
            return True
        for _ in self._backend.list_prefix(full, recursive=False):
            return True
        return False

    def __iter__(self) -> Iterator[str]:
        """Yield names of immediate sub-group / array children (sorted).

        Hidden entries (``.zattrs`` and other dotfiles) are skipped.  Only
        container entries are yielded — leaf files are excluded.
        """
        seen: set[str] = set()
        for entry in self._backend.list_prefix(self._prefix, recursive=False):
            if not entry.endswith("/"):
                continue
            rel = entry[len(self._prefix) + 1:] if self._prefix else entry
            name = rel.rstrip("/").split("/", 1)[0]
            if not name or name.startswith(".") or name in seen:
                continue
            seen.add(name)
            yield name

    # ---------------- chunk I/O ----------------

    def write_bytes(self, array_name: str, chunk_key: str, data: bytes) -> None:
        """Write raw bytes to a chunk file under ``array_name/chunk_key``."""
        self._backend.put_bytes(self._join(array_name, chunk_key), data)

    def read_bytes(self, array_name: str, chunk_key: str) -> bytes:
        """Read raw bytes from ``array_name/chunk_key``.

        Raises:
            StoreError: If the chunk does not exist.
        """
        key = self._join(array_name, chunk_key)
        try:
            return self._backend.get_bytes(key)
        except KeyError:
            raise StoreError(
                f"Chunk {array_name!r}/{chunk_key!r} not found at "
                f"{self._backend.url}/{key}"
            ) from None

    def chunk_exists(self, array_name: str, chunk_key: str) -> bool:
        """Return True if a chunk file exists."""
        return self._backend.exists(self._join(array_name, chunk_key))

    def list_chunks(self, array_name: str) -> list[str]:
        """Return sorted chunk keys for ``array_name``.

        Excludes ``.zattrs`` and any sub-directories.
        """
        arr_prefix = self._join(array_name)
        chunks: list[str] = []
        for entry in self._backend.list_prefix(arr_prefix, recursive=False):
            if entry.endswith("/"):
                continue
            name = entry.rsplit("/", 1)[-1]
            if not name or name.startswith("."):
                continue
            chunks.append(name)
        return sorted(chunks)

    # ---------------- array metadata ----------------

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
        """Write array metadata to ``<array>/.zattrs``."""
        self._backend.ensure_prefix(self._join(array_name))
        self._backend.put_bytes(
            self._join(array_name, _ZATTRS),
            _dump_json(meta),
        )

    def read_array_meta(self, array_name: str) -> dict[str, Any]:
        """Read array metadata from ``<array>/.zattrs``.  Returns ``{}`` if absent."""
        try:
            data = self._backend.get_bytes(self._join(array_name, _ZATTRS))
        except KeyError:
            return {}
        return json.loads(data)

    def array_exists(self, array_name: str) -> bool:
        """Return True if an array container exists."""
        full = self._join(array_name)
        return self._is_container(full)

    # ---------------- path / url ----------------

    @property
    def path(self) -> Path:
        """Filesystem :class:`Path` of this group.

        Only available for backends backed by a local filesystem.  Use
        :attr:`url` for a portable identifier.

        Raises:
            StoreError: If the active backend is not a local filesystem.
        """
        if not isinstance(self._backend, LocalBackend):
            raise StoreError(
                f"Group.path is only available for LocalBackend; got "
                f"{type(self._backend).__name__}.  Use Group.url instead."
            )
        if self._prefix:
            return self._backend.root.joinpath(*self._prefix.split("/"))
        return self._backend.root

    @property
    def url(self) -> str:
        """Canonical URL of this group's location.

        Equal to ``backend.url`` for the root group; appended with a
        slash-path for sub-groups.
        """
        base = self._backend.url.rstrip("/")
        return base if not self._prefix else f"{base}/{self._prefix}"

    @property
    def backend(self) -> StorageBackend:
        """The underlying :class:`StorageBackend`."""
        return self._backend

    @property
    def prefix(self) -> str:
        """The key prefix within the backend."""
        return self._prefix

    # ---------------- delete ----------------

    def delete_subtree(self, name: str) -> None:
        """Remove a sub-group / array and all its contents."""
        self._backend.delete_prefix(self._join(name))

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"Group({self.url!r})"


# ===================================================================
# .zattrs dict-like wrapper
# ===================================================================


class _Attrs:
    """Dict-like wrapper around a backend-stored ``.zattrs`` JSON document."""

    def __init__(self, backend: StorageBackend, key: str) -> None:
        self._backend = backend
        self._key = key

    def _load(self) -> dict[str, Any]:
        try:
            data = self._backend.get_bytes(self._key)
        except KeyError:
            return {}
        return json.loads(data)

    def _save(self, d: dict[str, Any]) -> None:
        self._backend.put_bytes(self._key, _dump_json(d))

    def __getitem__(self, key: str) -> Any:
        d = self._load()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __setitem__(self, key: str, value: Any) -> None:
        d = self._load()
        d[key] = value
        self._save(d)

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def update(self, other: dict[str, Any]) -> None:
        d = self._load()
        d.update(other)
        self._save(d)

    def to_dict(self) -> dict[str, Any]:
        return self._load()

    def __repr__(self) -> str:
        return f"_Attrs({self._key!r})"


# ===================================================================
# JSON serialisation
# ===================================================================


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy scalar / array types."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dump_json(d: dict[str, Any]) -> bytes:
    return json.dumps(d, indent=2, default=_json_default).encode("utf-8")
