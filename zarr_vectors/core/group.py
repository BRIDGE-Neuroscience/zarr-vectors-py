"""Backend-agnostic group abstraction wrapping a :class:`zarr.Group`.

This is the format seam: all ZV array I/O routes through this class.
The underlying Zarr store can be any :class:`zarr.abc.store.Store` —
``LocalStore``, ``MemoryStore``, ``FsspecStore``, ``ObjectStore``,
``IcechunkStore``.

Per-chunk byte blobs are stored as tiny single-chunk 1D ``uint8`` Zarr
arrays under a per-array Zarr group (Option G in the design doc):

    level/vertices/0.1.2          → Zarr 1D uint8 array, shape=(N,), chunks=(N,)
    level/vertex_group_offsets/0.1.2  → likewise
    level/object_index/data       → likewise (one blob per slot)

Per-array metadata (the ``zv_array`` discriminator and friends) lives on
the *group* node (``vertices/.attrs`` in v3 maps to ``zarr.json``
``attributes``).

Public surface mirrors the legacy :class:`FsGroup` for back-compat:

* ``attrs`` — dict-like access to this group's attributes
* ``create_group`` / ``require_group`` / ``__getitem__`` / ``__contains__``
  / ``__iter__`` — hierarchy navigation (sub-groups only)
* ``write_bytes`` / ``read_bytes`` / ``chunk_exists`` / ``list_chunks`` —
  per-chunk byte I/O
* ``write_array_meta`` / ``read_array_meta`` / ``array_exists`` —
  per-array metadata
* ``path`` — :class:`pathlib.Path` when the backing store is a Zarr
  ``LocalStore``, raises otherwise (use ``url`` instead)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from zarr.storage import LocalStore

from zarr_vectors.exceptions import StoreError


class Group:
    """A ZV group wrapping an underlying :class:`zarr.Group`."""

    def __init__(self, zarr_group: zarr.Group) -> None:
        self._zarr = zarr_group

    @classmethod
    def _from_zarr(cls, zarr_group: zarr.Group) -> Group:
        instance = cls.__new__(cls)
        instance._zarr = zarr_group
        return instance

    @classmethod
    def _from_backend(cls, store_or_shim: Any, prefix: str = "") -> Group:
        """Build a Group from a Zarr store (or a legacy ``_BackendShim``).

        Kept for back-compat with callers in :mod:`zarr_vectors.lazy`
        that resurrect a root-level Group from a stored backend handle.
        Always returns a write-capable handle — a read-only store
        reference (left over from an ``open_store(mode='r')`` flow) is
        unwrapped via ``store.with_read_only(False)``.
        """
        store = store_or_shim._store if isinstance(store_or_shim, _BackendShim) else store_or_shim
        if getattr(store, "read_only", False) and hasattr(store, "with_read_only"):
            store = store.with_read_only(False)
        path = "/" + prefix.strip("/") if prefix else "/"
        zg = zarr.open_group(store, path=path, mode="r+")
        return cls._from_zarr(zg)

    # ---------------- attributes ----------------

    @property
    def attrs(self) -> _Attrs:
        return _Attrs(self._zarr.attrs)

    # ---------------- sub-groups ----------------

    def create_group(self, name: str, **_kwargs: Any) -> Group:
        zg = self._zarr.require_group(name)
        return type(self)._from_zarr(zg)

    def require_group(self, name: str) -> Group:
        zg = self._zarr.require_group(name)
        return type(self)._from_zarr(zg)

    def __getitem__(self, key: str) -> Group:
        try:
            node = self._zarr[key]
        except KeyError:
            raise StoreError(
                f"Group {key!r} not found under {self._zarr.path or '<root>'}"
            ) from None
        if not isinstance(node, zarr.Group):
            raise StoreError(
                f"{key!r} under {self._zarr.path or '<root>'} is a "
                f"{type(node).__name__}, not a Group"
            )
        return type(self)._from_zarr(node)

    def __contains__(self, key: str) -> bool:
        return key in self._zarr

    def __iter__(self) -> Iterator[str]:
        yield from sorted(self._zarr.group_keys())

    # ---------------- chunk I/O (Option G: 1 tiny array per chunk) ----------------

    def write_bytes(self, array_name: str, chunk_key: str, data: bytes) -> None:
        arr_group = self._zarr.require_group(array_name)
        if chunk_key in arr_group:
            del arr_group[chunk_key]
        n = len(data)
        if n == 0:
            arr_group.create_array(
                chunk_key, shape=(0,), chunks=(1,), dtype="uint8",
            )
            return
        a = arr_group.create_array(
            chunk_key, shape=(n,), chunks=(n,), dtype="uint8",
        )
        a[:] = np.frombuffer(data, dtype="uint8")

    def read_bytes(self, array_name: str, chunk_key: str) -> bytes:
        path = f"{array_name}/{chunk_key}"
        try:
            arr = self._zarr[path]
        except KeyError:
            shard_bytes = _read_from_shard(self._zarr, array_name, chunk_key)
            if shard_bytes is not None:
                return shard_bytes
            raise StoreError(
                f"Chunk {array_name!r}/{chunk_key!r} not found in "
                f"{self._zarr.path or '<root>'}"
            ) from None
        if not isinstance(arr, zarr.Array):
            raise StoreError(
                f"{path!r} is a {type(arr).__name__}, not an Array"
            )
        if arr.shape[0] == 0:
            return b""
        return bytes(np.asarray(arr[:]).tobytes())

    def chunk_exists(self, array_name: str, chunk_key: str) -> bool:
        if f"{array_name}/{chunk_key}" in self._zarr:
            return True
        return _shard_index_contains(self._zarr, array_name, chunk_key)

    def list_chunks(self, array_name: str) -> list[str]:
        if array_name not in self._zarr:
            return []
        try:
            node = self._zarr[array_name]
        except KeyError:
            return []
        if not isinstance(node, zarr.Group):
            return []
        keys: set[str] = set()
        for k in node.array_keys():
            if k.startswith("__shard_"):
                shard_arr = node[k]
                index = shard_arr.attrs.get("shard_index", None)
                if index:
                    keys.update(index.keys())
            else:
                keys.add(k)
        return sorted(keys)

    # ---------------- array metadata ----------------

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
        arr_group = self._zarr.require_group(array_name)
        arr_group.attrs.update(_json_safe(meta))

    def read_array_meta(self, array_name: str) -> dict[str, Any]:
        if array_name not in self._zarr:
            return {}
        try:
            node = self._zarr[array_name]
        except KeyError:
            return {}
        if not isinstance(node, zarr.Group):
            return {}
        return dict(node.attrs)

    def array_exists(self, array_name: str) -> bool:
        if array_name not in self._zarr:
            return False
        try:
            node = self._zarr[array_name]
        except KeyError:
            return False
        return isinstance(node, zarr.Group)

    # ---------------- delete ----------------

    def delete_subtree(self, name: str) -> None:
        if name in self._zarr:
            del self._zarr[name]

    # ---------------- path / url ----------------

    @property
    def path(self) -> Path:
        store = self._zarr.store
        if not isinstance(store, LocalStore):
            raise StoreError(
                f"Group.path is only available for LocalStore; got "
                f"{type(store).__name__}. Use Group.url instead."
            )
        root = _local_root(store)
        if self._zarr.path:
            return root.joinpath(*self._zarr.path.strip("/").split("/"))
        return root

    @property
    def url(self) -> str:
        store = self._zarr.store
        if isinstance(store, LocalStore):
            base = _local_root(store).absolute().as_uri()
        else:
            base = repr(store)
        if self._zarr.path:
            return f"{base.rstrip('/')}/{self._zarr.path.strip('/')}"
        return base

    @property
    def prefix(self) -> str:
        return self._zarr.path

    # ---------------- back-compat shims (used by lazy/) ----------------

    @property
    def backend(self) -> _BackendShim:
        return _BackendShim(self._zarr.store)

    @property
    def _backend(self) -> _BackendShim:  # noqa: D401  (legacy callers)
        return _BackendShim(self._zarr.store)

    @property
    def zarr_group(self) -> zarr.Group:
        """The underlying :class:`zarr.Group`."""
        return self._zarr

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"Group({self.url!r})"


# ===================================================================
# .attrs dict-like wrapper
# ===================================================================


class _Attrs:
    """Dict-like wrapper around :attr:`zarr.Group.attrs`.

    The wrapper exists for API parity with the legacy on-disk-JSON
    ``_Attrs`` — callers use ``attrs.to_dict()``, ``attrs.update(d)``,
    ``attrs[k]``, ``attrs.get(k, default)``, ``k in attrs``.
    """

    def __init__(self, zarr_attrs: Any) -> None:
        self._attrs = zarr_attrs

    def __getitem__(self, key: str) -> Any:
        return self._attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._attrs[key] = _json_safe_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._attrs

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._attrs[key]
        except KeyError:
            return default

    def update(self, other: dict[str, Any]) -> None:
        self._attrs.update(_json_safe(other))

    def to_dict(self) -> dict[str, Any]:
        return dict(self._attrs)

    def __repr__(self) -> str:
        return f"_Attrs({dict(self._attrs)!r})"


# ===================================================================
# Back-compat shim for `group._backend.url` / `Group._from_backend`
# ===================================================================


class _BackendShim:
    """Minimal compat shim for callers that reach for ``group._backend``.

    Provides the ``url`` accessor and identity needed by
    :class:`Group._from_backend`.  Anything else raises ``AttributeError``
    so we notice if some caller depends on the deeper legacy surface.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @property
    def url(self) -> str:
        if isinstance(self._store, LocalStore):
            return _local_root(self._store).absolute().as_uri()
        return repr(self._store)

    def close(self) -> None:
        try:
            close = getattr(self._store, "close", None)
            if close is not None:
                close()
        except Exception:  # pragma: no cover  (defensive)
            pass


# ===================================================================
# Helpers
# ===================================================================


def _local_root(store: LocalStore) -> Path:
    raw = store.root
    return raw if isinstance(raw, Path) else Path(raw)


# ---------------- shard fallback (transparent read of packed chunks) ----------

def _read_from_shard(
    zarr_group: zarr.Group, array_name: str, chunk_key: str,
) -> bytes | None:
    """Locate ``chunk_key`` inside any ``__shard_<id>`` array under
    ``array_name`` and return its bytes.  Returns ``None`` when no
    shard contains the key.
    """
    if array_name not in zarr_group:
        return None
    try:
        arr_group = zarr_group[array_name]
    except KeyError:
        return None
    if not isinstance(arr_group, zarr.Group):
        return None
    for name in arr_group.array_keys():
        if not name.startswith("__shard_"):
            continue
        shard_arr = arr_group[name]
        index = shard_arr.attrs.get("shard_index", None)
        if not index or chunk_key not in index:
            continue
        offset, nbytes = index[chunk_key]
        packed = np.asarray(shard_arr[offset:offset + nbytes])
        return bytes(packed.tobytes())
    return None


def _shard_index_contains(
    zarr_group: zarr.Group, array_name: str, chunk_key: str,
) -> bool:
    if array_name not in zarr_group:
        return False
    try:
        arr_group = zarr_group[array_name]
    except KeyError:
        return False
    if not isinstance(arr_group, zarr.Group):
        return False
    for name in arr_group.array_keys():
        if not name.startswith("__shard_"):
            continue
        index = arr_group[name].attrs.get("shard_index", None)
        if index and chunk_key in index:
            return True
    return False


def _json_safe(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_safe_value(v) for k, v in d.items()}


def _json_safe_value(v: Any) -> Any:
    """Coerce numpy scalars / arrays to JSON-native types for zarr attrs."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return _json_safe(v)
    return v
