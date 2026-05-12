"""Tests for the AsyncStorageBackend protocol and LocalBackend's async path."""

from __future__ import annotations

import asyncio
import sys

import pytest

from zarr_vectors.core.backends import (
    AsyncStorageBackend,
    LocalBackend,
    make_async_backend,
)
from zarr_vectors.exceptions import StoreError


def _run(coro):
    return asyncio.run(coro)


def test_local_backend_satisfies_async_protocol(tmp_path):
    be = LocalBackend(tmp_path)
    assert isinstance(be, AsyncStorageBackend)


def test_async_round_trip(tmp_path):
    be = LocalBackend(tmp_path)

    async def go():
        await be.aput_bytes("a/b/c.bin", b"hello")
        data = await be.aget_bytes("a/b/c.bin")
        assert data == b"hello"
        assert await be.aexists("a/b/c.bin")
        assert not await be.aexists("missing")

    _run(go())


def test_async_get_missing_raises_keyerror(tmp_path):
    be = LocalBackend(tmp_path)

    async def go():
        with pytest.raises(KeyError):
            await be.aget_bytes("nope")

    _run(go())


def test_async_parallel_puts_via_gather(tmp_path):
    """``asyncio.gather`` over many puts should land them all."""
    be = LocalBackend(tmp_path)

    async def go():
        await asyncio.gather(*[
            be.aput_bytes(f"par/{i}.bin", bytes([i % 256]))
            for i in range(100)
        ])
        entries = [
            e async for e in be.alist_prefix("par")
            if not e.endswith("/")
        ]
        assert len(entries) == 100

    _run(go())


def test_async_delete_and_delete_prefix(tmp_path):
    be = LocalBackend(tmp_path)

    async def go():
        await be.aput_bytes("g/a.bin", b"x")
        await be.aput_bytes("g/b.bin", b"y")
        await be.aput_bytes("g/sub/c.bin", b"z")
        await be.adelete("g/a.bin")
        assert not await be.aexists("g/a.bin")
        await be.adelete_prefix("g")
        assert not await be.aexists("g/b.bin")
        assert not await be.aexists("g/sub/c.bin")

    _run(go())


def test_async_list_prefix_marks_directories(tmp_path):
    be = LocalBackend(tmp_path)

    async def go():
        await be.aput_bytes("g/file.bin", b"x")
        await be.aput_bytes("g/sub/inner.bin", b"y")
        entries = [e async for e in be.alist_prefix("g", recursive=False)]
        assert any(e == "g/file.bin" for e in entries)
        assert any(e.endswith("/") and e.startswith("g/sub") for e in entries)

    _run(go())


def test_async_list_prefix_recursive_files_only(tmp_path):
    be = LocalBackend(tmp_path)

    async def go():
        await be.aput_bytes("g/file.bin", b"x")
        await be.aput_bytes("g/sub/inner.bin", b"y")
        files = sorted([
            e async for e in be.alist_prefix("g", recursive=True)
        ])
        assert files == ["g/file.bin", "g/sub/inner.bin"]
        assert not any(f.endswith("/") for f in files)

    _run(go())


def test_make_async_backend_returns_async_protocol(tmp_path):
    be = make_async_backend(tmp_path)
    assert isinstance(be, AsyncStorageBackend)


def test_make_async_backend_missing_cloud_dep_message(monkeypatch):
    """Asking for obstore async on a cloud URL with no obstore should
    raise the same helpful error as the sync path."""
    monkeypatch.setitem(sys.modules, "obstore", None)
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(StoreError, match="requires a cloud backend"):
        make_async_backend("s3://bucket/k")
