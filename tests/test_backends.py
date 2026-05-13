"""Tests for the storage backend system.

Covers:

* URL-scheme detection
* Backend resolution precedence (explicit kwarg / env var / auto)
* ``LocalBackend`` round-trip and listing semantics
* ``Group`` operations against ``LocalBackend``
* Helpful error messages when an optional backend is missing
* ``rebind()`` semantics
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from zarr_vectors.core.backends import (
    LocalBackend,
    SCHEMES_LOCAL,
    SCHEMES_OBJECT_STORE,
    detect_scheme,
    make_backend,
    resolve_backend_name,
)
from zarr_vectors.core.backends.base import StorageBackend
from zarr_vectors.core.group import Group
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.store import create_store, open_store, rebind
from zarr_vectors.exceptions import StoreError


# ===================================================================
# URL scheme detection
# ===================================================================


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/abs/path/store.zvr", ""),
        ("relative/path", ""),
        (r"C:\Users\me\store.zvr", ""),   # bare Windows drive — not a scheme
        ("file:///C:/Users/me/store.zvr", "file"),
        ("file:///tmp/store.zvr", "file"),
        ("s3://bucket/path", "s3"),
        ("gs://bucket/path", "gs"),
        ("gcs://bucket/path", "gcs"),
        ("az://container/path", "az"),
        ("azure://container/path", "azure"),
        ("abfs://container/path", "abfs"),
        ("http://host/path", "http"),
        ("https://host/path", "https"),
    ],
)
def test_detect_scheme(url, expected):
    assert detect_scheme(url) == expected


def test_detect_scheme_path_object(tmp_path):
    assert detect_scheme(tmp_path) == ""


def test_scheme_categories_disjoint():
    assert SCHEMES_LOCAL.isdisjoint(SCHEMES_OBJECT_STORE)


# ===================================================================
# Backend resolution precedence
# ===================================================================


def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_BACKEND", "obstore")
    # Explicit kwarg overrides env.
    assert resolve_backend_name("/local/path", explicit="local") == "local"


def test_resolve_env_var_wins_over_auto(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_BACKEND", "fsspec")
    assert resolve_backend_name("/local/path") == "fsspec"


def test_resolve_auto_local_for_filesystem_path():
    assert resolve_backend_name("/some/path", env_override="") == "local"
    assert resolve_backend_name("file:///tmp/x", env_override="") == "local"


def test_resolve_cloud_without_extras_raises(monkeypatch):
    """s3:// URL with neither obstore nor fsspec installed must error helpfully."""
    # Hide both potential cloud backends from the registry's importability
    # check by injecting None into sys.modules.
    monkeypatch.setitem(sys.modules, "obstore", None)
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(StoreError, match="requires a cloud backend"):
        resolve_backend_name("s3://bucket/key", env_override="")


# ===================================================================
# LocalBackend round-trip
# ===================================================================


def test_local_backend_round_trip(tmp_path):
    be = LocalBackend(tmp_path)

    assert be.url.startswith("file://")
    assert be.exists("") is True

    be.put_bytes("a/b/c.bin", b"hello")
    assert be.get_bytes("a/b/c.bin") == b"hello"
    assert be.exists("a/b/c.bin")
    assert be.exists("a/b")           # intermediate dirs auto-created
    assert not be.exists("a/b/missing.bin")


def test_local_backend_get_missing_raises_keyerror(tmp_path):
    be = LocalBackend(tmp_path)
    with pytest.raises(KeyError):
        be.get_bytes("does-not-exist")


def test_local_backend_list_prefix_marks_directories(tmp_path):
    be = LocalBackend(tmp_path)
    be.put_bytes("g/file.bin", b"x")
    be.put_bytes("g/sub/inner.bin", b"y")

    entries = sorted(be.list_prefix("g", recursive=False))
    # "g/file.bin" is a leaf file; "g/sub/" should be flagged as a container.
    assert any(e == "g/file.bin" for e in entries)
    assert any(e.endswith("/") and e.startswith("g/sub") for e in entries)


def test_local_backend_list_prefix_recursive_files_only(tmp_path):
    be = LocalBackend(tmp_path)
    be.put_bytes("g/file.bin", b"x")
    be.put_bytes("g/sub/inner.bin", b"y")

    files = sorted(be.list_prefix("g", recursive=True))
    assert files == ["g/file.bin", "g/sub/inner.bin"]
    assert not any(f.endswith("/") for f in files)


def test_local_backend_rejects_dotdot(tmp_path):
    be = LocalBackend(tmp_path)
    with pytest.raises(ValueError):
        be.put_bytes("../escape.bin", b"x")


def test_local_backend_url_is_round_trippable(tmp_path):
    be1 = LocalBackend(tmp_path)
    be2 = LocalBackend(be1.url)
    assert be1.url == be2.url


# ===================================================================
# Group on LocalBackend (compact smoke test — full coverage is in test_core.py)
# ===================================================================


def test_group_create_subgroup(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    child = root.create_group("child")
    assert "child" in root
    assert isinstance(child, Group)
    assert child.prefix == "child"


def test_group_url_property(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    child = root.create_group("a").create_group("b")
    assert child.url.endswith("/a/b")


def test_group_path_only_for_local(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    # Local-backed store → .path returns a Path
    assert isinstance(root.path, Path)


# ===================================================================
# make_backend integration
# ===================================================================


def test_make_backend_local_for_path(tmp_path):
    be = make_backend(tmp_path)
    assert isinstance(be, LocalBackend)


def test_make_backend_explicit_local_for_url(tmp_path):
    be = make_backend(str(tmp_path), backend="local")
    assert isinstance(be, LocalBackend)


def test_make_backend_satisfies_protocol(tmp_path):
    be = make_backend(tmp_path)
    assert isinstance(be, StorageBackend)


def test_obstore_missing_dep_message(monkeypatch):
    """Asking for the obstore backend explicitly when not installed should
    raise a helpful StoreError, not a bare ImportError."""
    monkeypatch.setitem(sys.modules, "obstore", None)
    with pytest.raises(StoreError, match="obstore is not installed"):
        make_backend("s3://bucket/x", backend="obstore")


def test_fsspec_missing_dep_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(StoreError, match="fsspec is not installed"):
        make_backend("s3://bucket/x", backend="fsspec")


# ===================================================================
# rebind
# ===================================================================


def _minimal_root_meta():
    return RootMetadata(
        spatial_index_dims=[
            {"name": "x", "type": "space", "unit": "unit"},
            {"name": "y", "type": "space", "unit": "unit"},
            {"name": "z", "type": "space", "unit": "unit"},
        ],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0, 0, 0], [100, 100, 100]),
        geometry_types=["point_cloud"],
    )


def test_rebind_swap_local_for_local(tmp_path):
    """Rebind from one LocalBackend to a fresh LocalBackend at the same URL.

    Trivial but proves the rebind mechanics: same URL must be preserved,
    cached handles must continue to resolve.
    """
    store_path = tmp_path / "test.zvr"
    root = create_store(store_path, _minimal_root_meta())
    original_url = root.url

    rebind(root, "local")
    assert root.url == original_url

    # Read back works after rebind.
    reopened = open_store(store_path)
    assert reopened.attrs["zarr_vectors"]["format_version"]


def test_rebind_url_mismatch_raises(tmp_path):
    """Rebinding to a different URL is a programming error."""
    store_path = tmp_path / "test.zvr"
    root = create_store(store_path, _minimal_root_meta())

    other = LocalBackend(tmp_path / "other.zvr")
    with pytest.raises(StoreError, match="matching URLs"):
        rebind(root, other)


# ===================================================================
# create_store / open_store backend= kwarg routing
# ===================================================================


def test_create_store_with_explicit_local_backend(tmp_path):
    root = create_store(tmp_path / "x.zvr", _minimal_root_meta(), backend="local")
    assert "zarr_vectors" in root.attrs


def test_open_store_with_explicit_local_backend(tmp_path):
    p = tmp_path / "x.zvr"
    create_store(p, _minimal_root_meta())
    root = open_store(p, backend="local")
    assert "zarr_vectors" in root.attrs


def test_cloud_url_routes_to_correct_backend_name(monkeypatch):
    """When obstore is importable, an s3:// URL resolves to obstore."""
    monkeypatch.setitem(sys.modules, "fsspec", None)
    # Pretend obstore is installed (but don't actually call make_backend).
    fake_module = type(sys)("obstore")
    monkeypatch.setitem(sys.modules, "obstore", fake_module)
    assert (
        resolve_backend_name("s3://bucket/path", env_override="")
        == "obstore"
    )


def test_cloud_url_falls_back_to_fsspec_when_obstore_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "obstore", None)
    fake_module = type(sys)("fsspec")
    monkeypatch.setitem(sys.modules, "fsspec", fake_module)
    assert (
        resolve_backend_name("s3://bucket/path", env_override="")
        == "fsspec"
    )
