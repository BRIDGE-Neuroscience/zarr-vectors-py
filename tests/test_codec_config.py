"""Tests for the ``compressor=`` kwarg on public write functions.

Covers each `compressor` shape accepted by
:func:`zarr_vectors.encoding.compression.resolve_compressor`:

* ``None`` / ``"none"`` / ``False`` — no compression (``bytes`` only).
  This is the default; keeps the fast async PUT path.
* ``"zstd"`` — zarr v3's default compressor (``bytes`` + ``zstd``).
* ``"blosc"`` — Blosc(Zstd, BitShuffle, l5) shorthand.
* Caller-supplied codec list.

For each codec configuration we (a) inspect the per-chunk
``zarr.json`` and assert the on-disk ``codecs`` list matches what the
resolver returns, and (b) round-trip the written data through
``read_points`` to confirm the chunks decode correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.encoding.compression import (
    ZARR_V3_DEFAULT_ZSTD_CODEC,
    resolve_compressor,
)
from zarr_vectors.types.points import read_points, write_points


CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


def _new_store(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"codec_{prefix}_")) / "store.zarrvectors"


def _read_first_chunk_codecs(store: Path) -> list[dict]:
    """Return the ``codecs`` list from the first vertex-chunk ``zarr.json``."""
    vertices_dir = store / "0" / "vertices"
    for chunk in sorted(vertices_dir.iterdir()):
        inner = chunk / "zarr.json"
        if inner.exists():
            return json.loads(inner.read_text())["codecs"]
    raise RuntimeError(f"no vertex chunks under {vertices_dir}")


def _sorted_positions(p: np.ndarray) -> np.ndarray:
    """Sort rows of a (N, D) array so reorder-on-read can be compared."""
    return p[np.lexsort(p.T)]


@pytest.fixture(scope="module")
def positions() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1000, (5_000, 3)).astype(np.float32)


def test_resolve_compressor_default_is_bytes_only() -> None:
    """``compressor=None`` resolves to BytesCodec-only (no compression).

    The default is the fast async PUT path; compression is opt-in.
    """
    assert resolve_compressor(None) == [{"name": "bytes"}]
    assert resolve_compressor("none") == [{"name": "bytes"}]
    assert resolve_compressor(False) == [{"name": "bytes"}]


def test_resolve_compressor_zstd_shorthand() -> None:
    """``compressor='zstd'`` resolves to zarr v3's default compressor."""
    codecs = resolve_compressor("zstd")
    assert codecs == [{"name": "bytes"}, dict(ZARR_V3_DEFAULT_ZSTD_CODEC)]


def test_resolve_compressor_blosc_shorthand() -> None:
    codecs = resolve_compressor("blosc")
    assert codecs[0] == {"name": "bytes"}
    assert codecs[1]["name"] == "blosc"
    assert codecs[1]["configuration"]["cname"] == "zstd"
    assert codecs[1]["configuration"]["shuffle"] == "bitshuffle"


def test_resolve_compressor_passes_through_list() -> None:
    custom = [{"name": "blosc", "configuration": {
        "cname": "lz4", "clevel": 3, "shuffle": "shuffle",
        "typesize": 4, "blocksize": 0,
    }}]
    out = resolve_compressor(custom)
    # BytesCodec serializer prepended automatically.
    assert out[0] == {"name": "bytes"}
    assert out[1] == custom[0]


def test_resolve_compressor_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        resolve_compressor(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolve_compressor([{"missing_name_key": True}])


def test_write_points_default_is_uncompressed(positions: np.ndarray) -> None:
    """Default ``compressor=None`` writes BytesCodec-only chunks (no
    compression).  Keeps the fast async PUT path active.
    """
    store = _new_store("default")
    write_points(store, positions, chunk_shape=CHUNK, bin_shape=BIN)
    codecs = _read_first_chunk_codecs(store)
    assert codecs == [{"name": "bytes"}]
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_zstd_round_trips(positions: np.ndarray) -> None:
    """``compressor='zstd'`` writes Zstd-compressed chunks and round-trips."""
    store = _new_store("zstd")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="zstd",
    )
    codecs = _read_first_chunk_codecs(store)
    assert codecs[0]["name"] == "bytes"
    assert codecs[1]["name"] == "zstd"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_blosc_shorthand_round_trips(positions: np.ndarray) -> None:
    store = _new_store("blosc")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="blosc",
    )
    codecs = _read_first_chunk_codecs(store)
    assert codecs[0]["name"] == "bytes"
    assert codecs[1]["name"] == "blosc"
    assert codecs[1]["configuration"]["shuffle"] == "bitshuffle"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_custom_list_passes_through(positions: np.ndarray) -> None:
    """Caller-supplied codec list lands on disk verbatim and round-trips."""
    custom = [{"name": "blosc", "configuration": {
        "cname": "lz4", "clevel": 3, "shuffle": "shuffle",
        "typesize": 4, "blocksize": 0,
    }}]
    store = _new_store("custom")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor=custom,
    )
    codecs = _read_first_chunk_codecs(store)
    assert codecs[0] == {"name": "bytes"}
    assert codecs[1]["name"] == "blosc"
    assert codecs[1]["configuration"]["cname"] == "lz4"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_compression_reduces_disk_size(positions: np.ndarray) -> None:
    """``compressor='zstd'`` produces a strictly smaller store than the
    default (BytesCodec-only) for the same input."""
    def store_bytes(store: Path) -> int:
        return sum(f.stat().st_size for f in store.rglob("*") if f.is_file())

    s_default = _new_store("size_default")
    write_points(
        s_default, positions, chunk_shape=CHUNK, bin_shape=BIN,
    )
    s_zstd = _new_store("size_zstd")
    write_points(
        s_zstd, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="zstd",
    )
    assert store_bytes(s_zstd) < store_bytes(s_default), (
        f"zstd store ({store_bytes(s_zstd)} B) was not smaller than "
        f"default uncompressed store ({store_bytes(s_default)} B)"
    )
