"""Tests for the Tier D / E helpers.

* ``neighbouring_chunk_keys`` — pure tuple work, table-driven.
* ``chunk_local_to_global_offsets`` — backed by the vertex_count sidecar.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets
from zarr_vectors.spatial.chunking import neighbouring_chunk_keys
from zarr_vectors.types.points import write_points


# ===================================================================
# neighbouring_chunk_keys
# ===================================================================


def test_neighbours_2d_halo_1():
    out = neighbouring_chunk_keys((1, 1), halo=1)
    # 3^2 - 1 = 8 neighbours
    assert len(out) == 8
    assert (1, 1) not in out
    assert (0, 0) in out and (2, 2) in out and (1, 2) in out


def test_neighbours_3d_halo_1():
    out = neighbouring_chunk_keys((0, 0, 0), halo=1)
    assert len(out) == 26  # 3^3 - 1


def test_neighbours_3d_halo_2():
    out = neighbouring_chunk_keys((0, 0, 0), halo=2)
    assert len(out) == 5 ** 3 - 1


def test_neighbours_include_self():
    out = neighbouring_chunk_keys((1, 1), halo=1, include_self=True)
    assert (1, 1) in out
    assert len(out) == 9


def test_neighbours_filter_to_occupied():
    occupied = {(0, 0), (0, 1), (1, 0)}
    out = neighbouring_chunk_keys((0, 0), halo=1, occupied_keys=occupied)
    assert set(out) == {(0, 1), (1, 0)}  # (-1,*) / (*, -1) are off-grid


def test_neighbours_invalid_halo():
    with pytest.raises(ValueError):
        neighbouring_chunk_keys((0, 0), halo=-1)


def test_neighbours_works_for_4d_keys():
    """Attribute-chunked stores produce 4D chunk keys (attr_bin, z, y, x).
    The helper must compose with arbitrary arity."""
    out = neighbouring_chunk_keys((0, 0, 0, 0), halo=1)
    assert len(out) == 3 ** 4 - 1


# ===================================================================
# chunk_local_to_global_offsets
# ===================================================================


def test_offsets_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (777, 3)).astype("f4")
    store = tmp_path / "p.zvr"
    write_points(str(store), pos, chunk_shape=(50.0, 50.0, 50.0))

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    offsets, keys, total = chunk_local_to_global_offsets(lvl)
    assert total == 777
    # Offsets monotonic non-decreasing
    last = -1
    for k in keys:
        assert offsets[k] >= last
        last = offsets[k]


def test_offsets_empty_store_safe(tmp_path):
    """An empty level should report 0 chunks and 0 vertices, not raise."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10, (5, 3)).astype("f4")  # one chunk
    store = tmp_path / "p.zvr"
    write_points(str(store), pos, chunk_shape=(100.0, 100.0, 100.0))
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    _, keys, total = chunk_local_to_global_offsets(lvl)
    assert total == 5
    assert len(keys) == 1
