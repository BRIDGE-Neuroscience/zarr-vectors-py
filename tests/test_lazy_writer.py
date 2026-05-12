"""Tests for the ZVRWriter (Tier A + append_vertices + pending sidecars)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    compact_object_index,
    read_all_object_manifests,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.lazy.store import open_zvr
from zarr_vectors.types.points import read_points, write_points


def _run(coro):
    return asyncio.run(coro)


def _make_store(tmp_path, n=200):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    store = tmp_path / "p.zvr"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.arange(n, dtype=np.int64),
    )
    return store, pos


# ===================================================================
# Tier A — add_attribute
# ===================================================================


def test_add_attribute_round_trip(tmp_path):
    store, pos = _make_store(tmp_path, n=200)
    normals = np.random.default_rng(1).normal(size=(200, 3)).astype("f4")

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.add_attribute("normal", normals)

    _run(go())

    out = read_points(str(store), attribute_names=["normal"])
    assert "normal" in out["attributes"]
    # Data flattens via read_points's ncols=1 path; total count matches.
    assert out["attributes"]["normal"].size == 200 * 3


def test_add_attribute_sync_mirror(tmp_path):
    store, _ = _make_store(tmp_path, n=120)
    rng = np.random.default_rng(2)
    intensities = rng.uniform(0, 1, 120).astype("f4")

    zvr = open_zvr(str(store))
    with zvr[0].writer() as w:
        w.add_attribute_sync("intensity", intensities)

    out = read_points(str(store), attribute_names=["intensity"])
    assert out["attributes"]["intensity"].size == 120


def test_add_attribute_length_mismatch_raises(tmp_path):
    store, _ = _make_store(tmp_path, n=50)
    bad = np.zeros(51, dtype="f4")  # one too many

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.add_attribute("bad", bad)

    from zarr_vectors.exceptions import ArrayError
    with pytest.raises(ArrayError, match="!= level vertex count"):
        _run(go())


def test_add_object_attribute(tmp_path):
    from zarr_vectors.core.arrays import read_object_attributes
    store, _ = _make_store(tmp_path, n=80)

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.add_object_attribute("score", np.arange(80, dtype="f4"))

    _run(go())

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    scores = read_object_attributes(lvl, "score")
    assert scores.shape == (80,)
    assert float(scores[5]) == 5.0


# ===================================================================
# append_vertices + pending sidecar
# ===================================================================


def test_append_vertices_grows_store(tmp_path):
    store, _ = _make_store(tmp_path, n=100)
    new_pos = np.random.default_rng(3).uniform(0, 100, (40, 3)).astype("f4")

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            result = await w.append_vertices(new_pos)
            return result

    summary = _run(go())
    assert summary["vertices_added"] == 40
    assert summary["new_objects"] == 40

    out = read_points(str(store))
    assert out["vertex_count"] == 140

    root = open_store(str(store))
    rm = read_root_metadata(root)
    # vertex_count_cache always present; pending should be cleared by
    # the auto-compact path... actually we only compact on explicit
    # call.  The capability is still present after the first commit.
    assert "vertex_count_cache" in rm.format_capabilities


def test_append_then_compact_clears_pending(tmp_path):
    store, _ = _make_store(tmp_path, n=60)

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(4).uniform(0, 100, (10, 3)).astype("f4")
            )

    _run(go())
    rm_before = read_root_metadata(open_store(str(store)))
    assert "object_index_pending" in rm_before.format_capabilities

    # Compact
    async def do_compact():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            return await w.compact()

    result = _run(do_compact())
    assert result["compacted"] is True
    assert result["batches_folded"] == 1
    assert result["num_objects"] == 70

    rm_after = read_root_metadata(open_store(str(store)))
    assert "object_index_pending" not in rm_after.format_capabilities

    # Still readable after compact
    assert read_points(str(store))["vertex_count"] == 70


def test_two_pending_batches_unioned_on_read(tmp_path):
    store, _ = _make_store(tmp_path, n=30)

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(5).uniform(0, 100, (5, 3)).astype("f4")
            )
        # Re-open: each writer commit creates one batch
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(6).uniform(0, 100, (7, 3)).astype("f4")
            )

    _run(go())

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    manifests = read_all_object_manifests(lvl)
    assert len(manifests) == 42


def test_append_vertices_overlap_oid_raises(tmp_path):
    store, _ = _make_store(tmp_path, n=20)
    overlap = np.array([5, 6, 7], dtype=np.int64)  # collide with existing

    async def go():
        zvr = open_zvr(str(store))
        async with zvr[0].writer() as w:
            await w.append_vertices(
                np.zeros((3, 3), dtype="f4"),
                object_ids=overlap,
            )

    from zarr_vectors.exceptions import ArrayError
    with pytest.raises(ArrayError, match="overlap existing"):
        _run(go())
