"""Tests for cross-chunk face records.

In 0.6.0 cross-chunk faces are stored as ``link_width=3`` records
under ``cross_chunk_links/<delta=0>/`` instead of a separate
``cross_chunk_faces`` array.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import read_cross_chunk_links
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh


def _tetra_straddling_chunks(tmp_path):
    """A 4-vertex tetrahedron whose vertices land in different 50³ chunks."""
    verts = np.array([
        [40, 40, 40],  # chunk (0,0,0)
        [60, 40, 40],  # chunk (1,0,0)
        [50, 50, 50],  # chunk (1,1,1)  (exactly on boundary; rounds up)
        [50, 50, 60],  # chunk (1,1,1)
    ], dtype="f4")
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 3],
    ], dtype=np.int64)
    store = tmp_path / "m.zv"
    write_mesh(
        str(store), verts, faces,
        chunk_shape=(50.0, 50.0, 50.0),
    )
    return store, verts, faces


def test_read_mesh_returns_cross_chunk_faces(tmp_path):
    store, _, faces_in = _tetra_straddling_chunks(tmp_path)
    out = read_mesh(str(store))
    assert out["face_count"] == len(faces_in)
    assert out["vertex_count"] == 4


def test_cross_chunk_face_records_round_trip(tmp_path):
    store, _, _ = _tetra_straddling_chunks(tmp_path)
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    records = read_cross_chunk_links(lvl, delta=0)
    # All 4 faces of the tetrahedron span chunks → all 4 appear here.
    assert len(records) == 4
    # Every face has 3 endpoints (triangle, link_width=3).
    for face in records:
        assert len(face) == 3
        for cc, local_idx in face:
            assert len(cc) == 3
            assert local_idx >= 0


def test_intra_chunk_mesh_writes_no_cross_chunk_links(tmp_path):
    """A mesh wholly inside one chunk should not write cross_chunk_links."""
    verts = np.array([
        [10, 10, 10], [20, 10, 10], [15, 20, 10], [15, 15, 20],
    ], dtype="f4")
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
    ], dtype=np.int64)
    store = tmp_path / "m.zv"
    write_mesh(str(store), verts, faces, chunk_shape=(50.0, 50.0, 50.0))
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    assert read_cross_chunk_links(lvl, delta=0) == []
