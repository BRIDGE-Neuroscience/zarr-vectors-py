"""Tests for the cross_chunk_faces format extension (Tier C)."""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import read_cross_chunk_faces
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_root_metadata,
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
    store = tmp_path / "m.zvr"
    write_mesh(
        str(store), verts, faces,
        chunk_shape=(50.0, 50.0, 50.0),
    )
    return store, verts, faces


def test_capability_stamped_when_cross_faces_present(tmp_path):
    store, _, _ = _tetra_straddling_chunks(tmp_path)
    rm = read_root_metadata(open_store(str(store)))
    assert "cross_chunk_faces" in rm.format_capabilities


def test_read_mesh_returns_cross_chunk_faces(tmp_path):
    store, _, faces_in = _tetra_straddling_chunks(tmp_path)
    out = read_mesh(str(store))
    assert out["face_count"] == len(faces_in)
    assert out["vertex_count"] == 4


def test_cross_chunk_faces_array_round_trips(tmp_path):
    store, _, _ = _tetra_straddling_chunks(tmp_path)
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    records = read_cross_chunk_faces(lvl)
    # All 4 faces of the tetrahedron span chunks → all 4 appear here.
    assert len(records) == 4
    # Every face has 3 vertex records (triangle)
    for face in records:
        assert len(face) == 3
        # Each record is (chunk_coords, local_idx)
        for cc, local_idx in face:
            assert len(cc) == 3
            # Local index is non-negative; exact value depends on
            # vertex-to-chunk assignment which we don't pin here.
            assert local_idx >= 0


def test_legacy_no_cross_chunk_array_when_all_intra(tmp_path):
    """A mesh wholly inside one chunk should not stamp the capability."""
    verts = np.array([
        [10, 10, 10], [20, 10, 10], [15, 20, 10], [15, 15, 20],
    ], dtype="f4")
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
    ], dtype=np.int64)
    store = tmp_path / "m.zvr"
    write_mesh(str(store), verts, faces, chunk_shape=(50.0, 50.0, 50.0))
    rm = read_root_metadata(open_store(str(store)))
    assert "cross_chunk_faces" not in rm.format_capabilities

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    assert read_cross_chunk_faces(lvl) == []
