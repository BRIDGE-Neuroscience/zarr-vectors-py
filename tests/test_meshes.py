"""Step 10 tests: mesh write/read core API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.meshes import write_mesh, read_mesh
from zarr_vectors.exceptions import ArrayError


class TestMeshBasic:

    def test_tetrahedron(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[10,0,0],[5,10,0],[5,5,10]], dtype=np.float32)
        f = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["vertex_count"] == 4 and s["face_count"] == 4
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 4 and r["face_count"] == 4

    def test_large_single_chunk(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        v = rng.uniform(0, 80, size=(200, 3)).astype(np.float32)
        f = rng.integers(0, 200, size=(400, 3)).astype(np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 200 and r["face_count"] == 400

    def test_quad_mesh(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[10,0,0],[10,10,0],[0,10,0],
                       [0,0,10],[10,0,10],[10,10,10],[0,10,10]], dtype=np.float32)
        f = np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["face_count"] == 6 and r["faces"].shape[1] == 4


class TestMeshCrossChunk:

    def test_cross_face(self, tmp_path: Path) -> None:
        v = np.array([[10,50,50],[50,50,50],[110,50,50]], dtype=np.float32)
        f = np.array([[0,1,2]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["cross_face_count"] == 1

    def test_cluster_faces_intra(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        c1 = rng.uniform(0, 50, size=(50, 3)).astype(np.float32)
        c2 = rng.uniform(100, 150, size=(50, 3)).astype(np.float32)
        v = np.vstack([c1, c2])
        f = np.vstack([rng.integers(0,50,size=(30,3)), rng.integers(50,100,size=(30,3))]).astype(np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["intra_face_count"] == 60


class TestMeshBbox:

    def test_bbox_filter(self, tmp_path: Path) -> None:
        v = np.array([[10,10,10],[20,20,20],[30,30,30],
                       [150,150,150],[160,160,160],[170,170,170]], dtype=np.float32)
        f = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
        write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(200.,200.,200.))
        r = read_mesh(str(tmp_path / "m.zv"), bbox=(np.array([0,0,0]), np.array([50,50,50])))
        assert r["vertex_count"] == 3 and r["face_count"] == 1


class TestMeshEdgeCases:

    def test_bad_face_shape(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        try:
            write_mesh(str(tmp_path / "m.zv"), v, np.zeros((3,2), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
            assert False
        except ArrayError:
            pass

    def test_single_triangle(self, tmp_path: Path) -> None:
        v = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
        f = np.array([[0,1,2]], dtype=np.int64)
        write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 3 and r["face_count"] == 1
