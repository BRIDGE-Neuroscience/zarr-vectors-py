"""Step 10 tests: mesh write, read, OBJ/STL ingest, OBJ export."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from zarr_vectors.types.meshes import write_mesh, read_mesh
from zarr_vectors.ingest.obj import ingest_obj
from zarr_vectors.ingest.stl import ingest_stl
from zarr_vectors.export.obj import export_obj
from zarr_vectors.exceptions import ArrayError, IngestError


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


class TestOBJIngest:

    def test_triangle_obj(self, tmp_path: Path) -> None:
        obj = tmp_path / "t.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 5 10 0\nv 5 5 10\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 1 3 4\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["vertex_count"] == 4 and s["face_count"] == 4

    def test_quad_obj(self, tmp_path: Path) -> None:
        obj = tmp_path / "q.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nf 1 2 3 4\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 1

    def test_polygon_fan(self, tmp_path: Path) -> None:
        obj = tmp_path / "p.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 10 10 0\nv 5 15 0\nv 0 10 0\nf 1 2 3 4 5\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 3

    def test_not_found(self, tmp_path: Path) -> None:
        try:
            ingest_obj(tmp_path / "x.obj", tmp_path / "x.zv", (100.,100.,100.))
            assert False
        except IngestError:
            pass


class TestSTLIngest:

    def test_ascii(self, tmp_path: Path) -> None:
        stl = tmp_path / "a.stl"
        stl.write_text(
            "solid t\n"
            "  facet normal 0 0 1\n    outer loop\n"
            "      vertex 0 0 0\n      vertex 10 0 0\n      vertex 5 10 0\n"
            "    endloop\n  endfacet\n"
            "  facet normal 0 0 -1\n    outer loop\n"
            "      vertex 0 0 0\n      vertex 10 0 0\n      vertex 5 5 10\n"
            "    endloop\n  endfacet\n"
            "endsolid t\n"
        )
        s = ingest_stl(stl, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 2

    def test_binary(self, tmp_path: Path) -> None:
        stl = tmp_path / "b.stl"
        with open(stl, "wb") as f:
            f.write(b"\x00" * 80)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<3f", 0, 0, 1))
            for v in [(0,0,0),(10,0,0),(5,10,0)]:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
        s = ingest_stl(stl, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 1

    def test_not_found(self, tmp_path: Path) -> None:
        try:
            ingest_stl(tmp_path / "x.stl", tmp_path / "x.zv", (100.,100.,100.))
            assert False
        except IngestError:
            pass


class TestOBJExport:

    def test_export(self, tmp_path: Path) -> None:
        obj = tmp_path / "t.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 5 10 0\nv 5 5 10\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 1 3 4\n")
        ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        out = tmp_path / "out.obj"
        export_obj(tmp_path / "m.zv", out)
        lines = out.read_text().strip().split("\n")
        assert len([l for l in lines if l.startswith("v ")]) == 4
        assert len([l for l in lines if l.startswith("f ")]) == 4

    def test_round_trip(self, tmp_path: Path) -> None:
        obj = tmp_path / "rt.obj"
        obj.write_text("v 1.5 2.5 3.5\nv 4.5 5.5 6.5\nv 7.5 8.5 9.5\nf 1 2 3\n")
        ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        out = tmp_path / "out.obj"
        export_obj(tmp_path / "m.zv", out)
        rv = []
        for line in out.read_text().split("\n"):
            if line.startswith("v "):
                parts = line.split()
                rv.append([float(parts[1]), float(parts[2]), float(parts[3])])
        expected = [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]
        np.testing.assert_allclose(np.sort(rv, axis=0), np.sort(expected, axis=0), atol=1e-3)


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
