"""Step 09 tests: graph/skeleton write, read, SWC ingest/export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import write_graph, read_graph
from zarr_vectors.ingest.swc import ingest_swc
from zarr_vectors.export.swc import export_swc
from zarr_vectors.exceptions import ArrayError, IngestError


class TestGeneralGraph:

    def test_single_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[20,20,20],[30,30,30],[40,40,40]], dtype=np.float32)
        edges = np.array([[0,1],[1,2],[2,3],[0,2]], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.))
        assert s["node_count"] == 4 and s["edge_count"] == 4
        r = read_graph(str(tmp_path / "g.zv"))
        assert r["node_count"] == 4 and r["edge_count"] > 0

    def test_cross_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,50,50],[20,50,50],[110,50,50],[120,50,50]], dtype=np.float32)
        edges = np.array([[0,1],[2,3],[0,2],[1,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.))
        assert s["cross_edge_count"] >= 2

    def test_empty_edges(self, tmp_path: Path) -> None:
        pos = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        s = write_graph(str(tmp_path / "g.zv"), pos, np.zeros((0,2), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
        assert s["edge_count"] == 0

    def test_bad_edge_shape(self, tmp_path: Path) -> None:
        pos = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        try:
            write_graph(str(tmp_path / "g.zv"), pos, np.zeros((3,3), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
            assert False
        except ArrayError:
            pass


class TestSkeleton:

    def test_7node_tree(self, tmp_path: Path) -> None:
        pos = np.array([[50,50,50],[40,40,40],[60,60,60],[30,30,30],
                         [45,35,35],[65,65,65],[25,25,25]], dtype=np.float32)
        edges = np.array([[1,0],[2,0],[3,1],[4,1],[5,2],[6,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(200.,200.,200.), is_tree=True)
        r = read_graph(str(tmp_path / "s.zv"))
        assert r["node_count"] == 7 and r["edge_count"] == 6

    def test_with_attributes(self, tmp_path: Path) -> None:
        pos = np.array([[50,50,50],[40,40,40],[60,60,60]], dtype=np.float32)
        edges = np.array([[1,0],[2,0]], dtype=np.int64)
        radius = np.array([5.0,3.0,3.0], dtype=np.float32)
        write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(200.,200.,200.),
                    is_tree=True, node_attributes={"radius": radius})

    def test_cross_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,50,50],[20,50,50],[30,50,50],[110,50,50],[120,50,50]], dtype=np.float32)
        edges = np.array([[1,0],[2,1],[3,2],[4,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(100.,100.,100.), is_tree=True)
        assert s["cross_edge_count"] >= 1
        r = read_graph(str(tmp_path / "s.zv"))
        assert r["node_count"] == 5

    def test_multiple_objects(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[20,20,20],[30,30,30],[50,50,50],[60,60,60]], dtype=np.float32)
        edges = np.array([[1,0],[2,1],[4,3]], dtype=np.int64)
        oids = np.array([0,0,0,1,1], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.),
                        is_tree=True, object_ids=oids)
        assert s["object_count"] == 2


class TestSWC:

    def test_ingest(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text("# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n4 3 15 10 0 2 2\n5 2 -10 0 0 2.5 1\n")
        s = ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        assert s["node_count"] == 5 and s["is_tree"]

    def test_export(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text("# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n")
        ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        out = tmp_path / "out.swc"
        export_swc(tmp_path / "n.zv", out)
        lines = [l for l in out.read_text().strip().split("\n") if not l.startswith("#")]
        assert len(lines) == 3


class TestBboxFilter:

    def test_bbox(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[50,50,50],[150,150,150]], dtype=np.float32)
        edges = np.array([[0,1],[1,2]], dtype=np.int64)
        write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(200.,200.,200.))
        r = read_graph(str(tmp_path / "g.zv"), bbox=(np.array([0,0,0]), np.array([100,100,100])))
        assert r["node_count"] == 2
