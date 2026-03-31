"""Integration tests: end-to-end pipelines for all geometry types.

Each test exercises the full pipeline:
  create test data → write to store → validate → read back → verify
  → (optional) build pyramid → validate pyramid → read coarser level

These tests use only in-memory data (no external file dependencies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ===================================================================
# Point cloud: write → validate → read → pyramid → export
# ===================================================================

class TestPointCloudPipeline:

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.coarsen import build_pyramid
        from zarr_vectors.export.csv_points import export_csv
        from zarr_vectors.core.store import list_resolution_levels, open_store

        rng = np.random.default_rng(42)
        store = str(tmp_path / "points.zarrvectors")

        # Write
        positions = rng.uniform(0, 1000, size=(5000, 3)).astype(np.float32)
        intensity = rng.uniform(0, 1, size=5000).astype(np.float32)
        summary = write_points(
            store, positions,
            chunk_shape=(200.0, 200.0, 200.0),
            attributes={"intensity": intensity},
        )
        assert summary["vertex_count"] == 5000
        assert summary["chunk_count"] > 1

        # Validate L4
        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        # Read all
        result = read_points(store)
        assert result["vertex_count"] == 5000

        # Read bbox subset
        result_bbox = read_points(
            store,
            bbox=(np.array([0, 0, 0]), np.array([200, 200, 200])),
        )
        assert 0 < result_bbox["vertex_count"] < 5000

        # Build pyramid
        pyr = build_pyramid(store, reduction_factor=8)
        assert pyr["levels_created"] >= 1

        # Validate pyramid L5
        vr5 = validate(store, level=5)
        assert vr5.ok, vr5.summary()

        # Read coarser level
        levels = list_resolution_levels(open_store(store))
        assert len(levels) >= 2
        coarse = read_points(store, level=1)
        assert 0 < coarse["vertex_count"] < 5000

        # Export
        csv_out = tmp_path / "exported.csv"
        export_csv(store, csv_out)
        assert csv_out.exists()
        lines = csv_out.read_text().strip().split("\n")
        assert len(lines) == 5001  # header + 5000 rows


# ===================================================================
# CSV ingest → read → export round-trip
# ===================================================================

class TestCSVRoundTrip:

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        from zarr_vectors.ingest.csv_points import ingest_csv
        from zarr_vectors.export.csv_points import export_csv
        from zarr_vectors.types.points import read_points

        rng = np.random.default_rng(99)
        positions = rng.uniform(0, 100, size=(200, 3))
        temperature = rng.uniform(20, 40, size=200)
        data = np.column_stack([positions, temperature])

        csv_in = tmp_path / "sensor.csv"
        np.savetxt(csv_in, data, delimiter=",",
                   header="x,y,z,temperature", comments="")

        store = str(tmp_path / "sensor.zarrvectors")
        ingest_csv(csv_in, store, (50.0, 50.0, 50.0),
                   position_columns=["x", "y", "z"],
                   attribute_columns=["temperature"])

        result = read_points(store)
        assert result["vertex_count"] == 200

        csv_out = tmp_path / "sensor_out.csv"
        export_csv(store, csv_out)
        exported = np.loadtxt(csv_out, delimiter=",", skiprows=1)
        assert exported.shape[0] == 200


# ===================================================================
# Streamlines: write → validate → read → simplify
# ===================================================================

class TestStreamlinePipeline:

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines, read_polylines
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.strategies.polylines import coarsen_polylines

        rng = np.random.default_rng(42)
        store = str(tmp_path / "tracts.zarrvectors")

        # Generate 50 streamlines
        polylines = []
        for _ in range(50):
            n_pts = rng.integers(30, 80)
            start = rng.uniform(0, 200, size=(1, 3)).astype(np.float32)
            steps = rng.normal(0, 2, size=(n_pts - 1, 3)).astype(np.float32)
            pts = np.concatenate([start, start + np.cumsum(steps, axis=0)])
            polylines.append(np.clip(pts, 0, 299).astype(np.float32))

        # Write with groups
        summary = write_polylines(
            store, polylines,
            chunk_shape=(100.0, 100.0, 100.0),
            groups={0: list(range(25)), 1: list(range(25, 50))},
        )
        assert summary["polyline_count"] == 50
        assert summary["group_count"] == 2

        # Validate
        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        # Read all
        result = read_polylines(store)
        assert result["polyline_count"] == 50

        # Read by group
        g0 = read_polylines(store, group_ids=[0])
        assert g0["polyline_count"] == 25

        # Read single streamline
        s1 = read_polylines(store, object_ids=[10])
        assert s1["polyline_count"] == 1

        # Coarsen: simplify + subsample
        full_polys = [np.concatenate(segs) for segs in result["polylines"]]
        coarsened = coarsen_polylines(
            full_polys,
            simplify_epsilon=3.0,
            subsample_bin_size=100.0,
            max_per_bin=2,
        )
        assert coarsened["polyline_count"] <= 50
        assert coarsened["vertex_count"] < sum(len(p) for p in full_polys)


# ===================================================================
# Skeleton: SWC write → validate → read → prune → export
# ===================================================================

class TestSkeletonPipeline:

    def test_swc_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.ingest.swc import ingest_swc
        from zarr_vectors.export.swc import export_swc
        from zarr_vectors.types.graphs import read_graph
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.strategies.graphs import prune_skeleton

        # Create a synthetic SWC
        swc_in = tmp_path / "neuron.swc"
        rng = np.random.default_rng(42)
        lines = ["# synthetic neuron"]
        n_nodes = 100
        positions = [[0.0, 0.0, 0.0]]
        parents = [-1]
        for i in range(1, n_nodes):
            p = max(0, i - rng.integers(1, min(i + 1, 4)))
            px, py, pz = positions[p]
            dx, dy, dz = rng.normal(0, 5, size=3)
            positions.append([px + dx, py + dy, pz + dz])
            parents.append(p)

        for i in range(n_nodes):
            comp = 1 if i == 0 else (2 if positions[i][0] < 0 else 3)
            r = 5.0 if i == 0 else rng.uniform(0.5, 3.0)
            x, y, z = positions[i]
            lines.append(f"{i + 1} {comp} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {parents[i] + 1 if parents[i] >= 0 else -1}")

        swc_in.write_text("\n".join(lines))

        # Ingest
        store = str(tmp_path / "neuron.zarrvectors")
        summary = ingest_swc(swc_in, store, (100.0, 100.0, 100.0))
        assert summary["node_count"] == n_nodes

        # Validate
        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        # Read
        result = read_graph(store)
        assert result["node_count"] == n_nodes

        # Prune short branches
        pruned = prune_skeleton(
            result["positions"], result["edges"],
            min_branch_length=8.0,
        )
        assert pruned["node_count"] <= n_nodes

        # Export
        swc_out = tmp_path / "neuron_out.swc"
        export_swc(store, swc_out)
        assert swc_out.exists()
        data_lines = [l for l in swc_out.read_text().strip().split("\n")
                      if not l.startswith("#")]
        assert len(data_lines) == n_nodes


# ===================================================================
# Mesh: OBJ write → validate → read → coarsen → export
# ===================================================================

class TestMeshPipeline:

    def test_obj_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.ingest.obj import ingest_obj
        from zarr_vectors.export.obj import export_obj
        from zarr_vectors.types.meshes import read_mesh
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.strategies.meshes import coarsen_mesh_cluster

        # Create a synthetic OBJ (10x10 grid)
        obj_in = tmp_path / "grid.obj"
        lines = ["# 10x10 grid"]
        for iy in range(10):
            for ix in range(10):
                lines.append(f"v {ix} {iy} 0")
        for iy in range(9):
            for ix in range(9):
                v0 = iy * 10 + ix + 1
                lines.append(f"f {v0} {v0 + 1} {v0 + 11}")
                lines.append(f"f {v0} {v0 + 11} {v0 + 10}")
        obj_in.write_text("\n".join(lines))

        # Ingest
        store = str(tmp_path / "grid.zarrvectors")
        summary = ingest_obj(obj_in, store, (5.0, 5.0, 5.0))
        assert summary["vertex_count"] == 100
        assert summary["face_count"] == 162

        # Validate
        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        # Read
        result = read_mesh(store)
        assert result["vertex_count"] == 100

        # Coarsen
        coarsened = coarsen_mesh_cluster(
            result["vertices"], result["faces"], 3.0,
        )
        assert coarsened["vertex_count"] < 100
        assert coarsened["face_count"] < 162

        # Export
        obj_out = tmp_path / "grid_out.obj"
        export_obj(store, obj_out)
        assert obj_out.exists()
        out_v = sum(1 for l in obj_out.read_text().split("\n") if l.startswith("v "))
        assert out_v == 100


# ===================================================================
# Lines: write → validate → read
# ===================================================================

class TestLinesPipeline:

    def test_lines_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.lines import write_lines, read_lines
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "lines.zarrvectors")

        # 200 random line segments, some cross-chunk
        endpoints = rng.uniform(0, 200, size=(200, 2, 3)).astype(np.float32)
        summary = write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["line_count"] == 200

        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        result = read_lines(store)
        assert result["line_count"] == 200

        # Bbox filter
        filtered = read_lines(
            store,
            bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
        )
        assert filtered["line_count"] < 200


# ===================================================================
# Parametric: write → read
# ===================================================================

class TestParametricPipeline:

    def test_parametric_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.parametric import (
            write_parametric_objects, read_parametric_objects,
        )

        store = str(tmp_path / "planes.zarrvectors")

        write_parametric_objects(store, [
            {"type": "plane", "coefficients": [0, 1, 0, -50], "name": "coronal"},
            {"type": "plane", "coefficients": [1, 0, 0, -25], "name": "sagittal"},
            {"type": "sphere", "coefficients": [0, 0, 0, 100], "name": "bounding"},
        ], create_new_store=True)

        objects = read_parametric_objects(store)
        assert len(objects) == 3
        assert objects[0]["type"] == "plane"
        assert objects[2]["type"] == "sphere"


# ===================================================================
# CLI: argument parsing and help
# ===================================================================

class TestCLI:

    def test_parser_builds(self) -> None:
        from zarr_vectors.cli.main import build_parser
        parser = build_parser()
        assert parser is not None

    def test_help_does_not_crash(self) -> None:
        from zarr_vectors.cli.main import build_parser
        parser = build_parser()
        try:
            parser.parse_args(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_validate_cli(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors.cli.main import main

        store = str(tmp_path / "cli_test.zarrvectors")
        write_points(store, np.array([[1, 2, 3]], dtype=np.float32),
                     chunk_shape=(100.0, 100.0, 100.0))

        try:
            main(["validate", store, "--level", "3"])
        except SystemExit as e:
            assert e.code == 0

    def test_info_cli(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors.cli.main import main

        store = str(tmp_path / "cli_info.zarrvectors")
        write_points(store, np.array([[1, 2, 3]], dtype=np.float32),
                     chunk_shape=(100.0, 100.0, 100.0))

        try:
            main(["info", store])
        except SystemExit:
            pass  # info doesn't call sys.exit normally

    def test_ingest_csv_cli(self, tmp_path: Path) -> None:
        from zarr_vectors.cli.main import main

        csv_path = tmp_path / "data.csv"
        np.savetxt(csv_path, np.array([[1, 2, 3], [4, 5, 6]]),
                   delimiter=",", header="x,y,z", comments="")
        store = str(tmp_path / "cli_csv.zarrvectors")

        main(["ingest", "points", str(csv_path), store,
              "--chunk-shape", "100,100,100"])

        from zarr_vectors.types.points import read_points
        assert read_points(store)["vertex_count"] == 2

    def test_export_csv_cli(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors.cli.main import main

        store = str(tmp_path / "cli_exp.zarrvectors")
        write_points(store, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                     chunk_shape=(100.0, 100.0, 100.0))

        csv_out = str(tmp_path / "out.csv")
        main(["export", "csv", store, csv_out])
        assert Path(csv_out).exists()

    def test_build_pyramid_cli(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors.cli.main import main
        from zarr_vectors.core.store import list_resolution_levels, open_store

        rng = np.random.default_rng(42)
        store = str(tmp_path / "cli_pyr.zarrvectors")
        write_points(store, rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32),
                     chunk_shape=(100.0, 100.0, 100.0))

        main(["build-pyramid", store])
        levels = list_resolution_levels(open_store(store))
        assert len(levels) >= 2
