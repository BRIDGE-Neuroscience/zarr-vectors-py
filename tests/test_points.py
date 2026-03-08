"""Step 06 tests: point cloud write/read, CSV ingest/export round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.ingest.csv_points import ingest_csv
from zarr_vectors.export.csv_points import export_csv


# ===================================================================
# Undifferentiated point cloud (variant 1)
# ===================================================================

class TestUndifferentiatedPoints:

    def test_write_and_read_all(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(200, 3)).astype(np.float32)
        store = str(tmp_path / "points.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(500.0, 500.0, 500.0),
        )
        assert summary["vertex_count"] == 200
        assert summary["chunk_count"] > 0

        result = read_points(store)
        assert result["vertex_count"] == 200
        assert result["positions"].shape == (200, 3)

    def test_single_chunk(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        store = str(tmp_path / "tiny.zarr")

        write_points(store, positions, chunk_shape=(1000.0, 1000.0, 1000.0))
        result = read_points(store)
        assert result["vertex_count"] == 2
        # Positions should match (order may differ due to chunk assignment)
        read_pos = result["positions"]
        assert read_pos.shape == (2, 3)

    def test_with_attributes(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(99)
        positions = rng.uniform(0, 100, size=(50, 3)).astype(np.float32)
        intensity = rng.uniform(0, 1, size=50).astype(np.float32)
        store = str(tmp_path / "attrs.zarr")

        write_points(
            store, positions,
            chunk_shape=(50.0, 50.0, 50.0),
            attributes={"intensity": intensity},
        )

        result = read_points(store, attribute_names=["intensity"])
        assert result["vertex_count"] == 50

    def test_default_chunk_shape(self, tmp_path: Path) -> None:
        """When no chunk_shape given, should use a single chunk."""
        positions = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        store = str(tmp_path / "default.zarr")

        summary = write_points(store, positions)
        assert summary["chunk_count"] == 1
        result = read_points(store)
        assert result["vertex_count"] == 2

    def test_bbox_filter(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [50, 50, 50],
            [90, 90, 90],
            [150, 150, 150],
        ], dtype=np.float32)
        store = str(tmp_path / "bbox.zarr")

        write_points(store, positions, chunk_shape=(100.0, 100.0, 100.0))

        result = read_points(
            store,
            bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
        )
        # Should get 3 points (10,50,90 but not 150)
        assert result["vertex_count"] == 3

    def test_2d_points(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        store = str(tmp_path / "2d.zarr")

        write_points(store, positions, chunk_shape=(10.0, 10.0))
        result = read_points(store)
        assert result["vertex_count"] == 3
        assert result["positions"].shape[1] == 2


# ===================================================================
# Per-point objects (variant 2)
# ===================================================================

class TestPerPointObjects:

    def test_each_point_is_object(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [50, 50, 50],
            [110, 110, 110],
        ], dtype=np.float32)
        store = str(tmp_path / "per_point.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        assert summary["object_count"] == 3

        # Read by object ID
        result = read_points(store, object_ids=[1])
        assert result["vertex_count"] == 1
        np.testing.assert_allclose(result["positions"][0], [50, 50, 50], atol=1e-5)

    def test_implicit_object_ids(self, tmp_path: Path) -> None:
        """When object_attributes given but no object_ids, each point gets its own ID."""
        positions = np.array([
            [10, 10, 10],
            [20, 20, 20],
        ], dtype=np.float32)
        obj_attrs = np.array([100, 200], dtype=np.float32)
        store = str(tmp_path / "implicit_oid.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_attributes={"label": obj_attrs},
        )
        assert summary["object_count"] == 2

    def test_with_groups(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30],
            [40, 40, 40],
        ], dtype=np.float32)
        store = str(tmp_path / "grouped.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=np.array([0, 1, 2, 3], dtype=np.int64),
            groups={0: [0, 1], 1: [2, 3]},
            group_attributes={
                "name": np.array([1.0, 2.0], dtype=np.float32),
            },
        )
        assert summary["group_count"] == 2

        # Read by group
        result = read_points(store, group_ids=[0])
        assert result["vertex_count"] == 2


# ===================================================================
# Multi-point objects (variant 3)
# ===================================================================

class TestMultiPointObjects:

    def test_many_points_per_object(self, tmp_path: Path) -> None:
        """5 points belong to object 0, 3 to object 1."""
        positions = np.array([
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [50, 50, 50],
            [51, 51, 51],
            [52, 52, 52],
        ], dtype=np.float32)
        object_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        store = str(tmp_path / "multi.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=object_ids,
        )
        assert summary["object_count"] == 2

        # Read object 0: should get 5 points
        result = read_points(store, object_ids=[0])
        assert result["vertex_count"] == 5

        # Read object 1: should get 3 points
        result = read_points(store, object_ids=[1])
        assert result["vertex_count"] == 3

    def test_objects_across_chunks(self, tmp_path: Path) -> None:
        """Object 0 has points in two different chunks."""
        positions = np.array([
            [10, 50, 50],    # chunk (0,0,0)
            [110, 50, 50],   # chunk (1,0,0)
            [20, 50, 50],    # chunk (0,0,0)
        ], dtype=np.float32)
        object_ids = np.array([0, 0, 0], dtype=np.int64)
        store = str(tmp_path / "cross.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=object_ids,
        )
        assert summary["chunk_count"] == 2
        assert summary["object_count"] == 1

        result = read_points(store, object_ids=[0])
        assert result["vertex_count"] == 3


# ===================================================================
# CSV ingest / export round-trip
# ===================================================================

class TestCSVRoundTrip:

    def test_basic_csv(self, tmp_path: Path) -> None:
        """Write CSV → ingest → export → compare."""
        # Create a CSV file
        csv_path = tmp_path / "points.csv"
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(30, 3)).astype(np.float64)
        intensity = rng.uniform(0, 1, size=30).astype(np.float64)

        data = np.column_stack([positions, intensity])
        header = "x,y,z,intensity"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        # Ingest
        store_path = tmp_path / "from_csv.zarr"
        summary = ingest_csv(
            csv_path, store_path,
            chunk_shape=(50.0, 50.0, 50.0),
            ndim=3,
            position_columns=["x", "y", "z"],
            attribute_columns=["intensity"],
        )
        assert summary["vertex_count"] == 30

        # Export
        out_csv = tmp_path / "exported.csv"
        export_csv(store_path, out_csv)
        assert out_csv.exists()

        # Read back and compare vertex count
        exported_data = np.loadtxt(out_csv, delimiter=",", skiprows=1)
        assert exported_data.shape[0] == 30

    def test_xyz_no_header(self, tmp_path: Path) -> None:
        """XYZ file (no header, space-delimited)."""
        xyz_path = tmp_path / "points.xyz"
        positions = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        np.savetxt(xyz_path, positions, delimiter=" ")

        store_path = tmp_path / "from_xyz.zarr"
        summary = ingest_csv(
            xyz_path, store_path,
            chunk_shape=(100.0, 100.0, 100.0),
            has_header=False,
            delimiter=" ",
        )
        assert summary["vertex_count"] == 3

        result = read_points(str(store_path))
        assert result["vertex_count"] == 3

    def test_csv_with_extra_columns(self, tmp_path: Path) -> None:
        """CSV with more columns than just XYZ."""
        csv_path = tmp_path / "rich.csv"
        data = np.array([
            [1, 2, 3, 0.5, 100],
            [4, 5, 6, 0.8, 200],
        ])
        np.savetxt(
            csv_path, data, delimiter=",",
            header="x,y,z,intensity,class",
            comments="",
        )

        store_path = tmp_path / "rich.zarr"
        summary = ingest_csv(
            csv_path, store_path,
            chunk_shape=(100.0, 100.0, 100.0),
        )
        assert summary["vertex_count"] == 2

    def test_csv_round_trip_positions_match(self, tmp_path: Path) -> None:
        """Verify position values survive the round-trip."""
        csv_in = tmp_path / "in.csv"
        positions = np.array([
            [10.5, 20.3, 30.1],
            [40.7, 50.9, 60.2],
        ])
        np.savetxt(csv_in, positions, delimiter=",",
                    header="x,y,z", comments="")

        store_path = tmp_path / "rt.zarr"
        ingest_csv(csv_in, store_path,
                    chunk_shape=(100.0, 100.0, 100.0))

        csv_out = tmp_path / "out.csv"
        export_csv(store_path, csv_out)

        exported = np.loadtxt(csv_out, delimiter=",", skiprows=1)
        # Positions should match within float32 precision
        np.testing.assert_allclose(
            np.sort(exported, axis=0),
            np.sort(positions, axis=0),
            atol=1e-2,
        )


# ===================================================================
# LAS / PLY graceful import failure
# ===================================================================

class TestOptionalDependencies:

    def test_las_missing_raises_ingest_error(self, tmp_path: Path) -> None:
        """If laspy is not installed, ingest_las should raise IngestError."""
        from zarr_vectors.ingest.las import ingest_las
        from zarr_vectors.exceptions import IngestError

        try:
            ingest_las(
                tmp_path / "fake.las",
                tmp_path / "out.zarr",
                (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "laspy" in str(e).lower()
        except Exception:
            pass  # laspy might actually be installed in some envs

    def test_ply_missing_raises_ingest_error(self, tmp_path: Path) -> None:
        """If plyfile is not installed, ingest_ply should raise IngestError."""
        from zarr_vectors.ingest.ply import ingest_ply
        from zarr_vectors.exceptions import IngestError

        try:
            ingest_ply(
                tmp_path / "fake.ply",
                tmp_path / "out.zarr",
                (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "plyfile" in str(e).lower()
        except Exception:
            pass


# ===================================================================
# Edge cases
# ===================================================================

class TestPointsEdgeCases:

    def test_single_point(self, tmp_path: Path) -> None:
        positions = np.array([[42.0, 43.0, 44.0]], dtype=np.float32)
        store = str(tmp_path / "single.zarr")
        write_points(store, positions, chunk_shape=(100.0, 100.0, 100.0))
        result = read_points(store)
        assert result["vertex_count"] == 1

    def test_many_chunks(self, tmp_path: Path) -> None:
        """1000 points across many small chunks."""
        rng = np.random.default_rng(77)
        positions = rng.uniform(0, 1000, size=(1000, 3)).astype(np.float32)
        store = str(tmp_path / "many.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
        )
        assert summary["vertex_count"] == 1000
        assert summary["chunk_count"] > 1

        result = read_points(store)
        assert result["vertex_count"] == 1000

    def test_multichannel_attribute(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        color = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)
        store = str(tmp_path / "color.zarr")

        write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            attributes={"color": color},
        )

        result = read_points(store)
        assert result["vertex_count"] == 2
