"""Step 04 tests: core array creation, write, and read round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    count_vertex_groups,
    create_attribute_array,
    create_cross_chunk_links_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_link_attributes_array,
    create_links_array,
    create_metanode_children_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_group_object_ids,
    read_groupings_attributes,
    read_metanode_children,
    read_object_attributes,
    read_object_manifest,
    read_object_vertices,
    read_vertex_group,
    write_chunk_attributes,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_groupings,
    write_groupings_attributes,
    write_metanode_children,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.exceptions import ArrayError


def _make_level_group(tmp_path: Path, name: str = "resolution_0") -> FsGroup:
    root = FsGroup(tmp_path / "store.zarr", create=True)
    return root.create_group(name)


# ===================================================================
# Vertex write/read round-trips
# ===================================================================

class TestVertexArrays:

    def test_single_chunk_single_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [pts])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], pts)

    def test_single_chunk_multiple_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        g1 = np.array([[10, 10, 10]], dtype=np.float32)
        g2 = np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1, g2])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 3
        np.testing.assert_array_equal(groups[0], g0)
        np.testing.assert_array_equal(groups[1], g1)
        np.testing.assert_array_equal(groups[2], g2)

    def test_multiple_chunks(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        pts_a = [np.array([[1, 2, 3]], dtype=np.float32)]
        pts_b = [np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), pts_a)
        write_chunk_vertices(lg, (1, 0, 0), pts_b)

        keys = list_chunk_keys(lg)
        assert keys == [(0, 0, 0), (1, 0, 0)]

        g_a = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        g_b = read_chunk_vertices(lg, (1, 0, 0), dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(g_a[0], pts_a[0])
        np.testing.assert_array_equal(g_b[0], pts_b[0])

    def test_read_single_vertex_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        g1 = np.array([[10, 10, 10], [11, 11, 11], [12, 12, 12]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1])

        vg0 = read_vertex_group(lg, (0, 0, 0), 0, dtype=np.float32, ndim=3)
        vg1 = read_vertex_group(lg, (0, 0, 0), 1, dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(vg0, g0)
        np.testing.assert_array_equal(vg1, g1)

    def test_read_vertex_group_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)])
        try:
            read_vertex_group(lg, (0, 0, 0), 5, dtype=np.float32, ndim=3)
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_empty_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[1, 2, 3]], dtype=np.float32)
        g_empty = np.zeros((0, 3), dtype=np.float32)
        g2 = np.array([[7, 8, 9]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g_empty, g2])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 3
        assert groups[1].shape == (0, 3)

    def test_count_vertex_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(lg, (0, 0, 0), [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        ])
        assert count_vertex_groups(lg, (0, 0, 0)) == 3

    def test_2d_points(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg, dtype="float64")

        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        write_chunk_vertices(lg, (0, 0), [pts], dtype=np.float64)

        groups = read_chunk_vertices(lg, (0, 0), dtype=np.float64, ndim=2)
        np.testing.assert_array_equal(groups[0], pts)


# ===================================================================
# Links write/read
# ===================================================================

class TestLinkArrays:

    def test_triangle_faces(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=3)

        verts = [np.zeros((4, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        faces = [np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), faces)

        read_faces = read_chunk_links(lg, (0, 0, 0), link_width=3)
        assert len(read_faces) == 1
        np.testing.assert_array_equal(read_faces[0], faces[0])

    def test_edge_list(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        edges = [np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), edges)

        read_edges = read_chunk_links(lg, (0, 0, 0), link_width=2)
        np.testing.assert_array_equal(read_edges[0], edges[0])

    def test_multiple_link_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        v0 = np.zeros((3, 3), dtype=np.float32)
        v1 = np.zeros((2, 3), dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [v0, v1])

        e0 = np.array([[0, 1], [1, 2]], dtype=np.int64)
        e1 = np.array([[0, 1]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [e0, e1])

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2)
        assert len(groups) == 2
        np.testing.assert_array_equal(groups[0], e0)
        np.testing.assert_array_equal(groups[1], e1)


# ===================================================================
# Attribute write/read
# ===================================================================

class TestAttributeArrays:

    def test_scalar_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "intensity")

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        intensity = [np.array([0.5, 0.8, 0.3], dtype=np.float32)]
        write_chunk_attributes(lg, "intensity", (0, 0, 0), intensity)

        read_back = read_chunk_attributes(lg, "intensity", (0, 0, 0),
                                          dtype=np.float32, ncols=1)
        assert len(read_back) == 1
        np.testing.assert_allclose(read_back[0], intensity[0])

    def test_multichannel_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "color", channel_names=["r", "g", "b"])

        verts = [np.zeros((2, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        color = [np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)]
        write_chunk_attributes(lg, "color", (0, 0, 0), color, dtype=np.float32)

        read_back = read_chunk_attributes(lg, "color", (0, 0, 0),
                                          dtype=np.float32, ncols=3)
        np.testing.assert_array_equal(read_back[0], color[0])

    def test_multiple_attribute_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "radius")

        v0 = np.zeros((2, 3), dtype=np.float32)
        v1 = np.zeros((3, 3), dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [v0, v1])

        r0 = np.array([1.0, 2.0], dtype=np.float32)
        r1 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        write_chunk_attributes(lg, "radius", (0, 0, 0), [r0, r1])

        read_back = read_chunk_attributes(lg, "radius", (0, 0, 0),
                                          dtype=np.float32, ncols=1)
        assert len(read_back) == 2
        np.testing.assert_array_equal(read_back[0], r0)
        np.testing.assert_array_equal(read_back[1], r1)


# ===================================================================
# Object index write/read
# ===================================================================

class TestObjectIndex:

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)

        manifests = {
            0: [((0, 0, 0), 0), ((0, 0, 1), 2)],
            1: [((1, 1, 1), 0)],
            2: [((0, 0, 0), 1), ((0, 1, 0), 0)],
        }
        write_object_index(lg, manifests, sid_ndim=3)

        m0 = read_object_manifest(lg, 0)
        m1 = read_object_manifest(lg, 1)
        m2 = read_object_manifest(lg, 2)
        assert m0 == manifests[0]
        assert m1 == manifests[1]
        assert m2 == manifests[2]

    def test_read_all_manifests(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)

        manifests = {
            0: [((0, 0, 0), 0)],
            1: [((1, 0, 0), 0), ((1, 0, 1), 0)],
        }
        write_object_index(lg, manifests, sid_ndim=3)

        all_m = read_all_object_manifests(lg)
        assert len(all_m) == 2
        assert all_m[0] == manifests[0]
        assert all_m[1] == manifests[1]

    def test_object_id_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)
        write_object_index(lg, {0: [((0, 0, 0), 0)]}, sid_ndim=3)
        try:
            read_object_manifest(lg, 99)
            assert False
        except ArrayError:
            pass

    def test_read_object_vertices(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)

        g0 = np.array([[1, 2, 3]], dtype=np.float32)
        g1 = np.array([[4, 5, 6]], dtype=np.float32)
        g2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1])
        write_chunk_vertices(lg, (1, 0, 0), [g2])

        manifests = {
            0: [((0, 0, 0), 0), ((1, 0, 0), 0)],  # object spans 2 chunks
            1: [((0, 0, 0), 1)],                     # single chunk
        }
        write_object_index(lg, manifests, sid_ndim=3)

        verts_0 = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        assert len(verts_0) == 2
        np.testing.assert_array_equal(verts_0[0], g0)
        np.testing.assert_array_equal(verts_0[1], g2)

        verts_1 = read_object_vertices(lg, 1, dtype=np.float32, ndim=3)
        assert len(verts_1) == 1
        np.testing.assert_array_equal(verts_1[0], g1)


# ===================================================================
# Object attributes write/read
# ===================================================================

class TestObjectAttributes:

    def test_1d_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        write_object_attributes(lg, "score", data)
        read_back = read_object_attributes(lg, "score")
        np.testing.assert_array_equal(read_back, data)

    def test_2d_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        write_object_attributes(lg, "termination", data)
        read_back = read_object_attributes(lg, "termination")
        np.testing.assert_array_equal(read_back, data)
        assert read_back.shape == (3, 2)


# ===================================================================
# Groupings write/read
# ===================================================================

class TestGroupings:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)

        groups = {
            0: [0, 1, 5],
            1: [2, 3],
            2: [4],
        }
        write_groupings(lg, groups)

        g0 = read_group_object_ids(lg, 0)
        g1 = read_group_object_ids(lg, 1)
        g2 = read_group_object_ids(lg, 2)
        assert g0 == [0, 1, 5]
        assert g1 == [2, 3]
        assert g2 == [4]

    def test_read_all(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)

        groups = {0: [10, 20], 1: [30]}
        write_groupings(lg, groups)

        all_g = read_all_groupings(lg)
        assert all_g == [[10, 20], [30]]

    def test_group_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)
        write_groupings(lg, {0: [0]})
        try:
            read_group_object_ids(lg, 99)
            assert False
        except ArrayError:
            pass


# ===================================================================
# Groupings attributes
# ===================================================================

class TestGroupingsAttributes:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        write_groupings_attributes(lg, "mean_fa", data)
        read_back = read_groupings_attributes(lg, "mean_fa")
        np.testing.assert_array_equal(read_back, data)


# ===================================================================
# Cross-chunk links
# ===================================================================

class TestCrossChunkLinks:

    def test_basic_3d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg)

        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=3)

        read_back = read_cross_chunk_links(lg)
        assert len(read_back) == 2
        assert read_back[0] == links[0]
        assert read_back[1] == links[1]

    def test_2d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg)

        links = [
            (((0, 1), 3), ((1, 1), 0)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=2)

        read_back = read_cross_chunk_links(lg)
        assert read_back[0] == links[0]


# ===================================================================
# Link attributes
# ===================================================================

class TestLinkAttributes:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        create_link_attributes_array(lg, "weight")

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        edges = [np.array([[0, 1], [1, 2]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), edges)

        weights = [np.array([0.5, 0.8], dtype=np.float32)]
        write_chunk_link_attributes(lg, "weight", (0, 0, 0), weights)

        # Read back via raw bytes (link_attributes use same encoding as vertex groups)
        key = "0.0.0"
        raw = lg.read_bytes("link_attributes/weight", key)
        arr = np.frombuffer(raw, dtype=np.float32)
        np.testing.assert_allclose(arr, [0.5, 0.8])


# ===================================================================
# Metanode children
# ===================================================================

class TestMetanodeChildren:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_metanode_children_array(lg)

        children = {
            0: [((0, 0, 0), 0), ((0, 0, 0), 1), ((0, 0, 0), 2)],
            1: [((0, 0, 1), 0), ((0, 0, 1), 1)],
            2: [((1, 0, 0), 0)],
        }
        write_metanode_children(lg, children, sid_ndim=3)

        c0 = read_metanode_children(lg, metanode_id=0)
        c1 = read_metanode_children(lg, metanode_id=1)
        assert c0 == children[0]
        assert c1 == children[1]

        all_c = read_metanode_children(lg)
        assert isinstance(all_c, dict)
        assert all_c[2] == children[2]

    def test_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_metanode_children_array(lg)
        write_metanode_children(lg, {0: [((0, 0, 0), 0)]}, sid_ndim=3)
        try:
            read_metanode_children(lg, metanode_id=99)
            assert False
        except ArrayError:
            pass


# ===================================================================
# Integration: full object reconstruction
# ===================================================================

class TestObjectReconstruction:
    """Simulate a streamline spanning 2 chunks and verify full read path."""

    def test_streamline_across_chunks(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)
        create_cross_chunk_links_array(lg)

        # Segment A in chunk (0,0,0): 3 points
        seg_a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        # Segment B in chunk (0,0,1): 2 points
        seg_b = np.array([[3, 3, 3], [4, 4, 4]], dtype=np.float32)

        write_chunk_vertices(lg, (0, 0, 0), [seg_a])
        write_chunk_vertices(lg, (0, 0, 1), [seg_b])

        # Object 0 = this streamline, spanning both chunks
        write_object_index(lg, {
            0: [((0, 0, 0), 0), ((0, 0, 1), 0)],
        }, sid_ndim=3)

        # Cross-chunk link: last vertex of seg_a → first vertex of seg_b
        write_cross_chunk_links(lg, [
            (((0, 0, 0), 2), ((0, 0, 1), 0)),
        ], sid_ndim=3)

        # Read back full object
        verts = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        assert len(verts) == 2
        np.testing.assert_array_equal(verts[0], seg_a)
        np.testing.assert_array_equal(verts[1], seg_b)

        # Concatenate to get full streamline
        full = np.concatenate(verts, axis=0)
        assert full.shape == (5, 3)
        np.testing.assert_array_equal(full[0], [0, 0, 0])
        np.testing.assert_array_equal(full[4], [4, 4, 4])

        # Verify cross-chunk link
        ccl = read_cross_chunk_links(lg)
        assert len(ccl) == 1
        assert ccl[0] == (((0, 0, 0), 2), ((0, 0, 1), 0))

    def test_graph_multiple_objects_shared_chunks(self, tmp_path: Path) -> None:
        """Two objects (neurons) sharing the same spatial chunks."""
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)

        # Chunk (0,0,0): vertex group 0 = neuron A, group 1 = neuron B
        vg_a0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        vg_b0 = np.array([[5, 5, 5]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [vg_a0, vg_b0])

        # Chunk (1,0,0): vertex group 0 = neuron A, group 1 = neuron B
        vg_a1 = np.array([[10, 10, 10]], dtype=np.float32)
        vg_b1 = np.array([[15, 15, 15], [16, 16, 16]], dtype=np.float32)
        write_chunk_vertices(lg, (1, 0, 0), [vg_a1, vg_b1])

        write_object_index(lg, {
            0: [((0, 0, 0), 0), ((1, 0, 0), 0)],   # neuron A
            1: [((0, 0, 0), 1), ((1, 0, 0), 1)],   # neuron B
        }, sid_ndim=3)

        # Read neuron A
        verts_a = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        full_a = np.concatenate(verts_a)
        assert full_a.shape == (3, 3)  # 2 + 1 vertices
        np.testing.assert_array_equal(full_a[0], [0, 0, 0])

        # Read neuron B
        verts_b = read_object_vertices(lg, 1, dtype=np.float32, ndim=3)
        full_b = np.concatenate(verts_b)
        assert full_b.shape == (3, 3)  # 1 + 2 vertices
        np.testing.assert_array_equal(full_b[0], [5, 5, 5])
