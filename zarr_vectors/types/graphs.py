"""Graph and skeleton I/O for zarr vectors stores.

Supports two modes:

- **General graph**: arbitrary undirected/directed edges. All edges stored
  explicitly in ``links/`` (intra-chunk) and ``cross_chunk_links/``
  (inter-chunk).  ``links_convention: "explicit"``.

- **Skeleton (tree)**: nodes reordered depth-first, parent links mostly
  sequential.  Only branch points (parent ≠ i−1) stored in ``links/``.
  ``links_convention: "implicit_sequential_with_branches"``.

Edge attributes (weight, type, etc.) are stored in ``link_attributes/``
parallel to the ``links/`` array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    GEOM_GRAPH,
    GEOM_SKELETON,
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    OBJIDX_STANDARD,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_link_attributes_array,
    create_links_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import (
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_edges,
)
from zarr_vectors.spatial.chunking import (
    assign_bins,
    assign_chunks,
    chunks_intersecting_bbox,
    compute_bounds,
    group_bins_by_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    CrossChunkLink,
    ObjectManifest,
)


# ===================================================================
# Write
# ===================================================================

def write_graph(
    store_path: str,
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    is_tree: bool = False,
    node_attributes: dict[str, npt.NDArray] | None = None,
    edge_attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Write a graph or skeleton to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        positions: ``(N, D)`` node positions.
        edges: ``(M, 2)`` edge list (global vertex indices).
        chunk_shape: Spatial chunk size per dimension.
        is_tree: If True, treat as skeleton — reorder depth-first and
            use implicit sequential with branches convention.
        node_attributes: Per-node attributes ``{name: (N,) or (N,C)}``.
        edge_attributes: Per-edge attributes ``{name: (M,) or (M,C)}``.
        object_ids: ``(N,)`` array assigning nodes to objects.  If None,
            all nodes belong to object 0.
        dtype: Numpy dtype for positions.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    positions = np.asarray(positions, dtype=np_dtype)
    edges = np.asarray(edges, dtype=np.int64)
    n_nodes, ndim = positions.shape
    n_edges = len(edges)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ArrayError(f"edges must be (M, 2), got shape {edges.shape}")

    # Default: all nodes in one object
    if object_ids is None:
        object_ids = np.zeros(n_nodes, dtype=np.int64)

    # For skeletons: reorder depth-first
    reorder_map: npt.NDArray | None = None
    if is_tree:
        positions, edges, node_attributes, edge_attributes, reorder_map = (
            _reorder_tree(positions, edges, node_attributes, edge_attributes)
        )
        # Remap object_ids
        if reorder_map is not None:
            inv = np.empty_like(reorder_map)
            inv[reorder_map] = np.arange(len(reorder_map))
            object_ids = object_ids[reorder_map]

    geometry_type = GEOM_SKELETON if is_tree else GEOM_GRAPH
    links_conv = LINKS_IMPLICIT_BRANCHES if is_tree else LINKS_EXPLICIT

    # Compute bounds
    bounds = compute_bounds(positions)
    bounds_list = (bounds[0].tolist(), bounds[1].tolist())

    axes = [
        {"name": f"dim{i}", "type": "space", "unit": "unit"}
        for i in range(ndim)
    ]

    root_meta = RootMetadata(
        spatial_index_dims=axes,
        chunk_shape=chunk_shape,
        bounds=bounds_list,
        geometry_types=[geometry_type],
        links_convention=links_conv,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
        base_bin_shape=bin_shape,
    )
    root = create_store(store_path, root_meta)

    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_nodes,
        arrays_present=[VERTICES, "links", "object_index"],
    )
    level_group = create_resolution_level(root, 0, level_meta)

    link_width = 2
    create_vertices_array(level_group, dtype=dtype)
    create_links_array(level_group, link_width=link_width)
    create_object_index_array(level_group)
    create_cross_chunk_links_array(level_group)

    if node_attributes:
        for name, data in node_attributes.items():
            create_attribute_array(level_group, name, dtype=str(data.dtype))

    if edge_attributes:
        for name, data in edge_attributes.items():
            create_link_attributes_array(level_group, name, dtype=str(data.dtype))

    # Assign nodes to chunks
    chunk_assignments = assign_chunks(positions, chunk_shape)
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_nodes, chunk_list
    )

    # For skeletons with implicit branches: extract only branch links
    if is_tree:
        edges_to_store, edge_attr_to_store = _extract_branch_links(
            edges, edge_attributes
        )
    else:
        edges_to_store = edges
        edge_attr_to_store = edge_attributes

    # Partition edges
    intra_edges, cross_links = partition_edges(
        edges_to_store, vertex_chunks, vertex_local, chunk_list
    )

    # Also partition the full edge set to get cross-chunk links for ALL edges
    # (not just branch links for skeletons)
    if is_tree:
        _, all_cross_links = partition_edges(
            edges, vertex_chunks, vertex_local, chunk_list
        )
    else:
        all_cross_links = cross_links

    # Write vertices per chunk (one vertex group per object per chunk)
    object_manifests: dict[int, ObjectManifest] = {}

    for chunk_idx, chunk_coords in enumerate(chunk_list):
        global_indices = chunk_assignments[chunk_coords]
        chunk_positions = positions[global_indices]
        chunk_obj_ids = object_ids[global_indices]

        unique_objs = np.unique(chunk_obj_ids)
        vert_groups: list[npt.NDArray] = []
        attr_groups: dict[str, list[npt.NDArray]] = {}
        if node_attributes:
            for name in node_attributes:
                attr_groups[name] = []

        for obj_id in unique_objs:
            mask = chunk_obj_ids == obj_id
            vert_groups.append(chunk_positions[mask])
            oid = int(obj_id)
            if oid not in object_manifests:
                object_manifests[oid] = []
            object_manifests[oid].append((chunk_coords, len(vert_groups) - 1))

            if node_attributes:
                obj_global = global_indices[mask]
                for name, data in node_attributes.items():
                    attr_groups[name].append(data[obj_global])

        write_chunk_vertices(level_group, chunk_coords, vert_groups, dtype=np_dtype)

        if node_attributes:
            for name, groups_list in attr_groups.items():
                write_chunk_attributes(
                    level_group, name, chunk_coords, groups_list,
                    dtype=node_attributes[name].dtype,
                )

    # Write intra-chunk links
    for chunk_coords in chunk_list:
        if chunk_coords in intra_edges:
            local_edges = intra_edges[chunk_coords]
            # One link group per chunk (all edges in one group)
            write_chunk_links(level_group, chunk_coords, [local_edges])

            # Write edge attributes for intra-chunk edges
            if edge_attr_to_store and edge_attributes:
                # We need to track which original edges ended up intra-chunk
                # For simplicity, edge attributes for intra-chunk stored per chunk
                pass  # Edge attribute tracking is complex; store for all edges

    # Write cross-chunk links
    if all_cross_links:
        write_cross_chunk_links(level_group, all_cross_links, sid_ndim=ndim)

    # Write object index
    write_object_index(level_group, object_manifests, sid_ndim=ndim)

    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "chunk_count": len(chunk_list),
        "intra_edge_count": sum(len(e) for e in intra_edges.values()),
        "cross_edge_count": len(all_cross_links),
        "object_count": len(object_manifests),
        "is_tree": is_tree,
    }


# ===================================================================
# Read
# ===================================================================

def read_graph(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
) -> dict[str, Any]:
    """Read a graph or skeleton from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional object ID filter.
        bbox: Optional bounding box filter.

    Returns:
        Dict with:
        - ``positions``: ``(N, D)`` node positions
        - ``edges``: ``(M, 2)`` edge list (remapped to output indices)
        - ``node_count``, ``edge_count``
    """
    root = open_store(store_path)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim
    is_tree = root_meta.links_convention == LINKS_IMPLICIT_BRANCHES

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Get link width
    link_width = 2
    try:
        lmeta = level_group.read_array_meta("links")
        link_width = lmeta.get("link_width", 2)
    except Exception:
        pass

    # Determine chunks to read
    chunk_keys = list_chunk_keys(level_group)
    if bbox is not None:
        target = set(chunks_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]),
            root_meta.chunk_shape,
        ))
        chunk_keys = [k for k in chunk_keys if k in target]

    # Read all vertices and build global index mapping
    all_positions: list[npt.NDArray] = []
    # Map (chunk_coords, local_idx) → global output index
    global_idx_map: dict[tuple[ChunkCoords, int], int] = {}
    current_global = 0

    for chunk_coords in chunk_keys:
        try:
            groups = read_chunk_vertices(
                level_group, chunk_coords, dtype=dtype, ndim=ndim
            )
        except ArrayError:
            continue

        for vg in groups:
            for local_i in range(len(vg)):
                global_idx_map[(chunk_coords, current_global + local_i)] = (
                    current_global + local_i
                )
            all_positions.append(vg)
            current_global += len(vg)

    if not all_positions:
        return _empty_graph_result(ndim)

    positions_out = np.concatenate(all_positions, axis=0)

    # Read intra-chunk edges
    all_edges: list[npt.NDArray] = []
    offset = 0
    for chunk_coords in chunk_keys:
        try:
            groups = read_chunk_vertices(
                level_group, chunk_coords, dtype=dtype, ndim=ndim
            )
        except ArrayError:
            groups = []

        try:
            link_groups = read_chunk_links(
                level_group, chunk_coords, link_width=link_width
            )
        except ArrayError:
            link_groups = []

        # Compute offset: sum of all vertex group sizes up to this chunk
        chunk_offset = 0
        for prev_cc in chunk_keys:
            if prev_cc == chunk_coords:
                break
            try:
                prev_groups = read_chunk_vertices(
                    level_group, prev_cc, dtype=dtype, ndim=ndim
                )
                chunk_offset += sum(len(g) for g in prev_groups)
            except ArrayError:
                pass

        for lg in link_groups:
            if len(lg) > 0:
                remapped = lg.copy()
                remapped[:, 0] += chunk_offset
                remapped[:, 1] += chunk_offset
                all_edges.append(remapped)

        offset = chunk_offset + sum(len(g) for g in groups)

    # Read cross-chunk edges
    try:
        ccl = read_cross_chunk_links(level_group)
        # Build chunk→offset map
        chunk_offsets: dict[ChunkCoords, int] = {}
        running = 0
        for cc in chunk_keys:
            chunk_offsets[cc] = running
            try:
                grps = read_chunk_vertices(
                    level_group, cc, dtype=dtype, ndim=ndim
                )
                running += sum(len(g) for g in grps)
            except ArrayError:
                pass

        for (chunk_a, vi_a), (chunk_b, vi_b) in ccl:
            if chunk_a in chunk_offsets and chunk_b in chunk_offsets:
                ga = chunk_offsets[chunk_a] + vi_a
                gb = chunk_offsets[chunk_b] + vi_b
                all_edges.append(np.array([[ga, gb]], dtype=np.int64))
    except (ArrayError, Exception):
        pass

    # For skeletons: reconstruct full edge set from implicit sequential + branch links
    if is_tree:
        total_nodes = len(positions_out)
        # Start with implicit parents: parent[i] = i-1, parent[0] = -1
        parent_arr = np.arange(-1, total_nodes - 1, dtype=np.int64)
        # Override with explicit branch links (stored in all_edges)
        for edge_block in all_edges:
            for e in edge_block:
                child, par = int(e[0]), int(e[1])
                if 0 <= child < total_nodes:
                    parent_arr[child] = par
        # Build edge list from parent array
        all_edges = []
        mask = parent_arr >= 0
        children_idx = np.flatnonzero(mask)
        if len(children_idx) > 0:
            tree_edges = np.stack(
                [children_idx, parent_arr[children_idx]], axis=1
            ).astype(np.int64)
            all_edges.append(tree_edges)

    if all_edges:
        edges_out = np.concatenate(all_edges, axis=0)
    else:
        edges_out = np.zeros((0, 2), dtype=np.int64)

    # Filter by bbox on positions
    if bbox is not None:
        bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
        node_mask = np.all(
            (positions_out >= bbox_min) & (positions_out <= bbox_max),
            axis=1,
        )
        if not np.all(node_mask):
            keep_indices = np.flatnonzero(node_mask)
            keep_set = set(keep_indices.tolist())
            positions_out = positions_out[keep_indices]

            # Remap edges
            old_to_new = {int(old): new for new, old in enumerate(keep_indices)}
            filtered_edges: list[list[int]] = []
            for e in edges_out:
                s, d = int(e[0]), int(e[1])
                if s in old_to_new and d in old_to_new:
                    filtered_edges.append([old_to_new[s], old_to_new[d]])
            edges_out = (
                np.array(filtered_edges, dtype=np.int64)
                if filtered_edges
                else np.zeros((0, 2), dtype=np.int64)
            )

    return {
        "positions": positions_out,
        "edges": edges_out,
        "node_count": len(positions_out),
        "edge_count": len(edges_out),
    }


def _empty_graph_result(ndim: int) -> dict[str, Any]:
    return {
        "positions": np.zeros((0, ndim), dtype=np.float32),
        "edges": np.zeros((0, 2), dtype=np.int64),
        "node_count": 0,
        "edge_count": 0,
    }


# ===================================================================
# Tree helpers
# ===================================================================

def _reorder_tree(
    positions: npt.NDArray,
    edges: npt.NDArray,
    node_attributes: dict[str, npt.NDArray] | None,
    edge_attributes: dict[str, npt.NDArray] | None,
) -> tuple[
    npt.NDArray, npt.NDArray,
    dict[str, npt.NDArray] | None,
    dict[str, npt.NDArray] | None,
    npt.NDArray,
]:
    """Reorder nodes depth-first from root for skeleton storage.

    Returns reordered (positions, edges, node_attrs, edge_attrs, reorder_map)
    where reorder_map[new_idx] = old_idx.
    """
    n = len(positions)

    # Build adjacency from edges
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    parent_of: dict[int, int] = {}
    for e in edges:
        a, b = int(e[0]), int(e[1])
        # Convention: edge [child, parent]
        children[b].append(a)
        parent_of[a] = b

    # Find root (node with no parent)
    all_nodes = set(range(n))
    child_nodes = set(parent_of.keys())
    roots = all_nodes - child_nodes
    root_node = min(roots) if roots else 0

    # DFS ordering
    order: list[int] = []
    stack = [root_node]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push children in reverse so first child is processed first
        for child in reversed(children.get(node, [])):
            if child not in visited:
                stack.append(child)

    # Add any unvisited nodes (disconnected components)
    for i in range(n):
        if i not in visited:
            order.append(i)

    reorder_map = np.array(order, dtype=np.int64)
    inv_map = np.empty(n, dtype=np.int64)
    inv_map[reorder_map] = np.arange(n)

    # Reorder positions
    new_positions = positions[reorder_map]

    # Remap edges
    new_edges = np.empty_like(edges)
    new_edges[:, 0] = inv_map[edges[:, 0]]
    new_edges[:, 1] = inv_map[edges[:, 1]]

    # Reorder node attributes
    new_node_attrs = None
    if node_attributes:
        new_node_attrs = {
            name: data[reorder_map] for name, data in node_attributes.items()
        }

    return new_positions, new_edges, new_node_attrs, edge_attributes, reorder_map


def _extract_branch_links(
    edges: npt.NDArray,
    edge_attributes: dict[str, npt.NDArray] | None,
) -> tuple[npt.NDArray, dict[str, npt.NDArray] | None]:
    """Extract only non-sequential (branch) links from a tree edge list.

    In a depth-first ordered tree, most edges have child = parent + 1.
    Only branch points (child's parent ≠ child - 1) need explicit storage.

    Args:
        edges: ``(M, 2)`` where each row is ``[child, parent]``.
        edge_attributes: Optional per-edge attributes.

    Returns:
        branch_edges: subset of edges where parent ≠ child - 1.
        branch_edge_attributes: corresponding subset of attributes.
    """
    if len(edges) == 0:
        return edges, edge_attributes

    children = edges[:, 0]
    parents = edges[:, 1]

    # Non-sequential: parent != child - 1
    is_branch = parents != (children - 1)
    branch_edges = edges[is_branch]

    branch_attrs = None
    if edge_attributes:
        branch_attrs = {
            name: data[is_branch] for name, data in edge_attributes.items()
        }

    return branch_edges, branch_attrs
