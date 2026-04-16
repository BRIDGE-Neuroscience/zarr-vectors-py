"""Point cloud I/O for ZVF stores.

Supports three point cloud variants:

1. **Undifferentiated** — no per-point object identity.  All points in
   a chunk form a single vertex group.  No object_index.
2. **Per-point objects** — each point is its own object (e.g. cell
   centroids).  One vertex group per point per chunk.  Object index
   maps point ID → (chunk, vg_index).
3. **Multi-point objects** — many points per object (e.g. transcript
   spots grouped into cells).  Points sharing an object_id are stored
   in one vertex group per chunk.  Object index maps object → chunks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    GEOM_POINT_CLOUD,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_IDENTITY,
    OBJIDX_STANDARD,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_vertices,
    read_group_object_ids,
    read_groupings_attributes,
    read_object_attributes,
    read_object_vertices,
    read_vertex_group,
    write_chunk_attributes,
    write_chunk_vertices,
    write_groupings,
    write_groupings_attributes,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import (
    FsGroup,
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
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
    ObjectManifest,
    VertexGroupRef,
)


def write_points(
    store_path: str,
    positions: npt.NDArray[np.floating],
    *,
    chunk_shape: ChunkShape | None = None,
    bin_shape: BinShape | None = None,
    attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Write a point cloud to a new ZVF store.

    Args:
        store_path: Path for the new store (must not exist).
        positions: ``(N, D)`` vertex positions.
        chunk_shape: Spatial chunk size per dimension.  If None, a
            single chunk containing all points is used.
        bin_shape: Supervoxel bin edge lengths.  If None, defaults to
            ``chunk_shape`` (one bin per chunk — backward compatible).
        attributes: Per-vertex attributes ``{name: array}``.
        object_ids: ``(N,)`` integer per-point object assignment.
        object_attributes: Per-object attributes ``{name: array}``.
        groups: Group memberships ``{group_id: [object_id, ...]}``.
        group_attributes: Per-group attributes ``{name: array}``.
        dtype: Numpy dtype string for vertex positions.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    positions = np.asarray(positions, dtype=np_dtype)
    n_vertices, ndim = positions.shape

    if chunk_shape is None:
        bounds = compute_bounds(positions)
        extent = bounds[1] - bounds[0]
        max_extent = float(np.max(extent))
        side = max_extent + abs(float(np.max(bounds[1]))) + 1.0
        chunk_shape = tuple(side for _ in range(ndim))

    effective_bin = bin_shape if bin_shape is not None else chunk_shape
    bins_per_chunk = tuple(
        int(round(cs / bs)) for cs, bs in zip(chunk_shape, effective_bin)
    )
    total_bins_per_chunk = 1
    for b in bins_per_chunk:
        total_bins_per_chunk *= b

    needs_objects = (
        object_ids is not None
        or object_attributes is not None
        or groups is not None
    )
    if needs_objects and object_ids is None:
        object_ids = np.arange(n_vertices, dtype=np.int64)

    bounds = compute_bounds(positions)
    bounds_list = (bounds[0].tolist(), bounds[1].tolist())
    axes = [{"name": f"dim{i}", "type": "space", "unit": "unit"} for i in range(ndim)]

    root_meta = RootMetadata(
        spatial_index_dims=axes, chunk_shape=chunk_shape, bounds=bounds_list,
        geometry_types=[GEOM_POINT_CLOUD],
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
        base_bin_shape=bin_shape,
    )
    root = create_store(store_path, root_meta)

    bin_assignments = assign_bins(positions, effective_bin)
    chunked_bins = group_bins_by_chunk(bin_assignments, bins_per_chunk)
    n_chunks = len(chunked_bins)

    arrays_present = [VERTICES]
    if attributes:
        arrays_present.append("attributes")
    if needs_objects:
        arrays_present.append("object_index")

    level_meta = LevelMetadata(level=0, vertex_count=n_vertices, arrays_present=arrays_present)
    level_group = create_resolution_level(root, 0, level_meta)
    create_vertices_array(level_group, dtype=dtype)

    if attributes:
        for attr_name, attr_data in attributes.items():
            channel_names = None
            if attr_data.ndim == 2:
                channel_names = [f"ch{i}" for i in range(attr_data.shape[1])]
            create_attribute_array(level_group, attr_name, dtype=str(attr_data.dtype),
                                   channel_names=channel_names)

    if needs_objects:
        create_object_index_array(level_group)

    object_manifests: dict[int, ObjectManifest] = {}

    if object_ids is not None:
        # With objects: use per-object vertex groups per chunk (not per-bin)
        # This preserves correct object_index semantics for read_object_vertices
        chunk_assignments = assign_chunks(positions, chunk_shape)
        for chunk_coords, global_indices in sorted(chunk_assignments.items()):
            chunk_positions = positions[global_indices]
            chunk_obj_ids = object_ids[global_indices]
            unique_objs = np.unique(chunk_obj_ids)
            vert_groups: list[npt.NDArray] = []
            attr_groups_per_name: dict[str, list[npt.NDArray]] = {}
            if attributes:
                for attr_name in attributes:
                    attr_groups_per_name[attr_name] = []

            vg_idx = 0
            for obj_id in unique_objs:
                mask = chunk_obj_ids == obj_id
                vert_groups.append(chunk_positions[mask])
                oid = int(obj_id)
                if oid not in object_manifests:
                    object_manifests[oid] = []
                object_manifests[oid].append((chunk_coords, vg_idx))
                if attributes:
                    obj_global = global_indices[mask]
                    for attr_name, attr_data in attributes.items():
                        attr_groups_per_name[attr_name].append(attr_data[obj_global])
                vg_idx += 1

            write_chunk_vertices(level_group, chunk_coords, vert_groups, dtype=np_dtype)
            if attributes:
                for attr_name, groups_list in attr_groups_per_name.items():
                    write_chunk_attributes(level_group, attr_name, chunk_coords, groups_list,
                                           dtype=attributes[attr_name].dtype)
    else:
        # Undifferentiated: use per-bin vertex groups
        for chunk_coords, vg_dict in sorted(chunked_bins.items()):
            vert_groups_bin: list[npt.NDArray] = [
                np.zeros((0, ndim), dtype=np_dtype) for _ in range(total_bins_per_chunk)
            ]
            attr_groups_per_name_bin: dict[str, list[npt.NDArray]] = {}
            if attributes:
                for attr_name, attr_data in attributes.items():
                    shape_tail = attr_data.shape[1:] if attr_data.ndim > 1 else ()
                    attr_groups_per_name_bin[attr_name] = [
                        np.zeros((0,) + shape_tail, dtype=attr_data.dtype)
                        for _ in range(total_bins_per_chunk)
                    ]

            for vg_idx, global_indices in vg_dict.items():
                vert_groups_bin[vg_idx] = positions[global_indices]
                if attributes:
                    for attr_name, attr_data in attributes.items():
                        attr_groups_per_name_bin[attr_name][vg_idx] = attr_data[global_indices]

            write_chunk_vertices(level_group, chunk_coords, vert_groups_bin, dtype=np_dtype)
            if attributes:
                for attr_name, groups_list in attr_groups_per_name_bin.items():
                    write_chunk_attributes(level_group, attr_name, chunk_coords, groups_list,
                                           dtype=attributes[attr_name].dtype)

    if needs_objects and object_manifests:
        write_object_index(level_group, object_manifests, sid_ndim=ndim)

    n_objects = len(object_manifests) if object_manifests else 0
    if object_attributes:
        for name, data in object_attributes.items():
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, data)

    n_groups = 0
    if groups:
        create_groupings_array(level_group)
        write_groupings(level_group, groups)
        n_groups = len(groups)

    if group_attributes:
        for name, data in group_attributes.items():
            create_groupings_attributes_array(level_group, name)
            write_groupings_attributes(level_group, name, data)

    return {
        "vertex_count": n_vertices,
        "chunk_count": n_chunks,
        "object_count": n_objects,
        "group_count": n_groups,
        "bins_per_chunk": bins_per_chunk,
    }


def read_points(
    store_path: str,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    attribute_names: list[str] | None = None,
) -> dict[str, Any]:
    """Read point cloud data from a ZVF store.

    Supports filtering by bounding box, object ID, or group ID.
    Filters are applied in order: group → object → spatial.

    Args:
        store_path: Path to the ZVF store.
        level: Resolution level to read (default 0).
        bbox: Optional bounding box filter as ``(min_corner, max_corner)``.
        object_ids: Optional list of object IDs to read.
        group_ids: Optional list of group IDs — expands to their object IDs.
        attribute_names: Optional list of attribute names to read.
            If None, reads all available attributes.

    Returns:
        Dict with keys:
        - ``positions``: ``(M, D)`` array of vertex positions
        - ``attributes``: ``{name: array}`` of per-vertex attributes
        - ``object_ids``: ``(M,)`` array of object IDs per vertex (if objects exist)
        - ``vertex_count``: total vertices returned
    """
    root = open_store(store_path)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim
    dtype = np.float32  # default; could read from array meta

    # Read vertex dtype from metadata if available
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Resolve group_ids → object_ids
    if group_ids is not None:
        resolved_obj_ids: set[int] = set()
        for gid in group_ids:
            members = read_group_object_ids(level_group, gid)
            resolved_obj_ids.update(members)
        if object_ids is not None:
            resolved_obj_ids &= set(object_ids)
        object_ids = sorted(resolved_obj_ids)

    # Path 1: read by object ID (via object_index)
    if object_ids is not None:
        all_positions: list[npt.NDArray] = []
        all_obj_labels: list[npt.NDArray] = []
        all_attrs: dict[str, list[npt.NDArray]] = {}

        for oid in object_ids:
            try:
                verts_list = read_object_vertices(
                    level_group, oid, dtype=dtype, ndim=ndim
                )
            except ArrayError:
                continue

            for vg in verts_list:
                n_pts = len(vg)
                all_positions.append(vg)
                all_obj_labels.append(np.full(n_pts, oid, dtype=np.int64))

        if not all_positions:
            return _empty_result(ndim)

        positions_out = np.concatenate(all_positions, axis=0)

        # Apply bbox filter if needed
        if bbox is not None:
            mask = np.all(
                (positions_out >= bbox[0]) & (positions_out <= bbox[1]),
                axis=1,
            )
            positions_out = positions_out[mask]
            all_obj_labels = [
                np.concatenate(all_obj_labels)[mask]
            ]

        result: dict[str, Any] = {
            "positions": positions_out,
            "object_ids": np.concatenate(all_obj_labels) if all_obj_labels else np.array([], dtype=np.int64),
            "attributes": {},
            "vertex_count": len(positions_out),
        }
        return result

    # Path 2: read by bounding box or read all
    chunk_keys = list_chunk_keys(level_group)
    effective_bin = root_meta.effective_bin_shape
    bins_per_chunk = root_meta.bins_per_chunk
    has_bins = any(b > 1 for b in bins_per_chunk)

    if bbox is not None and has_bins:
        # Bin-level targeting: only decode matching vertex groups
        from zarr_vectors.spatial.chunking import (
            bins_intersecting_bbox,
            bin_to_chunk,
            bin_to_vg_index,
        )
        target_bins = bins_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]),
            effective_bin,
        )
        # Group target bins by chunk
        chunk_vg_targets: dict[ChunkCoords, list[int]] = {}
        for bc in target_bins:
            cc = bin_to_chunk(bc, bins_per_chunk)
            vgi = bin_to_vg_index(bc, cc, bins_per_chunk)
            if cc not in chunk_vg_targets:
                chunk_vg_targets[cc] = []
            chunk_vg_targets[cc].append(vgi)

        # Only read from chunks that have data
        chunk_keys_set = set(chunk_keys)
        all_positions = []
        for cc, vg_indices in chunk_vg_targets.items():
            if cc not in chunk_keys_set:
                continue
            for vgi in vg_indices:
                try:
                    vg = read_vertex_group(
                        level_group, cc, vgi, dtype=dtype, ndim=ndim,
                    )
                    if len(vg) > 0:
                        all_positions.append(vg)
                except ArrayError:
                    continue

    elif bbox is not None:
        # Chunk-level targeting (old stores without bins)
        target_chunks = set(chunks_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]),
            root_meta.chunk_shape,
        ))
        chunk_keys = [k for k in chunk_keys if k in target_chunks]

        all_positions = []
        for chunk_coords in chunk_keys:
            try:
                groups = read_chunk_vertices(
                    level_group, chunk_coords, dtype=dtype, ndim=ndim
                )
            except ArrayError:
                continue
            for vg in groups:
                if len(vg) > 0:
                    all_positions.append(vg)

    else:
        # Read all
        all_positions = []
        for chunk_coords in chunk_keys:
            try:
                groups = read_chunk_vertices(
                    level_group, chunk_coords, dtype=dtype, ndim=ndim
                )
            except ArrayError:
                continue
            for vg in groups:
                if len(vg) > 0:
                    all_positions.append(vg)

    if not all_positions:
        return _empty_result(ndim)

    positions_out = np.concatenate(all_positions, axis=0)

    # Apply precise bbox filter (chunk-level is coarse)
    if bbox is not None:
        mask = np.all(
            (positions_out >= bbox[0]) & (positions_out <= bbox[1]),
            axis=1,
        )
        positions_out = positions_out[mask]

    # Read attributes
    attrs_out: dict[str, npt.NDArray] = {}
    if attribute_names:
        for attr_name in attribute_names:
            attr_parts: list[npt.NDArray] = []
            for chunk_coords in chunk_keys:
                try:
                    attr_groups = read_chunk_attributes(
                        level_group, attr_name, chunk_coords,
                        dtype=np.float32, ncols=1,
                    )
                    for ag in attr_groups:
                        attr_parts.append(ag)
                except ArrayError:
                    continue
            if attr_parts:
                attr_all = np.concatenate(attr_parts, axis=0)
                if bbox is not None:
                    attr_all = attr_all[mask]
                attrs_out[attr_name] = attr_all

    return {
        "positions": positions_out,
        "attributes": attrs_out,
        "vertex_count": len(positions_out),
    }


def _empty_result(ndim: int) -> dict[str, Any]:
    """Return an empty result dict."""
    return {
        "positions": np.zeros((0, ndim), dtype=np.float32),
        "attributes": {},
        "vertex_count": 0,
    }
