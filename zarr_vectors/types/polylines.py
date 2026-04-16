"""Polyline and streamline I/O for zarr vectors stores.

A polyline is an ordered sequence of vertices forming a connected path.
Streamlines (from tractography) are polylines with additional per-object
attributes like termination regions.

Polylines that cross chunk boundaries are split into segments.  The
``object_index`` stores the ordered segment sequence for each polyline,
and ``cross_chunk_links`` connects the last vertex of one segment to
the first vertex of the next.  Within each segment, connectivity is
implicit sequential (vertex i → vertex i+1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_STANDARD,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_group_object_ids,
    read_object_attributes,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_vertices,
    write_cross_chunk_links,
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
from zarr_vectors.spatial.boundary import (
    cross_chunk_links_for_segments,
    split_polyline_at_boundaries,
)
from zarr_vectors.spatial.chunking import (
    chunks_intersecting_bbox,
    compute_bounds,
    bin_to_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    CrossChunkLink,
    ObjectManifest,
    VertexGroupRef,
)


def write_polylines(
    store_path: str,
    polylines: list[npt.NDArray[np.floating]],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    vertex_attributes: dict[str, list[npt.NDArray]] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
    geometry_type: str = GEOM_STREAMLINE,
) -> dict[str, Any]:
    """Write polylines/streamlines to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        polylines: List of arrays, each ``(N_k, D)`` — one per polyline.
        chunk_shape: Spatial chunk size per dimension.
        vertex_attributes: Per-vertex attributes aligned with polylines.
            ``{name: [array_for_polyline_0, array_for_polyline_1, ...]}``
            where each array is ``(N_k,)`` or ``(N_k, C)``.
        object_attributes: Per-polyline attributes.
            ``{name: (O,) or (O, C)}`` where O = number of polylines.
        groups: Group memberships ``{group_id: [polyline_indices]}``.
        group_attributes: Per-group attributes ``{name: (G,) or (G,C)}``.
        dtype: Numpy dtype for positions.
        geometry_type: ``"streamline"`` or ``"polyline"``.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    n_polylines = len(polylines)

    if n_polylines == 0:
        raise ArrayError("Cannot write empty polyline list")

    # Determine dimensionality from first polyline
    ndim = polylines[0].shape[1]

    # Compute global bounds from all vertices
    all_pts = np.concatenate(polylines, axis=0)
    bounds = compute_bounds(all_pts)
    bounds_list = (bounds[0].tolist(), bounds[1].tolist())
    total_vertices = len(all_pts)

    axes = [
        {"name": f"dim{i}", "type": "space", "unit": "unit"}
        for i in range(ndim)
    ]

    effective_bin = bin_shape if bin_shape is not None else chunk_shape
    bins_per_chunk = tuple(
        int(round(cs / bs)) for cs, bs in zip(chunk_shape, effective_bin)
    )

    root_meta = RootMetadata(
        spatial_index_dims=axes,
        chunk_shape=chunk_shape,
        bounds=bounds_list,
        geometry_types=[geometry_type],
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
        base_bin_shape=bin_shape,
    )
    root = create_store(store_path, root_meta)

    arrays_present = [VERTICES, "object_index"]
    level_meta = LevelMetadata(
        level=0,
        vertex_count=total_vertices,
        arrays_present=arrays_present,
    )
    level_group = create_resolution_level(root, 0, level_meta)
    create_vertices_array(level_group, dtype=dtype)
    create_object_index_array(level_group)
    create_cross_chunk_links_array(level_group)

    # Create attribute arrays
    if vertex_attributes:
        for attr_name, attr_list in vertex_attributes.items():
            sample = attr_list[0]
            create_attribute_array(
                level_group, attr_name,
                dtype=str(sample.dtype),
            )

    # Split each polyline at chunk boundaries and accumulate per-chunk data
    # chunk_data[chunk_coords] = list of (polyline_id, segment_vertices, segment_attrs)
    chunk_data: dict[ChunkCoords, list[tuple[int, npt.NDArray, dict[str, npt.NDArray]]]] = {}
    object_manifests: dict[int, ObjectManifest] = {}
    all_cross_links: list[CrossChunkLink] = []

    for poly_id, poly_verts in enumerate(polylines):
        poly_verts = np.asarray(poly_verts, dtype=np_dtype)

        # Split at bin boundaries (finer than chunk boundaries when bin_shape < chunk_shape)
        segments = split_polyline_at_boundaries(poly_verts, effective_bin)

        if not segments:
            object_manifests[poly_id] = []
            continue

        # Split vertex attributes to match segments
        seg_attrs_list: list[dict[str, npt.NDArray]] = []
        if vertex_attributes:
            # Compute cumulative vertex counts to split attributes
            seg_lengths = [len(s[1]) for s in segments]
            for seg_idx in range(len(segments)):
                seg_a: dict[str, npt.NDArray] = {}
                offset = sum(seg_lengths[:seg_idx])
                length = seg_lengths[seg_idx]
                for attr_name, attr_list in vertex_attributes.items():
                    attr_data = attr_list[poly_id]
                    seg_a[attr_name] = attr_data[offset:offset + length]
                seg_attrs_list.append(seg_a)
        else:
            seg_attrs_list = [{} for _ in segments]

        # Assign segments to chunks (bin_coords from splitter → chunk_coords)
        manifest: ObjectManifest = []
        for seg_idx, (bin_coords, seg_verts) in enumerate(segments):
            chunk_coords = bin_to_chunk(bin_coords, bins_per_chunk)
            if chunk_coords not in chunk_data:
                chunk_data[chunk_coords] = []
            vg_idx = len(chunk_data[chunk_coords])
            chunk_data[chunk_coords].append(
                (poly_id, seg_verts, seg_attrs_list[seg_idx])
            )
            manifest.append((chunk_coords, vg_idx))

        object_manifests[poly_id] = manifest

        # Cross-chunk links: only between consecutive segments in DIFFERENT chunks
        if len(manifest) > 1:
            for i in range(len(manifest) - 1):
                cc_a, vg_a = manifest[i]
                cc_b, vg_b = manifest[i + 1]
                if cc_a != cc_b:
                    all_cross_links.append(((cc_a, 0), (cc_b, 0)))

    # Write vertex groups per chunk
    for chunk_coords in sorted(chunk_data.keys()):
        entries = chunk_data[chunk_coords]
        vert_groups = [e[1] for e in entries]
        write_chunk_vertices(level_group, chunk_coords, vert_groups, dtype=np_dtype)

        # Write attributes per chunk
        if vertex_attributes:
            for attr_name in vertex_attributes:
                attr_groups = [e[2].get(attr_name) for e in entries]
                # Filter out None (shouldn't happen but be safe)
                attr_groups = [a for a in attr_groups if a is not None]
                if attr_groups:
                    write_chunk_attributes(
                        level_group, attr_name, chunk_coords, attr_groups,
                        dtype=attr_groups[0].dtype,
                    )

    # Write object index
    write_object_index(level_group, object_manifests, sid_ndim=ndim)

    # Write cross-chunk links
    if all_cross_links:
        write_cross_chunk_links(level_group, all_cross_links, sid_ndim=ndim)

    # Write object attributes
    if object_attributes:
        for name, data in object_attributes.items():
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, np.asarray(data))

    # Write groupings
    n_groups = 0
    if groups:
        create_groupings_array(level_group)
        write_groupings(level_group, groups)
        n_groups = len(groups)

    if group_attributes:
        for name, data in group_attributes.items():
            create_groupings_attributes_array(level_group, name)
            write_groupings_attributes(level_group, name, np.asarray(data))

    return {
        "polyline_count": n_polylines,
        "vertex_count": total_vertices,
        "chunk_count": len(chunk_data),
        "cross_chunk_link_count": len(all_cross_links),
        "group_count": n_groups,
    }


def read_polylines(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
) -> dict[str, Any]:
    """Read polylines/streamlines from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional list of polyline (object) IDs.
        group_ids: Optional group IDs — expands to their object IDs.
        bbox: Optional bounding box filter. Returns polylines that have
            at least one segment in a matching chunk.

    Returns:
        Dict with:
        - ``polylines``: list of lists of arrays. ``polylines[i]`` is
          a list of segment arrays for polyline i (concatenate for full path).
        - ``polyline_count``: number of polylines returned.
        - ``vertex_count``: total vertices across all returned polylines.
    """
    root = open_store(store_path)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Resolve group_ids → object_ids
    if group_ids is not None:
        resolved: set[int] = set()
        for gid in group_ids:
            members = read_group_object_ids(level_group, gid)
            resolved.update(members)
        if object_ids is not None:
            resolved &= set(object_ids)
        object_ids = sorted(resolved)

    # If no filter, read all objects
    if object_ids is None:
        try:
            meta = level_group.read_array_meta("object_index")
            object_ids = list(range(meta["num_objects"]))
        except Exception:
            return _empty_polyline_result()

    # If bbox, find which chunks are relevant
    target_chunks: set[ChunkCoords] | None = None
    if bbox is not None:
        target_chunks = set(chunks_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]),
            root_meta.chunk_shape,
        ))

    result_polylines: list[list[npt.NDArray]] = []
    total_verts = 0

    for oid in object_ids:
        try:
            vg_list = read_object_vertices(
                level_group, oid, dtype=dtype, ndim=ndim
            )
        except ArrayError:
            continue

        if not vg_list:
            continue

        # If bbox filter, check if any segment is in a target chunk
        if target_chunks is not None:
            try:
                manifest = read_all_object_manifests(level_group)
                obj_manifest = manifest[oid] if oid < len(manifest) else []
                has_match = any(
                    chunk_coords in target_chunks
                    for chunk_coords, _ in obj_manifest
                )
                if not has_match:
                    continue
            except Exception:
                continue

        result_polylines.append(vg_list)
        total_verts += sum(len(vg) for vg in vg_list)

    return {
        "polylines": result_polylines,
        "polyline_count": len(result_polylines),
        "vertex_count": total_verts,
    }


def _empty_polyline_result() -> dict[str, Any]:
    return {
        "polylines": [],
        "polyline_count": 0,
        "vertex_count": 0,
    }
