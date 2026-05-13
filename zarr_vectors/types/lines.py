"""Finite line and vector I/O for zarr vectors stores.

A finite line is two endpoints stored as a two-vertex object.
Connectivity is implicit sequential (vertex 0 → vertex 1), so no
``links`` array is needed.

Lines that cross a chunk boundary are split into two single-vertex
vertex groups in their respective chunks, with the ``object_index``
tracking both and a ``cross_chunk_links`` entry bridging them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    GEOM_LINE,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_STANDARD,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_attributes,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.attr_chunking import (
    assign_attribute_bins,
    compute_chunk_dim_names,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import (
    FsGroup,
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    compute_bounds,
    compute_chunk_coords,
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


def write_lines(
    store_path: str,
    endpoints: npt.NDArray[np.floating],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    attributes: dict[str, npt.NDArray] | None = None,
    line_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
) -> dict[str, Any]:
    """Write finite line segments to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        endpoints: ``(N, 2, D)`` array — N lines, each with 2 endpoints
            of D dimensions.
        chunk_shape: Spatial chunk size per dimension.
        attributes: Per-vertex attributes as ``{name: (N, 2) or (N, 2, C)}``.
            Two values per line (one per endpoint).
        line_attributes: Per-line (per-object) attributes as
            ``{name: (N,) or (N, C)}``.
        dtype: Numpy dtype string for positions.

    Returns:
        Summary dict with ``line_count``, ``chunk_count``,
        ``cross_chunk_count``.
    """
    np_dtype = np.dtype(dtype)
    endpoints = np.asarray(endpoints, dtype=np_dtype)

    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ArrayError(
            f"endpoints must be (N, 2, D), got shape {endpoints.shape}"
        )

    n_lines, _, ndim = endpoints.shape

    # Per-line attribute chunking: each line goes into one bin.  The
    # attribute name must appear in ``line_attributes`` (per-line) — or
    # in per-endpoint ``attributes`` with both endpoints sharing the
    # same value.
    line_attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        src_values: npt.NDArray | None = None
        if line_attributes and chunk_by_attribute in line_attributes:
            src_values = np.asarray(line_attributes[chunk_by_attribute])
        elif attributes and chunk_by_attribute in attributes:
            per_endpoint = np.asarray(attributes[chunk_by_attribute])
            if per_endpoint.shape[:2] != (n_lines, 2):
                raise ArrayError(
                    f"per-endpoint attribute {chunk_by_attribute!r} has "
                    f"shape {per_endpoint.shape}; expected (N, 2[, C])"
                )
            mismatched = np.any(per_endpoint[:, 0] != per_endpoint[:, 1])
            if mismatched:
                raise ArrayError(
                    f"chunk_by_attribute={chunk_by_attribute!r} on a "
                    f"per-endpoint attribute requires both endpoints of "
                    f"each line to share the same value (got mixed lines)"
                )
            src_values = per_endpoint[:, 0]
        if src_values is None:
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `line_attributes` (per-line) or in `attributes` "
                f"(per-endpoint, uniform per line)"
            )
        if src_values.shape[0] != n_lines:
            raise ArrayError(
                f"chunk_by_attribute array length {src_values.shape[0]} "
                f"!= n_lines {n_lines}"
            )
        line_attr_bins, attr_bin_values = assign_attribute_bins(src_values)

    effective_bin = bin_shape if bin_shape is not None else chunk_shape
    bins_per_chunk = tuple(
        int(round(cs / bs)) for cs, bs in zip(chunk_shape, effective_bin)
    )

    all_pts = endpoints.reshape(-1, ndim)
    bounds = compute_bounds(all_pts)
    bounds_list = (bounds[0].tolist(), bounds[1].tolist())

    root = _create_or_open_store(store_path, backend=backend)
    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        inferred_bounds=bounds_list,
        chunk_shape_hint=chunk_shape,
        geometry_type=GEOM_LINE,
        base_bin_shape=bin_shape,
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    arrays_present = [VERTICES, "object_index"]
    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )
    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_lines * 2,
        arrays_present=arrays_present,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)
    create_vertices_array(level_group, dtype=dtype)
    create_object_index_array(level_group)
    create_cross_chunk_links_array(level_group, delta=0)

    # Classify endpoints by bin → chunk
    chunk_a_arr = np.array([
        bin_to_chunk(compute_chunk_coords(endpoints[i, 0], effective_bin), bins_per_chunk)
        for i in range(n_lines)
    ])
    chunk_b_arr = np.array([
        bin_to_chunk(compute_chunk_coords(endpoints[i, 1], effective_bin), bins_per_chunk)
        for i in range(n_lines)
    ])

    chunk_groups: dict[ChunkCoords, list[tuple[int, npt.NDArray]]] = {}
    object_manifests: dict[int, ObjectManifest] = {}
    cross_links: list[CrossChunkLink] = []

    def _prefix(cc: tuple) -> ChunkCoords:
        if line_attr_bins is None:
            return cc
        return (int(line_attr_bins[i]),) + cc

    for i in range(n_lines):
        ca = _prefix(tuple(int(x) for x in chunk_a_arr[i]))
        cb = _prefix(tuple(int(x) for x in chunk_b_arr[i]))

        if ca == cb:
            # Both endpoints in same chunk — one vertex group of 2 points
            if ca not in chunk_groups:
                chunk_groups[ca] = []
            vg_idx = len(chunk_groups[ca])
            chunk_groups[ca].append((i, endpoints[i]))  # (N=2, D)
            object_manifests[i] = [(ca, vg_idx)]
        else:
            # Cross-chunk: one vertex group per chunk, one vertex each
            if ca not in chunk_groups:
                chunk_groups[ca] = []
            vg_idx_a = len(chunk_groups[ca])
            chunk_groups[ca].append((i, endpoints[i, 0:1]))  # (1, D)

            if cb not in chunk_groups:
                chunk_groups[cb] = []
            vg_idx_b = len(chunk_groups[cb])
            chunk_groups[cb].append((i, endpoints[i, 1:2]))  # (1, D)

            object_manifests[i] = [(ca, vg_idx_a), (cb, vg_idx_b)]

            # Cross-chunk link: last vertex of group A → first vertex of group B
            cross_links.append(((ca, 0), (cb, 0)))

    # Write vertex groups per chunk
    for chunk_coords, groups_list in sorted(chunk_groups.items()):
        vert_arrays = [g[1] for g in groups_list]
        write_chunk_vertices(level_group, chunk_coords, vert_arrays, dtype=np_dtype)

    # Write object index — widen sid_ndim when attribute-chunked.
    idx_ndim = ndim + 1 if line_attr_bins is not None else ndim
    write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

    # Write cross-chunk links
    if cross_links:
        write_cross_chunk_links(
            level_group, cross_links, sid_ndim=idx_ndim, delta=0,
        )

    # Write line attributes (per-object)
    if line_attributes:
        for name, data in line_attributes.items():
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, np.asarray(data))

    return {
        "line_count": n_lines,
        "chunk_count": len(chunk_groups),
        "cross_chunk_count": len(cross_links),
    }


def read_lines(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read finite lines from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional list of line (object) IDs to read.
        bbox: Optional bounding box filter (lines with any endpoint
            inside the box are returned).

    Returns:
        Dict with:
        - ``endpoints``: ``(M, 2, D)`` array of line endpoints
        - ``line_count``: number of lines returned
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Resolve attribute_filter → leading-bin index (per-object lines:
    # all of an object's chunks share the same leading coord).
    filter_bin: int | None = None
    if attribute_filter:
        try:
            lm = read_level_metadata(root, level)
        except Exception:
            lm = None
        if (
            lm is None
            or lm.chunk_attribute_name is None
            or lm.chunk_attribute_values is None
        ):
            raise ArrayError(
                "attribute_filter requires a store written with chunk_by_attribute"
            )
        if len(attribute_filter) != 1:
            raise ArrayError(
                "attribute_filter must specify exactly one attribute"
            )
        fname, fvalue = next(iter(attribute_filter.items()))
        if fname != lm.chunk_attribute_name:
            raise ArrayError(
                f"attribute_filter key {fname!r} does not match the "
                f"store's chunk_attribute_name {lm.chunk_attribute_name!r}"
            )
        try:
            filter_bin = lm.chunk_attribute_values.index(fvalue)
        except ValueError:
            return {
                "endpoints": np.zeros((0, 2, ndim), dtype=dtype),
                "line_count": 0,
            }

    # Determine which objects to read
    if object_ids is None:
        meta = level_group.read_array_meta("object_index")
        object_ids = list(range(meta["num_objects"]))

    # When filtering by attribute, prune object_ids whose manifest
    # entries don't start with the matching bin.
    if filter_bin is not None:
        from zarr_vectors.core.arrays import read_all_object_manifests
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            manifests = []
        kept: list[int] = []
        for oid in object_ids:
            if oid < len(manifests) and manifests[oid]:
                if all(cc and cc[0] == filter_bin for cc, _ in manifests[oid]):
                    kept.append(oid)
        object_ids = kept

    result_endpoints: list[npt.NDArray] = []

    for oid in object_ids:
        try:
            vg_list = read_object_vertices(
                level_group, oid, dtype=dtype, ndim=ndim
            )
        except ArrayError:
            continue

        # Concatenate vertex groups to get the full line (2 vertices)
        all_verts = np.concatenate(vg_list, axis=0)

        if len(all_verts) < 2:
            continue

        # Take first and last as endpoints (handles both same-chunk
        # and cross-chunk cases)
        ep = np.stack([all_verts[0], all_verts[-1]], axis=0)  # (2, D)
        result_endpoints.append(ep)

    if not result_endpoints:
        return {
            "endpoints": np.zeros((0, 2, ndim), dtype=dtype),
            "line_count": 0,
        }

    endpoints_out = np.stack(result_endpoints, axis=0)  # (M, 2, D)

    # Apply bbox filter
    if bbox is not None:
        bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
        # Keep line if either endpoint is inside the bbox
        in_a = np.all(
            (endpoints_out[:, 0] >= bbox_min) & (endpoints_out[:, 0] <= bbox_max),
            axis=1,
        )
        in_b = np.all(
            (endpoints_out[:, 1] >= bbox_min) & (endpoints_out[:, 1] <= bbox_max),
            axis=1,
        )
        mask = in_a | in_b
        endpoints_out = endpoints_out[mask]

    return {
        "endpoints": endpoints_out,
        "line_count": len(endpoints_out),
    }
