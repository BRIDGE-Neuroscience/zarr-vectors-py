"""Mesh I/O for zarr vectors stores.

Supports two encoding modes:

- **Raw**: vertex positions in ``vertices/``, face indices in ``links/``
  (L=3 for triangles, L=4 for quads).  Faces spanning chunk boundaries
  go into ``cross_chunk_links/``.

- **Draco**: each vertex group is encoded as a Draco bitstream containing
  both positions and faces.  ``links/`` may be omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    ENCODING_DRACO,
    ENCODING_RAW,
    GEOM_MESH,
    LINKS_EXPLICIT,
    OBJIDX_STANDARD,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    resolve_chunk_keys,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_faces,
    read_cross_chunk_links,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_faces,
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
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_faces,
)
from zarr_vectors.spatial.chunking import (
    assign_bins,
    assign_chunks,
    compute_bounds,
    group_bins_by_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    ObjectManifest,
)


def _stamp_root_capability(root_group, cap: str) -> None:
    """Add ``cap`` to root metadata's ``format_capabilities`` (idempotent)."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    caps = list(zv.get("format_capabilities", []))
    if cap not in caps:
        caps.append(cap)
        zv["format_capabilities"] = caps
        root_group.attrs.update({"zarr_vectors": zv})


def write_mesh(
    store_path: str,
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    encoding: str = ENCODING_RAW,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    dtype: str = "float32",
    draco_quantization_bits: int = 11,
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
) -> dict[str, Any]:
    """Write a mesh to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        vertices: ``(V, D)`` vertex positions.
        faces: ``(F, L)`` face index array (L=3 triangles, L=4 quads).
        chunk_shape: Spatial chunk size per dimension.
        encoding: ``"raw"`` or ``"draco"``.
        vertex_attributes: Per-vertex attributes ``{name: (V,) or (V,C)}``.
        object_ids: ``(V,)`` array assigning vertices to mesh objects.
            If None, all vertices belong to object 0.
        dtype: Numpy dtype for positions.
        draco_quantization_bits: For Draco encoding only.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    vertices = np.asarray(vertices, dtype=np_dtype)
    faces = np.asarray(faces, dtype=np.int64)
    n_verts, ndim = vertices.shape
    n_faces, link_width = faces.shape

    if faces.ndim != 2 or link_width < 3:
        raise ArrayError(f"faces must be (F, L) with L≥3, got shape {faces.shape}")

    if object_ids is None:
        # When attribute-chunking, each vertex becomes its own object so
        # the per-object uniformity check is trivially satisfied.
        if chunk_by_attribute is not None:
            object_ids = np.arange(n_verts, dtype=np.int64)
        else:
            object_ids = np.zeros(n_verts, dtype=np.int64)

    bounds = compute_bounds(vertices)
    bounds_list = (bounds[0].tolist(), bounds[1].tolist())
    axes = [{"name": f"dim{i}", "type": "space", "unit": "unit"} for i in range(ndim)]

    root_meta = RootMetadata(
        spatial_index_dims=axes,
        chunk_shape=chunk_shape,
        bounds=bounds_list,
        geometry_types=[GEOM_MESH],
        links_convention=LINKS_EXPLICIT,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
        base_bin_shape=bin_shape,
    )
    root = create_store(store_path, root_meta, backend=backend)

    # Attribute chunking: per-object uniformity required.
    vertex_attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        if (
            not vertex_attributes
            or chunk_by_attribute not in vertex_attributes
        ):
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `vertex_attributes`"
            )
        src_values = np.asarray(vertex_attributes[chunk_by_attribute])
        if src_values.ndim != 1 or src_values.shape[0] != n_verts:
            raise ArrayError(
                f"vertex_attributes[{chunk_by_attribute!r}] must be 1D "
                f"of length n_verts={n_verts}, got shape {src_values.shape}"
            )
        vertex_attr_bins, attr_bin_values = assign_attribute_bins(src_values)
        for oid in np.unique(object_ids):
            mask = object_ids == oid
            unique_bins = np.unique(vertex_attr_bins[mask])
            if len(unique_bins) > 1:
                raise ArrayError(
                    f"chunk_by_attribute={chunk_by_attribute!r} requires "
                    f"per-object uniformity for meshes; object {int(oid)} "
                    f"has {len(unique_bins)} distinct attribute values"
                )

    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )

    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_verts,
        arrays_present=[VERTICES, "links", "object_index"],
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)
    create_vertices_array(level_group, dtype=dtype, encoding=encoding)
    create_links_array(level_group, link_width=link_width, delta=0)
    create_object_index_array(level_group)
    create_cross_chunk_links_array(level_group, delta=0)

    if vertex_attributes:
        for name, data in vertex_attributes.items():
            create_attribute_array(level_group, name, dtype=str(data.dtype))

    # Assign vertices to chunks.  When attribute-chunked, prefix each
    # spatial chunk key with the per-vertex bin.
    chunk_assignments = assign_chunks(vertices, chunk_shape)
    if vertex_attr_bins is not None:
        prefixed: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
        for spatial_cc, gi in chunk_assignments.items():
            gi = np.asarray(gi, dtype=np.int64)
            chunk_bins = vertex_attr_bins[gi]
            for ab in np.unique(chunk_bins):
                mask = chunk_bins == ab
                prefixed[(int(ab),) + spatial_cc] = gi[mask]
        chunk_assignments = prefixed
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_verts, chunk_list
    )

    # Partition faces
    intra_faces, cross_faces = partition_faces(
        faces, vertex_chunks, vertex_local, chunk_list
    )

    # Write vertices per chunk (one vertex group per chunk for simplicity)
    object_manifests: dict[int, ObjectManifest] = {}
    for chunk_idx, chunk_coords in enumerate(chunk_list):
        global_indices = chunk_assignments[chunk_coords]
        chunk_verts = vertices[global_indices]

        if encoding == ENCODING_DRACO and ndim == 3:
            # Draco mode: encode positions + local faces together
            local_faces_arr = intra_faces.get(chunk_coords)
            _write_draco_chunk(
                level_group, chunk_coords, chunk_verts,
                local_faces_arr, draco_quantization_bits, np_dtype,
            )
        else:
            write_chunk_vertices(level_group, chunk_coords, [chunk_verts], dtype=np_dtype)

        # Track manifests: one vertex group per chunk per unique object in chunk
        chunk_obj_ids = object_ids[global_indices]
        for oid in np.unique(chunk_obj_ids):
            oid_int = int(oid)
            if oid_int not in object_manifests:
                object_manifests[oid_int] = []
            object_manifests[oid_int].append((chunk_coords, 0))

        # Write vertex attributes
        if vertex_attributes:
            for name, data in vertex_attributes.items():
                chunk_attrs = data[global_indices]
                write_chunk_attributes(
                    level_group, name, chunk_coords, [chunk_attrs],
                    dtype=data.dtype,
                )

    # Write intra-chunk faces (raw mode)
    if encoding != ENCODING_DRACO:
        for chunk_coords in chunk_list:
            if chunk_coords in intra_faces:
                write_chunk_links(
                    level_group, chunk_coords, [intra_faces[chunk_coords]], delta=0,
                )

    # Write cross-chunk faces
    # Convert cross-face refs to cross_chunk_links format
    # Each cross face is a list of (chunk, local_idx) tuples
    cross_links: list[Any] = []
    for face_ref in cross_faces:
        # Store as pairs: each consecutive pair of face vertices
        for i in range(len(face_ref) - 1):
            cross_links.append((face_ref[i], face_ref[i + 1]))
        # Close the face: last vertex to first
        if len(face_ref) >= 3:
            cross_links.append((face_ref[-1], face_ref[0]))

    idx_ndim = ndim + 1 if vertex_attr_bins is not None else ndim
    if cross_links:
        write_cross_chunk_links(
            level_group, cross_links, sid_ndim=idx_ndim, delta=0,
        )

    # Tier C: persist cross-chunk face identity alongside the edge-pair
    # fallback.  Old readers that ignore the new array still see
    # connectivity via the existing cross_chunk_links; new readers can
    # reconstruct boundary faces exactly.
    if cross_faces:
        write_cross_chunk_faces(level_group, cross_faces, sid_ndim=idx_ndim)
        _stamp_root_capability(root, "cross_chunk_faces")

    # Write object index
    write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

    # Per-object attributes (Tier B): {name: (O,) or (O, C)}.
    if object_attributes:
        for _name, _data in object_attributes.items():
            create_object_attributes_array(level_group, _name)
            write_object_attributes(level_group, _name, np.asarray(_data))

    return {
        "vertex_count": n_verts,
        "face_count": n_faces,
        "chunk_count": len(chunk_list),
        "intra_face_count": sum(len(f) for f in intra_faces.values()),
        "cross_face_count": len(cross_faces),
        "encoding": encoding,
    }


def read_mesh(
    store_path: str,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read a mesh from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            in those chunks is returned. AND-ed with ``bbox`` and
            ``object_ids``. ``chunks=[]`` yields an empty result;
            ``chunks=None`` (default) applies no chunk filter.

    Returns:
        Dict with:
        - ``vertices``: ``(V, D)`` positions
        - ``faces``: ``(F, L)`` face indices (remapped to output vertex order)
        - ``vertex_count``, ``face_count``
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

    link_width = 3
    try:
        lmeta = level_group.read_array_meta("links/0")
        link_width = lmeta.get("link_width", 3)
    except Exception:
        pass

    # Determine chunks to read (intersection of physical keys, bbox-implied
    # set, and explicit chunks whitelist).
    chunk_keys = resolve_chunk_keys(
        level_group, root_meta.chunk_shape, bbox=bbox, chunks=chunks,
    )

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
            return _empty_mesh_result(ndim, link_width)
        chunk_keys = [k for k in chunk_keys if k and k[0] == filter_bin]

    # Read vertices and build offset map
    all_positions: list[npt.NDArray] = []
    chunk_offsets: dict[ChunkCoords, int] = {}
    running = 0

    for chunk_coords in chunk_keys:
        chunk_offsets[chunk_coords] = running
        try:
            groups = read_chunk_vertices(
                level_group, chunk_coords, dtype=dtype, ndim=ndim
            )
            for vg in groups:
                all_positions.append(vg)
                running += len(vg)
        except ArrayError:
            pass

    if not all_positions:
        return _empty_mesh_result(ndim, link_width)

    positions_out = np.concatenate(all_positions, axis=0)

    # Read intra-chunk faces
    all_faces: list[npt.NDArray] = []
    for chunk_coords in chunk_keys:
        try:
            link_groups = read_chunk_links(
                level_group, chunk_coords, link_width=link_width, delta=0,
            )
            offset = chunk_offsets.get(chunk_coords, 0)
            for lg in link_groups:
                if len(lg) > 0:
                    remapped = lg.copy() + offset
                    all_faces.append(remapped)
        except ArrayError:
            pass

    # Tier C: emit cross-chunk faces using preserved identity records.
    # Map each (chunk, local_idx) record into the global vertex index
    # via ``chunk_offsets`` (built from the chunks we just read).  When
    # the array is absent (0.2 stores or no boundary faces), reads are
    # untouched.
    cross_face_records = read_cross_chunk_faces(level_group)
    for face in cross_face_records:
        vertex_ids: list[int] = []
        for cc, local_idx in face:
            if cc not in chunk_offsets:
                vertex_ids = []
                break
            vertex_ids.append(int(chunk_offsets[cc]) + int(local_idx))
        if len(vertex_ids) == link_width:
            all_faces.append(np.asarray(vertex_ids, dtype=np.int64)[None, :])

    if all_faces:
        faces_out = np.concatenate(all_faces, axis=0)
    else:
        faces_out = np.zeros((0, link_width), dtype=np.int64)

    # Apply bbox filter on vertices
    if bbox is not None:
        bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
        node_mask = np.all(
            (positions_out >= bbox_min) & (positions_out <= bbox_max),
            axis=1,
        )
        if not np.all(node_mask):
            keep = np.flatnonzero(node_mask)
            keep_set = set(keep.tolist())
            positions_out = positions_out[keep]

            old_to_new = {int(old): new for new, old in enumerate(keep)}
            filtered: list[npt.NDArray] = []
            for f in faces_out:
                if all(int(v) in keep_set for v in f):
                    filtered.append(np.array([old_to_new[int(v)] for v in f]))
            faces_out = (
                np.stack(filtered).astype(np.int64)
                if filtered
                else np.zeros((0, link_width), dtype=np.int64)
            )

    return {
        "vertices": positions_out,
        "faces": faces_out,
        "vertex_count": len(positions_out),
        "face_count": len(faces_out),
    }


def _empty_mesh_result(ndim: int, link_width: int) -> dict[str, Any]:
    return {
        "vertices": np.zeros((0, ndim), dtype=np.float32),
        "faces": np.zeros((0, link_width), dtype=np.int64),
        "vertex_count": 0,
        "face_count": 0,
    }


def _write_draco_chunk(
    level_group: Any,
    chunk_coords: ChunkCoords,
    positions: npt.NDArray,
    faces: npt.NDArray | None,
    qbits: int,
    np_dtype: np.dtype,
) -> None:
    """Encode a chunk as Draco and write as raw bytes."""
    from zarr_vectors.encoding.draco import draco_encode_mesh

    if faces is None or len(faces) == 0:
        # Point cloud mode
        from zarr_vectors.encoding.draco import draco_encode_point_cloud
        blob = draco_encode_point_cloud(positions, quantization_bits=qbits)
    else:
        blob = draco_encode_mesh(positions, faces, quantization_bits=qbits)

    # Store as raw bytes in the vertices chunk
    from zarr_vectors.core.arrays import _chunk_key
    from zarr_vectors.encoding.ragged import encode_paired_offsets

    key = _chunk_key(chunk_coords)
    level_group.write_bytes("vertices", key, blob)

    # Single vertex group spanning whole chunk
    v_off = np.array([0], dtype=np.int64)
    l_off = np.array([-1], dtype=np.int64)
    level_group.write_bytes(
        "vertex_group_offsets", key,
        encode_paired_offsets(v_off, l_off),
    )
