"""Cross-chunk boundary handling: splitting, partitioning, and linking.

All functions are pure numpy.  They take vertex positions and chunk
assignments as input and produce the data structures needed to write
cross-chunk links, partition edges/faces into intra- vs inter-chunk,
and split ordered polylines at chunk boundaries.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ChunkingError
from zarr_vectors.spatial.chunking import compute_chunk_coords
from zarr_vectors.typing import ChunkCoords, ChunkShape, CrossChunkLink


# ===================================================================
# Polyline / streamline splitting
# ===================================================================

def split_polyline_at_boundaries(
    vertices: npt.NDArray[np.floating],
    chunk_shape: ChunkShape,
) -> list[tuple[ChunkCoords, npt.NDArray[np.floating]]]:
    """Split an ordered polyline into segments at chunk boundaries.

    Consecutive vertices in the same chunk form one segment.  The
    returned list preserves the original vertex order — concatenating
    all segment arrays recovers the input.

    Args:
        vertices: ``(N, D)`` ordered vertex positions.
        chunk_shape: Physical chunk size per dimension.

    Returns:
        List of ``(chunk_coords, segment_vertices)`` in polyline order.
        Each ``segment_vertices`` is ``(N_k, D)``.

    Raises:
        ChunkingError: If vertices is empty or dimensions mismatch.
    """
    if len(vertices) == 0:
        return []

    ndim = vertices.shape[1]
    if len(chunk_shape) != ndim:
        raise ChunkingError(
            f"chunk_shape length {len(chunk_shape)} != vertex ndim {ndim}"
        )

    cs = np.array(chunk_shape, dtype=np.float64)
    # Compute chunk coords for every vertex — vectorised
    chunk_ints = np.floor(vertices / cs).astype(np.int64)  # (N, D)

    # Find where chunk coords change between consecutive vertices
    changes = np.any(chunk_ints[1:] != chunk_ints[:-1], axis=1)  # (N-1,)
    boundaries = np.flatnonzero(changes) + 1  # indices where new segment starts

    # Split vertex array at boundaries
    segments_idx = np.split(np.arange(len(vertices)), boundaries)

    result: list[tuple[ChunkCoords, npt.NDArray[np.floating]]] = []
    for seg_indices in segments_idx:
        if len(seg_indices) == 0:
            continue
        first = seg_indices[0]
        coord = tuple(int(x) for x in chunk_ints[first])
        result.append((coord, vertices[seg_indices]))

    return result


def cross_chunk_links_for_segments(
    segments: list[tuple[ChunkCoords, npt.NDArray[np.floating]]],
    vg_indices: list[int],
) -> list[CrossChunkLink]:
    """Compute cross-chunk links connecting adjacent polyline segments.

    The link connects the last vertex of segment k to the first vertex
    of segment k+1 (using local indices within each vertex group).

    Args:
        segments: Output of :func:`split_polyline_at_boundaries`.
        vg_indices: Vertex group index assigned to each segment within
            its chunk.  Must be same length as *segments*.

    Returns:
        List of :data:`CrossChunkLink` tuples.  Length is
        ``len(segments) - 1`` (one link per boundary crossing).

    Raises:
        ChunkingError: If lengths don't match.
    """
    if len(vg_indices) != len(segments):
        raise ChunkingError(
            f"vg_indices length {len(vg_indices)} != segments length {len(segments)}"
        )

    links: list[CrossChunkLink] = []
    for i in range(len(segments) - 1):
        chunk_a, verts_a = segments[i]
        chunk_b, verts_b = segments[i + 1]
        # Last vertex of segment i (local index within vertex group)
        last_idx_a = len(verts_a) - 1
        # First vertex of segment i+1
        first_idx_b = 0
        links.append((
            (chunk_a, last_idx_a),
            (chunk_b, first_idx_b),
        ))

    return links


# ===================================================================
# Edge partitioning (graphs)
# ===================================================================

def partition_edges(
    edges: npt.NDArray[np.integer],
    vertex_chunks: npt.NDArray[np.int64],
    vertex_local_indices: npt.NDArray[np.int64],
    chunk_coords_list: list[ChunkCoords],
) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], list[CrossChunkLink]]:
    """Partition edges into intra-chunk and cross-chunk.

    Args:
        edges: ``(M, 2)`` global vertex index pairs.
        vertex_chunks: ``(N,)`` array where ``vertex_chunks[i]`` is the
            index into *chunk_coords_list* for vertex *i*.
        vertex_local_indices: ``(N,)`` array where
            ``vertex_local_indices[i]`` is vertex *i*'s local index
            within its chunk's vertex group.
        chunk_coords_list: Ordered list of unique chunk coordinates.
            ``vertex_chunks[i]`` indexes into this list.

    Returns:
        intra_edges: Dict mapping ``chunk_coords`` → ``(M_local, 2)``
            array of local-index edge pairs (both endpoints in this chunk).
        cross_links: List of :data:`CrossChunkLink` for edges spanning
            chunk boundaries.
    """
    src = edges[:, 0]
    dst = edges[:, 1]

    src_chunk = vertex_chunks[src]  # (M,)
    dst_chunk = vertex_chunks[dst]  # (M,)

    same_chunk = src_chunk == dst_chunk  # (M,)

    # --- Intra-chunk edges ---
    intra_mask = same_chunk
    intra_edges_global = edges[intra_mask]
    intra_src_chunk = src_chunk[intra_mask]

    intra: dict[ChunkCoords, list[tuple[int, int]]] = {}
    # Vectorised: group by chunk
    for chunk_idx in np.unique(intra_src_chunk):
        mask_c = intra_src_chunk == chunk_idx
        e_global = intra_edges_global[mask_c]
        # Remap to local indices
        local_src = vertex_local_indices[e_global[:, 0]]
        local_dst = vertex_local_indices[e_global[:, 1]]
        local_edges = np.stack([local_src, local_dst], axis=1)
        coord = chunk_coords_list[int(chunk_idx)]
        intra[coord] = local_edges

    # --- Cross-chunk edges ---
    cross_mask = ~same_chunk
    cross_edges = edges[cross_mask]
    cross_src_chunk = src_chunk[cross_mask]
    cross_dst_chunk = dst_chunk[cross_mask]

    cross_links: list[CrossChunkLink] = []
    for i in range(len(cross_edges)):
        s, d = int(cross_edges[i, 0]), int(cross_edges[i, 1])
        chunk_a = chunk_coords_list[int(cross_src_chunk[i])]
        chunk_b = chunk_coords_list[int(cross_dst_chunk[i])]
        local_a = int(vertex_local_indices[s])
        local_b = int(vertex_local_indices[d])
        cross_links.append(((chunk_a, local_a), (chunk_b, local_b)))

    return intra, cross_links


def partition_faces(
    faces: npt.NDArray[np.integer],
    vertex_chunks: npt.NDArray[np.int64],
    vertex_local_indices: npt.NDArray[np.int64],
    chunk_coords_list: list[ChunkCoords],
) -> tuple[
    dict[ChunkCoords, npt.NDArray[np.int64]],
    list[list[tuple[ChunkCoords, int]]],
]:
    """Partition faces into intra-chunk and cross-chunk.

    Args:
        faces: ``(F, L)`` global vertex index array.  L=3 for triangles,
            L=4 for quads.
        vertex_chunks: ``(N,)`` chunk index per vertex.
        vertex_local_indices: ``(N,)`` local index per vertex.
        chunk_coords_list: Ordered unique chunk coordinates.

    Returns:
        intra_faces: Dict mapping ``chunk_coords`` → ``(F_local, L)``
            array of local-index face definitions.
        cross_faces: List of cross-chunk face references.  Each element
            is a list of ``L`` tuples ``(chunk_coords, local_vertex_index)``
            — one per face vertex.
    """
    f_count, l = faces.shape

    # Get chunk index for every vertex of every face
    face_chunks = vertex_chunks[faces]  # (F, L)

    # A face is intra-chunk if all vertices are in the same chunk
    all_same = np.all(face_chunks == face_chunks[:, :1], axis=1)  # (F,)

    # --- Intra-chunk faces ---
    intra_faces = faces[all_same]
    intra_chunk_ids = face_chunks[all_same, 0]

    intra: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    for chunk_idx in np.unique(intra_chunk_ids):
        mask = intra_chunk_ids == chunk_idx
        f_global = intra_faces[mask]
        # Remap all vertices to local indices
        local_f = vertex_local_indices[f_global]
        coord = chunk_coords_list[int(chunk_idx)]
        intra[coord] = local_f

    # --- Cross-chunk faces ---
    cross_face_indices = np.flatnonzero(~all_same)
    cross: list[list[tuple[ChunkCoords, int]]] = []
    for fi in cross_face_indices:
        face_ref: list[tuple[ChunkCoords, int]] = []
        for vi in range(l):
            global_vi = int(faces[fi, vi])
            chunk_ci = int(vertex_chunks[global_vi])
            coord = chunk_coords_list[chunk_ci]
            local_vi = int(vertex_local_indices[global_vi])
            face_ref.append((coord, local_vi))
        cross.append(face_ref)

    return intra, cross


# ===================================================================
# Vertex assignment helpers
# ===================================================================

def build_vertex_chunk_mapping(
    chunk_assignments: dict[ChunkCoords, npt.NDArray[np.intp]],
    n_vertices: int,
    chunk_coords_list: list[ChunkCoords] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], list[ChunkCoords]]:
    """Build per-vertex chunk index and local index arrays.

    Given the output of ``chunking.assign_chunks``, builds the arrays
    needed by :func:`partition_edges` and :func:`partition_faces`.

    Args:
        chunk_assignments: ``{chunk_coords: vertex_indices_array}``.
        n_vertices: Total number of vertices.
        chunk_coords_list: If provided, use this ordering.  Otherwise
            sorted from chunk_assignments keys.

    Returns:
        vertex_chunks: ``(N,)`` int64 — index into *chunk_coords_list*
            for each vertex.
        vertex_local_indices: ``(N,)`` int64 — local index within the
            chunk for each vertex (i.e. position within the chunk's
            vertex array).
        chunk_coords_list: The chunk coordinate ordering used.
    """
    if chunk_coords_list is None:
        chunk_coords_list = sorted(chunk_assignments.keys())

    coord_to_idx = {c: i for i, c in enumerate(chunk_coords_list)}

    vertex_chunks = np.full(n_vertices, -1, dtype=np.int64)
    vertex_local_indices = np.full(n_vertices, -1, dtype=np.int64)

    for coord, global_indices in chunk_assignments.items():
        chunk_idx = coord_to_idx[coord]
        for local_idx, global_idx in enumerate(global_indices):
            vertex_chunks[global_idx] = chunk_idx
            vertex_local_indices[global_idx] = local_idx

    if np.any(vertex_chunks == -1):
        missing = int(np.sum(vertex_chunks == -1))
        raise ChunkingError(
            f"{missing} vertices not assigned to any chunk"
        )

    return vertex_chunks, vertex_local_indices, chunk_coords_list


def build_reindex_map(
    chunk_assignments: dict[ChunkCoords, npt.NDArray[np.intp]],
) -> dict[ChunkCoords, dict[int, int]]:
    """Build global→local index mapping per chunk.

    Args:
        chunk_assignments: ``{chunk_coords: vertex_indices_array}``.

    Returns:
        Dict mapping ``chunk_coords`` → dict of ``{global_idx: local_idx}``.
    """
    result: dict[ChunkCoords, dict[int, int]] = {}
    for coord, indices in chunk_assignments.items():
        result[coord] = {int(g): i for i, g in enumerate(indices)}
    return result
