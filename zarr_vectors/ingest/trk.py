"""Ingest streamlines from TrackVis TRK files into zarr vectors.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.typing import ChunkShape


def ingest_trk(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Ingest a TRK file into a zarr vectors streamline store.

    Args:
        input_path: Path to the input .trk file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.

    Returns:
        Summary dict from :func:`write_polylines`.

    Raises:
        IngestError: If nibabel is not installed or the file is unreadable.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise IngestError(
            "nibabel is required for TRK ingest. "
            "Install with: pip install nibabel"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        trk = nib.streamlines.load(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read TRK '{input_path}': {e}") from e

    streamlines = trk.streamlines
    np_dtype = np.dtype(dtype)
    polylines = [np.asarray(s, dtype=np_dtype) for s in streamlines]

    if len(polylines) == 0:
        raise IngestError(f"TRK file contains no streamlines: {input_path}")

    # Extract per-vertex scalars if present
    vertex_attributes: dict[str, list[np.ndarray]] | None = None
    if hasattr(trk, "tractogram") and trk.tractogram.data_per_point:
        vertex_attributes = {}
        for key, values in trk.tractogram.data_per_point.items():
            vertex_attributes[key] = [
                np.asarray(v, dtype=np.float32) for v in values
            ]

    # Extract per-streamline properties if present
    object_attributes: dict[str, np.ndarray] | None = None
    if hasattr(trk, "tractogram") and trk.tractogram.data_per_streamline:
        object_attributes = {}
        for key, values in trk.tractogram.data_per_streamline.items():
            object_attributes[key] = np.asarray(values, dtype=np.float32)

    return write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )
