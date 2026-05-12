"""Attribute-based chunking helpers.

When a writer is invoked with ``chunk_by_attribute="<name>"``, the
per-vertex categorical attribute named ``<name>`` becomes the **leading
chunk axis**.  Chunk keys go from ``z.y.x`` to ``attr_bin.z.y.x``, and
each chunk holds vertices that share both the attribute value and a
spatial cell.

This module is the single source of truth for that mapping: it converts
a per-vertex value array into a contiguous bin index, and produces the
ordered list mapping bin → original value (which gets persisted in
:class:`LevelMetadata.chunk_attribute_values`).

v1 is **categorical only**:

- Accepts integer or string (object/bytes/unicode) dtypes.
- Rejects floating-point arrays — for continuous values use the rechunk
  module's bin-edge support instead.

Per-vertex semantics: an "object" (e.g. a polyline) whose vertices have
mixed attribute values will be split across the corresponding chunks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ArrayError


def assign_attribute_bins(
    values: npt.NDArray,
) -> tuple[npt.NDArray[np.int64], list[Any]]:
    """Map a per-vertex categorical attribute array to bin indices.

    The output bin indices are dense (``[0, K)``), and the returned
    ``bin_values`` list lets a reader recover the original value for any
    chunk (``bin_values[i]`` is the value of bin ``i``).  Bin order is
    the lexicographic / sorted order of unique values.

    Args:
        values: ``(N,)`` per-vertex attribute array.  Integer or string
            dtype.

    Returns:
        ``(bin_indices, bin_values)`` where:
        - ``bin_indices``: ``(N,) int64`` array in ``[0, K)``.
        - ``bin_values``: list of length ``K``; ``bin_values[i]`` is the
          attribute value mapped to bin ``i``, in sorted order.

    Raises:
        ArrayError: If ``values`` has the wrong shape or an unsupported
            dtype (e.g. floating-point).
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ArrayError(
            f"chunk_by_attribute requires a 1D per-vertex array, "
            f"got shape {arr.shape}"
        )

    kind = arr.dtype.kind
    # Accept ints (i/u), booleans (b), strings (U/S), and Python objects (O).
    if kind in ("f", "c"):
        raise ArrayError(
            f"chunk_by_attribute is categorical-only in v1; got "
            f"floating-point dtype {arr.dtype}.  For continuous values, "
            f"use the rechunk module's bin-edge support instead."
        )
    if kind not in ("i", "u", "b", "U", "S", "O"):
        raise ArrayError(
            f"Unsupported attribute dtype {arr.dtype} for chunk_by_attribute "
            f"(supported: int, uint, bool, str)"
        )

    # np.unique gives a sorted unique array + inverse (the bin index for
    # each input element).  Inverse is the right shape and is already a
    # dense [0, K) encoding.
    unique_vals, inverse = np.unique(arr, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False).reshape(-1)
    # Convert numpy scalars to native Python values for the value list,
    # so it round-trips through JSON metadata.
    bin_values = [_to_native(v) for v in unique_vals]
    return inverse, bin_values


def _to_native(v: Any) -> Any:
    """Convert a numpy scalar to a JSON-serialisable Python value."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        # Bytestrings come from S-dtype arrays; decode for JSON.
        return v.decode("utf-8")
    return v


def compute_chunk_dim_names(
    attribute_name: str,
    sid_ndim: int,
    spatial_dim_names: list[str] | None = None,
) -> list[str]:
    """Compute the ``chunk_dims`` list for an attribute-chunked level.

    Args:
        attribute_name: The leading-axis attribute name.
        sid_ndim: Number of spatial dimensions.
        spatial_dim_names: Optional explicit spatial axis names taken
            from root metadata.  Falls back to ``["dim0", "dim1", ...]``.

    Returns:
        ``[attribute_name, *spatial_dim_names]``.
    """
    if spatial_dim_names is None:
        spatial_dim_names = [f"dim{i}" for i in range(sid_ndim)]
    return [attribute_name, *spatial_dim_names]
