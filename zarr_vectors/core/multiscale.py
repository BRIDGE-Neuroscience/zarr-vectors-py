"""OME-Zarr compatible multiscale metadata for zarr vectors stores.

Generates and reads ``multiscales`` metadata blocks in the root
``.zattrs``, following the OME-NGFF spec (v0.5).  This allows
OME-Zarr-aware viewers to discover the resolution pyramid structure.

The metadata is informational and coexists with zarr vectors-specific
metadata — it does not replace the ``zarr_vectors_level`` entries.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.core.metadata import (
    LevelMetadata,
    compute_bin_ratio,
)
from zarr_vectors.core.store import (
    FsGroup,
    list_resolution_levels,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.constants import RESOLUTION_PREFIX
from zarr_vectors.exceptions import MetadataError


def write_multiscale_metadata(root: FsGroup) -> dict[str, Any]:
    """Generate and write OME-Zarr multiscale metadata to root .zattrs.

    Reads all existing resolution levels and their bin shapes to
    compute scale and translation transforms.

    Args:
        root: Root store group.

    Returns:
        The ``multiscales`` list written to ``.zattrs``.
    """
    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)

    # Build axes from spatial_index_dims
    axes = []
    for ax in meta.spatial_index_dims:
        axes.append({
            "name": ax.get("name", f"dim{len(axes)}"),
            "type": ax.get("type", "space"),
            "unit": ax.get("unit", "unit"),
        })

    # Build datasets array (one per level)
    datasets: list[dict[str, Any]] = []
    for lvl in levels:
        path = f"{RESOLUTION_PREFIX}{lvl}"

        if lvl == 0:
            # Level 0: scale = 1.0, translation = base_bin/2
            scale = [1.0] * ndim
            translation = [bs / 2.0 for bs in base_bin]
        else:
            try:
                lm = read_level_metadata(root, lvl)
                if lm.bin_ratio is not None:
                    scale = [float(r) for r in lm.bin_ratio]
                elif lm.bin_shape is not None:
                    ratio = compute_bin_ratio(base_bin, lm.bin_shape)
                    scale = [float(r) for r in ratio]
                else:
                    scale = [1.0] * ndim

                if lm.bin_shape is not None:
                    translation = [bs / 2.0 for bs in lm.bin_shape]
                else:
                    translation = [bs / 2.0 for bs in base_bin]
            except Exception:
                scale = [1.0] * ndim
                translation = [bs / 2.0 for bs in base_bin]

        transforms: list[dict[str, Any]] = [
            {"type": "scale", "scale": scale},
        ]
        if any(t != 0 for t in translation):
            transforms.append({"type": "translation", "translation": translation})

        datasets.append({
            "path": path,
            "coordinateTransformations": transforms,
        })

    multiscales = [{
        "version": "0.5",
        "name": "default",
        "type": "zarr_vectors",
        "axes": axes,
        "datasets": datasets,
    }]

    # Write to root attrs (alongside existing zarr_vectors metadata)
    attrs = root.attrs.to_dict()
    attrs["multiscales"] = multiscales
    root.attrs.update(attrs)

    return multiscales


def read_multiscale_metadata(root: FsGroup) -> list[dict[str, Any]] | None:
    """Read OME-Zarr multiscale metadata from root .zattrs.

    Args:
        root: Root store group.

    Returns:
        The ``multiscales`` list, or None if not present.
    """
    attrs = root.attrs.to_dict()
    return attrs.get("multiscales")


def get_level_scale(
    root: FsGroup,
    level: int,
) -> list[float] | None:
    """Get the scale factors for a specific level from multiscale metadata.

    Args:
        root: Root store group.
        level: Resolution level index.

    Returns:
        List of scale factors per dimension, or None if not available.
    """
    ms = read_multiscale_metadata(root)
    if ms is None:
        return None

    path = f"{RESOLUTION_PREFIX}{level}"
    for ms_entry in ms:
        for ds in ms_entry.get("datasets", []):
            if ds.get("path") == path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        return t.get("scale")
    return None


def get_level_translation(
    root: FsGroup,
    level: int,
) -> list[float] | None:
    """Get the translation offset for a specific level.

    Args:
        root: Root store group.
        level: Resolution level index.

    Returns:
        List of translation offsets per dimension, or None.
    """
    ms = read_multiscale_metadata(root)
    if ms is None:
        return None

    path = f"{RESOLUTION_PREFIX}{level}"
    for ms_entry in ms:
        for ds in ms_entry.get("datasets", []):
            if ds.get("path") == path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "translation":
                        return t.get("translation")
    return None
