"""Level 3 consistency validation — verify data arrays are internally consistent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    list_chunk_keys, read_all_object_manifests, read_chunk_vertices, read_cross_chunk_links,
)
from zarr_vectors.core.store import (
    get_resolution_level, list_resolution_levels, open_store, read_root_metadata,
)
from zarr_vectors.validate.structure import ValidationResult


def validate_consistency(store_path: str | Path) -> ValidationResult:
    """Level 3: verify internal data consistency."""
    result = ValidationResult(level=3)

    try:
        root = open_store(str(store_path))
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)

    for li in levels:
        prefix = f"resolution_{li}"
        try:
            lg = get_resolution_level(root, li)
        except Exception as e:
            result.add_error(f"{prefix}: cannot open: {e}")
            continue

        chunk_keys = list_chunk_keys(lg)
        if not chunk_keys:
            result.add_warning(f"{prefix}: no chunk data")
            continue

        total_verts = 0
        chunk_vg_counts: dict[tuple, int] = {}

        for ck in chunk_keys:
            try:
                groups = read_chunk_vertices(lg, ck, dtype=np.float32, ndim=ndim)
            except Exception as e:
                result.add_error(f"{prefix}: chunk {ck} decode failed: {e}")
                continue

            chunk_vg_counts[ck] = len(groups)
            for vi, vg in enumerate(groups):
                if vg.ndim != 2 or vg.shape[1] != ndim:
                    result.add_error(f"{prefix}: chunk {ck} vg[{vi}] shape {vg.shape}")
                if np.any(~np.isfinite(vg)):
                    result.add_warning(f"{prefix}: chunk {ck} vg[{vi}] NaN/Inf")
                total_verts += len(vg)

        result.add_pass(f"{prefix}: {len(chunk_keys)} chunks decoded, {total_verts} vertices")

        try:
            la = lg.attrs
            evc = la.get("vertex_count")
            if evc is not None:
                if total_verts != evc:
                    result.add_error(f"{prefix}: metadata vertex_count={evc}, actual={total_verts}")
                else:
                    result.add_pass(f"{prefix}: vertex_count matches")
        except Exception:
            pass

        try:
            manifests = read_all_object_manifests(lg)
            for oid, mf in enumerate(manifests):
                for cc, vgi in mf:
                    if cc not in chunk_vg_counts:
                        result.add_error(f"{prefix}: obj {oid} refs non-existent chunk {cc}")
                    elif vgi >= chunk_vg_counts[cc]:
                        result.add_error(f"{prefix}: obj {oid} refs vg_idx={vgi} >= {chunk_vg_counts[cc]}")
            result.add_pass(f"{prefix}: object_index validated ({len(manifests)} objects)")
        except Exception:
            pass

        try:
            ccl = read_cross_chunk_links(lg)
            for (ca, _), (cb, _) in ccl:
                if ca not in chunk_vg_counts:
                    result.add_error(f"{prefix}: ccl refs non-existent chunk {ca}")
                if cb not in chunk_vg_counts:
                    result.add_error(f"{prefix}: ccl refs non-existent chunk {cb}")
            result.add_pass(f"{prefix}: ccl validated ({len(ccl)} links)")
        except Exception:
            pass

    return result
