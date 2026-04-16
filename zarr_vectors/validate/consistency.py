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

        # Determine effective bin shape for this level
        try:
            la = lg.attrs
            level_bin_shape = la.get("bin_shape") or la.get("bin_size")
            if level_bin_shape is not None:
                level_bin_shape = tuple(float(x) for x in level_bin_shape)
            elif li == 0:
                level_bin_shape = meta.effective_bin_shape
            else:
                level_bin_shape = meta.chunk_shape  # unknown — skip bin checks
        except Exception:
            level_bin_shape = meta.chunk_shape

        # Compute bins_per_chunk for this level
        level_bins_per_chunk = tuple(
            int(round(cs / bs))
            for cs, bs in zip(meta.chunk_shape, level_bin_shape)
        )
        max_vgs = 1
        for b in level_bins_per_chunk:
            max_vgs *= b

        # Per-bin VG layout only applies to undifferentiated point-cloud stores.
        # Polylines / lines / graphs / meshes use VGs to represent segments,
        # endpoints, or per-object partitions — not bins.
        geom_types = meta.geometry_types or []
        is_point_cloud_only = (
            "point_cloud" in geom_types
            and not any(gt in geom_types for gt in [
                "polyline", "streamline", "line", "graph",
                "skeleton", "mesh",
            ])
        )
        # Also skip if the store has object_index (VGs are per-object, not per-bin)
        try:
            has_object_index = "object_index" in lg
        except Exception:
            has_object_index = False

        check_bin_layout = is_point_cloud_only and not has_object_index
        chunks_checked_for_bin_bounds = 0

        for ck in chunk_keys:
            try:
                groups = read_chunk_vertices(lg, ck, dtype=np.float32, ndim=ndim)
            except Exception as e:
                result.add_error(f"{prefix}: chunk {ck} decode failed: {e}")
                continue

            chunk_vg_counts[ck] = len(groups)

            # Check VG count doesn't exceed bins_per_chunk
            # (only for undifferentiated point clouds with explicit bins)
            has_bins = any(b > 1 for b in level_bins_per_chunk)
            if check_bin_layout and has_bins and len(groups) > max_vgs:
                result.add_error(
                    f"{prefix}: chunk {ck} has {len(groups)} VGs, "
                    f"exceeds bins_per_chunk product {max_vgs}"
                )

            for vi, vg in enumerate(groups):
                if vg.ndim != 2 or vg.shape[1] != ndim:
                    result.add_error(f"{prefix}: chunk {ck} vg[{vi}] shape {vg.shape}")
                if len(vg) > 0 and np.any(~np.isfinite(vg)):
                    result.add_warning(f"{prefix}: chunk {ck} vg[{vi}] NaN/Inf")
                total_verts += len(vg)

            # Spot-check bin bounds for point clouds only
            if check_bin_layout and has_bins and chunks_checked_for_bin_bounds < 3:
                from zarr_vectors.spatial.chunking import vg_index_to_bin
                chunks_checked_for_bin_bounds += 1
                for vi, vg in enumerate(groups):
                    if len(vg) == 0:
                        continue
                    try:
                        bin_coords = vg_index_to_bin(vi, ck, level_bins_per_chunk)
                    except Exception:
                        continue
                    bin_lo = np.array(
                        [bc * bs for bc, bs in zip(bin_coords, level_bin_shape)],
                        dtype=np.float64,
                    )
                    bin_hi = bin_lo + np.array(level_bin_shape, dtype=np.float64)
                    # Allow small tolerance for float rounding
                    tol = 1e-4
                    out_of_bin = np.any(
                        (vg < bin_lo - tol) | (vg >= bin_hi + tol),
                        axis=1,
                    )
                    n_out = int(np.sum(out_of_bin))
                    if n_out > 0:
                        result.add_warning(
                            f"{prefix}: chunk {ck} vg[{vi}] has {n_out} points "
                            f"outside bin {bin_coords} bounds"
                        )

        result.add_pass(
            f"{prefix}: {len(chunk_keys)} chunks decoded, {total_verts} vertices"
        )

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
