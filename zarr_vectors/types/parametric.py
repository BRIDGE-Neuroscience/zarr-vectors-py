"""Parametric object I/O for zarr vectors stores.

Parametric objects are algebraic entities (planes, lines, spheres)
defined by coefficients rather than sampled vertices.  They live in
the ``/parametric/`` group at the store root, outside the resolution
level hierarchy — they are resolution-independent.

Each object is stored as a type tag followed by its coefficients.
A type registry in ``/parametric/.zattrs`` maps type IDs to names
and coefficient schemas so readers can parse any object without
hardcoded knowledge of every type.

Supported built-in types:

- **Plane** (type 0): ``Ax + By + Cz + D = 0`` → coefficients ``[A, B, C, D]``
- **Line** (type 1): ``P(t) = P₀ + t·d`` → coefficients ``[x0, y0, z0, dx, dy, dz]``
- **Sphere** (type 2): ``|P - C|² = r²`` → coefficients ``[cx, cy, cz, r]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import PARAMETRIC_GROUP
from zarr_vectors.core.metadata import (
    DEFAULT_PARAMETRIC_TYPES,
    PARAMETRIC_LINE,
    PARAMETRIC_PLANE,
    PARAMETRIC_SPHERE,
    ParametricTypeDef,
    deserialise_parametric_types,
    serialise_parametric_types,
)
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    get_parametric_group,
    open_store,
    read_root_metadata,
    write_parametric_types,
    read_parametric_types,
)
from zarr_vectors.exceptions import ArrayError, MetadataError


# ===================================================================
# Write
# ===================================================================

def write_parametric_objects(
    store_path: str,
    objects: list[dict[str, Any]],
    *,
    object_attributes: dict[str, npt.NDArray] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    custom_types: list[ParametricTypeDef] | None = None,
    create_new_store: bool = False,
    store_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write parametric objects to a zarr vectors store.

    Can write to an existing store (appending to ``/parametric/``) or
    create a new store if ``create_new_store=True``.

    Args:
        store_path: Path to the store.
        objects: List of dicts, each with:
            - ``"type"``: type name (``"plane"``, ``"line"``, ``"sphere"``)
              or integer type ID.
            - ``"coefficients"``: list of floats matching the type's schema.
            - Optionally ``"name"``: human-readable name for the object.
        object_attributes: Per-object attributes as ``{name: (O,) or (O,C)}``.
        groups: Group memberships ``{group_id: [object_indices]}``.
        group_attributes: Per-group attributes ``{name: (G,) or (G,C)}``.
        custom_types: Additional parametric type definitions beyond the
            built-in plane/line/sphere.
        create_new_store: If True, create a new store.  Otherwise open
            existing.
        store_kwargs: Extra kwargs passed to ``create_store`` (e.g.
            ``root_metadata`` fields).

    Returns:
        Summary dict with ``object_count``, ``type_counts``.

    Raises:
        ArrayError: If an object has an unknown type or wrong coefficient count.
    """
    # Build type registry
    all_types = list(DEFAULT_PARAMETRIC_TYPES)
    if custom_types:
        all_types.extend(custom_types)

    type_by_name = {t.name: t for t in all_types}
    type_by_id = {t.type_id: t for t in all_types}

    # Open or create store
    if create_new_store:
        from zarr_vectors.core.metadata import RootMetadata
        kw = store_kwargs or {}
        if "spatial_index_dims" not in kw:
            kw["spatial_index_dims"] = [
                {"name": "x", "type": "space", "unit": "unit"},
                {"name": "y", "type": "space", "unit": "unit"},
                {"name": "z", "type": "space", "unit": "unit"},
            ]
        if "chunk_shape" not in kw:
            kw["chunk_shape"] = (1000.0, 1000.0, 1000.0)
        if "bounds" not in kw:
            kw["bounds"] = ([0, 0, 0], [1000, 1000, 1000])
        if "geometry_types" not in kw:
            kw["geometry_types"] = ["point_cloud"]
        root = create_store(store_path, RootMetadata(**kw))
    else:
        root = open_store(store_path, mode="r+")

    # Write type registry
    write_parametric_types(root, all_types)

    para = get_parametric_group(root)

    # Encode objects
    n_objects = len(objects)
    type_counts: dict[str, int] = {}
    encoded_rows: list[list[float]] = []
    names: list[str] = []

    for i, obj in enumerate(objects):
        # Resolve type
        obj_type = obj.get("type")
        if isinstance(obj_type, str):
            if obj_type not in type_by_name:
                raise ArrayError(
                    f"Object {i}: unknown type '{obj_type}'. "
                    f"Known types: {list(type_by_name.keys())}"
                )
            tdef = type_by_name[obj_type]
        elif isinstance(obj_type, int):
            if obj_type not in type_by_id:
                raise ArrayError(
                    f"Object {i}: unknown type ID {obj_type}. "
                    f"Known IDs: {list(type_by_id.keys())}"
                )
            tdef = type_by_id[obj_type]
        else:
            raise ArrayError(
                f"Object {i}: 'type' must be a string name or integer ID, "
                f"got {type(obj_type)}"
            )

        coeffs = obj.get("coefficients", [])
        if len(coeffs) != len(tdef.coefficients):
            raise ArrayError(
                f"Object {i} (type '{tdef.name}'): expected "
                f"{len(tdef.coefficients)} coefficients "
                f"({tdef.coefficients}), got {len(coeffs)}"
            )

        # Row: [type_id, coeff0, coeff1, ...]
        row = [float(tdef.type_id)] + [float(c) for c in coeffs]
        encoded_rows.append(row)

        type_counts[tdef.name] = type_counts.get(tdef.name, 0) + 1
        names.append(obj.get("name", f"{tdef.name}_{i}"))

    # Pad rows to same length (different types may have different coeff counts)
    max_len = max(len(r) for r in encoded_rows) if encoded_rows else 0
    for r in encoded_rows:
        while len(r) < max_len:
            r.append(float("nan"))

    # Write encoded objects as a dense float64 array
    if encoded_rows:
        data = np.array(encoded_rows, dtype=np.float64)
        para.write_bytes("objects", "data", data.tobytes())
        para.write_array_meta("objects", {
            "zvf_array": "parametric_objects",
            "num_objects": n_objects,
            "max_row_length": max_len,
            "dtype": "float64",
        })

    # Write names as object attribute
    if names:
        # Store names as a newline-joined byte string
        names_str = "\n".join(names)
        para.write_bytes("names", "data", names_str.encode("utf-8"))
        para.write_array_meta("names", {
            "zvf_array": "parametric_names",
            "num_objects": n_objects,
        })

    # Write additional object attributes
    if object_attributes:
        for attr_name, attr_data in object_attributes.items():
            arr = np.asarray(attr_data)
            full_name = f"object_attributes/{attr_name}"
            para.require_group("object_attributes")
            para.write_bytes(full_name, "data", arr.tobytes())
            para.write_array_meta(full_name, {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            })

    # Write groupings
    if groups:
        from zarr_vectors.encoding.ragged import encode_ragged_ints
        max_gid = max(groups.keys())
        group_list = [
            np.array(groups.get(gid, []), dtype=np.int64)
            for gid in range(max_gid + 1)
        ]
        raw, offsets = encode_ragged_ints(group_list)
        para.write_bytes("groupings", "data", raw)
        para.write_bytes("groupings", "offsets", offsets.tobytes())
        para.write_array_meta("groupings", {
            "num_groups": max_gid + 1,
        })

    if group_attributes:
        para.require_group("groupings_attributes")
        for attr_name, attr_data in group_attributes.items():
            arr = np.asarray(attr_data)
            full_name = f"groupings_attributes/{attr_name}"
            para.write_bytes(full_name, "data", arr.tobytes())
            para.write_array_meta(full_name, {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            })

    return {
        "object_count": n_objects,
        "type_counts": type_counts,
    }


# ===================================================================
# Read
# ===================================================================

def read_parametric_objects(
    store_path: str,
) -> list[dict[str, Any]]:
    """Read all parametric objects from a zarr vectors store.

    Returns:
        List of dicts, each with:
        - ``"type"``: type name string
        - ``"type_id"``: integer type ID
        - ``"coefficients"``: list of float coefficient values
        - ``"coefficient_names"``: list of coefficient name strings
        - ``"name"``: object name (if stored)
    """
    root = open_store(store_path)
    para = get_parametric_group(root)

    # Read type registry
    types = read_parametric_types(root)
    type_by_id = {t.type_id: t for t in types}

    # Read encoded objects
    try:
        meta = para.read_array_meta("objects")
    except Exception:
        return []

    if "num_objects" not in meta:
        return []

    n_objects = meta["num_objects"]
    max_row_len = meta["max_row_length"]
    dtype = np.dtype(meta.get("dtype", "float64"))

    raw = para.read_bytes("objects", "data")
    data = np.frombuffer(raw, dtype=dtype).reshape(n_objects, max_row_len)

    # Read names if available
    names: list[str] = []
    try:
        names_raw = para.read_bytes("names", "data")
        names = names_raw.decode("utf-8").split("\n")
    except Exception:
        names = [f"object_{i}" for i in range(n_objects)]

    # Decode objects
    result: list[dict[str, Any]] = []
    for i in range(n_objects):
        row = data[i]
        type_id = int(row[0])
        tdef = type_by_id.get(type_id)
        if tdef is None:
            result.append({
                "type": f"unknown_{type_id}",
                "type_id": type_id,
                "coefficients": row[1:].tolist(),
                "coefficient_names": [],
                "name": names[i] if i < len(names) else f"object_{i}",
            })
            continue

        n_coeffs = len(tdef.coefficients)
        coeffs = row[1 : 1 + n_coeffs].tolist()

        result.append({
            "type": tdef.name,
            "type_id": type_id,
            "coefficients": coeffs,
            "coefficient_names": tdef.coefficients,
            "name": names[i] if i < len(names) else f"object_{i}",
        })

    return result
