"""ZVF metadata: dataclasses, serialisation, and validation.

This module is pure Python — it does not import zarr.  Metadata objects
are serialised to/from plain dicts (JSON-compatible) and written to
stores by ``core.store``.

Three tiers of metadata:
- **RootMetadata**: format version, spatial index, CRS, conventions
- **LevelMetadata**: per-resolution-level configuration
- **ArrayMetadata**: per-array dtype, encoding, compression

Plus helpers for OME-Zarr–compatible axes and coordinate transforms,
convention validation, and parametric type registries.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from zarr_vectors.constants import (
    DEFAULT_COARSENING_METHOD,
    DEFAULT_REDUCTION_FACTOR,
    FORMAT_VERSION,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_STANDARD,
    CROSS_CHUNK_EXPLICIT,
    VALID_CROSS_CHUNK_STRATEGIES,
    VALID_GEOMETRY_TYPES,
    VALID_LINKS_CONVENTIONS,
    VALID_OBJIDX_CONVENTIONS,
    VALID_ENCODINGS,
    ENCODING_RAW,
)
from zarr_vectors.exceptions import ConventionError, MetadataError


# ===================================================================
# Axes / CRS helpers (OME-Zarr RFC 4/5)
# ===================================================================

def build_axes_metadata(
    dim_names: list[str],
    dim_types: list[str],
    dim_units: list[str],
) -> list[dict[str, str]]:
    """Build OME-Zarr–style axes list.

    Args:
        dim_names: Axis names, e.g. ``["x", "y", "z"]``.
        dim_types: Axis types, e.g. ``["space", "space", "space"]``.
        dim_units: Axis units, e.g. ``["um", "um", "um"]``.

    Returns:
        List of axis dicts: ``[{"name":"x","type":"space","unit":"um"}, ...]``

    Raises:
        MetadataError: If list lengths are inconsistent or no space axes.
    """
    if not (len(dim_names) == len(dim_types) == len(dim_units)):
        raise MetadataError(
            f"Axis list lengths must match: names={len(dim_names)}, "
            f"types={len(dim_types)}, units={len(dim_units)}"
        )
    if not any(t == "space" for t in dim_types):
        raise MetadataError("At least one axis must have type 'space'")
    return [
        {"name": n, "type": t, "unit": u}
        for n, t, u in zip(dim_names, dim_types, dim_units)
    ]


def validate_axes(axes: list[dict[str, str]]) -> None:
    """Validate axes metadata structure.

    Raises:
        MetadataError: If axes are malformed.
    """
    if not axes or len(axes) < 2:
        raise MetadataError("At least 2 axes required")
    space_count = sum(1 for a in axes if a.get("type") == "space")
    if space_count < 2:
        raise MetadataError(f"Need ≥2 space axes, found {space_count}")
    for i, a in enumerate(axes):
        if "name" not in a:
            raise MetadataError(f"Axis {i} missing 'name'")


def build_coordinate_transforms(
    scale: list[float],
    translation: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build coordinateTransformations list for a resolution level.

    Args:
        scale: Scale factors per axis (voxel-to-physical).
        translation: Optional offset per axis.

    Returns:
        List of transform dicts.
    """
    transforms: list[dict[str, Any]] = [{"type": "scale", "scale": list(scale)}]
    if translation is not None:
        transforms.append({"type": "translation", "translation": list(translation)})
    return transforms


# ===================================================================
# Convention validation
# ===================================================================

def validate_conventions(
    links_convention: str,
    object_index_convention: str,
    cross_chunk_strategy: str,
    num_spatial_chunks: int | None = None,
    has_links_array: bool | None = None,
    has_object_index: bool | None = None,
    geometry_type: str | None = None,
) -> None:
    """Validate that convention flags are consistent with store contents.

    Args:
        links_convention: Value of links_convention metadata field.
        object_index_convention: Value of object_index_convention field.
        cross_chunk_strategy: Value of cross_chunk_strategy field.
        num_spatial_chunks: Number of spatial chunks (if known).
        has_links_array: Whether the links array is present (if known).
        has_object_index: Whether the object_index array is present (if known).
        geometry_type: Primary geometry type (if known).

    Raises:
        ConventionError: If any convention is invalid or inconsistent.
    """
    if links_convention not in VALID_LINKS_CONVENTIONS:
        raise ConventionError(
            f"Invalid links_convention '{links_convention}', "
            f"must be one of {VALID_LINKS_CONVENTIONS}"
        )
    if object_index_convention not in VALID_OBJIDX_CONVENTIONS:
        raise ConventionError(
            f"Invalid object_index_convention '{object_index_convention}', "
            f"must be one of {VALID_OBJIDX_CONVENTIONS}"
        )
    if cross_chunk_strategy not in VALID_CROSS_CHUNK_STRATEGIES:
        raise ConventionError(
            f"Invalid cross_chunk_strategy '{cross_chunk_strategy}', "
            f"must be one of {VALID_CROSS_CHUNK_STRATEGIES}"
        )
    # Identity convention only valid for single-chunk stores
    if object_index_convention == "identity":
        if num_spatial_chunks is not None and num_spatial_chunks > 1:
            raise ConventionError(
                "object_index_convention='identity' is only valid for "
                f"single-chunk stores, but found {num_spatial_chunks} chunks"
            )


def requires_links_array(convention: str) -> bool:
    """Return whether the links array is required for this convention."""
    return convention == "explicit"


def requires_object_index(convention: str, num_chunks: int) -> bool:
    """Return whether the object_index array is required."""
    if convention == "identity":
        return False
    return num_chunks > 1 or convention == "standard"


# ===================================================================
# RootMetadata
# ===================================================================

@dataclass
class RootMetadata:
    """Root-level metadata for a ZVF store.

    Attributes:
        format_version: ZVF specification version.
        spatial_index_dims: Axes definitions (OME-Zarr style).
        chunk_shape: Spatial chunk size per dimension.
        bounds: Global bounding box as ``(min_corner, max_corner)``.
        geometry_types: List of geometry types in the store.
        crs: Coordinate reference system dict (or None).
        links_convention: How intra-chunk links are stored.
        object_index_convention: How object→vertex-group mapping works.
        cross_chunk_strategy: How cross-chunk connectivity is handled.
        reduction_factor: Multi-resolution threshold factor.
    """

    spatial_index_dims: list[dict[str, str]]
    chunk_shape: tuple[float, ...]
    bounds: tuple[list[float], list[float]]
    geometry_types: list[str]
    format_version: str = FORMAT_VERSION
    crs: dict[str, Any] | None = None
    links_convention: str = LINKS_IMPLICIT_SEQUENTIAL
    object_index_convention: str = OBJIDX_STANDARD
    cross_chunk_strategy: str = CROSS_CHUNK_EXPLICIT
    reduction_factor: int = DEFAULT_REDUCTION_FACTOR
    base_bin_shape: tuple[float, ...] | None = None
    """Supervoxel bin edge lengths at level 0. When None, defaults to
    chunk_shape (one bin per chunk — backward compatible)."""

    def validate(self) -> None:
        """Validate this metadata object.

        Raises:
            MetadataError: If any field is invalid.
            ConventionError: If conventions are inconsistent.
        """
        if not self.format_version:
            raise MetadataError("format_version is required")

        validate_axes(self.spatial_index_dims)

        sid_ndim = len(self.spatial_index_dims)
        if len(self.chunk_shape) != sid_ndim:
            raise MetadataError(
                f"chunk_shape length {len(self.chunk_shape)} != "
                f"spatial_index_dims length {sid_ndim}"
            )
        if any(c <= 0 for c in self.chunk_shape):
            raise MetadataError("All chunk_shape values must be > 0")

        if len(self.bounds) != 2:
            raise MetadataError("bounds must be (min_corner, max_corner)")
        if len(self.bounds[0]) != sid_ndim or len(self.bounds[1]) != sid_ndim:
            raise MetadataError(
                f"bounds dimensions must match sid_ndim={sid_ndim}"
            )

        for gt in self.geometry_types:
            if gt not in VALID_GEOMETRY_TYPES:
                raise MetadataError(
                    f"Unknown geometry_type '{gt}', "
                    f"must be one of {VALID_GEOMETRY_TYPES}"
                )

        validate_conventions(
            self.links_convention,
            self.object_index_convention,
            self.cross_chunk_strategy,
        )

        if self.reduction_factor < 2:
            raise MetadataError(
                f"reduction_factor must be ≥2, got {self.reduction_factor}"
            )

        # Validate base_bin_shape if set
        if self.base_bin_shape is not None:
            if len(self.base_bin_shape) != sid_ndim:
                raise MetadataError(
                    f"base_bin_shape length {len(self.base_bin_shape)} != "
                    f"spatial_index_dims length {sid_ndim}"
                )
            if any(b <= 0 for b in self.base_bin_shape):
                raise MetadataError("All base_bin_shape values must be > 0")
            # chunk_shape must be an integer multiple of base_bin_shape
            for i, (cs, bs) in enumerate(
                zip(self.chunk_shape, self.base_bin_shape)
            ):
                ratio = cs / bs
                if abs(ratio - round(ratio)) > 1e-9:
                    raise MetadataError(
                        f"chunk_shape[{i}]={cs} is not an integer multiple "
                        f"of base_bin_shape[{i}]={bs} (ratio={ratio:.6f})"
                    )

    @property
    def effective_bin_shape(self) -> tuple[float, ...]:
        """Base bin shape, defaulting to chunk_shape if not set."""
        return self.base_bin_shape if self.base_bin_shape is not None else self.chunk_shape

    @property
    def bins_per_chunk(self) -> tuple[int, ...]:
        """Number of supervoxel bins per chunk in each dimension."""
        bbs = self.effective_bin_shape
        return tuple(
            int(round(cs / bs))
            for cs, bs in zip(self.chunk_shape, bbs)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d = {
            "zarr_vectors": {
                "format_version": self.format_version,
                "spatial_index_dims": self.spatial_index_dims,
                "chunk_shape": list(self.chunk_shape),
                "bounds": [list(self.bounds[0]), list(self.bounds[1])],
                "geometry_types": self.geometry_types,
                "crs": self.crs,
                "links_convention": self.links_convention,
                "object_index_convention": self.object_index_convention,
                "cross_chunk_strategy": self.cross_chunk_strategy,
                "reduction_factor": self.reduction_factor,
            }
        }
        if self.base_bin_shape is not None:
            d["zarr_vectors"]["base_bin_shape"] = list(self.base_bin_shape)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RootMetadata:
        """Deserialise from a dict (as stored in ``.zattrs``).

        Args:
            d: Dict with a ``"zarr_vectors"`` key containing the metadata.

        Raises:
            MetadataError: If the dict is malformed or missing required keys.
        """
        if "zarr_vectors" not in d:
            raise MetadataError("Root metadata must contain 'zarr_vectors' key")
        zv = d["zarr_vectors"]

        required = [
            "format_version", "spatial_index_dims", "chunk_shape",
            "bounds", "geometry_types",
        ]
        for key in required:
            if key not in zv:
                raise MetadataError(f"Missing required root metadata key: '{key}'")

        bbs = zv.get("base_bin_shape")
        return cls(
            format_version=zv["format_version"],
            spatial_index_dims=zv["spatial_index_dims"],
            chunk_shape=tuple(zv["chunk_shape"]),
            bounds=(list(zv["bounds"][0]), list(zv["bounds"][1])),
            geometry_types=zv["geometry_types"],
            crs=zv.get("crs"),
            links_convention=zv.get("links_convention", LINKS_IMPLICIT_SEQUENTIAL),
            object_index_convention=zv.get("object_index_convention", OBJIDX_STANDARD),
            cross_chunk_strategy=zv.get("cross_chunk_strategy", CROSS_CHUNK_EXPLICIT),
            reduction_factor=zv.get("reduction_factor", DEFAULT_REDUCTION_FACTOR),
            base_bin_shape=tuple(bbs) if bbs else None,
        )

    @property
    def sid_ndim(self) -> int:
        """Number of spatial index dimensions."""
        return len(self.spatial_index_dims)


# ===================================================================
# LevelMetadata
# ===================================================================

@dataclass
class LevelMetadata:
    """Per-resolution-level metadata.

    Attributes:
        level: Integer level index (0 = full resolution).
        vertex_count: Total vertices at this level.
        arrays_present: List of array names present at this level.
        bin_shape: Supervoxel edge lengths at this level (None for level 0,
            which inherits base_bin_shape from root).
        bin_ratio: Integer fold-change per axis relative to level 0.
            ``(1,1,1)`` for level 0, ``(2,2,2)`` for 2× coarser bins, etc.
        object_sparsity: Fraction of objects retained at this level (0,1].
        coarsening_method: How this level was generated.
        parent_level: Index of the source level (None for level 0).
    """

    level: int
    vertex_count: int
    arrays_present: list[str]
    bin_shape: tuple[float, ...] | None = None
    bin_ratio: tuple[int, ...] | None = None
    object_sparsity: float = 1.0
    coarsening_method: str = "none"
    parent_level: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "zarr_vectors_level": {
                "level": self.level,
                "bin_shape": list(self.bin_shape) if self.bin_shape else None,
                "bin_ratio": list(self.bin_ratio) if self.bin_ratio else None,
                "object_sparsity": self.object_sparsity,
                "vertex_count": self.vertex_count,
                "coarsening_method": self.coarsening_method,
                "parent_level": self.parent_level,
                "arrays_present": self.arrays_present,
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LevelMetadata:
        """Deserialise from a dict.

        Supports both new ``bin_shape``/``bin_ratio`` fields and the
        legacy ``bin_size`` field for backward compatibility.

        Raises:
            MetadataError: If the dict is malformed.
        """
        if "zarr_vectors_level" not in d:
            raise MetadataError(
                "Level metadata must contain 'zarr_vectors_level' key"
            )
        lv = d["zarr_vectors_level"]
        required = ["level", "vertex_count", "arrays_present"]
        for key in required:
            if key not in lv:
                raise MetadataError(
                    f"Missing required level metadata key: '{key}'"
                )
        # Read bin_shape — fall back to legacy bin_size
        bs = lv.get("bin_shape") or lv.get("bin_size")
        br = lv.get("bin_ratio")
        return cls(
            level=lv["level"],
            vertex_count=lv["vertex_count"],
            arrays_present=lv["arrays_present"],
            bin_shape=tuple(bs) if bs else None,
            bin_ratio=tuple(int(x) for x in br) if br else None,
            object_sparsity=lv.get("object_sparsity", 1.0),
            coarsening_method=lv.get("coarsening_method", "none"),
            parent_level=lv.get("parent_level"),
        )

    def validate(self) -> None:
        """Validate this metadata object.

        Raises:
            MetadataError: If fields are invalid.
        """
        if self.level < 0:
            raise MetadataError(f"Level must be ≥0, got {self.level}")
        if self.vertex_count < 0:
            raise MetadataError(
                f"vertex_count must be ≥0, got {self.vertex_count}"
            )
        if not (0.0 < self.object_sparsity <= 1.0):
            raise MetadataError(
                f"object_sparsity must be in (0, 1], got {self.object_sparsity}"
            )
        if self.level == 0:
            # Level 0: bin_shape/bin_ratio may be None (inherits from root)
            if self.parent_level is not None:
                raise MetadataError("Level 0 should have parent_level=None")
        else:
            if self.bin_shape is None:
                raise MetadataError(
                    f"Level {self.level} must have a bin_shape"
                )
            if self.parent_level is None:
                raise MetadataError(
                    f"Level {self.level} must have a parent_level"
                )
            if self.bin_ratio is not None:
                if any(r < 1 for r in self.bin_ratio):
                    raise MetadataError(
                        f"bin_ratio values must be ≥1, got {self.bin_ratio}"
                    )


# ===================================================================
# Bin shape / ratio helpers
# ===================================================================

def compute_bin_shape(
    base_bin_shape: tuple[float, ...],
    bin_ratio: tuple[int, ...],
) -> tuple[float, ...]:
    """Compute the bin shape at a coarser level.

    Args:
        base_bin_shape: Supervoxel edge lengths at level 0.
        bin_ratio: Integer fold-change per dimension.

    Returns:
        ``tuple(base * ratio for each dimension)``.

    Raises:
        MetadataError: If lengths don't match.
    """
    if len(base_bin_shape) != len(bin_ratio):
        raise MetadataError(
            f"base_bin_shape has {len(base_bin_shape)} dims, "
            f"bin_ratio has {len(bin_ratio)} dims"
        )
    return tuple(b * r for b, r in zip(base_bin_shape, bin_ratio))


def compute_bin_ratio(
    base_bin_shape: tuple[float, ...],
    level_bin_shape: tuple[float, ...],
) -> tuple[int, ...]:
    """Compute the bin ratio from base and level bin shapes.

    Args:
        base_bin_shape: Supervoxel edge lengths at level 0.
        level_bin_shape: Supervoxel edge lengths at this level.

    Returns:
        Integer ratio per dimension.

    Raises:
        MetadataError: If the ratio is not integer in any dimension.
    """
    if len(base_bin_shape) != len(level_bin_shape):
        raise MetadataError(
            f"base_bin_shape has {len(base_bin_shape)} dims, "
            f"level_bin_shape has {len(level_bin_shape)} dims"
        )
    ratios: list[int] = []
    for i, (base, level) in enumerate(zip(base_bin_shape, level_bin_shape)):
        r = level / base
        if abs(r - round(r)) > 1e-9:
            raise MetadataError(
                f"Dimension {i}: level_bin_shape={level} / "
                f"base_bin_shape={base} = {r:.6f} is not an integer"
            )
        ratios.append(int(round(r)))
    return tuple(ratios)


def validate_bin_shape_divides_chunk(
    chunk_shape: tuple[float, ...],
    bin_shape: tuple[float, ...],
) -> None:
    """Validate that chunk_shape is an integer multiple of bin_shape.

    Raises:
        MetadataError: If any dimension is not evenly divisible.
    """
    for i, (cs, bs) in enumerate(zip(chunk_shape, bin_shape)):
        ratio = cs / bs
        if abs(ratio - round(ratio)) > 1e-9:
            raise MetadataError(
                f"chunk_shape[{i}]={cs} is not an integer multiple "
                f"of bin_shape[{i}]={bs} (ratio={ratio:.6f})"
            )


# ===================================================================
# ArrayMetadata
# ===================================================================

@dataclass
class ArrayMetadata:
    """Per-array metadata (stored in each array's ``.zattrs``).

    Attributes:
        name: Array name (e.g. ``"vertices"``, ``"attributes/radius"``).
        dtype: Numpy dtype string (e.g. ``"float32"``, ``"int64"``).
        encoding: ``"raw"`` or ``"draco"``.
        compression_codec: Codec name or None.
        chunk_shape: Zarr chunk shape or None (for ragged).
        channel_names: For attribute arrays, list of channel names.
        channel_dtype: For attribute arrays, the channel dtype.
    """

    name: str
    dtype: str
    encoding: str = ENCODING_RAW
    compression_codec: str | None = None
    chunk_shape: tuple[int, ...] | None = None
    channel_names: list[str] | None = None
    channel_dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "encoding": self.encoding,
        }
        if self.compression_codec is not None:
            d["compression_codec"] = self.compression_codec
        if self.chunk_shape is not None:
            d["chunk_shape"] = list(self.chunk_shape)
        if self.channel_names is not None:
            d["channel_names"] = self.channel_names
        if self.channel_dtype is not None:
            d["channel_dtype"] = self.channel_dtype
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArrayMetadata:
        """Deserialise from a dict."""
        if "name" not in d or "dtype" not in d:
            raise MetadataError("ArrayMetadata requires 'name' and 'dtype'")
        cs = d.get("chunk_shape")
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            encoding=d.get("encoding", ENCODING_RAW),
            compression_codec=d.get("compression_codec"),
            chunk_shape=tuple(cs) if cs else None,
            channel_names=d.get("channel_names"),
            channel_dtype=d.get("channel_dtype"),
        )

    def validate(self) -> None:
        """Validate this metadata object."""
        if self.encoding not in VALID_ENCODINGS:
            raise MetadataError(
                f"Invalid encoding '{self.encoding}', "
                f"must be one of {VALID_ENCODINGS}"
            )


# ===================================================================
# ParametricTypeDef
# ===================================================================

@dataclass
class ParametricTypeDef:
    """Definition of a parametric object type for the type registry.

    Attributes:
        type_id: Integer type tag stored in the objects array.
        name: Human-readable name (e.g. ``"plane"``, ``"line"``).
        coefficients: Ordered list of coefficient names.
    """

    type_id: int
    name: str
    coefficients: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "coefficients": self.coefficients,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParametricTypeDef:
        return cls(
            type_id=d["type_id"],
            name=d["name"],
            coefficients=d["coefficients"],
        )


# ===================================================================
# Standard parametric type definitions
# ===================================================================

PARAMETRIC_PLANE = ParametricTypeDef(
    type_id=0, name="plane", coefficients=["A", "B", "C", "D"]
)
PARAMETRIC_LINE = ParametricTypeDef(
    type_id=1, name="line", coefficients=["x0", "y0", "z0", "dx", "dy", "dz"]
)
PARAMETRIC_SPHERE = ParametricTypeDef(
    type_id=2, name="sphere", coefficients=["cx", "cy", "cz", "r"]
)

DEFAULT_PARAMETRIC_TYPES = [PARAMETRIC_PLANE, PARAMETRIC_LINE, PARAMETRIC_SPHERE]


# ===================================================================
# Serialisation helpers (for store read/write)
# ===================================================================

def serialise_parametric_types(
    types: list[ParametricTypeDef],
) -> dict[str, Any]:
    """Serialise parametric type registry to a dict.

    Returns:
        ``{"parametric_types": {"0": {...}, "1": {...}, ...}}``
    """
    return {
        "parametric_types": {
            str(t.type_id): {"name": t.name, "coefficients": t.coefficients}
            for t in types
        }
    }


def deserialise_parametric_types(
    d: dict[str, Any],
) -> list[ParametricTypeDef]:
    """Deserialise parametric type registry from a dict.

    Args:
        d: Dict with ``"parametric_types"`` key.

    Returns:
        List of ParametricTypeDef objects.
    """
    if "parametric_types" not in d:
        return []
    registry = d["parametric_types"]
    types: list[ParametricTypeDef] = []
    for tid_str, info in registry.items():
        types.append(ParametricTypeDef(
            type_id=int(tid_str),
            name=info["name"],
            coefficients=info["coefficients"],
        ))
    return sorted(types, key=lambda t: t.type_id)
