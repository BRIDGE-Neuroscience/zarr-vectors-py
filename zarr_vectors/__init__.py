"""
zarr-vectors: Python utilities for the Zarr Vectors (ZV) format.

Cloud-native storage for points, lines, streamlines, graphs, and meshes
built on Zarr v3.
"""

from zarr_vectors.core.backends import StorageBackend, detect_scheme
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    open_store,
    rebind,
)
from zarr_vectors.lazy.writer import ZVRWriter
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute

__version__ = "0.1.0"

__all__ = [
    "StorageBackend",
    "Group",
    "FsGroup",
    "create_store",
    "open_store",
    "rebind",
    "detect_scheme",
    "RechunkSpec",
    "rechunk",
    "rechunk_by_attribute",
    "ZVRWriter",
    "__version__",
]
