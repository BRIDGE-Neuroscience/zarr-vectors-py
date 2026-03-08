"""Custom exceptions for the zarr-vectors package.

All exceptions inherit from ZVFError so callers can catch broadly
or narrowly as needed.
"""


class ZVFError(Exception):
    """Base exception for all zarr-vectors errors."""


class StoreError(ZVFError):
    """Raised when a ZVF store cannot be created, opened, or is structurally invalid."""


class MetadataError(ZVFError):
    """Raised when metadata is missing, malformed, or fails schema validation."""


class ArrayError(ZVFError):
    """Raised when an array has wrong shape, dtype, or cannot be read/written."""


class ChunkingError(ZVFError):
    """Raised when spatial chunk assignment fails or produces invalid results."""


class ConventionError(ZVFError):
    """Raised when convention flags are inconsistent with the store contents.

    For example, using object_index_convention='identity' with multiple
    spatial chunks, or links_convention='implicit_sequential' on a mesh.
    """


class ValidationError(ZVFError):
    """Raised when a store fails conformance or consistency validation."""


class IngestError(ZVFError):
    """Raised when reading from an external format fails during ingest."""


class ExportError(ZVFError):
    """Raised when writing to an external format fails during export."""


class CoarseningError(ZVFError):
    """Raised when multi-resolution pyramid construction fails."""


class DracoError(ZVFError):
    """Raised when Draco encoding or decoding fails."""
