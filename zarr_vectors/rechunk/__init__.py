"""N-dimensional rechunking for zarr vectors stores.

Rechunking reorganises data so that objects sharing a common
dimension value (group, attribute bin, object ID range) are
physically contiguous on disk. This turns O(N) manifest scans
into O(1) prefix-based reads for that dimension.

Usage::

    from zarr_vectors.rechunk import rechunk, RechunkSpec

    rechunk("tracts.zarrvectors", RechunkSpec(by="group"))
"""

from zarr_vectors.rechunk.spec import RechunkSpec
from zarr_vectors.rechunk.engine import rechunk

__all__ = ["RechunkSpec", "rechunk"]
