"""Lazy zarr vectors store access.

Opens a store without reading data. All vertex, attribute, and object
access is deferred until ``.compute()`` is called, optionally using
Dask for parallel I/O.

Usage::

    from zarr_vectors.lazy import open_zvr

    zvr = open_zvr("scan.zarrvectors")
    zvr[0].vertices.compute()          # materialise all level-0 vertices
    zvr[0].vertices[0, 0, 0].compute() # single chunk
    zvr[0].attributes["intensity"].compute()
"""

from zarr_vectors.lazy.store import ZVRStore, open_zvr
from zarr_vectors.lazy.views import ZVRView, ZVRPolylineCollection

__all__ = ["ZVRStore", "ZVRView", "ZVRPolylineCollection", "open_zvr"]
