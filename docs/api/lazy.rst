Lazy loading
============

The ``ZarrVectorStore`` class provides a lazy, level-of-detail-aware
interface for reading ZVF stores without loading all data into memory.
It is the recommended interface for remote stores, large datasets, and
interactive applications.

.. automodule:: zarr_vectors.lazy
   :members:
   :undoc-members:
   :show-inheritance:

ZarrVectorStore
---------------

.. autoclass:: zarr_vectors.lazy.ZarrVectorStore
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__

LazyArray
---------

.. autoclass:: zarr_vectors.lazy.LazyArray
   :members:
   :undoc-members:
   :show-inheritance:
