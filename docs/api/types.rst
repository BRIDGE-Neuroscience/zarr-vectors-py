Geometry types
==============

Read and write functions for each ZVF geometry type. All write functions
accept either a local path string, a ``zarr.storage.Store`` object, or
an fsspec mapper as the first argument. All read functions return a typed
result dictionary documented on each function's page.

Point clouds
------------

.. automodule:: zarr_vectors.types.points
   :members:
   :undoc-members:
   :show-inheritance:

Lines
-----

.. automodule:: zarr_vectors.types.lines
   :members:
   :undoc-members:
   :show-inheritance:

Polylines and streamlines
-------------------------

.. automodule:: zarr_vectors.types.polylines
   :members:
   :undoc-members:
   :show-inheritance:

Graphs and skeletons
--------------------

.. automodule:: zarr_vectors.types.graphs
   :members:
   :undoc-members:
   :show-inheritance:

Meshes
------

.. automodule:: zarr_vectors.types.meshes
   :members:
   :undoc-members:
   :show-inheritance:

Multi-resolution coarsening
---------------------------

.. automodule:: zarr_vectors.multiresolution.coarsen
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: zarr_vectors.multiresolution.object_selection
   :members:
   :undoc-members:
   :show-inheritance:
