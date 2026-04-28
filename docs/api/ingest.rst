Ingest and export
=================

Format converters for reading third-party formats into ZVF stores and
exporting ZVF stores back to third-party formats. Most ingest functions
require ``zarr-vectors[ingest]``. Draco export requires
``zarr-vectors[draco]``.

.. note::

   If a required extra is not installed, importing the relevant module
   raises ``ImportError`` with a message indicating which ``pip install``
   command is needed.

Ingest
------

.. automodule:: zarr_vectors.ingest.las
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.ply
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.csv_points
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.trk
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.tck
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.trx
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.swc
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.graphml
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.obj
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.ingest.stl
   :members:
   :undoc-members:

Export
------

.. automodule:: zarr_vectors.export.csv_points
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.export.ply
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.export.trk
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.export.trx
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.export.swc
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.export.obj
   :members:
   :undoc-members:

Precomputed export (Neuroglancer)
---------------------------------

.. automodule:: zarr_vectors.export.precomputed
   :members:
   :undoc-members:

Repair utilities
----------------

.. automodule:: zarr_vectors.repair
   :members:
   :undoc-members:
