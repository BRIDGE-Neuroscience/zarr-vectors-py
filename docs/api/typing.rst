Type aliases
============

Type aliases used throughout the ``zarr-vectors`` API. All are defined
in ``zarr_vectors.typing`` and re-exported from the top-level package.

.. automodule:: zarr_vectors.typing
   :members:
   :undoc-members:
   :show-inheritance:

Reference table
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Alias
     - Definition
     - Notes
   * - ``ChunkShape``
     - ``tuple[float, ...]``
     - D-tuple of positive floats (physical units)
   * - ``BinShape``
     - ``tuple[float, ...]``
     - D-tuple; must evenly divide ``ChunkShape``
   * - ``BinRatio``
     - ``tuple[int, ...]``
     - D-tuple of positive integers
   * - ``ChunkCoords``
     - ``tuple[int, ...]``
     - D-tuple; chunk grid coordinate
   * - ``BoundingBox``
     - ``tuple[ndarray, ndarray]``
     - ``(lo, hi)`` arrays of shape ``(D,)``
   * - ``CrossChunkLink``
     - ``tuple[int, int]``
     - ``(src_global_id, dst_global_id)``
   * - ``ObjectManifest``
     - ``list[tuple[int, int]]``
     - Ordered ``(chunk_flat, vg_flat)`` pairs
   * - ``AttributeDict``
     - ``dict[str, ndarray]``
     - Named per-vertex or per-object attribute arrays
   * - ``LevelConfig``
     - ``TypedDict``
     - ``bin_ratio``, ``object_sparsity``, ``sparsity_strategy``
   * - ``StorePath``
     - ``str | zarr.storage.Store``
     - Accepts local path, fsspec mapper, or ``MemoryStore``
