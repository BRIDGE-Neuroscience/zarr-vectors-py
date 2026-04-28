.. zarr-vectors documentation master file

.. image:: zarr-vectors.png
   :width: 55%
   :align: center
   :alt: zarr-vectors

----

**zarr-vectors** stores three-dimensional spatial geometry — point clouds,
streamlines, graphs, skeletons, and meshes — in spatially indexed Zarr v3
stores. Spatial queries touch only the chunks they need, whether the store
sits on a local filesystem or a cloud object store (S3, GCS). Resolution
pyramids are encoded natively so viewers like Neuroglancer can stream data
progressively at any scale.

The library implements the `Zarr Vector Format
<https://github.com/AllenInstitute/zarr_vectors>`_ originally specified by
Forest Collman at the Allen Institute for Brain Sciences, extended with
separated chunk/bin sizes, per-level sparsity, and OME-Zarr-compatible
multiscale metadata.

----

| `GitHub <https://github.com/BRIDGE-Neuroscience/zarr-vectors-py>`__

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`getting_started/quickstart`
     - Write and query your first vector store in a few lines of Python.
   * - :doc:`getting_started/concepts`
     - The mental model: chunk shapes, bin shapes, and resolution pyramids.
   * - :doc:`spec/index`
     - Full technical specification for the Zarr Vector Format.
   * - :doc:`api/index`
     - Auto-generated reference for all public functions and classes.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts
   getting_started/faq

.. toctree::
   :maxdepth: 1
   :caption: Specification
   :hidden:

   spec/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/data_types/point_clouds
   tutorials/data_types/polylines_streamlines
   tutorials/data_types/graphs_skeletons
   tutorials/data_types/meshes
   tutorials/multiscale/building_pyramids
   tutorials/multiscale/lazy_loading
   tutorials/io/ingest_formats
   tutorials/io/cloud_stores
   tutorials/io/validation_and_repair
   tutorials/neuroglancer/overview
   tutorials/neuroglancer/zv_ngtools_install
   tutorials/neuroglancer/local_viewer
   tutorials/neuroglancer/shell_console
   tutorials/neuroglancer/layer_api
   tutorials/neuroglancer/precomputed_export

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides
   :hidden:

   how_to/choose_chunk_and_bin
   how_to/memory_efficient_writes
   how_to/hpc_pipelines
   how_to/cite
