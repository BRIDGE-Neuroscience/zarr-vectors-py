.. zarr-vectors documentation master file

.. image:: _static/zarr-vectors-logo.png
   :width: 55%
   :align: center
   :alt: zarr-vectors

|

.. raw:: html

   <p style="text-align:center;font-size:1.05rem;color:var(--color-foreground-secondary)">
     A chunked, cloud-native format for multiscale spatial vector geometry —
     built on Zarr v3.
   </p>

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts
   getting_started/faq

.. toctree::
   :maxdepth: 1
   :caption: Specification

   spec/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

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

   api/index

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides

   how_to/choose_chunk_and_bin
   how_to/memory_efficient_writes
   how_to/hpc_pipelines
   how_to/cite
