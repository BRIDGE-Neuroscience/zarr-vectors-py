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
   :maxdepth: 2
   :caption: Specification

   spec/index
   spec/foundations/zarr_v3_primer
   spec/foundations/store_types
   spec/foundations/dimensionality
   spec/foundations/codec_pipeline
   spec/layout/directory_structure
   spec/layout/root_metadata
   spec/layout/level_groups
   spec/layout/chunk_arrays
   spec/layout/vg_index_arrays
   spec/chunking/chunk_shape
   spec/chunking/bin_shape
   spec/chunking/chunk_vs_bin
   spec/chunking/rechunking
   spec/chunking/sharding
   spec/multiscale/multiscale_metadata
   spec/multiscale/pyramid_construction
   spec/multiscale/sparsity
   spec/geometry_types/index
   spec/geometry_types/point_cloud
   spec/geometry_types/line
   spec/geometry_types/polyline
   spec/geometry_types/streamline
   spec/geometry_types/graph
   spec/geometry_types/skeleton
   spec/geometry_types/mesh
   spec/object_model/vertex_groups
   spec/object_model/object_manifest
   spec/object_model/cross_chunk_links
   spec/object_model/object_attributes
   spec/validation/overview
   spec/validation/l1_structural
   spec/validation/l2_metadata
   spec/validation/l3_consistency
   spec/comparisons/neuroglancer_precomputed
   spec/comparisons/trx_format
   spec/comparisons/ome_zarr
   spec/contributing/spec_change_process
   spec/contributing/adding_geometry_types
   spec/contributing/test_compliance

.. toctree::
   :maxdepth: 2
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
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/zarr_vectors
   api/types
   api/lazy
   api/spatial
   api/ingest
   api/validate
   api/constants
   api/typing

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides

   how_to/choose_chunk_and_bin
   how_to/memory_efficient_writes
   how_to/hpc_pipelines
   how_to/cite
