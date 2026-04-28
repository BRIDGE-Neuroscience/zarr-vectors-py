Constants
=========

Geometry type string constants. Import these instead of using raw strings
to ensure forward-compatibility if constant values change.

.. automodule:: zarr_vectors.constants
   :members:
   :undoc-members:
   :show-inheritance:

Geometry type constant values
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Constant
     - Value
     - Use
   * - ``GEOM_POINT_CLOUD``
     - ``"point_cloud"``
     - Unconnected vertices
   * - ``GEOM_LINE``
     - ``"line"``
     - Independent line segment pairs
   * - ``GEOM_POLYLINE``
     - ``"polyline"``
     - Ordered vertex paths
   * - ``GEOM_STREAMLINE``
     - ``"streamline"``
     - Tractography paths with metadata
   * - ``GEOM_GRAPH``
     - ``"graph"``
     - Arbitrary vertex–edge graph
   * - ``GEOM_SKELETON``
     - ``"skeleton"``
     - Tree-structured morphology (SWC-compatible)
   * - ``GEOM_MESH``
     - ``"mesh"``
     - Triangulated surface mesh
