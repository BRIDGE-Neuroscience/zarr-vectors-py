> [!NOTE]
> This package is under active development.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />

**Tools for Zarr Vectors Data**

`zarr-vectors-py` is a Python package for reading, writing, and managing large-scale vector geometry data in the zarr vectors format — a chunked, cloud-native format built on Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons, and meshes.

The package supports supervoxel-level spatial binning with separated chunk and bin sizes, per-level object sparsity for balanced multi-resolution pyramids, and OME-Zarr-compatible multiscale metadata.

*Aligned to the Zarr Vectors specification by Forest Collman, Allen Institute for Brain Sciences*
[Link to specification GitHub](https://github.com/AllenInstitute/zarr_vectors)

---
## Documentation

**link to readthedocs:** https://zarr-vectors-py.readthedocs.io/en/latest

---

## Install

```bash
pip install zarr-vectors
```

---

## Store Layout

```
dataset.zarrvectors/
├── .zattrs                          # root metadata: SID, CRS, conventions, base_bin_shape
├── resolution_0/                    # full resolution (bin_ratio = 1,1,1)
│   ├── vertices/                    # spatial positions (ragged per bin)
│   ├── vertex_group_offsets/        # byte offsets for sub-bin random access
│   ├── links/                       # connectivity (edges, faces, parents)
│   ├── attributes/                  # per-vertex data
│   │   ├── intensity/
│   │   └── gene_expression/
│   ├── object_index/                # object ID → (chunk, vertex_group) mapping
│   ├── object_attributes/           # per-object metadata
│   ├── groupings/                   # group ID → [object IDs]
│   ├── groupings_attributes/        # per-group metadata
│   └── cross_chunk_links/           # connectivity across chunk boundaries
├── resolution_1/                    # coarsened (bin_ratio, object_sparsity in .zattrs)
│   └── [same arrays]
├── parametric/                      # algebraic objects (planes, spheres)
│   ├── objects/
│   └── object_attributes/
└── metadata.json
```
