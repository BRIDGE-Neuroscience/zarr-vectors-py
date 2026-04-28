# Precomputed export

The Neuroglancer precomputed format is a static binary format that can be
served over HTTP without any running Python process. Converting a ZVF store
to precomputed format enables sharing with Neuroglancer users who do not
have `zv-ngtools` installed, embedding in publications, or serving at scale
from a CDN.

Precomputed export is one-way: the output directory is a read-only
Neuroglancer data source. To continue writing or updating the data, work
with the original ZVF store and re-export as needed.

---

## When to use precomputed export

| Scenario | Path A (LocalNeuroglancer) | Path B (Precomputed export) |
|----------|--------------------------|----------------------------|
| Local analysis and exploration | ✓ preferred | — |
| Collaborative review (team has zv-ngtools) | ✓ | — |
| Public dataset sharing | — | ✓ preferred |
| Embedded Neuroglancer in a publication | — | ✓ |
| CI-generated figures (no browser) | ✓ | — |
| Cloud-scale public serving (no server) | — | ✓ |

---

## Supported export targets

| ZVF geometry type | Precomputed format |
|------------------|--------------------|
| `skeleton` | Neuroglancer sharded skeleton |
| `mesh` | Neuroglancer sharded mesh (Draco optional) |
| `point_cloud` | Neuroglancer annotation (point) |
| `line` | Neuroglancer annotation (line) |
| `polyline` / `streamline` | Neuroglancer annotation (line) |

---

## Skeleton export

### CLI

```bash
zarr-vectors export precomputed-skeletons \
    neurons.zarrvectors \
    gs://my-bucket/neurons_precomputed/ \
    --sharded \
    --vertex-attributes radius,swc_type
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--sharded` | True | Use Neuroglancer sharded skeleton format (recommended for > 1000 skeletons) |
| `--vertex-attributes` | all | Comma-separated list of per-vertex attributes to export |
| `--level` | 0 | Which ZVF resolution level to export |
| `--chunk-size` | 256 | Shard spatial chunk size (number of objects per shard) |

### Python API

```python
from zarr_vectors.export.precomputed import export_precomputed_skeletons

export_precomputed_skeletons(
    "neurons.zarrvectors",
    "gs://my-bucket/neurons_precomputed/",
    sharded=True,
    vertex_attributes=["radius", "swc_type"],
    level=0,
)
```

### Output directory structure

```
neurons_precomputed/
├── info                    ← JSON: declares layer type, scales, vertex attrs
├── transform.json          ← coordinate transform (from ZVF multiscale metadata)
└── skeletons/
    ├── spatial0/           ← shard index for the spatial grid
    │   ├── 0_0_0           ← shard file: packed skeletons in this block
    │   ├── 0_0_1
    │   └── …
    └── @1                  ← sharding specification file
```

The `info` file mirrors the ZVF `multiscales` coordinate transforms so
that Neuroglancer displays skeletons at the correct physical coordinates:

```json
{
  "@type": "neuroglancer_skeletons",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0],
  "vertex_attributes": [
    {"id": "radius",   "data_type": "float32", "num_components": 1},
    {"id": "swc_type", "data_type": "int32",   "num_components": 1}
  ],
  "sharding": {
    "@type": "neuroglancer_uint64_sharded_v1",
    "preshift_bits": 9,
    "hash": "identity",
    "minishard_bits": 0,
    "shard_bits": 11
  }
}
```

---

## Mesh export

### CLI

```bash
zarr-vectors export precomputed-meshes \
    brain.zarrvectors \
    gs://my-bucket/brain_mesh_precomputed/ \
    --draco \
    --quantization 11 \
    --sharded
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--draco` | False | Apply Draco compression to mesh fragments |
| `--quantization` | 11 | Draco quantisation bits (ignored if `--draco` not set) |
| `--sharded` | True | Use sharded mesh format |
| `--level` | 0 | ZVF resolution level to export |

### Python API

```python
from zarr_vectors.export.precomputed import export_precomputed_meshes

export_precomputed_meshes(
    "brain.zarrvectors",
    "gs://my-bucket/brain_mesh_precomputed/",
    use_draco=True,
    draco_quantization=11,
    sharded=True,
)
```

### Output directory structure

```
brain_mesh_precomputed/
├── info
└── mesh/
    ├── <object_id>:0.index   ← per-object manifest JSON (legacy format)
    │   or
    ├── spatial0/             ← shard files (sharded format)
    └── @1
```

The `info` file for meshes:

```json
{
  "@type": "neuroglancer_multilod_draco",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0],
  "lod_scale_multiplier": 1.0,
  "vertex_quantization_bits": 11,
  "input_transform": [...]
}
```

---

## Point cloud and streamline annotation export

### CLI

```bash
# Point cloud → Neuroglancer annotation (points)
zarr-vectors export precomputed-annotations \
    scan.zarrvectors \
    gs://my-bucket/scan_annotations/ \
    --annotation-type point

# Streamlines → Neuroglancer annotation (lines)
zarr-vectors export precomputed-annotations \
    tracts.zarrvectors \
    gs://my-bucket/tracts_annotations/ \
    --annotation-type line
```

### Python API

```python
from zarr_vectors.export.precomputed import export_precomputed_annotations

export_precomputed_annotations(
    "scan.zarrvectors",
    "gs://my-bucket/scan_annotations/",
    annotation_type="point",
    properties=["intensity", "label"],   # attributes to include as properties
)
```

---

## Constructing the Neuroglancer layer URL

After exporting, construct the URL to load the layer in Neuroglancer:

### Using the Neuroglancer web app

```
https://neuroglancer-demo.appspot.com/#!
{
  "layers": [
    {
      "type": "skeletons",
      "source": "precomputed://https://storage.googleapis.com/my-bucket/neurons_precomputed",
      "name": "neurons",
      "tab": "rendering",
      "shader": "#uicontrol vec3 color color(default=\"#22d97a\")\nvoid main() { emitRGB(color); }"
    }
  ],
  "layout": "3d"
}
```

### Using the Python helper

```python
from zarr_vectors.export.precomputed import get_neuroglancer_url

url = get_neuroglancer_url(
    layers=[
        {
            "name": "neurons",
            "source": "gs://my-bucket/neurons_precomputed",
            "color": "#22d97a",
        },
        {
            "name": "EM",
            "source": "zarr://gs://my-bucket/em_volume.zarr",
        },
    ],
    position=[500., 300., 200.],
    zoom=4.0,
    layout="xy-3d",
    neuroglancer_url="https://neuroglancer-demo.appspot.com",
)
print(url)
# https://neuroglancer-demo.appspot.com/#!{...}
```

---

## Hosting on GCS or S3

### GCS — set public read and CORS

```bash
# Make bucket contents publicly readable
gsutil iam ch allUsers:objectViewer gs://my-bucket

# Set CORS
cat > cors.json << 'EOF'
[{
  "origin": ["*"],
  "method": ["GET", "HEAD"],
  "responseHeader": ["Content-Type", "Range", "X-Requested-With"],
  "maxAgeSeconds": 3600
}]
EOF
gsutil cors set cors.json gs://my-bucket
```

The precomputed source URL for Neuroglancer is:
`precomputed://https://storage.googleapis.com/my-bucket/neurons_precomputed`

### S3 — public read and CORS

```bash
# Enable public read (adjust per your security requirements)
aws s3api put-bucket-acl --bucket my-bucket --acl public-read

# CORS
aws s3api put-bucket-cors --bucket my-bucket \
    --cors-configuration '{
      "CORSRules": [{
        "AllowedMethods": ["GET","HEAD"],
        "AllowedOrigins": ["*"],
        "AllowedHeaders": ["*"],
        "ExposeHeaders": ["ETag","Content-Length"],
        "MaxAgeSeconds": 3600
      }]
    }'
```

The precomputed source URL:
`precomputed://https://my-bucket.s3.amazonaws.com/neurons_precomputed`

---

## Automating export in a pipeline

For datasets that are updated regularly, automate precomputed export
as a post-processing step:

```bash
#!/bin/bash
# export_pipeline.sh

STORE="neurons.zarrvectors"
DEST="gs://my-bucket/neurons_precomputed"

# 1. Validate store before export
zarr-vectors validate "$STORE" --level 3 || exit 1

# 2. Export
zarr-vectors export precomputed-skeletons "$STORE" "$DEST" \
    --sharded --vertex-attributes radius,swc_type

# 3. Regenerate Neuroglancer link
python - << 'EOF'
from zarr_vectors.export.precomputed import get_neuroglancer_url
url = get_neuroglancer_url(
    layers=[{"name":"neurons","source":"gs://my-bucket/neurons_precomputed"}]
)
print(f"Neuroglancer link: {url}")
EOF
```
