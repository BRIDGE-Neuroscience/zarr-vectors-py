# Machine-readable schema

The narrative spec pages in `docs/spec/` describe the Zarr Vectors
on-disk format in prose.  For a machine-readable reference — usable
from any language with a JSON Schema validator — see the [`schema/`]
directory in the repository root:

- [`schema/zarr_vectors.linkml.yaml`] — the canonical [LinkML] source.
- [`schema/zarr_vectors.schema.json`] — generated JSON Schema (Draft
  2020-12).
- [`schema/reference.md`] — generated Markdown reference.

The runtime Python source of truth is the dataclass tree in
[`zarr_vectors/core/metadata.py`].  A pytest sanity check
(`tests/test_linkml_schema.py`) prevents the LinkML schema from
drifting from the runtime dataclasses.

For details on coverage, vocabulary mappings (OME-Zarr / Schema.org),
and the build-time regeneration workflow, see
[`schema/README.md`].

[`schema/`]:                            ../../schema/
[`schema/zarr_vectors.linkml.yaml`]:    ../../schema/zarr_vectors.linkml.yaml
[`schema/zarr_vectors.schema.json`]:    ../../schema/zarr_vectors.schema.json
[`schema/reference.md`]:                ../../schema/reference.md
[`schema/README.md`]:                   ../../schema/README.md
[`zarr_vectors/core/metadata.py`]:      ../../zarr_vectors/core/metadata.py
[LinkML]:                               https://linkml.io/
