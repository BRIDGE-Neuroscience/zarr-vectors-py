"""Regenerate the JSON Schema + Markdown reference from the LinkML source.

Usage::

    python schema/regen.py            # regenerate artefacts in place
    python schema/regen.py --check    # fail (exit 1) if regen would change files

CI runs ``--check``; if the generated files in this directory drift from
what the LinkML source would emit, the build fails and the contributor
re-runs without ``--check`` and commits the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# LinkML's DocGenerator inlines a Stack Overflow link about tabbed code
# blocks as an HTML comment on every "Examples" section it can't render.
# Strip these — they're tool noise, not user-facing.
_DOCGEN_TODO_RE = re.compile(
    r"<!--\s*TODO: investigate https://stackoverflow\.com/[^>]*-->\s*\n?"
)


def _strip_docgen_noise(text: str) -> str:
    return _DOCGEN_TODO_RE.sub("", text)

SCHEMA_DIR = Path(__file__).resolve().parent
SOURCE = SCHEMA_DIR / "zarr_vectors.linkml.yaml"
JSON_SCHEMA_OUT = SCHEMA_DIR / "zarr_vectors.schema.json"
DOC_OUT = SCHEMA_DIR / "reference.md"


def _gen_json_schema(source: Path, out: Path) -> None:
    """Render the LinkML schema as JSON Schema (Draft 2020-12)."""
    from linkml.generators.jsonschemagen import JsonSchemaGenerator

    gen = JsonSchemaGenerator(str(source))
    rendered = gen.serialize()
    # LinkML returns a string; round-trip through json to normalise
    # whitespace so diffs aren't noisy.
    parsed = json.loads(rendered)
    out.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gen_doc(source: Path, out: Path) -> None:
    """Render a single-file Markdown reference for the schema."""
    from linkml.generators.docgen import DocGenerator

    # DocGenerator writes one .md per class/slot/enum.  For a small
    # reference we instead generate to a temp dir and concatenate the
    # files in a stable order, producing one ``reference.md``.
    with tempfile.TemporaryDirectory() as td:
        gen = DocGenerator(str(source), directory=td, mergeimports=True)
        gen.serialize()
        td_path = Path(td)
        sections: list[str] = []
        order = ["index.md", "schema.md"]
        for name in order:
            p = td_path / name
            if p.exists():
                sections.append(p.read_text(encoding="utf-8"))
        # Append every other generated file in sorted order so classes,
        # enums and slots are all captured.
        seen = set(order)
        for p in sorted(td_path.glob("*.md")):
            if p.name in seen:
                continue
            sections.append(f"\n\n---\n\n{p.read_text(encoding='utf-8')}")
        out.write_text("\n".join(sections), encoding="utf-8")


def regen(check: bool) -> int:
    if not SOURCE.exists():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 2

    if check:
        # Write the JSON Schema to a scratch path and byte-compare.
        # Only the JSON Schema is byte-checked: LinkML's DocGenerator
        # currently emits class lists in non-deterministic order, so
        # the Markdown reference is regenerated freely and not strictly
        # gated.
        with tempfile.TemporaryDirectory() as td:
            tmp_json = Path(td) / "schema.json"
            _gen_json_schema(SOURCE, tmp_json)
            if not JSON_SCHEMA_OUT.exists():
                print(f"missing: {JSON_SCHEMA_OUT}", file=sys.stderr)
                print("Run `python schema/regen.py` to refresh.", file=sys.stderr)
                return 1
            if JSON_SCHEMA_OUT.read_bytes() != tmp_json.read_bytes():
                print(f"stale: {JSON_SCHEMA_OUT}", file=sys.stderr)
                print("Run `python schema/regen.py` to refresh.", file=sys.stderr)
                return 1
        print("schema artefacts up to date")
        return 0

    _gen_json_schema(SOURCE, JSON_SCHEMA_OUT)
    _gen_doc(SOURCE, DOC_OUT)
    print(f"wrote {JSON_SCHEMA_OUT.name}")
    print(f"wrote {DOC_OUT.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated artefacts are out of date.",
    )
    args = parser.parse_args(argv)
    return regen(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
