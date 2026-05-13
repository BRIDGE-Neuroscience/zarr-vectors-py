"""Generate examples/07_multiscale_links.ipynb from a single source.

Run once and commit the output.  Keeps the notebook deterministic and
avoids hand-editing JSON.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "07_multiscale_links.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ===================================================================
# Notebook content
# ===================================================================

md(
    "# Multiscale links (cross-pyramid-level edges)\n"
    "\n"
    "**Geometry type:** `graph` · **Schema version:** `0.4`\n"
    "\n"
    "This notebook is a deep dive into the **multiscale links layout** "
    "introduced in schema `0.4` — how graph edges, cross-chunk links, "
    "and their attributes are organised across pyramid levels and how "
    "you can read each piece directly.\n"
    "\n"
    "Topics:\n"
    "\n"
    "1. The on-disk layout: `links/<delta>/<chunk>` and friends\n"
    "2. Write a small graph and build a 3-level pyramid\n"
    "3. Inspect the directory tree — `0`, `+1`, `-1` arrays per level\n"
    "4. Read intra-level edges (`delta=0`) — current behaviour\n"
    "5. Read cross-level edges (`delta=+1`) — fine → coarse drill-up\n"
    "6. Cross-chunk links across levels (`cross_chunk_links/+1`)\n"
    "7. Per-link attributes, intra- *and* cross-chunk\n"
    "8. Storage modes: `none` vs `implicit` vs `explicit`\n"
    "9. Depth knob: `cross_level_depth=2`\n"
    "10. Validate"
)

code(
    "import numpy as np\n"
    "import tempfile, os\n"
    "from pathlib import Path\n"
    "\n"
    "_tmpdir = tempfile.mkdtemp(prefix=\"zvf_multiscale_\")\n"
    "STORE = os.path.join(_tmpdir, \"graph.zarrvectors\")\n"
    "print(\"Store:\", STORE)"
)

md(
    "## 1 · The on-disk layout\n"
    "\n"
    "Under the 0.4 schema, every link-family array gets a `<level_delta>` "
    "path segment that says how many pyramid levels the edges span:\n"
    "\n"
    "```\n"
    "/resolution_N/links/<delta>/<chunk_key>\n"
    "/resolution_N/cross_chunk_links/<delta>/data\n"
    "/resolution_N/link_attributes/<name>/<delta>/<chunk_key>\n"
    "/resolution_N/cross_chunk_link_attributes/<name>/<delta>/data\n"
    "```\n"
    "\n"
    "Convention for `<delta>`:\n"
    "\n"
    "| Segment | Meaning |\n"
    "|---------|---------|\n"
    "| `0`     | intra-level (the only kind written pre-0.4) |\n"
    "| `+1`    | edges from this level to `this_level + 1` (one step coarser) |\n"
    "| `-1`    | edges from this level to `this_level - 1` (one step finer) |\n"
    "| `+N` / `-N` | jumps of N levels |\n"
    "\n"
    "Sides of an edge:\n"
    "\n"
    "- For `links/<delta>/<chunk>`: source endpoint is local to the chunk "
    "at the owning level; target endpoint is local to the **same chunk key** "
    "at level `+delta`. Only used when the two endpoints share a chunk_key.\n"
    "- For `cross_chunk_links/<delta>/data`: each row is "
    "`((chunk_a, local_a), (chunk_b, local_b))` — endpoint A at the owning level, "
    "endpoint B at level `+delta`. Used when chunk keys differ across levels.\n"
    "\n"
    "Use the path helpers in `zarr_vectors.core.paths` to compose paths — never "
    "hard-code the `<delta>` formatting yourself."
)

code(
    "from zarr_vectors.core.paths import (\n"
    "    format_delta, parse_delta,\n"
    "    links_path, cross_chunk_links_path,\n"
    "    link_attributes_path, cross_chunk_link_attributes_path,\n"
    ")\n"
    "\n"
    "print(\"format_delta(0)  =\", format_delta(0))\n"
    "print(\"format_delta(+1) =\", format_delta(1))\n"
    "print(\"format_delta(-2) =\", format_delta(-2))\n"
    "print()\n"
    "print(\"links_path(0)              =\", links_path(0))\n"
    "print(\"links_path(+1)             =\", links_path(1))\n"
    "print(\"cross_chunk_links_path(-1) =\", cross_chunk_links_path(-1))\n"
    "print(\"link_attributes_path('weight', +1)            =\", link_attributes_path('weight', 1))\n"
    "print(\"cross_chunk_link_attributes_path('weight', 0) =\", cross_chunk_link_attributes_path('weight', 0))"
)

md(
    "## 2 · Write a small graph and build a 3-level pyramid\n"
    "\n"
    "We use a small (500-node) graph in a 400 µm cube so the pyramid produces a "
    "handful of metanodes per level — easy to eyeball.  Each level will roughly "
    "8× coarsen the previous one.\n"
    "\n"
    "Defaults pick up `cross_level_depth=1` and `cross_level_storage=\"explicit\"` — "
    "we'll override those below to compare modes.  Default `explicit` writes both "
    "`+1` at the finer level and `-1` at the coarser level for every adjacent pair."
)

code(
    "from zarr_vectors.types.graphs import write_graph\n"
    "from zarr_vectors.multiresolution.coarsen import build_pyramid\n"
    "from zarr_vectors.constants import XLEVEL_EXPLICIT, XLEVEL_IMPLICIT, XLEVEL_NONE\n"
    "\n"
    "rng = np.random.default_rng(0)\n"
    "N = 500\n"
    "positions = rng.uniform(0.0, 400.0, size=(N, 3)).astype(np.float32)\n"
    "edges = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int64)\n"
    "edge_weights = rng.uniform(0.1, 1.0, size=len(edges)).astype(np.float32)\n"
    "\n"
    "write_graph(\n"
    "    STORE,\n"
    "    positions=positions,\n"
    "    edges=edges,\n"
    "    object_ids=np.zeros(N, dtype=np.int64),\n"
    "    chunk_shape=(100.0, 100.0, 100.0),\n"
    "    bounds=([0.0, 0.0, 0.0], [400.0, 400.0, 400.0]),\n"
    "    edge_attributes={\"weight\": edge_weights},\n"
    ")\n"
    "\n"
    "build_pyramid(\n"
    "    STORE,\n"
    "    factors=[(2.0, 1.0), (2.0, 1.0)],\n"
    "    cross_level_depth=1,                # ±1 between every adjacent pair\n"
    "    cross_level_storage=XLEVEL_EXPLICIT,  # store both +1 and -1\n"
    ")\n"
    "print(\"Build complete.\")"
)

md(
    "## 3 · Walk the on-disk tree\n"
    "\n"
    "Each resolution level should now carry `links/0` (intra-level edges) "
    "and, depending on its position in the pyramid, some combination of "
    "`links/+1`, `links/-1`, `cross_chunk_links/+1`, `cross_chunk_links/-1`."
)

code(
    "from zarr_vectors.core.store import open_store, list_resolution_levels, get_resolution_level\n"
    "from zarr_vectors.constants import LINKS, CROSS_CHUNK_LINKS\n"
    "from zarr_vectors.core.arrays import list_link_deltas, list_cross_link_deltas\n"
    "\n"
    "root = open_store(STORE)\n"
    "levels = sorted(list_resolution_levels(root))\n"
    "print(f\"Pyramid levels: {levels}\")\n"
    "print()\n"
    "for lvl in levels:\n"
    "    lg = get_resolution_level(root, lvl)\n"
    "    ld = list_link_deltas(lg)\n"
    "    cd = list_cross_link_deltas(lg)\n"
    "    print(f\"  resolution_{lvl}: links/<delta> = {ld}    cross_chunk_links/<delta> = {cd}\")"
)

md(
    "Reading this:\n"
    "\n"
    "- Level 0 has `+1` (drill *up* to level 1) but no `-1` (nothing below).\n"
    "- Mid levels carry both `+1` and `-1`.\n"
    "- The top level has `-1` but no `+1` (nothing above).\n"
    "\n"
    "If you peek at the actual directory you'll see one subdir per `<delta>`:"
)

code(
    "from pathlib import Path\n"
    "level0_links = Path(STORE) / \"resolution_0\" / \"links\"\n"
    "print(f\"contents of {level0_links}:\")\n"
    "for child in sorted(level0_links.iterdir()):\n"
    "    print(\" \", child.name)"
)

md(
    "## 4 · Intra-level edges — `delta=0` (unchanged behaviour)\n"
    "\n"
    "Reading `delta=0` matches the pre-0.4 behaviour of `read_chunk_links`.  "
    "You get one list of `(M_k, 2)` arrays per spatial chunk; each row is a "
    "pair of local-vertex indices."
)

code(
    "from zarr_vectors.core.arrays import read_chunk_links, list_chunk_keys\n"
    "\n"
    "lg0 = get_resolution_level(root, 0)\n"
    "chunk_keys = list_chunk_keys(lg0, LINKS + \"/0\")\n"
    "print(f\"level 0 has links/0 in {len(chunk_keys)} chunks\")\n"
    "for ck in chunk_keys[:3]:\n"
    "    groups = read_chunk_links(lg0, ck, link_width=2, delta=0)\n"
    "    n = sum(len(g) for g in groups)\n"
    "    print(f\"  chunk {ck}: {n} intra-chunk edges (groups: {len(groups)})\")"
)

md(
    "## 5 · Cross-level edges — `delta=+1` (drill up)\n"
    "\n"
    "Cross-level edges are conceptually trivial: every fine vertex has one "
    "edge to its coarse parent metanode.  The build splits those edges into:\n"
    "\n"
    "- **chunk-aligned** edges → `links/+1/<chunk_key>` when the source chunk "
    "key matches the coarse target chunk key;\n"
    "- **cross-chunk** edges → `cross_chunk_links/+1/data` otherwise.\n"
    "\n"
    "For a `links/+1` row, *column 0* is the local vertex index in the source "
    "chunk **at the owning level**; *column 1* is the local vertex index in "
    "the same chunk key **at level + 1**."
)

code(
    "plus1_chunks = list_chunk_keys(lg0, LINKS + \"/+1\")\n"
    "print(f\"level 0 has links/+1 in {len(plus1_chunks)} chunks (chunk-aligned cross-level edges)\")\n"
    "total_plus1 = 0\n"
    "for ck in plus1_chunks:\n"
    "    g = read_chunk_links(lg0, ck, link_width=2, delta=1)\n"
    "    n = sum(len(x) for x in g)\n"
    "    total_plus1 += n\n"
    "    if n:\n"
    "        sample = g[0][:3]\n"
    "        print(f\"  chunk {ck}: {n} edges, sample rows (fine_local, coarse_local):\")\n"
    "        for row in sample:\n"
    "            print(f\"     {tuple(row)}\")\n"
    "print(f\"\\nTotal chunk-aligned +1 edges at level 0: {total_plus1}\")"
)

md(
    "## 6 · Cross-chunk + cross-level: `cross_chunk_links/+1`\n"
    "\n"
    "When a fine vertex's coarse parent lives in a *different* chunk grid cell, "
    "the edge can't be expressed by a per-chunk row — it goes into the global "
    "`cross_chunk_links/+1/data` blob.  Each entry encodes both endpoint sides "
    "explicitly:\n"
    "\n"
    "```\n"
    "((source_chunk_coords, source_local_idx),  # at this level\n"
    " (target_chunk_coords, target_local_idx))  # at this_level + delta\n"
    "```"
)

code(
    "from zarr_vectors.core.arrays import read_cross_chunk_links\n"
    "\n"
    "ccl_plus1 = read_cross_chunk_links(lg0, delta=1)\n"
    "print(f\"level 0 has {len(ccl_plus1)} cross-chunk +1 edges\")\n"
    "for a, b in ccl_plus1[:3]:\n"
    "    (ca, la), (cb, lb) = a, b\n"
    "    print(f\"  src chunk={ca} local={la}  ->  tgt chunk={cb} local={lb}\")"
)

md(
    "## 7 · Per-link attributes — intra- and cross-chunk\n"
    "\n"
    "Two parallel attribute namespaces exist:\n"
    "\n"
    "- `link_attributes/<name>/<delta>/<chunk_key>` — parallel to `links/<delta>/<chunk_key>`, "
    "one ragged group per spatial chunk.\n"
    "- `cross_chunk_link_attributes/<name>/<delta>/data` — *new in 0.4*, parallel to "
    "`cross_chunk_links/<delta>/data`; one flat row per cross-chunk link in the same order.\n"
    "\n"
    "The build wrote `delta=0` attributes from the `edge_attributes={'weight': ...}` we passed "
    "into `write_graph`.  We'll also write a cross-chunk attribute by hand to show the new API.\n"
    "\n"
    "Note: cross-chunk link attribute writes enforce `len(values) == num_links` at runtime — "
    "a misaligned write fails loudly instead of silently corrupting the parallel array."
)

code(
    "from zarr_vectors.core.arrays import (\n"
    "    create_cross_chunk_link_attributes_array,\n"
    "    write_cross_chunk_link_attributes,\n"
    "    read_cross_chunk_link_attributes,\n"
    ")\n"
    "\n"
    "# Re-open for writing.\n"
    "root_rw = open_store(STORE, mode=\"r+\")\n"
    "lg0_rw = get_resolution_level(root_rw, 0)\n"
    "\n"
    "# Cross-chunk +1 link attributes: one float per cross-chunk link, in path order.\n"
    "num_ccl_plus1 = len(ccl_plus1)\n"
    "if num_ccl_plus1:\n"
    "    create_cross_chunk_link_attributes_array(lg0_rw, \"weight\", dtype=\"float32\", delta=1)\n"
    "    weights = np.linspace(0.0, 1.0, num_ccl_plus1, dtype=np.float32)\n"
    "    write_cross_chunk_link_attributes(\n"
    "        lg0_rw, \"weight\", weights, num_links=num_ccl_plus1, delta=1,\n"
    "    )\n"
    "    back = read_cross_chunk_link_attributes(lg0_rw, \"weight\", delta=1)\n"
    "    print(f\"wrote/read {len(back)} cross-chunk-link weights at delta=+1\")\n"
    "    print(f\"first 5: {back[:5]}\")\n"
    "else:\n"
    "    print(\"no cross-chunk +1 edges at level 0; skipping attribute round-trip\")"
)

md(
    "Length-invariant check (this *should* raise):"
)

code(
    "from zarr_vectors.exceptions import ArrayError\n"
    "\n"
    "if num_ccl_plus1:\n"
    "    try:\n"
    "        bad = np.zeros(num_ccl_plus1 + 7, dtype=np.float32)\n"
    "        write_cross_chunk_link_attributes(\n"
    "            lg0_rw, \"weight\", bad, num_links=num_ccl_plus1, delta=1,\n"
    "        )\n"
    "    except ArrayError as e:\n"
    "        print(f\"ArrayError raised as expected:\\n  {e}\")"
)

md(
    "## 8 · Storage modes: `explicit` vs `implicit` vs `none`\n"
    "\n"
    "Three modes control whether `-N` arrays are materialised at all:\n"
    "\n"
    "| Mode | Writes `+N` at fine level? | Writes `-N` at coarse level? |\n"
    "|------|----------------------------|------------------------------|\n"
    "| `none`     | no  | no  |\n"
    "| `implicit` | yes | no  |\n"
    "| `explicit` | yes | yes |\n"
    "\n"
    "`implicit` saves storage; the `-N` direction is reconstructed by reading the `+N` array "
    "at the target level and swapping endpoints.  `explicit` materialises both, paying disk "
    "for O(1) reads in both directions.\n"
    "\n"
    "Let's rebuild against fresh stores to compare on-disk footprints."
)

code(
    "import shutil\n"
    "\n"
    "def _build_one(name, *, depth, storage):\n"
    "    path = os.path.join(_tmpdir, f\"{name}.zarrvectors\")\n"
    "    if os.path.exists(path):\n"
    "        shutil.rmtree(path)\n"
    "    write_graph(\n"
    "        path,\n"
    "        positions=positions,\n"
    "        edges=edges,\n"
    "        object_ids=np.zeros(N, dtype=np.int64),\n"
    "        chunk_shape=(100.0, 100.0, 100.0),\n"
    "        bin_shape=(25.0, 25.0, 25.0),\n"
    "        edge_attributes={\"weight\": edge_weights},\n"
    "    )\n"
    "    build_pyramid(\n"
    "        path, factors=[(2.0, 1.0), (2.0, 1.0)],\n"
    "        cross_level_depth=depth, cross_level_storage=storage,\n"
    "    )\n"
    "    return path\n"
    "\n"
    "def _scan(path):\n"
    "    r = open_store(path)\n"
    "    out = []\n"
    "    for lvl in sorted(list_resolution_levels(r)):\n"
    "        lg = get_resolution_level(r, lvl)\n"
    "        out.append((lvl, list_link_deltas(lg), list_cross_link_deltas(lg)))\n"
    "    return out\n"
    "\n"
    "for mode in (XLEVEL_NONE, XLEVEL_IMPLICIT, XLEVEL_EXPLICIT):\n"
    "    p = _build_one(f\"graph_{mode}\", depth=1, storage=mode)\n"
    "    print(f\"\\n--- cross_level_storage = {mode!r} ---\")\n"
    "    for lvl, ld, cd in _scan(p):\n"
    "        print(f\"  resolution_{lvl}: links/<delta>={ld}  cross_chunk_links/<delta>={cd}\")"
)

md(
    "## 9 · Depth knob: `cross_level_depth=2`\n"
    "\n"
    "`cross_level_depth` controls how far the cross-level emission reaches:\n"
    "\n"
    "- `0` — disabled (same as `storage=\"none\"`).\n"
    "- `N` — materialise up to `±N` for every adjacent pair we can reach.\n"
    "- `-1` — walk *all* available pyramid levels.\n"
    "\n"
    "At `depth=2` the writer composes the fine→parent map across two coarsening "
    "steps (`grandparent[i] = parent_at_L1[parent_at_L0[i]]`) so a single edge "
    "goes from a level-0 vertex straight to its level-2 metanode."
)

code(
    "p2 = _build_one(\"graph_depth2\", depth=2, storage=XLEVEL_EXPLICIT)\n"
    "print(\"depth=2, explicit:\")\n"
    "for lvl, ld, cd in _scan(p2):\n"
    "    print(f\"  resolution_{lvl}: links/<delta>={ld}  cross_chunk_links/<delta>={cd}\")"
)

md(
    "Expected (with a 3-level pyramid):\n"
    "\n"
    "- Level 0 → `+1`, `+2`\n"
    "- Level 1 → `-1`, `+1`\n"
    "- Level 2 → `-1`, `-2`\n"
    "\n"
    "Plus `0` everywhere from the original `write_graph` call."
)

md(
    "## 10 · Validate\n"
    "\n"
    "The validator walks each `<delta>` subdir under `links/` and `cross_chunk_links/` and "
    "checks that endpoint chunk keys are present in the level's chunk grid."
)

code(
    "from zarr_vectors.validate import validate\n"
    "\n"
    "rv = validate(STORE, level=3)\n"
    "print(rv.summary())"
)

md(
    "## Summary\n"
    "\n"
    "| Concept | API |\n"
    "|---------|-----|\n"
    "| Pyramid with cross-level edges | `build_pyramid(path, cross_level_depth=N, cross_level_storage=\"explicit\")` |\n"
    "| Compose paths | `links_path(delta)`, `cross_chunk_links_path(delta)`, `link_attributes_path(name, delta)`, `cross_chunk_link_attributes_path(name, delta)` |\n"
    "| List deltas on disk | `list_link_deltas(level)`, `list_cross_link_deltas(level)` |\n"
    "| Read intra-level edges | `read_chunk_links(level, chunk, delta=0)` |\n"
    "| Read cross-level edges | `read_chunk_links(level, chunk, delta=+1)` |\n"
    "| Read cross-chunk links | `read_cross_chunk_links(level, delta=±N)` |\n"
    "| Write/read cross-chunk-link attrs | `write_cross_chunk_link_attributes(level, name, values, num_links, delta)` / `read_cross_chunk_link_attributes(level, name, delta)` |\n"
    "\n"
    "Endpoint convention recap:\n"
    "\n"
    "- `links/<delta>/<chunk>` rows: column 0 = source-level local index, "
    "column 1 = local index in the **same chunk key** at level `+delta`.\n"
    "- `cross_chunk_links/<delta>/data` rows: "
    "`((src_chunk, src_local), (tgt_chunk, tgt_local))` — `src_*` at the owning level, "
    "`tgt_*` at level `+delta`.\n"
    "\n"
    "See `docs/multiscale-links.md` (or the plan notes in the repo) for the design rationale "
    "and the schema-0.4 breaking change details."
)


# ===================================================================
# Build the JSON
# ===================================================================

def _to_source(text: str) -> list[str]:
    """Match the multi-line `source` list shape Jupyter writes."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    # Jupyter convention: no trailing newline on the last entry.
    if lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _build() -> dict:
    cells = []
    for kind, text in CELLS:
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown",
                "id": _cell_id(),
                "metadata": {},
                "source": _to_source(text),
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": _to_source(text),
            })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "zarr-vectors",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.15",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    NB_PATH.write_text(
        json.dumps(_build(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NB_PATH.name} ({NB_PATH.stat().st_size:,} bytes)")
