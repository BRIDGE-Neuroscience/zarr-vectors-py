"""Generate the three benchmark notebooks.

Run once and commit the output:

    python examples/benchmarks/_build.py

Each notebook follows a tight 4-section structure (setup → sweep →
table → plot).  Keep the per-cell strings short — these are
*benchmarks*, not tutorials.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ===================================================================
# Shared cells (same across all three notebooks)
# ===================================================================

SHARED_HELPERS = """import os, time, tempfile, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _time(fn, *args, **kwargs):
    \"\"\"Call fn(*args, **kwargs); return (elapsed_seconds, result).\"\"\"
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def _store_bytes(path):
    \"\"\"Total on-disk size of a store directory, in bytes.\"\"\"
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def _new_store(prefix):
    \"\"\"Fresh tempdir + zarrvectors path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'zvbench_{prefix}_')) / 'store.zarrvectors'
"""


# ===================================================================
# 01 · Size scaling — point cloud, vary N
# ===================================================================

SIZE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Size scaling — point cloud

Write/read/disk-size of point clouds at increasing `N`. Same
`chunk_shape` across runs so the only variable is vertex count.

Runtime: a few minutes on a laptop (the 1M case dominates).
"""),
    ("code", SHARED_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points

SIZES = [1_000, 10_000, 100_000, 1_000_000]
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)
rows = []
for n in SIZES:
    positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, n).astype(np.float32)

    store = _new_store(f'size_{n}')
    t_write, _ = _time(
        write_points, store, positions,
        chunk_shape=CHUNK, bin_shape=BIN,
        attributes={'intensity': intensity},
    )
    t_read, _ = _time(read_points, store, attribute_names=['intensity'])
    rows.append({
        'N': n,
        'write_s': round(t_write, 3),
        'read_s':  round(t_read,  3),
        'size_MB': round(_store_bytes(store) / 1e6, 2),
    })
    shutil.rmtree(Path(store).parent, ignore_errors=True)

df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(df['N'], df['write_s'], 'o-', label='write (s)')
ax.loglog(df['N'], df['read_s'],  's-', label='read (s)')
ax.loglog(df['N'], df['size_MB'], '^-', label='size (MB)')
ax.set_xlabel('N (vertices)')
ax.set_title('Point cloud: write/read time + disk footprint vs N')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
"""),
]


# ===================================================================
# 02 · Data types — fixed N, sweep the six types
# ===================================================================

TYPES_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Data-type scaling

Same `N = 50_000` vertices/nodes across all six geometry types.
Synthetic inputs are tiny per-type generators (random positions / a
spanning tree / a triangulated grid).

Numbers here are *not* a fair cross-type comparison of any
underlying algorithm — different types do genuinely different work.
The point is to see relative per-type cost on the write/read path.
"""),
    ("code", SHARED_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.types.lines import write_lines, read_lines
from zarr_vectors.types.polylines import write_polylines, read_polylines
from zarr_vectors.types.graphs import write_graph, read_graph
from zarr_vectors.types.meshes import write_mesh, read_mesh

N = 50_000
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0
rng = np.random.default_rng(SEED)
"""),
    ("md", "## 2 · Synthetic generators (one per type)"),
    ("code", """\
def _points_input():
    return rng.uniform(0, 1000, (N, 3)).astype(np.float32)


def _lines_input():
    # N lines, each two random endpoints
    return rng.uniform(0, 1000, (N, 2, 3)).astype(np.float32)


def _polylines_input():
    # ~N total vertices spread across short random walks
    counts = rng.integers(8, 16, size=N // 12)
    out = []
    for c in counts:
        steps = rng.normal(0, 5, (c, 3))
        start = rng.uniform(0, 1000, 3)
        out.append((start + steps.cumsum(axis=0)).astype(np.float32))
    return out


def _graph_input(is_tree=False):
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    if is_tree:
        # spanning-tree edges: each node i (i>=1) connects to a random ancestor
        parents = rng.integers(0, np.arange(1, N))
        edges = np.stack([np.arange(1, N), parents], axis=1).astype(np.int64)
    else:
        # ~3N/2 random undirected edges
        src = rng.integers(0, N, size=3 * N // 2)
        dst = rng.integers(0, N, size=3 * N // 2)
        mask = src != dst
        edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)
    return positions, edges


def _mesh_input():
    # Triangulated grid with ~N vertices
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(np.linspace(0, 1000, side), np.linspace(0, 1000, side))
    zs = rng.uniform(0, 100, (side, side))
    verts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    # Two triangles per grid cell.
    i = np.arange(side - 1)
    j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel()
    b = a + 1
    c = a + side
    d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
def bench_points():
    pts = _points_input()
    store = _new_store('points')
    tw, _ = _time(write_points, store, pts, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _ = _time(read_points, store)
    return tw, tr, _store_bytes(store), store

def bench_lines():
    eps = _lines_input()
    store = _new_store('lines')
    tw, _ = _time(write_lines, store, eps, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _ = _time(read_lines, store)
    return tw, tr, _store_bytes(store), store

def bench_polylines():
    plys = _polylines_input()
    store = _new_store('polylines')
    tw, _ = _time(write_polylines, store, plys, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _ = _time(read_polylines, store)
    return tw, tr, _store_bytes(store), store

def bench_graph(is_tree):
    pos, edges = _graph_input(is_tree=is_tree)
    store = _new_store('skeleton' if is_tree else 'graph')
    tw, _ = _time(
        write_graph, store, pos, edges,
        chunk_shape=CHUNK, bin_shape=BIN, is_tree=is_tree,
    )
    tr, _ = _time(read_graph, store)
    return tw, tr, _store_bytes(store), store

def bench_mesh():
    verts, faces = _mesh_input()
    store = _new_store('mesh')
    tw, _ = _time(write_mesh, store, verts, faces, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _ = _time(read_mesh, store)
    return tw, tr, _store_bytes(store), store


fns = [
    ('points',    bench_points),
    ('lines',     bench_lines),
    ('polylines', bench_polylines),
    ('graph',     lambda: bench_graph(is_tree=False)),
    ('skeleton',  lambda: bench_graph(is_tree=True)),
    ('mesh',      bench_mesh),
]

rows = []
for name, fn in fns:
    tw, tr, nbytes, store = fn()
    rows.append({
        'type':    name,
        'write_s': round(tw, 3),
        'read_s':  round(tr, 3),
        'size_MB': round(nbytes / 1e6, 2),
    })
    shutil.rmtree(Path(store).parent, ignore_errors=True)

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(df))
ax.bar(x - 0.18, df['write_s'], width=0.36, label='write (s)')
ax.bar(x + 0.18, df['read_s'],  width=0.36, label='read (s)')
ax.set_xticks(x)
ax.set_xticklabels(df['type'])
ax.set_ylabel('seconds')
ax.set_title(f'Write / read time per type (N = {N:,})')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
"""),
]


# ===================================================================
# 03 · Backends — point cloud at fixed N, vary backend
# ===================================================================

BACKEND_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Backend scaling

Point cloud at `N = 100_000`. Always benchmarks `local`. If
`ZV_BENCH_S3_URL` is set in the env (and `obstore` / `fsspec` are
installed), also benchmarks those against the URL.

Set up a cloud target with:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
```

When `ZV_BENCH_S3_URL` is unset, the notebook reports a one-row
local-only result.
"""),
    ("code", SHARED_HELPERS),
    ("md", "## 1 · Setup + backend availability"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.exceptions import StoreError

N = 100_000
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0

cloud_url = os.environ.get('ZV_BENCH_S3_URL')
print(f'local backend:  always benchmarked')
print(f'cloud backends: {\"benchmarked against \" + cloud_url if cloud_url else \"SKIPPED (set ZV_BENCH_S3_URL to enable)\"}')
"""),
    ("md", "## 2 · Synthetic input (shared across backends)"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, N).astype(np.float32)
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
def run_one(label, target, backend):
    \"\"\"Time write + read against `target` using `backend`. Returns row dict.\"\"\"
    try:
        tw, _ = _time(
            write_points, target, positions,
            chunk_shape=CHUNK, bin_shape=BIN,
            attributes={'intensity': intensity},
            backend=backend,
        )
        tr, _ = _time(read_points, target, attribute_names=['intensity'], backend=backend)
        size_MB = _store_bytes(target) / 1e6 if backend == 'local' else float('nan')
        return {
            'backend': label,
            'write_s': round(tw, 3),
            'read_s':  round(tr, 3),
            'size_MB': round(size_MB, 2) if size_MB == size_MB else None,  # nan check
        }
    except (ImportError, StoreError) as e:
        return {'backend': label, 'write_s': None, 'read_s': None, 'size_MB': None, 'skipped': str(e)[:60]}


rows = []

# Local (always)
local_store = _new_store('backend_local')
rows.append(run_one('local', str(local_store), 'local'))
shutil.rmtree(Path(local_store).parent, ignore_errors=True)

# obstore / fsspec (only if cloud_url set)
if cloud_url:
    for backend in ('obstore', 'fsspec'):
        # Use a per-backend subpath so runs don't collide.
        target = cloud_url.rstrip('/') + f'/run_{backend}'
        rows.append(run_one(backend, target, backend))

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
plot_df = df.dropna(subset=['write_s', 'read_s'])
if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.18, plot_df['write_s'], width=0.36, label='write (s)')
    ax.bar(x + 0.18, plot_df['read_s'],  width=0.36, label='read (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['backend'])
    ax.set_ylabel('seconds')
    ax.set_title(f'Write / read time per backend (N = {N:,})')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
else:
    print('no successful backends to plot')
"""),
]


# ===================================================================
# Notebook builder
# ===================================================================

def _to_source(text: str) -> list[str]:
    """Match the multi-line `source` list shape Jupyter writes."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    if lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _build(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, text in cells:
        if kind == "md":
            nb_cells.append({
                "cell_type": "markdown",
                "id": _cell_id(),
                "metadata": {},
                "source": _to_source(text),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": _to_source(text),
            })
    return {
        "cells": nb_cells,
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


def _write(name: str, cells: list[tuple[str, str]]) -> None:
    out = ROOT / name
    out.write_text(
        json.dumps(_build(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    _write("01_size_scaling.ipynb", SIZE_CELLS)
    _write("02_data_types.ipynb", TYPES_CELLS)
    _write("03_backends.ipynb", BACKEND_CELLS)
