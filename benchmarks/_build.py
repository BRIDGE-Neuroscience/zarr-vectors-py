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


# Stats helpers used by the averaged benchmarks (01, 02).  Not needed in 03
# (backends) which still uses a single-run sketch.
STATS_HELPERS = """N_RUNS = 10
T95_DF9 = 2.262  # scipy.stats.t.ppf(0.975, df=9) — hard-coded to avoid scipy dep


def _mean_ci95(samples):
    \"\"\"(mean, half-width) for a 1-D sample using Student's t, df=n-1.\"\"\"
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    hw = T95_DF9 * s / np.sqrt(arr.size)
    return float(m), float(hw)
"""


# ===================================================================
# 01 · Size scaling — point cloud, vary N
# ===================================================================

SIZE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Size scaling — point cloud

Write/read/disk-size of point clouds at increasing `N`, with **CSV as a
baseline** for context. Same `chunk_shape` across runs so the only
variable is vertex count.

Each timing is averaged over **`N_RUNS = 10` runs**; the plot shows the
mean with a shaded **95% confidence interval** (Student's t, df=9).

For each `N` we measure:

| Operation | zarr-vectors | CSV (baseline) |
| --- | --- | --- |
| Write     | `write_points`           | `pandas.to_csv` |
| Read all  | `read_points`            | `pandas.read_csv` |
| Read one  | one chunk via lazy API   | `read_csv` at a **random row index** |
| Disk size | store directory          | CSV file |

CSV "read one" uses a random row each run (CSV has no random access, so the
parser must scan from the file head — this is the honest comparison vs.
zarr-vectors' chunk-level lookup).

Runtime: ~5 minutes on a laptop (the 1M case dominates).
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.lazy import open_zv

SIZES = [1_000, 10_000, 100_000, 1_000_000]
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0


def _csv_path(prefix):
    \"\"\"Fresh tempdir + CSV path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'csvbench_{prefix}_')) / 'points.csv'


def _csv_write(path, positions, intensity):
    \"\"\"Baseline: write x,y,z,intensity columns to a CSV.\"\"\"
    pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'intensity': intensity,
    }).to_csv(path, index=False)


def _csv_read_all(path):
    \"\"\"Read every row back into memory.\"\"\"
    return pd.read_csv(path)


def _csv_read_one(path, row_idx):
    \"\"\"Random-index single-row read. CSV has no random access, so the
    parser must scan from the top — this exposes the linear-scan cost.\"\"\"
    return pd.read_csv(path, skiprows=range(1, row_idx + 1), nrows=1)


def _zv_read_one(store_path):
    \"\"\"Read just one chunk's worth of vertices via the lazy API.\"\"\"
    zv = open_zv(store_path)
    chunk_keys = zv[0].vertices._chunk_keys  # noqa: SLF001 — minimal demo
    if not chunk_keys:
        return None
    return zv[0].vertices[chunk_keys[0]].compute()
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)

# Per-(metric, prefix) raw timings; shape becomes (len(SIZES), N_RUNS) after fill.
metrics = ['write_s', 'read_all_s', 'read_one_s']
raw = {m: {p: np.zeros((len(SIZES), N_RUNS)) for p in ('zv', 'csv')} for m in metrics}
sizes_MB = {'zv': np.zeros(len(SIZES)), 'csv': np.zeros(len(SIZES))}

for i, n in enumerate(SIZES):
    # Generate input data once per N (re-running write_points with the same
    # input is the realistic re-run scenario; we still get fresh stores).
    positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, n).astype(np.float32)

    for run in range(N_RUNS):
        # ---- ZV: fresh store each run (cold-cache write) ----
        store = _new_store(f'size_{n}_r{run}')
        t_zv_write,    _ = _time(
            write_points, store, positions,
            chunk_shape=CHUNK, bin_shape=BIN,
            vertex_attributes={'intensity': intensity},
        )
        t_zv_read_all, _ = _time(read_points, store, attribute_names=['intensity'])
        t_zv_read_one, _ = _time(_zv_read_one, store)
        if run == 0:
            sizes_MB['zv'][i] = _store_bytes(store) / 1e6
        shutil.rmtree(Path(store).parent, ignore_errors=True)

        # ---- CSV baseline ----
        csv = _csv_path(f'size_{n}_r{run}')
        t_csv_write,    _ = _time(_csv_write, csv, positions, intensity)
        t_csv_read_all, _ = _time(_csv_read_all, csv)
        # Fresh random row index per run — CSV pays linear scan cost.
        row_idx = int(rng.integers(0, n))
        t_csv_read_one, _ = _time(_csv_read_one, csv, row_idx)
        if run == 0:
            sizes_MB['csv'][i] = csv.stat().st_size / 1e6
        shutil.rmtree(csv.parent, ignore_errors=True)

        raw['write_s']['zv'][i, run]     = t_zv_write
        raw['read_all_s']['zv'][i, run]  = t_zv_read_all
        raw['read_one_s']['zv'][i, run]  = t_zv_read_one
        raw['write_s']['csv'][i, run]    = t_csv_write
        raw['read_all_s']['csv'][i, run] = t_csv_read_all
        raw['read_one_s']['csv'][i, run] = t_csv_read_one

# Summarise into a tidy dataframe (mean + 95% CI half-width per metric).
rows = []
for i, n in enumerate(SIZES):
    row = {'N': n}
    for m in metrics:
        for p in ('zv', 'csv'):
            mean, hw = _mean_ci95(raw[m][p][i])
            row[f'{p}_{m}_mean'] = round(mean, 4)
            row[f'{p}_{m}_hw']   = round(hw, 4)
    row['zv_size_MB']  = round(sizes_MB['zv'][i],  2)
    row['csv_size_MB'] = round(sizes_MB['csv'][i], 2)
    rows.append(row)

df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
PANELS = [
    ('Write time', 'write_s',    's'),
    ('Read all',   'read_all_s', 's'),
    ('Read one',   'read_one_s', 's'),
]
SERIES = [
    ('zarr-vectors', 'zv',  'C0'),
    ('csv',          'csv', 'C1'),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Three timing panels: mean line + shaded 95% CI band on log-log.
for ax, (title, key, unit) in zip(axes[:3], PANELS):
    for label, prefix, color in SERIES:
        mean = df[f'{prefix}_{key}_mean'].values
        hw   = df[f'{prefix}_{key}_hw'].values
        ax.fill_between(df['N'], mean - hw, mean + hw, color=color, alpha=0.2)
        ax.loglog(df['N'], mean, 'o-', color=color, label=label)
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

# Disk size panel: single deterministic measurement per series — no CI.
ax = axes[3]
for label, prefix, color in SERIES:
    ax.loglog(df['N'], df[f'{prefix}_size_MB'], 'o-', color=color, label=label)
ax.set_xlabel('N (vertices)')
ax.set_ylabel('MB')
ax.set_title('Disk size')
ax.grid(True, which='both', alpha=0.3)

axes[0].legend(loc='best')
fig.suptitle(
    f'zarr-vectors vs CSV — point cloud scaling ({N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 02 · Data types — fixed N, sweep the six types
# ===================================================================

TYPES_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Data-type scaling

Six geometry types benchmarked at the same target vertex count
(`N = 50_000`, ± a few %).  Each type is run in two regimes:

- **representative** — natural per-type layout (polylines as locally
  coherent random walks, mesh as a triangulated grid, lines / graph /
  skeleton with scattered positions);
- **scattered** — every vertex drawn independently uniformly across
  the volume, with topology unchanged.  Forces near-maximal cross-chunk
  traffic for types whose representative layout is spatially coherent.

The two regimes isolate the two cost components: per-vertex / per-object
overhead intrinsic to the write/read path, and the cross-chunk cost
driven by spatial distribution.

Each timing is averaged over **`N_RUNS = 10` runs**; bars show the mean
with **95% confidence interval** error bars (Student's t, df=9).
The results table reports the structural metrics (`vertex_count`,
`object_count`, `cross_chunk_count`, `size_MB`) so residual differences
between types can be attributed to specific drivers.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
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
    ("md", "## 2 · Synthetic generators (representative + scattered)"),
    ("code", """\
# Representative — natural per-type layout, ~N total vertices.
def _rep_points():
    return rng.uniform(0, 1000, (N, 3)).astype(np.float32)


def _rep_lines():
    # N/2 segments, two random endpoints each → ~N total vertices.
    return rng.uniform(0, 1000, (N // 2, 2, 3)).astype(np.float32)


def _rep_polylines():
    # ~N total vertices spread across short, locally coherent walks.
    counts = rng.integers(8, 16, size=N // 12)
    out = []
    for c in counts:
        steps = rng.normal(0, 5, (c, 3))
        start = rng.uniform(0, 1000, 3)
        out.append((start + steps.cumsum(axis=0)).astype(np.float32))
    return out


def _rep_graph(is_tree=False):
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    if is_tree:
        parents = rng.integers(0, np.arange(1, N))
        edges = np.stack([np.arange(1, N), parents], axis=1).astype(np.int64)
    else:
        src = rng.integers(0, N, size=3 * N // 2)
        dst = rng.integers(0, N, size=3 * N // 2)
        mask = src != dst
        edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)
    return positions, edges


def _rep_mesh():
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(np.linspace(0, 1000, side), np.linspace(0, 1000, side))
    zs = rng.uniform(0, 100, (side, side))
    verts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(side - 1); j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces


# Scattered — same topology, every vertex independently uniform in [0, 1000]^3.
# Types whose representative is already uniform reuse the rep generator.
_scat_points = _rep_points
_scat_lines  = _rep_lines
_scat_graph  = _rep_graph


def _scat_polylines():
    counts = rng.integers(8, 16, size=N // 12)
    return [
        rng.uniform(0, 1000, (c, 3)).astype(np.float32)
        for c in counts
    ]


def _scat_mesh():
    # Same grid faces, but vertex positions are uniformly random.
    side = int(np.sqrt(N))
    n_verts = side * side
    verts = rng.uniform(0, 1000, (n_verts, 3)).astype(np.float32)
    i = np.arange(side - 1); j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
def _extract_stats(name, r):
    \"\"\"Normalize each write_*'s return dict into (vertex_count, object_count, cross_chunk_count).\"\"\"
    if name == 'points':
        return r['vertex_count'], r['object_count'], 0
    if name == 'lines':
        return r['line_count'] * 2, r['line_count'], r['cross_chunk_count']
    if name == 'polylines':
        return r['vertex_count'], r['polyline_count'], r['cross_chunk_link_count']
    if name in ('graph', 'skeleton'):
        return r['node_count'], r['node_count'], r['cross_edge_count']
    if name == 'mesh':
        return r['vertex_count'], r['face_count'], r['cross_face_count']
    raise ValueError(name)


def bench_points(gen):
    pts = gen()
    store = _new_store('points')
    tw, res = _time(write_points, store, pts, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_points, store)
    return tw, tr, res, _store_bytes(store), store

def bench_lines(gen):
    eps = gen()
    store = _new_store('lines')
    tw, res = _time(write_lines, store, eps, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_lines, store)
    return tw, tr, res, _store_bytes(store), store

def bench_polylines(gen):
    plys = gen()
    store = _new_store('polylines')
    tw, res = _time(write_polylines, store, plys, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_polylines, store)
    return tw, tr, res, _store_bytes(store), store

def bench_graph(gen, kind):
    pos, edges = gen(is_tree=(kind == 'skeleton'))
    store = _new_store(kind)
    tw, res = _time(
        write_graph, store, pos, edges,
        chunk_shape=CHUNK, bin_shape=BIN, kind=kind,
    )
    tr, _ = _time(read_graph, store)
    return tw, tr, res, _store_bytes(store), store

def bench_mesh(gen):
    verts, faces = gen()
    store = _new_store('mesh')
    tw, res = _time(write_mesh, store, verts, faces, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_mesh, store)
    return tw, tr, res, _store_bytes(store), store


REGIMES = {
    'representative': {
        'points':    lambda: bench_points(_rep_points),
        'lines':     lambda: bench_lines(_rep_lines),
        'polylines': lambda: bench_polylines(_rep_polylines),
        'graph':     lambda: bench_graph(_rep_graph, kind='graph'),
        'skeleton':  lambda: bench_graph(_rep_graph, kind='skeleton'),
        'mesh':      lambda: bench_mesh(_rep_mesh),
    },
    'scattered': {
        'points':    lambda: bench_points(_scat_points),
        'lines':     lambda: bench_lines(_scat_lines),
        'polylines': lambda: bench_polylines(_scat_polylines),
        'graph':     lambda: bench_graph(_scat_graph, kind='graph'),
        'skeleton':  lambda: bench_graph(_scat_graph, kind='skeleton'),
        'mesh':      lambda: bench_mesh(_scat_mesh),
    },
}
TYPES = ['points', 'lines', 'polylines', 'graph', 'skeleton', 'mesh']

raw    = {r: {name: {'write_s': [], 'read_s': []} for name in TYPES} for r in REGIMES}
struct = {r: {} for r in REGIMES}
sizes  = {r: {} for r in REGIMES}

for regime, fns in REGIMES.items():
    for name in TYPES:
        for run in range(N_RUNS):
            tw, tr, res, nbytes, store = fns[name]()
            raw[regime][name]['write_s'].append(tw)
            raw[regime][name]['read_s'].append(tr)
            if run == 0:
                struct[regime][name] = _extract_stats(name, res)
                sizes[regime][name]  = nbytes
            shutil.rmtree(Path(store).parent, ignore_errors=True)

# Tidy long-form df: one row per (regime, type, op).
rows = []
for regime in REGIMES:
    for name in TYPES:
        v_count, o_count, cc_count = struct[regime][name]
        for op in ('write_s', 'read_s'):
            mean, hw = _mean_ci95(raw[regime][name][op])
            rows.append({
                'regime':            regime,
                'type':              name,
                'op':                op.replace('_s', ''),
                'mean_s':            round(mean, 4),
                'ci_hw':             round(hw, 4),
                'vertex_count':      int(v_count),
                'object_count':      int(o_count),
                'cross_chunk_count': int(cc_count),
                'size_MB':           round(sizes[regime][name] / 1e6, 2),
            })

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
x = np.arange(len(TYPES))
w = 0.36

for ax, regime in zip(axes, list(REGIMES)):
    sub   = df[df['regime'] == regime]
    write = sub[sub['op'] == 'write'].set_index('type').loc[TYPES]
    read  = sub[sub['op'] == 'read' ].set_index('type').loc[TYPES]
    ax.bar(x - w/2, write['mean_s'], width=w, yerr=write['ci_hw'],
           capsize=3, label='write')
    ax.bar(x + w/2, read['mean_s'],  width=w, yerr=read['ci_hw'],
           capsize=3, label='read')
    ax.set_xticks(x)
    ax.set_xticklabels(TYPES, rotation=20)
    ax.set_ylabel('seconds')
    ax.set_title(regime)
    ax.grid(True, axis='y', alpha=0.3)

axes[0].legend()
fig.suptitle(
    f'Write / read time per type '
    f'(N ≈ {N:,}, {N_RUNS} runs, 95% CI)'
)
fig.tight_layout()
plt.show()
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
