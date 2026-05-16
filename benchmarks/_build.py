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
# 04 · Pyramid scaling — points + graphs, vary N and coarsening ratio
# ===================================================================

PYRAMID_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Pyramid construction scaling

Time `coarsen_level` for **point clouds** (vertex-only) and **graphs**
(N nodes + edges) at increasing vertex count `N`, comparing three
uniform coarsening ratios `R ∈ {2, 4, 8}` over 3 levels (resulting
pyramids span level 0 / 1 / 2 / 3).

Each `(N, R, geometry)` cell is averaged over **`N_RUNS = 10` runs**;
the plot shows the mean with a shaded **95% confidence interval**
(Student's t, df=9).

Per-level timings come from `coarsen_level(source, target,
coarsen_factor=R, sparsity_factor=1.0)` called in a loop — same
work as `build_pyramid(factors=[(R, 1)]*3)` but with one timing per
hop. `coarsen_level`'s default `cross_level_storage="none"` isolates
raw coarsening cost; full `build_pyramid` runs with
`cross_level_storage="explicit"` and is moderately slower.

Runtime: ~15–30 minutes on a laptop (the 1 M tier dominates).
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.multiresolution.coarsen import coarsen_level

SIZES  = [10_000, 100_000, 1_000_000]
RATIOS = [2, 4, 8]
N_LVLS = 3                    # builds levels 1, 2, 3 from level 0
CHUNK  = (200.0, 200.0, 200.0)
BIN    = (50.0,  50.0,  50.0)
SEED   = 0
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)

metrics = ['lvl1_s', 'lvl2_s', 'lvl3_s', 'total_s']
# raw[metric][(geom, ratio)] is shape (len(SIZES), N_RUNS)
raw = {
    m: {
        (g, r): np.zeros((len(SIZES), N_RUNS))
        for g in ('points', 'graph')
        for r in RATIOS
    }
    for m in metrics
}

for i, n in enumerate(SIZES):
    positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
    # ~3N/2 random undirected edges, dedup'd against self-loops.
    src = rng.integers(0, n, size=3 * n // 2)
    dst = rng.integers(0, n, size=3 * n // 2)
    mask = src != dst
    edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)

    geometries = [
        ('points', write_points, (positions,)),
        ('graph',  write_graph,  (positions, edges)),
    ]

    for ratio in RATIOS:
        for run in range(N_RUNS):
            for geom, write_fn, args in geometries:
                store = _new_store(f'pyr_{geom}_n{n}_r{ratio}_run{run}')
                write_fn(store, *args, chunk_shape=CHUNK, bin_shape=BIN)
                per_lvl = []
                for lvl in range(N_LVLS):
                    t, _ = _time(
                        coarsen_level, store,
                        source_level=lvl, target_level=lvl + 1,
                        coarsen_factor=float(ratio),
                        sparsity_factor=1.0,
                    )
                    per_lvl.append(t)
                raw['lvl1_s'][(geom, ratio)][i, run]  = per_lvl[0]
                raw['lvl2_s'][(geom, ratio)][i, run]  = per_lvl[1]
                raw['lvl3_s'][(geom, ratio)][i, run]  = per_lvl[2]
                raw['total_s'][(geom, ratio)][i, run] = sum(per_lvl)
                shutil.rmtree(Path(store).parent, ignore_errors=True)

# Summarise into a tidy dataframe: one row per (N, geom, ratio).
rows = []
for i, n in enumerate(SIZES):
    for geom in ('points', 'graph'):
        for ratio in RATIOS:
            row = {'N': n, 'geom': geom, 'ratio': ratio}
            for m in metrics:
                mean, hw = _mean_ci95(raw[m][(geom, ratio)][i])
                row[f'{m}_mean'] = round(mean, 4)
                row[f'{m}_hw']   = round(hw,   4)
            rows.append(row)

df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
COLORS = {2: 'C0', 4: 'C1', 8: 'C2'}
GEOMS  = ['points', 'graph']

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for row, geom in enumerate(GEOMS):
    # Left column: total build time vs N (one line per ratio, log-log).
    ax = axes[row, 0]
    for ratio in RATIOS:
        sub  = df[(df.geom == geom) & (df.ratio == ratio)].sort_values('N')
        mean = sub['total_s_mean'].values
        hw   = sub['total_s_hw'].values
        ax.fill_between(
            sub['N'], mean - hw, mean + hw, color=COLORS[ratio], alpha=0.2,
        )
        ax.loglog(
            sub['N'], mean, 'o-', color=COLORS[ratio], label=f'R={ratio}',
        )
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel('s')
    ax.set_title(f'{geom}: total build time vs N')
    ax.grid(True, which='both', alpha=0.3)

    # Right column: per-level breakdown at the largest N (grouped bar).
    ax = axes[row, 1]
    x_pos = np.arange(N_LVLS)
    for j, ratio in enumerate(RATIOS):
        sub = df[(df.geom == geom)
                 & (df.ratio == ratio)
                 & (df.N == max(SIZES))].iloc[0]
        means = [sub[f'lvl{k + 1}_s_mean'] for k in range(N_LVLS)]
        hws   = [sub[f'lvl{k + 1}_s_hw']   for k in range(N_LVLS)]
        ax.bar(
            x_pos + (j - 1) * 0.27, means, width=0.27,
            yerr=hws, capsize=3, color=COLORS[ratio], label=f'R={ratio}',
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['L0→L1', 'L1→L2', 'L2→L3'])
    ax.set_ylabel('s')
    ax.set_title(f'{geom}: per-level breakdown @ N={max(SIZES):,}')
    ax.grid(True, axis='y', alpha=0.3)

axes[0, 0].legend(loc='best')
fig.suptitle(
    f'Pyramid construction scaling ({N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 05 · Spatial bbox queries — zarr-vectors vs pandas/CSV
# ===================================================================

BBOX_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Spatial bbox queries

Fixed `N = 1_000_000` point cloud uniformly distributed in
`[0, 1000)³`.  Sweep bbox volume fraction `V ∈ {0.001, 0.01, 0.1, 1.0}`
of the domain and time each side reading just the points inside the
bbox:

- `zarr-vectors`: `open_zv(store)[0].filter(bbox=(lo, hi)).compute()`
  — only the chunks the bbox intersects are read.
- `pandas / CSV`: `pd.read_csv(path).query(...)` — no spatial index,
  must scan every row.

10 runs per `V`; each run draws a fresh random bbox location.  The
zarr-vectors line should stay roughly flat at small `V` and rise as
`V` approaches 1; the CSV line should be roughly flat at the
full-file-scan cost regardless of `V`.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points
from zarr_vectors.lazy import open_zv

N = 1_000_000
V_FRACTIONS = [0.001, 0.01, 0.1, 1.0]
DOMAIN = 1000.0
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0,  50.0,  50.0)
SEED  = 0


def _csv_path(prefix):
    return Path(tempfile.mkdtemp(prefix=f'csvbench_{prefix}_')) / 'points.csv'
"""),
    ("md", "## 2 · Write the shared input store and CSV baseline"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)

# zarr-vectors store: written once and reused across every (V, run).
zv_store = _new_store(f'bbox_zv_{N}')
write_points(zv_store, positions, chunk_shape=CHUNK, bin_shape=BIN)
zv_size_MB = _store_bytes(zv_store) / 1e6

# CSV baseline: same data dumped as text.
csv_path = _csv_path(f'bbox_csv_{N}')
pd.DataFrame({
    'x': positions[:, 0], 'y': positions[:, 1], 'z': positions[:, 2],
}).to_csv(csv_path, index=False)
csv_size_MB = csv_path.stat().st_size / 1e6

print(f'zarr-vectors store: {zv_size_MB:.2f} MB')
print(f'CSV baseline:       {csv_size_MB:.2f} MB')
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
metrics = ['zv_s', 'csv_s', 'returned_count']
raw = {m: np.zeros((len(V_FRACTIONS), N_RUNS)) for m in metrics}

for i, v in enumerate(V_FRACTIONS):
    edge = DOMAIN * (v ** (1 / 3))
    for run in range(N_RUNS):
        lo = rng.uniform(0, DOMAIN - edge, 3).astype(np.float32)
        hi = (lo + edge).astype(np.float32)

        zv = open_zv(zv_store)
        t_zv, zv_out = _time(
            lambda: zv[0].filter(bbox=(lo, hi)).compute()
        )
        t_csv, csv_out = _time(
            lambda: pd.read_csv(csv_path).query(
                f'x >= {lo[0]} and x <= {hi[0]} and '
                f'y >= {lo[1]} and y <= {hi[1]} and '
                f'z >= {lo[2]} and z <= {hi[2]}'
            )
        )
        raw['zv_s'][i, run]            = t_zv
        raw['csv_s'][i, run]           = t_csv
        raw['returned_count'][i, run]  = zv_out['vertex_count']

rows = []
for i, v in enumerate(V_FRACTIONS):
    row = {'V': v, 'V_pct': v * 100}
    for m in metrics:
        mean, hw = _mean_ci95(raw[m][i])
        row[f'{m}_mean'] = round(mean, 4)
        row[f'{m}_hw']   = round(hw,   4)
    rows.append(row)
df = pd.DataFrame(rows)
shutil.rmtree(Path(zv_store).parent, ignore_errors=True)
shutil.rmtree(csv_path.parent, ignore_errors=True)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

SERIES = [
    ('zarr-vectors', 'zv',  'C0'),
    ('csv',          'csv', 'C1'),
]

# Panel 1: time vs V on log-log.
ax = axes[0]
for label, prefix, color in SERIES:
    mean = df[f'{prefix}_s_mean'].values
    hw   = df[f'{prefix}_s_hw'].values
    ax.fill_between(df['V'], mean - hw, mean + hw, color=color, alpha=0.2)
    ax.loglog(df['V'], mean, 'o-', color=color, label=label)
ax.set_xlabel('bbox volume fraction')
ax.set_ylabel('seconds')
ax.set_title(f'Bbox read time vs V (N = {N:,})')
ax.grid(True, which='both', alpha=0.3)
ax.legend()

# Panel 2: returned vertex count vs V (linearity check).
ax = axes[1]
mean = df['returned_count_mean'].values
ax.loglog(df['V'], mean, 'o-', color='C2', label='vertex_count')
ax.set_xlabel('bbox volume fraction')
ax.set_ylabel('returned vertices')
ax.set_title('Linearity check: count ∝ V')
ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    f'Spatial bbox queries — zarr-vectors vs CSV ({N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 06 · Chunk-shape sensitivity — points, fixed N, sweep chunk_shape
# ===================================================================

CHUNK_SHAPE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Chunk-shape sensitivity

Fixed `N = 500_000` point cloud uniformly distributed in `[0, 1000)³`.
Sweep `chunk_shape` ∈ `{50, 100, 200, 400, 800}` (uniform 3-tuples) and
measure write, full read, **bbox read** (1% of volume), disk size, and
chunk count.  `bin_shape = chunk_shape / 4` for every run.

The bbox panel is the headline.  Expect a U-shape: tiny chunks pay
per-file metadata overhead, huge chunks waste I/O on data the bbox
didn't ask for.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.lazy import open_zv

N           = 500_000
CHUNK_SIZES = [50, 100, 200, 400, 800]
DOMAIN      = 1000.0
BBOX_V      = 0.01           # 1% volume read-bbox per run
SEED        = 0
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)

metrics = ['write_s', 'read_all_s', 'read_bbox_s']
raw = {m: np.zeros((len(CHUNK_SIZES), N_RUNS)) for m in metrics}
sizes_MB    = np.zeros(len(CHUNK_SIZES))
chunk_count = np.zeros(len(CHUNK_SIZES))

for i, cs in enumerate(CHUNK_SIZES):
    chunk_shape = (float(cs),) * 3
    bin_shape   = (cs / 4.0,) * 3
    edge = DOMAIN * (BBOX_V ** (1 / 3))

    for run in range(N_RUNS):
        store = _new_store(f'cs_{cs}_run{run}')

        t_w, _ = _time(
            write_points, store, positions,
            chunk_shape=chunk_shape, bin_shape=bin_shape,
        )
        t_r, _ = _time(read_points, store)

        lo = rng.uniform(0, DOMAIN - edge, 3).astype(np.float32)
        hi = (lo + edge).astype(np.float32)
        zv = open_zv(store)
        t_b, _ = _time(lambda: zv[0].filter(bbox=(lo, hi)).compute())

        raw['write_s'][i, run]     = t_w
        raw['read_all_s'][i, run]  = t_r
        raw['read_bbox_s'][i, run] = t_b

        if run == 0:
            sizes_MB[i]    = _store_bytes(store) / 1e6
            chunk_count[i] = len(zv[0].chunk_keys)

        shutil.rmtree(Path(store).parent, ignore_errors=True)

rows = []
for i, cs in enumerate(CHUNK_SIZES):
    row = {'chunk_shape': cs, 'size_MB': round(sizes_MB[i], 2),
           'chunk_count': int(chunk_count[i])}
    for m in metrics:
        mean, hw = _mean_ci95(raw[m][i])
        row[f'{m}_mean'] = round(mean, 4)
        row[f'{m}_hw']   = round(hw,   4)
    rows.append(row)
df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

def _line(ax, key, ylabel, title, color):
    mean = df[f'{key}_mean'].values
    hw   = df[f'{key}_hw'].values
    ax.fill_between(df['chunk_shape'], mean - hw, mean + hw,
                    color=color, alpha=0.2)
    ax.loglog(df['chunk_shape'], mean, 'o-', color=color)
    ax.set_xlabel('chunk_shape (per axis)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

_line(axes[0, 0], 'write_s',     's', 'Write time',       'C0')
_line(axes[0, 1], 'read_all_s',  's', 'Read all',         'C1')
_line(axes[1, 0], 'read_bbox_s', 's', f'Read bbox (V={BBOX_V*100}%)', 'C2')

# Disk size + chunk count on the same axis.
ax = axes[1, 1]
ax.loglog(df['chunk_shape'], df['size_MB'], 'o-', color='C3', label='disk MB')
ax.set_xlabel('chunk_shape (per axis)')
ax.set_ylabel('MB')
ax.grid(True, which='both', alpha=0.3)
ax_r = ax.twinx()
ax_r.loglog(df['chunk_shape'], df['chunk_count'], 's--', color='C4',
            label='chunk count')
ax_r.set_ylabel('chunk count')
ax.set_title('Disk MB and chunk count')
ax.legend(loc='upper left')
ax_r.legend(loc='lower right')

fig.suptitle(
    f'Chunk-shape sensitivity — point cloud (N = {N:,}, '
    f'{N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 07 · Compression / codec impact — fixed N, sweep compressor
# ===================================================================

COMPRESSION_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Compression / codec impact

Fixed `N = 1_000_000` point cloud.  Sweep the `compressor=` kwarg
(added in the production refactor) across five values and measure
write time, read time, and disk MB.

| Label | `compressor=` value |
|-------|--------------------|
| `none` | `'none'` — uncompressed chunks |
| `zstd` (default) | `None` — zarr v3's default |
| `blosc_l1` | `[{...}]` Blosc + Zstd level 1, no shuffle |
| `blosc_l5` | `[{...}]` Blosc + Zstd level 5, no shuffle |
| `blosc_l5_bitshuffle` | `'blosc'` — Blosc + Zstd level 5 + BitShuffle |

Compression CPU usually pays off on disk; bitshuffle is expected to
beat plain Zstd on float32 positions.  See
[`docs/spec/foundations/codec_pipeline.md`](../docs/spec/foundations/codec_pipeline.md)
for the kwarg semantics.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points

N      = 1_000_000
CHUNK  = (200.0, 200.0, 200.0)
BIN    = (50.0,  50.0,  50.0)
SEED   = 0

CONFIGS = [
    ('none',                'none'),
    ('zstd',                None),
    ('blosc_l1',            [{'name': 'blosc', 'configuration': {
                                'cname': 'zstd', 'clevel': 1,
                                'shuffle': 'noshuffle',
                                'typesize': 4, 'blocksize': 0}}]),
    ('blosc_l5',            [{'name': 'blosc', 'configuration': {
                                'cname': 'zstd', 'clevel': 5,
                                'shuffle': 'noshuffle',
                                'typesize': 4, 'blocksize': 0}}]),
    ('blosc_l5_bitshuffle', 'blosc'),
]
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)

metrics = ['write_s', 'read_all_s']
raw = {m: np.zeros((len(CONFIGS), N_RUNS)) for m in metrics}
disk_MB = np.zeros(len(CONFIGS))

for i, (label, compressor) in enumerate(CONFIGS):
    for run in range(N_RUNS):
        store = _new_store(f'codec_{label}_run{run}')
        t_w, _ = _time(
            write_points, store, positions,
            chunk_shape=CHUNK, bin_shape=BIN, compressor=compressor,
        )
        t_r, _ = _time(read_points, store)
        raw['write_s'][i, run]    = t_w
        raw['read_all_s'][i, run] = t_r
        if run == 0:
            disk_MB[i] = _store_bytes(store) / 1e6
        shutil.rmtree(Path(store).parent, ignore_errors=True)

rows = []
for i, (label, _) in enumerate(CONFIGS):
    row = {'config': label, 'disk_MB': round(disk_MB[i], 2)}
    for m in metrics:
        mean, hw = _mean_ci95(raw[m][i])
        row[f'{m}_mean'] = round(mean, 4)
        row[f'{m}_hw']   = round(hw,   4)
    rows.append(row)
df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x = np.arange(len(df))
ROT = 20

# Panel 1: write time.
ax = axes[0]
ax.bar(x, df['write_s_mean'], yerr=df['write_s_hw'],
       capsize=3, color='C0')
ax.set_xticks(x); ax.set_xticklabels(df['config'], rotation=ROT, ha='right')
ax.set_ylabel('s')
ax.set_title('Write time')
ax.grid(True, axis='y', alpha=0.3)

# Panel 2: read all time.
ax = axes[1]
ax.bar(x, df['read_all_s_mean'], yerr=df['read_all_s_hw'],
       capsize=3, color='C1')
ax.set_xticks(x); ax.set_xticklabels(df['config'], rotation=ROT, ha='right')
ax.set_ylabel('s')
ax.set_title('Read all')
ax.grid(True, axis='y', alpha=0.3)

# Panel 3: disk MB.
ax = axes[2]
ax.bar(x, df['disk_MB'], color='C3')
ax.set_xticks(x); ax.set_xticklabels(df['config'], rotation=ROT, ha='right')
ax.set_ylabel('MB')
ax.set_title('Disk size')
ax.grid(True, axis='y', alpha=0.3)

ratio = df.loc[df['config'] == 'none', 'disk_MB'].iloc[0] / \\
        df.loc[df['config'] == 'blosc_l5_bitshuffle', 'disk_MB'].iloc[0]
fig.suptitle(
    f'Compression impact — point cloud (N = {N:,}, '
    f'{N_RUNS} runs, 95% CI) — blosc bitshuffle gives {ratio:.1f}× smaller disk',
)
plt.tight_layout()
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
    _write("04_pyramid_scaling.ipynb", PYRAMID_CELLS)
    _write("05_bbox_queries.ipynb", BBOX_CELLS)
    _write("06_chunk_shape.ipynb", CHUNK_SHAPE_CELLS)
    _write("07_compression.ipynb", COMPRESSION_CELLS)
