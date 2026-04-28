# HPC pipelines

This guide covers writing ZVF stores from HPC (High Performance Computing)
environments: Slurm job arrays, MPI parallel writes, and Lustre/GPFS
file system optimisation.

---

## SLURM job array pattern

The simplest parallel write strategy: split the dataset into spatial
partitions and run one SLURM job per partition.

### Submit script

```bash
#!/bin/bash
#SBATCH --job-name=zvf_write
#SBATCH --array=0-19          # 20 z-slabs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

python write_partition.py \
    --slab-index ${SLURM_ARRAY_TASK_ID} \
    --n-slabs 20 \
    --output /scratch/scan.zarrvectors
```

### `write_partition.py`

```python
import argparse
import numpy as np
from zarr_vectors.types.points import write_points_partition
from zarr_vectors.core.store import open_store
from zarr_vectors.core.multiscale import write_multiscale_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slab-index", type=int)
    parser.add_argument("--n-slabs", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    chunk_shape = (500., 500., 500.)
    total_z = 10_000.0
    slab_dz  = total_z / args.n_slabs
    z_lo = args.slab_index * slab_dz
    z_hi = z_lo + slab_dz

    positions  = load_slab_positions(z_lo, z_hi)
    attributes = load_slab_attributes(z_lo, z_hi)

    write_points_partition(
        args.output,
        positions,
        chunk_shape=chunk_shape,
        bin_shape=(125., 125., 125.),
        attributes={"intensity": attributes},
        z_range=(z_lo, z_hi),
    )
    print(f"Slab {args.slab_index} done: {len(positions)} vertices")

if __name__ == "__main__":
    main()
```

### Finalisation step (after all array jobs complete)

```bash
#!/bin/bash
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}
#SBATCH --ntasks=1
#SBATCH --mem=4G

python finalise_store.py --output /scratch/scan.zarrvectors
```

```python
# finalise_store.py
import argparse, zarr
from zarr_vectors.core.store import open_store
from zarr_vectors.core.multiscale import write_multiscale_metadata
from zarr_vectors.multiresolution.coarsen import build_pyramid

parser = argparse.ArgumentParser()
parser.add_argument("--output")
args = parser.parse_args()

root = open_store(args.output, mode="r+")
write_multiscale_metadata(root)
zarr.consolidate_metadata(root.store)

build_pyramid(args.output,
              level_configs=[{"bin_ratio":(2,2,2)}, {"bin_ratio":(4,4,4)}])
print("Store finalised.")
```

---

## MPI parallel writes

For tightly coupled parallel writes using `mpi4py`:

```python
from mpi4py import MPI
import numpy as np
from zarr_vectors.types.points import write_points_partition
from zarr_vectors.core.store import open_store
from zarr_vectors.core.multiscale import write_multiscale_metadata
import zarr

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

# Partition coordinate space by x-slabs
total_x = 10_000.0
slab_dx  = total_x / size
x_lo = rank * slab_dx
x_hi = x_lo + slab_dx

# Each rank loads and writes its own x-slab
positions = load_positions_x_range(x_lo, x_hi)
attrs     = load_attrs_x_range(x_lo, x_hi)

write_points_partition(
    "/scratch/scan.zarrvectors",
    positions,
    chunk_shape=(500., 500., 500.),
    bin_shape=(125., 125., 125.),
    attributes={"intensity": attrs},
    x_range=(x_lo, x_hi),
)

# All ranks wait before rank 0 finalises
comm.Barrier()

if rank == 0:
    root = open_store("/scratch/scan.zarrvectors", mode="r+")
    write_multiscale_metadata(root)
    zarr.consolidate_metadata(root.store)
    print(f"Wrote {size} partitions successfully.")
```

Run with:

```bash
mpirun -n 16 python mpi_write.py
```

---

## Lustre / GPFS optimisation

### Striping

Stripe the output directory across multiple OSTs before writing:

```bash
# Lustre: stripe across 8 OSTs (adjust for your system)
mkdir /scratch/scan.zarrvectors
lfs setstripe -c 8 /scratch/scan.zarrvectors
```

For stores with many chunks per directory, increase the directory stripe
count to avoid metadata bottlenecks:

```bash
lfs setstripe -c 1 -S 1m /scratch/scan.zarrvectors  # metadata stripe
lfs setstripe -c 8 -S 4m /scratch/scan.zarrvectors/resolution_0/vertices/c/
```

### Parallel I/O considerations

- **Avoid writing to the same chunk from multiple processes.** Each
  partition should write to a non-overlapping set of chunk coordinates.
  `write_points_partition` enforces this via the `x_range`/`z_range`
  argument.
- **Use large `chunk_shape` on Lustre.** Stripe granularity is typically
  1–4 MB; chunks smaller than this see no parallelism benefit. A chunk
  of 200 000 float32 vertices (≈ 2.4 MB) saturates a single stripe
  efficiently.
- **Avoid `O_SYNC` writes.** Zarr v3 does not use synchronous writes by
  default. If your Lustre mount forces `O_SYNC`, contact your sysadmin.

### Temporary local SSD → copy to Lustre

On clusters with local NVMe scratch (e.g. `/local/scratch`), write to
local storage first and copy to Lustre at the end:

```bash
# SLURM: write to local NVMe, copy to Lustre at job end
python write_partition.py --output /local/scratch/scan_${SLURM_JOB_ID}

# After job
rsync -a /local/scratch/scan_${SLURM_JOB_ID}.zarrvectors/ \
         /lustre/project/scan.zarrvectors/
```

---

## Writing to S3 from HPC

For cloud-destined datasets, write directly to S3 from the compute nodes
(if outbound internet is available):

```bash
# Install cloud extras in your conda environment
pip install "zarr-vectors[cloud]"
```

```python
import s3fs
from zarr_vectors.types.points import write_points_partition

fs    = s3fs.S3FileSystem(anon=False)
store = fs.get_mapper("s3://my-bucket/scan.zarrvectors")

write_points_partition(store, positions, chunk_shape=..., z_range=...)
```

On clusters without outbound internet, write to Lustre first, then copy
to S3 as a post-processing step:

```bash
# After all SLURM partitions complete
aws s3 sync /lustre/project/scan.zarrvectors/ \
            s3://my-bucket/scan.zarrvectors/ \
            --no-progress
```

---

## Recommended job sizing

| Dataset size | Strategy | SLURM resources |
|-------------|----------|-----------------|
| < 10M vertices | Single job | 1 node, 32 GB RAM, 4 CPU |
| 10M–1B vertices | Job array (spatial slabs) | 16–64 tasks, 16–32 GB each |
| > 1B vertices | MPI + streaming writer | 64–256 ranks, 8 GB each |
| Any size, S3 destination | Job array with direct S3 write | As above; add `[cloud]` extra |
