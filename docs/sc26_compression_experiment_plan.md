# SC26 Compression Experiment Plan

## Goal

Make CDC compression fast, scalable, and storage-efficient enough for the SC26 poster draft planned for the end of May 2026.

## File Map

| Path | Use |
| --- | --- |
| `experiments/compression/run_compression_experiment.py` | Main runner for one experiment configuration |
| `experiments/compression/summarize_results.py` | Combines `summary.csv` files into one table |
| `experiments/compression/inspect_compression_controls.py` | Checks whether compression level is a runtime parameter |
| `experiments/compression/configs/deltaai_paths.env` | DeltaAI paths, checkpoint names, and default image counts |
| `experiments/compression/slurm/01_baseline_resolution_batch.sbatch` | Baseline, batch-size, and resolution sweep |
| `experiments/compression/slurm/02_checkpoint_level_sweep.sbatch` | Compression-level sweep through checkpoints |
| `experiments/compression/slurm/03_tiling_sweep.sbatch` | 512, 1024, and 2048 tiling experiment |
| `experiments/compression/slurm/04_hpc_scaling_array.sbatch` | Multiple jobs in parallel |
| `experiments/compression/slurm/05_storage_compare.sbatch` | Shared filesystem vs node-local staged storage |
| `experiments/compression/slurm/06_summarize_all.sbatch` | Final all-run summary table |
| `experiments/compression/slurm/run_all_suite.sh` | Submits the full suite |

## Timestamp Convention

All DeltaAI outputs go under a timestamped suite folder by default:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/
```

`RUN_STAMP` defaults to `YYYYMMDD_HHMMSS`. The Python runner also records:

- `run_start_utc`
- `run_end_utc`
- `run_start_local`
- `run_end_local`
- per-batch or per-image start/end timestamps in `results.csv`
- SLURM job ID and array task ID when available

You can set a human-readable timestamp label:

```bash
RUN_STAMP=20260504_meeting_prep N_IMAGES=8 experiments/compression/slurm/run_all_suite.sh
```

## Experiment Matrix

### Baseline

- native full-resolution images
- pretrained x-param checkpoint `b0.2048`
- batch sizes: 1, 2, 4
- metrics: runtime, peak GPU memory, BPP, compression ratio, PSNR, SSIM

### Resolution Sweep

- 4K: longest edge resized to 4096 pixels
- 2K: longest edge resized to 2048 pixels
- 1K: longest edge resized to 1024 pixels
- output: time vs resolution and size reduction vs resolution

### Compression-Level Sweep

The x-param workflow does not expose a direct runtime compression-level parameter. The suite therefore sweeps the available pretrained checkpoints and uses measured BPP to identify practical low, medium, and high compression settings.

Output:

- checkpoint vs runtime
- checkpoint vs BPP
- checkpoint vs compression ratio
- checkpoint vs PSNR and SSIM

### Batch-Size Sweep

- batch size 1
- batch size 2
- batch size 4

Use the summary table to decide whether batching improves throughput or mainly increases GPU memory.

### HPC Scaling

The scaling script runs a SLURM array with four independent jobs. Each job processes a different image shard.

Compare:

- one job processing `N` images
- four jobs processing four shards in parallel
- aggregate images/hour
- failure or queue behavior

### Storage Compare

The storage script runs the same setup twice:

- shared project filesystem
- node-local staged images and checkpoint

Use this to check whether I/O staging changes total wall time.

### Tiling

Tile sizes:

- 512 x 512
- 1024 x 1024
- 2048 x 2048 if memory allows

Each tile is compressed independently and stitched back together. The runner records seam artifact metrics and saves stitched visuals for inspection.

## Main Deliverable Table

Use `combined_summary.md` or `combined_summary.csv` from each run folder.

Minimum columns for the meeting:

| resolution | batch size | tile size | compression setting | time | peak GPU memory | compressed size | compression ratio | PSNR | SSIM | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 4K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 2K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 1K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | none |
| native | 4 | 512 | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | inspect seams |
| native | 4 | 1024 | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | inspect seams |

## First Commands to Run

Start with a small smoke-size run:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code
N_IMAGES=2 sbatch experiments/compression/slurm/01_baseline_resolution_batch.sbatch
N_IMAGES=2 sbatch experiments/compression/slurm/03_tiling_sweep.sbatch
```

If those succeed, run:

```bash
N_IMAGES=8 experiments/compression/slurm/run_all_suite.sh
```

For final poster numbers, rerun with:

```bash
N_IMAGES=100 experiments/compression/slurm/run_all_suite.sh
```
