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
| `experiments/compression/slurm/03_tiling_sweep.sbatch` | 256, 512, 1024, and 2048 tiling experiment |
| `experiments/compression/slurm/04_hpc_scaling_array.sbatch` | Multiple jobs in parallel |
| `experiments/compression/slurm/05_storage_compare.sbatch` | Shared filesystem vs node-local staged storage |
| `experiments/compression/slurm/06_summarize_all.sbatch` | Final all-run summary table |
| `experiments/compression/slurm/run_all_suite.sh` | Submits the full suite |
| `experiments/compression/slurm/run_jacob_compression_suite.sh` | Submits Jacob's compression-side suite without tiling |

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
- metrics: runtime, peak GPU memory, BPP, compression ratio, PSNR, SSIM, MSE, RMSE, MAE, high-percentile absolute error, mean bias

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
- checkpoint vs PSNR, SSIM, and pixel-error metrics

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

- 256 x 256
- 512 x 512
- 1024 x 1024
- 2048 x 2048 if memory allows

Each tile is compressed independently and stitched back together. The runner records seam artifact metrics and saves stitched visuals for inspection.

Status on 2026-05-12: the DeltaAI GH200 `N_IMAGES=8` pilot completed for no tiling, `512 x 512`, `1024 x 1024`, and `2048 x 2048` tiles. The `512 x 512` case reduced wall time from `143.55` seconds to `86.01` seconds and peak GPU memory from `52.0` GB to `3.0` GB, with PSNR `29.73` and SSIM `0.8822`.

Next experiment: add a `256 x 256` tiling case and rerun the same image subset first. This tests whether smaller tiles reduce memory further, whether many small tiles add enough overhead to slow the run, and whether fine-grained tile boundaries create visible artifacts. Keep `512 x 512` as the current baseline to beat.

Run order:

1. `256 x 256` smoke test on two images.
2. Full tiling sweep on eight images: no tiling, `256 x 256`, `512 x 512`, `1024 x 1024`, and `2048 x 2048`.
3. Larger selected-run validation after the best setup is clear.

Quality checks:

- numeric table: compression ratio, PSNR, SSIM, MSE, RMSE, MAE, `error_p95`, `error_p99`, seam metrics
- visual table: original preview, reconstruction preview, absolute-error heatmap, side-by-side comparison panel
- failure mode to watch: regular tile-boundary patterns in heatmaps or seam-region crops

## Main Deliverable Table

Use `combined_summary.md` or `combined_summary.csv` from each run folder.

Minimum columns for the meeting:

| resolution | batch size | tile size | compression setting | time | peak GPU memory | compressed size | compression ratio | PSNR | SSIM | MAE | error p99 | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 4K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 2K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | none |
| 1K | 1 | none | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | none |
| native | 8 | 256 | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | heatmap + seams |
| native | 4 | 512 | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | heatmap + seams |
| native | 4 | 1024 | baseline_b02048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | heatmap + seams |

## First Commands to Run

Start with a small smoke-size run:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code
N_IMAGES=2 sbatch experiments/compression/slurm/01_baseline_resolution_batch.sbatch
TILING_SIZES="256" N_IMAGES=2 SAVE_VISUAL_LIMIT=2 sbatch experiments/compression/slurm/03_tiling_sweep.sbatch
```

For Jacob's compression-side task, use the dedicated no-tiling launcher:

```bash
RUN_STAMP=20260515_jacob_compression_smoke N_IMAGES=4 SHARD_IMAGES=4 SAVE_VISUAL_LIMIT=1 experiments/compression/slurm/run_jacob_compression_suite.sh
```

If those succeed, run:

```bash
TILING_SIZES="256 512 1024 2048" N_IMAGES=8 sbatch experiments/compression/slurm/03_tiling_sweep.sbatch
```

For final poster numbers, rerun with:

```bash
TILING_SIZES="256 512" N_IMAGES=100 SAVE_VISUAL_LIMIT=8 sbatch experiments/compression/slurm/03_tiling_sweep.sbatch
```
