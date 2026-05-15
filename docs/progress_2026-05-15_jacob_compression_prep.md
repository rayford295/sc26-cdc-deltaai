# Jacob Compression-Side Experiment Prep for 2026-05-20

## Purpose

Prepare Yifan to run Jacob's compression-side experiments from the DeltaAI checkout before `2026-05-20`. The goal is to produce a small, defensible table for the group:

```text
resolution | batch size | compression setting | time | compression ratio | notes
```

The main question is simple: which compression setup is fast, scalable, and storage-efficient enough to support the shared CDC model workflow?

## What Is Already Ready

The repo already has runnable DeltaAI scripts for the requested tasks:

| Task from chat | Ready script | Output folder |
| --- | --- | --- |
| Baseline full-resolution run | `experiments/compression/slurm/01_baseline_resolution_batch.sbatch` | `01_baseline_resolution_batch/` |
| Resolution sweep, 4K to 2K to 1K | `experiments/compression/slurm/01_baseline_resolution_batch.sbatch` | `01_baseline_resolution_batch/` |
| Batch size 1, 2, 4 | `experiments/compression/slurm/01_baseline_resolution_batch.sbatch` | `01_baseline_resolution_batch/` |
| Compression level low/medium/high | `experiments/compression/slurm/02_checkpoint_level_sweep.sbatch` | `02_checkpoint_level_sweep/` |
| Multiple jobs in parallel | `experiments/compression/slurm/04_hpc_scaling_array.sbatch` | `04_hpc_scaling_array/` |
| Shared vs local storage | `experiments/compression/slurm/05_storage_compare.sbatch` | `05_storage_compare/` |
| Final combined summary | `experiments/compression/slurm/06_summarize_all.sbatch` | suite root |

There is also a Jacob-specific launcher:

```bash
experiments/compression/slurm/run_jacob_compression_suite.sh
```

This launcher intentionally skips the tiling sweep because Jacob's request is about compression-side baseline, parameter, scaling, and storage experiments.

## Important Compression-Level Note

The current SC26 scripts use the x-param compression path. That path does not expose a direct runtime parameter named `compression_level`, `quality`, `bitrate`, `lambda`, or `quantization`.

For this repo, define practical low, medium, and high compression from measured checkpoint results:

1. Run the checkpoint sweep.
2. Sort checkpoint rows by `avg_bpp` and `avg_compression_ratio`.
3. Label low/medium/high from the measured size-quality-speed trade-off.

Do not mix the epsilon-param `bitrate_scale` path into this x-param table unless the experiment is explicitly redesigned.

## Run Plan

Use the existing DeltaAI checkout:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
git pull origin main
```

First run a small smoke suite:

```bash
RUN_STAMP=20260515_jacob_compression_smoke \
N_IMAGES=4 \
SHARD_IMAGES=4 \
SAVE_VISUAL_LIMIT=1 \
experiments/compression/slurm/run_jacob_compression_suite.sh
```

If the smoke suite succeeds, run the meeting-scale suite:

```bash
RUN_STAMP=20260516_jacob_compression_n20 \
N_IMAGES=20 \
SHARD_IMAGES=10 \
SAVE_VISUAL_LIMIT=2 \
experiments/compression/slurm/run_jacob_compression_suite.sh
```

If queue time and allocation allow it, run the final pre-`2026-05-20` suite:

```bash
RUN_STAMP=20260518_jacob_compression_n50 \
N_IMAGES=50 \
SHARD_IMAGES=12 \
SAVE_VISUAL_LIMIT=2 \
experiments/compression/slurm/run_jacob_compression_suite.sh
```

## Monitoring

Use:

```bash
squeue -u $USER
tail -f experiments/compression/slurm/logs/*.log
```

For a specific job:

```bash
squeue -j JOB_ID
tail -f experiments/compression/slurm/logs/baseline_JOB_ID.log
tail -f experiments/compression/slurm/logs/checkpoints_JOB_ID.log
tail -f experiments/compression/slurm/logs/storage_JOB_ID.log
```

The result root will be:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/
```

## What to Read After Each Run

Start with:

```bash
cat /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/combined_summary.md
```

If the final summary has not run yet, inspect each stage:

```bash
cat /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/01_baseline_resolution_batch/combined_summary.md
cat /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/02_checkpoint_level_sweep/combined_summary.md
cat /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/04_hpc_scaling_array/combined_summary.md
cat /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/05_storage_compare/combined_summary.md
```

## Deliverables

By `2026-05-20`, bring back a lightweight result folder:

```text
results/2026-05-20-jacob-compression-suite/
├── README.md
├── tables/
│   ├── baseline_resolution_batch_summary.csv
│   ├── checkpoint_level_summary.csv
│   ├── hpc_scaling_summary.csv
│   ├── storage_compare_summary.csv
│   └── combined_summary.csv
└── visual_examples_small/
```

The README should answer:

- Which resolution gives the best time vs compression-ratio trade-off?
- Does batch size help, or does it mostly increase memory?
- Which checkpoint should be called practical low/medium/high compression?
- Does parallel job scaling improve images/hour?
- Does local staging improve wall time enough to matter?
- What is the best compression setup for the shared model workflow?

## Group-Chat Response Draft

Yifan can say:

```text
I prepared the compression-side experiment suite on DeltaAI. It covers baseline full-resolution compression, 4K/2K/1K resolution sweep, batch-size 1/2/4, checkpoint-based compression-level sweep, parallel scaling, and shared-vs-local storage. I also checked the compression-level question: the x-param path does not expose a direct runtime compression_level/quality/bitrate knob, so I will assign low/medium/high from measured BPP and compression ratio across checkpoints. I plan to finish the selected results before 5/20.
```
