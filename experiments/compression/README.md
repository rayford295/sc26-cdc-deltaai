# SC26 Compression Experiment Suite

This folder contains the runnable experiment scaffold for the next compression meeting task.

## What It Covers

| Task | Script |
| --- | --- |
| Baseline full-resolution run | `slurm/01_baseline_resolution_batch.sbatch` |
| Batch-size test, 1, 2, 4 | `slurm/01_baseline_resolution_batch.sbatch` |
| Resolution sweep, 4K, 2K, 1K | `slurm/01_baseline_resolution_batch.sbatch` |
| Compression-level check | `inspect_compression_controls.py` |
| Checkpoint-based compression sweep | `slurm/02_checkpoint_level_sweep.sbatch` |
| Tiling, 512, 1024, 2048 | `slurm/03_tiling_sweep.sbatch` |
| Multiple parallel jobs | `slurm/04_hpc_scaling_array.sbatch` |
| Shared vs local storage | `slurm/05_storage_compare.sbatch` |
| Final all-run summary | `slurm/06_summarize_all.sbatch` |
| Combined tables | `summarize_results.py` |

## Important Compression-Level Finding

The x-param path used by the current SC26 workflow does not expose a direct runtime parameter called `compression_level`, `quality`, `bitrate`, or `quantization`.

For this workflow, run the checkpoint sweep and assign practical low, medium, and high compression from the measured BPP and compression ratio. The epsilon-param code has a `bitrate_scale` argument, but that belongs to a different path and should not be mixed into the x-param results unless the experiment changes.

Generate the inspection report with:

```bash
python experiments/compression/inspect_compression_controls.py
```

It writes:

```text
experiments/compression/compression_controls_report.md
```

## Output Layout

On DeltaAI the default output root is:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/
```

`RUN_STAMP` defaults to `YYYYMMDD_HHMMSS`. The full suite uses one shared
timestamp for all submitted jobs, so baseline, checkpoint, tiling, scaling, and
storage results stay together.

Each run writes:

```text
experiment_name/
├── results.csv       # per-image metrics
├── summary.csv       # one-row aggregate
├── manifest.json     # full config and summary
├── report.md         # compact Markdown summary
└── visuals/          # first reconstructed outputs for visual inspection
```

After each SLURM script finishes, it also writes:

```text
combined_summary.csv
combined_summary.md
```

## Metrics

The core table columns are:

| Column | Meaning |
| --- | --- |
| `run_start_utc` | experiment start timestamp in UTC |
| `run_end_utc` | experiment end timestamp in UTC |
| `run_stamp` | suite timestamp used in the output folder |
| `slurm_job_id` | SLURM job ID, if available |
| `resolution_label` | native, 4k, 2k, or 1k |
| `batch_size` | full-image batch size, or tile batch size for tiled runs |
| `tile_size` | `0` for no tiling, otherwise tile edge length |
| `avg_wall_sec` | average wall time per image |
| `avg_inference_sec` | GPU-timed inference time |
| `avg_peak_gpu_mem_mb` | peak allocated GPU memory |
| `avg_bpp` | estimated model bitrate |
| `avg_compression_ratio` | `24 / avg_bpp` vs uncompressed RGB |
| `avg_psnr_db` | reconstruction PSNR |
| `avg_ssim` | reconstruction SSIM |
| `avg_seam_error_mean` | tiled-only seam artifact metric |

The compressed size is estimated from model BPP:

```text
estimated compressed bytes = width * height * bpp / 8
```

The reconstructed PNG size is recorded only for saved visual outputs. It is not the actual compressed representation.

## Run on DeltaAI

Clone or update the repo on DeltaAI:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code
git pull
```

Run the baseline, batch, and resolution sweep:

```bash
N_IMAGES=8 sbatch experiments/compression/slurm/01_baseline_resolution_batch.sbatch
```

Run the checkpoint compression-level sweep:

```bash
N_IMAGES=8 sbatch experiments/compression/slurm/02_checkpoint_level_sweep.sbatch
```

Run the tiling sweep:

```bash
N_IMAGES=8 sbatch experiments/compression/slurm/03_tiling_sweep.sbatch
```

Run multiple jobs in parallel:

```bash
SHARD_IMAGES=8 sbatch experiments/compression/slurm/04_hpc_scaling_array.sbatch
```

Run shared-vs-local storage comparison:

```bash
N_IMAGES=8 sbatch experiments/compression/slurm/05_storage_compare.sbatch
```

Submit the whole suite:

```bash
N_IMAGES=8 experiments/compression/slurm/run_all_suite.sh
```

To choose a timestamp yourself:

```bash
RUN_STAMP=20260504_meeting_prep N_IMAGES=8 experiments/compression/slurm/run_all_suite.sh
```

For poster-quality numbers, raise `N_IMAGES` after the smoke run succeeds:

```bash
N_IMAGES=100 experiments/compression/slurm/run_all_suite.sh
```

## Run One Local Interactive Test

Inside a GPU allocation:

```bash
module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
python -m pip install --user scikit-image compressai einops lpips ema-pytorch tqdm matplotlib pandas --quiet

cd /projects/bfod/$USER/cdc-deltaai/code

python experiments/compression/run_compression_experiment.py \
  --ckpt /projects/bfod/$USER/cdc-deltaai/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt \
  --img_dir /projects/bfod/$USER/cdc-deltaai/data/imgs \
  --out_dir /projects/bfod/$USER/cdc-deltaai/output/sc26_compression/smoke_test \
  --lpips_weight 0.9 \
  --experiment_name smoke_test \
  --compression_setting baseline_b02048 \
  --resolution_label native_full_resolution \
  --max_edge 0 \
  --batch_size 1 \
  --n_images 1 \
  --n_denoise_step 65
```

## Tiling Notes

Tiling uses independent tile compression, then stitches tiles back into one image. It records:

- total runtime
- peak GPU memory
- PSNR and SSIM
- estimated compressed size and compression ratio
- seam artifact metrics at tile boundaries

The first few stitched outputs are saved under `visuals/`. Inspect these images before using tiling results in the poster.
