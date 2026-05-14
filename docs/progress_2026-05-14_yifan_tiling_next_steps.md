# Yifan Tiling Next Steps for 2026-05-14

## Purpose

Prepare the next DeltaAI run after the `2026-05-12` tiling pilot. The immediate goal is to add a `256 x 256` tiling case, compare it with the current `512 x 512` recommendation, and use numeric metrics plus error heatmaps to decide which setup should move into the larger poster-scale run.

## Current State

The `2026-05-12` DeltaAI GH200 pilot finished on eight full-resolution drone images. The current best setup is `512 x 512` tiling:

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | Seam metric |
| --- | --- | --- | --- | --- | --- | --- |
| No tiling | 143.55 s | 52.0 GB | 72.74x | 29.88 | 0.8847 | n/a |
| `512 x 512` tile | 86.01 s | 3.0 GB | 68.79x | 29.73 | 0.8822 | 0.027796 |
| `1024 x 1024` tile | 88.35 s | 11.2 GB | 66.11x | 29.82 | 0.8835 | 0.028595 |
| `2048 x 2048` tile | 95.39 s | 43.8 GB | 66.04x | 29.90 | 0.8841 | 0.031026 |

The code is ready on `origin/main` at commit `d38b555`. It adds:

- `256 x 256` to the default tiling sweep.
- New quality metrics: MSE, RMSE, MAE, `error_p95`, `error_p99`, maximum absolute error, and mean bias.
- New visual QA files: original preview, reconstruction preview, error heatmap, and side-by-side comparison panel.

## Next Run Order

First update the DeltaAI checkout:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
git pull origin main
```

Run a `256 x 256` smoke test first:

```bash
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tile256_smoke,TILING_SIZES="256",N_IMAGES=2,SAVE_VISUAL_LIMIT=2 experiments/compression/slurm/03_tiling_sweep.sbatch
```

If the smoke test succeeds, run the eight-image comparison:

```bash
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tiling_with_256,TILING_SIZES="256 512 1024 2048",N_IMAGES=8,SAVE_VISUAL_LIMIT=4 experiments/compression/slurm/03_tiling_sweep.sbatch
```

If `256 x 256` looks promising, run a larger selected comparison:

```bash
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_selected_tiling_large,TILING_SIZES="256 512",N_IMAGES=50,SAVE_VISUAL_LIMIT=8 experiments/compression/slurm/03_tiling_sweep.sbatch
```

Use `N_IMAGES=100` instead of `50` if queue time and GPU hours allow it.

## What to Check

Use `combined_summary.csv` and `combined_summary.md` from:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/$RUN_STAMP/03_tiling_sweep/
```

Check these columns first:

| Metric | What it tells us |
| --- | --- |
| `avg_wall_sec` | practical time per image |
| `avg_peak_gpu_mem_mb` | peak GPU memory pressure |
| `avg_compression_ratio` | estimated storage savings against 24-bit RGB |
| `avg_psnr_db` | pixel-level reconstruction fidelity |
| `avg_ssim` | structural similarity |
| `avg_mae` | average absolute reconstruction error |
| `avg_error_p99` | strong local reconstruction errors |
| `avg_seam_error_mean` | tile-boundary artifact risk |

Then inspect the saved visual QA files under each run's `visuals/` folder:

| File suffix | Use |
| --- | --- |
| `_stitched.png` | reconstructed tiled output |
| `_original_preview.png` | original image preview |
| `_error_heatmap.png` | absolute RGB reconstruction-error heatmap |
| `_comparison.png` | original, reconstruction, and heatmap in one panel |

Dark heatmap regions mean low error. Yellow-white regions mark the largest residuals in that image. Watch for regular tile-boundary patterns, especially in the `256 x 256` case.

## Decision Rule

Promote `256 x 256` if it lowers memory beyond the `512 x 512` run, keeps time per image close to or better than `86.01 s`, keeps PSNR and SSIM close to the current pilot, and shows no regular tile-boundary artifacts in heatmaps.

Keep `512 x 512` as the recommendation if `256 x 256` is slower, has worse local errors, or shows visible grid artifacts.

Keep `1024 x 1024` as the quality-oriented backup if both smaller tile sizes show artifacts.

## Repo Artifacts to Bring Back

After the successful run, copy small outputs into a new dated result folder:

```text
results/2026-05-14-yifan-tiling-with-256/
├── README.md
├── tables/
│   ├── combined_summary.csv
│   └── combined_summary.md
└── visual_examples_small/
    ├── *_comparison.png
    ├── *_error_heatmap.png
    └── *_seam_region.jpg
```

Keep full-resolution raw outputs, checkpoints, full logs, and large visual folders on DeltaAI.

## One-Sentence Meeting Update

The next step is to test `256 x 256` tiling against the current `512 x 512` recommendation, using runtime, memory, compression ratio, PSNR, SSIM, pixel-error metrics, seam metrics, and heatmap panels to decide which setup should scale to the poster run.
