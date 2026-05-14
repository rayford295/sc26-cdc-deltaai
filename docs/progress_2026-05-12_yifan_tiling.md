# Yifan Tiling Progress for the 2026-05-12 Weekly Update

## Goal

Test whether tiling can make CDC compression faster and more memory efficient without causing unacceptable reconstruction quality loss or visible stitching artifacts.

## Completed This Week

- Fixed the SLURM path-resolution issue that appeared when DeltaAI copied `.sbatch` files into `/var/spool/slurmd`.
- Submitted a clean DeltaAI GH200 tiling smoke job from `code_tiling_fixed`.
- Completed the first one-image smoke test and then an `N_IMAGES=8` pilot sweep on `5440 x 3648` full-resolution drone images:
  - no-tiling reference
  - `512 x 512` tiles
  - `1024 x 1024` tiles
  - `2048 x 2048` tiles
- Generated `combined_summary.csv` and `combined_summary.md` under the timestamped DeltaAI output folder.
- Downloaded the saved stitched visuals and added lightweight overview and seam-region examples to GitHub.

## Pilot Result

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | Seam metric |
| --- | --- | --- | --- | --- | --- | --- |
| No tiling | 143.55 s | 52.0 GB | 72.74x | 29.88 | 0.8847 | n/a |
| 512 tile | 86.01 s | 3.0 GB | 68.79x | 29.73 | 0.8822 | 0.027796 |
| 1024 tile | 88.35 s | 11.2 GB | 66.11x | 29.82 | 0.8835 | 0.028595 |
| 2048 tile | 95.39 s | 43.8 GB | 66.04x | 29.90 | 0.8841 | 0.031026 |

## Main Interpretation

The tiling workflow is now runnable on DeltaAI and stable across the eight-image pilot. `512 x 512` tiling reduced wall time from `143.55` seconds to `86.01` seconds and reduced peak GPU memory from `52.0` GB to `3.0` GB. PSNR and SSIM stayed close to the no-tiling reference. Visual inspection of the saved overview and seam-region examples found no obvious grid-like stitching seams in the checked sample.

## Next Step

Add the missing `256 x 256` case before scaling the selected setup. This is the next experiment because `512 x 512` already looks strong, but `256 x 256` may lower memory further. The trade-off to measure is tile-management overhead and possible fine-grained boundary artifacts.

Smoke test only the new tile size first:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tile256_smoke,TILING_SIZES="256",N_IMAGES=2,SAVE_VISUAL_LIMIT=2 experiments/compression/slurm/03_tiling_sweep.sbatch
```

Then rerun the full tiling comparison on the same eight-image pilot scale:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tiling_with_256,TILING_SIZES="256 512 1024 2048",N_IMAGES=8,SAVE_VISUAL_LIMIT=4 experiments/compression/slurm/03_tiling_sweep.sbatch
```

Use `512 x 512` as the current recommended default until the `256 x 256` numbers are available. Keep `1024 x 1024` as the quality-oriented backup if larger-sample visual checks show artifacts for smaller tiles.

## Metric Notes for the Next Update

Compression ratio is the storage-savings estimate. The runner computes it as `24 / bpp`, because uncompressed RGB uses 24 bits per pixel. A higher value means smaller model representation, but it does not by itself prove better visual quality.

PSNR is a pixel-level fidelity score in decibels. Higher is better. It is useful for tracking reconstruction error, but it can miss artifacts that matter visually.

SSIM is a structural similarity score from 0 to 1. Higher is better. It is closer to perceptual structure than PSNR because it compares local luminance, contrast, and texture patterns.

The updated runner also records MSE, RMSE, MAE, `error_p95`, `error_p99`, maximum absolute error, and mean bias. These make the visual-quality story easier to defend because they show how large the residual errors are, not only whether PSNR and SSIM stayed close.

For saved examples, use the new `*_comparison.png` panels and `*_error_heatmap.png` files. The comparison panel is ordered as original preview, reconstruction preview, and absolute-error heatmap. In the heatmap, dark regions have low reconstruction error and yellow-white regions have the largest residuals in that image.
