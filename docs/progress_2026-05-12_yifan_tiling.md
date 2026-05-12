# Yifan Tiling Progress for the 2026-05-12 Weekly Update

## Goal

Test whether tiling can make CDC compression faster and more memory efficient without causing unacceptable reconstruction quality loss or visible stitching artifacts.

## Completed This Week

- Fixed the SLURM path-resolution issue that appeared when DeltaAI copied `.sbatch` files into `/var/spool/slurmd`.
- Submitted a clean DeltaAI GH200 tiling smoke job from `code_tiling_fixed`.
- Completed one full tiling sweep on a `5440 x 3648` full-resolution drone image:
  - no-tiling reference
  - `512 x 512` tiles
  - `1024 x 1024` tiles
  - `2048 x 2048` tiles
- Generated `combined_summary.csv` and `combined_summary.md` under the timestamped DeltaAI output folder.

## Smoke-Test Result

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | Seam metric |
| --- | --- | --- | --- | --- | --- | --- |
| No tiling | 144.17 s | 51.8 GB | 72.74x | 30.05 | 0.8838 | n/a |
| 512 tile | 86.43 s | 3.0 GB | 68.71x | 29.88 | 0.8813 | 0.028631 |
| 1024 tile | 88.39 s | 11.2 GB | 65.99x | 30.01 | 0.8824 | 0.028634 |
| 2048 tile | 95.98 s | 43.8 GB | 65.94x | 30.09 | 0.8832 | 0.028444 |

## Main Interpretation

The tiling workflow is now runnable on DeltaAI. In the one-image smoke test, `512 x 512` tiling reduced wall time from `144.17` seconds to `86.43` seconds and reduced peak GPU memory from `51.8` GB to `3.0` GB. PSNR and SSIM stayed close to the no-tiling reference. The seam metric was recorded for all tiled cases, but the saved stitched visuals still need a manual artifact check before using this as a final figure.

## Next Step

Run the pilot with more images:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260512_yifan_tiling_pilot,N_IMAGES=8,SAVE_VISUAL_LIMIT=2 experiments/compression/slurm/03_tiling_sweep.sbatch
```

Use the pilot to decide whether `512 x 512` or `1024 x 1024` should be the default tiling setup for the SC26 poster cycle.
