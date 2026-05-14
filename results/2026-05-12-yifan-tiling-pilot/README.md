# 2026-05-12 Yifan Tiling Pilot Result

This folder records the DeltaAI GH200 `N_IMAGES=8` pilot run for Yifan's tiling experiment in the SC26 compression workflow.

## Scope

- System: DeltaAI GH200, partition `ghx4`
- SLURM job: `2275024`
- Run stamp: `20260512_yifan_tiling_pilot`
- Input: eight full-resolution drone images cropped to `5440 x 3648`
- Checkpoint: `baseline_b02048`
- Denoising steps: `65`
- Precision: `fp32`
- Tiling cases: no tiling, `512 x 512`, `1024 x 1024`, and `2048 x 2048`
- Follow-up case to run next: `256 x 256`

The full raw output remains on DeltaAI:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/20260512_yifan_tiling_pilot/03_tiling_sweep/
```

## Key Finding

The `512 x 512` tiled run is the best current speed and memory setup. It reduced wall time by about 40 percent and peak GPU memory by about 17.2x compared with full-image compression, while keeping PSNR and SSIM close to the no-tiling reference.

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | Seam metric |
| --- | --- | --- | --- | --- | --- | --- |
| No tiling | 143.55 s | 52.0 GB | 72.74x | 29.88 | 0.8847 | n/a |
| 512 tile | 86.01 s | 3.0 GB | 68.79x | 29.73 | 0.8822 | 0.027796 |
| 1024 tile | 88.35 s | 11.2 GB | 66.11x | 29.82 | 0.8835 | 0.028595 |
| 2048 tile | 95.39 s | 43.8 GB | 66.04x | 29.90 | 0.8841 | 0.031026 |

## Interpretation for Weekly Progress

The pilot confirms the smoke-test trend on eight images. Smaller tiles give the largest memory reduction and the fastest runtime. The `1024 x 1024` and `2048 x 2048` cases keep PSNR and SSIM slightly closer to no tiling, but they use more GPU memory and run slower than `512 x 512`.

For the weekly update, use `512 x 512` as the recommended default for speed and memory. Before final poster reporting, rerun the selected setup on a larger image set.

## Next Experiment: Add `256 x 256`

The next run should add `256 x 256` tiles to the same comparison. This tests whether smaller tiles reduce memory below the `512 x 512` case, and whether the extra number of tiles adds enough overhead or boundary artifacts to hurt the result.

Recommended smoke command:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tile256_smoke,TILING_SIZES="256",N_IMAGES=2,SAVE_VISUAL_LIMIT=2 experiments/compression/slurm/03_tiling_sweep.sbatch
```

Recommended eight-image comparison:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code_tiling_fixed
sbatch --export=ALL,REPO_DIR=/projects/bfod/$USER/cdc-deltaai/code_tiling_fixed,RUN_STAMP=20260514_yifan_tiling_with_256,TILING_SIZES="256 512 1024 2048",N_IMAGES=8,SAVE_VISUAL_LIMIT=4 experiments/compression/slurm/03_tiling_sweep.sbatch
```

The updated code will write the same summary table plus new pixel-error metrics and visual comparison files for each saved example.

## Metric Meanings

Compression ratio estimates storage savings against uncompressed RGB. The runner computes it as `24 / bpp`, because uncompressed RGB uses 24 bits per pixel. Higher means smaller compressed representation.

PSNR measures pixel-level reconstruction fidelity in decibels. Higher is better. It is useful for comparing runs, but it can miss structural artifacts.

SSIM measures structural similarity on a 0 to 1 scale. Higher is better. It compares local luminance, contrast, and structure, so it helps catch visual changes that pure pixel error may not explain.

The follow-up runner also reports MSE, RMSE, MAE, `error_p95`, `error_p99`, maximum absolute error, and mean bias. These metrics make the error distribution clearer: MAE gives the average absolute residual, high-percentile errors show stronger local failures, and mean bias shows whether reconstructions are brighter or darker on average.

## Visual Artifact Check

Visual inspection of the saved overviews and same-region crops found no obvious grid-like stitching seams in the `512 x 512`, `1024 x 1024`, or `2048 x 2048` stitched outputs for image `100_0005_0001`. The road surface contains real lane markings, pavement boundaries, and compression haze, but the tiled outputs do not show a new regular tile boundary pattern in the checked region.

Use the seam-region examples below as the quick visual evidence:

![No-tiling seam region](visual_examples_small/100_0005_0001_no_tiling_seam_region.jpg)

![512 tile seam region](visual_examples_small/100_0005_0001_tile512_seam_region.jpg)

For the next run, use the new `*_comparison.png` panels and `*_error_heatmap.png` files. The comparison panel is ordered as original preview, reconstruction preview, and absolute-error heatmap. Dark heatmap areas indicate small residuals; yellow-white areas indicate the largest residuals in that image.

## Files

| File | Use |
| --- | --- |
| `tables/combined_summary.csv` | Machine-readable pilot summary |
| `tables/combined_summary.md` | Markdown table copied from the DeltaAI combined summary |
| `visual_examples_small/*_overview.jpg` | Downsampled overview images for no tiling and each tile size |
| `visual_examples_small/*_seam_region.jpg` | Crops from the same road/intersection region for visual seam inspection |
| future `visuals/*_error_heatmap.png` | Absolute RGB reconstruction-error heatmap |
| future `visuals/*_comparison.png` | Original/reconstruction/heatmap panel for visual QA |
