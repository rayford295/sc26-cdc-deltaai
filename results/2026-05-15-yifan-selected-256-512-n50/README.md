# 2026-05-15 Yifan Selected `256 x 256` vs `512 x 512` Tiling Result

This folder records the DeltaAI GH200 `N_IMAGES=50` selected comparison for Yifan's SC26 CDC tiling experiment.

## Scope

- System: DeltaAI GH200, partition `ghx4`
- SLURM job: `2287737`
- Run stamp: `20260515_yifan_selected_256_512_n50`
- Input: fifty full-resolution drone images cropped to `5440 x 3648`
- Checkpoint: `baseline_b02048`
- Denoising steps: `65`
- Precision: `fp32`
- Cases: no tiling reference, `256 x 256`, and `512 x 512`

The full raw output remains on DeltaAI:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/20260515_yifan_selected_256_512_n50/03_tiling_sweep/
```

## Key Finding

`256 x 256` is the best speed and memory candidate in this selected `N_IMAGES=50` run. It reduced wall time by about 44.7 percent and peak GPU memory by about 31.3x compared with full-image compression. Compared with `512 x 512`, it was about 9.0 percent faster and used about 45.0 percent less GPU memory.

Quality dropped slightly relative to `512 x 512`: PSNR was lower by `0.086 dB`, SSIM was lower by `0.0031`, MAE was higher by `0.000217`, and `error_p99` was higher by `0.001314`. The seam metric was nearly unchanged.

## Result Table

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | MAE | Error p99 | Seam metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No tiling | 144.35 s | 52.01 GB | 70.11x | 28.96 | 0.8616 | 0.027016 | 0.117676 | n/a |
| `256 x 256` tile | 79.84 s | 1.66 GB | 68.24x | 28.78 | 0.8552 | 0.027463 | 0.120211 | 0.032556 |
| `512 x 512` tile | 87.71 s | 3.02 GB | 66.31x | 28.86 | 0.8583 | 0.027246 | 0.118897 | 0.032224 |

## Interpretation

Use `256 x 256` as the new speed and memory candidate. It provides the lowest GPU memory and fastest throughput among the selected cases, while quality metrics remain close to `512 x 512`.

Keep `512 x 512` as the quality-safe backup. It has slightly better PSNR, SSIM, MAE, and high-percentile error, but the gains are small compared with its extra runtime and memory cost.

Before using `256 x 256` as the final poster recommendation, inspect the saved `*_comparison.png` and `*_error_heatmap.png` files from the DeltaAI output folder. The visual check should look for regular tile-boundary patterns that scalar metrics may hide.

## Files

| File | Use |
| --- | --- |
| `tables/combined_summary.csv` | Machine-readable selected-run summary |
| `tables/combined_summary.md` | Markdown table copied from the DeltaAI combined summary |
| `visual_examples_small/100_0005_0001_comparison.jpg` | No-tiling original/reconstruction/error-heatmap panel |
| `visual_examples_small/100_0005_0001_error_heatmap.jpg` | No-tiling absolute RGB reconstruction-error heatmap |
| `visual_examples_small/100_0005_0001_tile256_comparison.jpg` | `256 x 256` original/reconstruction/error-heatmap panel |
| `visual_examples_small/100_0005_0001_tile256_error_heatmap.jpg` | `256 x 256` absolute RGB reconstruction-error heatmap |
| `visual_examples_small/100_0005_0001_tile512_comparison.jpg` | `512 x 512` original/reconstruction/error-heatmap panel |
| `visual_examples_small/100_0005_0001_tile512_error_heatmap.jpg` | `512 x 512` absolute RGB reconstruction-error heatmap |

The committed visual examples are downsampled JPEG previews from image `100_0005_0001`. The full visual archive remains on DeltaAI.
