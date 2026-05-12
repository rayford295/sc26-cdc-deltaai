# 2026-05-12 Yifan Tiling Smoke Result

This folder records the first DeltaAI GH200 smoke test for Yifan's tiling experiment in the SC26 compression workflow.

## Scope

- System: DeltaAI GH200, partition `ghx4`
- SLURM job: `2274966`
- Run stamp: `20260512_yifan_tiling_smoke3`
- Input: one full-resolution drone image cropped to `5440 x 3648`
- Checkpoint: `baseline_b02048`
- Denoising steps: `65`
- Precision: `fp32`
- Tiling cases: no tiling, `512 x 512`, `1024 x 1024`, and `2048 x 2048`

The full raw output remains on DeltaAI:

```text
/projects/bfod/$USER/cdc-deltaai/output/sc26_compression/20260512_yifan_tiling_smoke3/03_tiling_sweep/
```

## Key Smoke-Test Finding

Tiling ran successfully end to end: the runner split the image into tiles, compressed each tile independently, stitched the image back together, recorded seam metrics, and wrote the combined summary table.

For this one-image smoke test, `512 x 512` tiling was the strongest speed and memory result:

| Setup | Time per image | Peak GPU memory | Compression ratio | PSNR | SSIM | Seam metric |
| --- | --- | --- | --- | --- | --- | --- |
| No tiling | 144.17 s | 51.8 GB | 72.74x | 30.05 | 0.8838 | n/a |
| 512 tile | 86.43 s | 3.0 GB | 68.71x | 29.88 | 0.8813 | 0.028631 |
| 1024 tile | 88.39 s | 11.2 GB | 65.99x | 30.01 | 0.8824 | 0.028634 |
| 2048 tile | 95.98 s | 43.8 GB | 65.94x | 30.09 | 0.8832 | 0.028444 |

Compared with full-image compression, the `512 x 512` tiled run reduced wall time by about 40 percent and peak GPU memory by about 17x, while keeping PSNR and SSIM close to the no-tiling reference.

## Interpretation for Weekly Progress

This result is a smoke test, not the final poster statistic. It validates that tiling is runnable on DeltaAI and gives a strong early signal that small tiles can reduce memory sharply and improve runtime. The next step is to repeat the same table with `N_IMAGES=8` and inspect the saved stitched visuals for visible boundary artifacts.

## Files

| File | Use |
| --- | --- |
| `tables/combined_summary.csv` | Machine-readable smoke-test summary |
| `tables/combined_summary.md` | Markdown table copied from the DeltaAI combined summary |
