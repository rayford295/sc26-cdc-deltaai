# 2026-04-26 Reconstruction Results

This folder contains the report-ready outputs from Yifan's reconstruction profiling run for the 2026-05-01 meeting cycle.

## Experiment Context

- Cluster: NSF ACCESS DeltaAI
- GPU target: GH200
- Pipeline: reconstruction / decoding / diffusion
- Image size used by the model: `5440 x 3648` cropped from full-resolution drone images
- Checkpoint: `b0.2048` x-parameterization checkpoint with `lpips_weight=0.9`
- Realistic full-image batch size: `1`

## Main Conclusion

Reconstruction is dominated by diffusion inference. Reducing denoising steps is the strongest speed lever. Batch size is limited by GPU memory for full-resolution images: `batch_size=2` caused CUDA OOM, so `batch_size=1` is the realistic workload setting.

The 5-step setting is the fastest measured setting so far and passed the first visual check on image `100_0005_0001`.

## Folder Contents

| Folder | Contents |
|--------|----------|
| `plots/` | Five generated figures for time, quality, memory, and speed-quality trade-off |
| `tables/` | Aggregated CSV summaries for the step sweep and batch pilot |
| `reports/` | Detailed fp32 and fp16 65-step profiling reports |
| `visual_examples_small/` | Compressed visual examples suitable for GitHub and meeting materials |

## Key Files

| File | Use |
|------|-----|
| `tables/sweep_summary.csv` | Main repeated step-sweep summary |
| `tables/batch_pilot_summary.csv` | Batch-size pilot summary |
| `plots/plot_time_vs_steps.png` | Time vs denoising steps |
| `plots/plot_quality_vs_speed.png` | Quality-speed trade-off |
| `plots/plot_memory_vs_steps.png` | GPU memory comparison |
| `reports/profile_report_fp32.txt` | Detailed 65-step fp32 profile |
| `reports/profile_report_fp16.txt` | Detailed 65-step fp16 profile |
| `visual_examples_small/comparison_100_0005_0001.jpg` | Slide-ready visual comparison |

## Representative Numbers

| Precision | Steps | Batch | Inference sec/image | Images/hour | Peak GPU memory | PSNR | SSIM | BPP |
|-----------|-------|-------|---------------------|-------------|-----------------|------|------|-----|
| fp32 | 5 | 1 | 11.27 | 319.3 | 52.2 GB | 31.57 | 0.9066 | 0.3300 |
| fp32 | 20 | 1 | 44.37 | 81.2 | 52.2 GB | 30.47 | 0.8961 | 0.3300 |
| fp32 | 65 | 1 | 143.67 | 25.1 | 52.2 GB | 29.92 | 0.8845 | 0.3300 |
| fp16 | 5 | 1 | 10.47 | 343.8 | 34.2 GB | 31.63 | 0.9063 | invalid |
| fp16 | 65 | 1 | 133.68 | 26.9 | 34.2 GB | 30.03 | 0.8839 | invalid |

## Notes

- Use fp32 BPP for bitrate and compression-ratio discussion.
- Use fp16 only for speed and memory discussion in the current run because fp16 BPP was numerically invalid.
- Full-resolution visual examples remain on DeltaAI under `/projects/bfod/$USER/cdc-deltaai/output/visual_examples`.
