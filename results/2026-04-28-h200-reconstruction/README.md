# 2026-04-28 H200 Reconstruction Results

This folder contains the report-ready outputs from the Delta H200 reconstruction check.
It extends the earlier DeltaAI GH200 experiment with the same reconstruction pipeline and a smaller quick sweep.

## Experiment Context

- System: NCSA Delta
- Login node checked: `dt-login02.delta.ncsa.illinois.edu`
- GPU node used: `gpue08.delta.ncsa.illinois.edu`
- GPU target: NVIDIA H200
- Account: `bfod-delta-gpu`
- Pipeline: reconstruction / decoding / diffusion
- Image size used by the model: `5440 x 3648`
- Checkpoint: `b0.2048` x-parameterization checkpoint with `lpips_weight=0.9`

## Main Findings

- Delta H200 is runnable for this CDC reconstruction workflow.
- H200 batch size 2 fits in memory, but throughput is worse than batch size 1.
- Use `batch_size=1` for H200 reconstruction comparisons.
- H200 is slightly faster than the prior DeltaAI GH200 sweep at the tested step counts, but the difference is modest.
- fp16 reduces memory and improves speed slightly, but fp16 BPP is invalid in the current output, so use fp32 for bitrate and compression-ratio discussion.

## Folder Contents

| Folder | Contents |
|--------|----------|
| `plots/` | H200 quick-sweep figures for time, quality, memory, and speed-quality trade-off |
| `tables/` | H200 batch pilot and quick-sweep summary CSVs |

## Key Files

| File | Use |
|------|-----|
| `tables/sweep_summary.csv` | Main H200 quick step-sweep summary |
| `tables/batch_pilot_summary.csv` | H200 batch-size pilot summary |
| `plots/plot_time_vs_steps.png` | H200 time vs denoising steps |
| `plots/plot_quality_vs_speed.png` | H200 quality-speed trade-off |
| `plots/plot_memory_vs_steps.png` | H200 GPU memory comparison |

## Batch Pilot

| Steps | Precision | Batch | Inference sec/image | Images/hour | Peak GPU memory | PSNR | SSIM | BPP | N |
|-------|-----------|-------|---------------------|-------------|-----------------|------|------|-----|---|
| 20 | fp32 | 1 | 42.55 | 84.6 | 52.1 GB | 30.52 | 0.8942 | 0.3306 | 2 |
| 20 | fp32 | 2 | 53.14 | 67.8 | 108.6 GB | 30.51 | 0.8946 | 0.3306 | 2 |

## Quick Step Sweep

| Precision | Steps | Batch | Inference sec/image | Images/hour | Peak GPU memory | PSNR | SSIM | BPP | N |
|-----------|-------|-------|---------------------|-------------|-----------------|------|------|-----|---|
| fp32 | 5 | 1 | 10.87 | 331.2 | 52.2 GB | 31.56 | 0.9066 | 0.3300 | 15 |
| fp32 | 20 | 1 | 42.74 | 84.2 | 52.2 GB | 30.48 | 0.8961 | 0.3300 | 15 |
| fp32 | 65 | 1 | 138.48 | 26.0 | 52.2 GB | 29.92 | 0.8846 | 0.3300 | 15 |
| fp16 | 5 | 1 | 10.29 | 350.1 | 34.2 GB | 31.63 | 0.9063 | invalid | 15 |
| fp16 | 20 | 1 | 40.32 | 89.3 | 34.2 GB | 30.54 | 0.8958 | invalid | 15 |
| fp16 | 65 | 1 | 130.64 | 27.6 | 34.2 GB | 30.02 | 0.8838 | invalid | 15 |

## Notes

- Use fp32 rows for BPP and compression-ratio reporting.
- Use fp16 rows for speed and memory discussion only.
- The full 7-step DeltaAI GH200 sweep remains in `results/2026-04-26-reconstruction/`.
