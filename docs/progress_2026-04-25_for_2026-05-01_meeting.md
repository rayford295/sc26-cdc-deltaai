# SC26 CDC Reconstruction Progress Log

Recorded: 2026-04-25 23:34:18 CDT
Next meeting: 2026-05-01
Owner: Yifan Yang
Scope: Reconstruction / decoding / diffusion experiments on NSF ACCESS DeltaAI

## Current Cycle Summary

This cycle moved the reconstruction experiment from planning to active DeltaAI execution.
The repository now contains scripts and documentation for the shared SC26 experiment format:

- single-image sanity test
- realistic batch-size pilot
- repeated runs with averaged results
- speed / quality / memory plots for reconstruction

The latest pushed commits for this work are:

- `1913f5c` Add reconstruction profiling experiment workflow
- `9ea1303` Fix DeltaAI profiling runtime setup

## DeltaAI Environment Status

The working DeltaAI login and compute setup is:

- Login node used: `gh-login01`
- Interactive GPU node tested: `gh061`
- Batch partition used: `ghx4`
- PyTorch module: `python/miniforge3_pytorch/2.10.0`
- Active conda environment: shared `base`
- Required user-site path:

```bash
export PYTHONPATH=/u/yyang48/.local/lib/python3.12/site-packages:$PYTHONPATH
```

The environment was validated with:

- `torch.__version__ = 2.10.0+cu129`
- `torch.cuda.is_available() = True`
- CDC dependencies imported successfully: `skimage`, `compressai`, `einops`, `lpips`, `ema_pytorch`, `tqdm`, `matplotlib`, `pandas`

## Data and Checkpoint Status

Drone image data is present on DeltaAI:

- Image directory: `/projects/bfod/$USER/cdc-deltaai/data/imgs`
- Image count checked: `363`
- Full-resolution input size: `5472 x 3648`
- Script crop size used for model compatibility: `5440 x 3648`

The x-param checkpoint used for reconstruction profiling is:

```bash
/projects/bfod/$USER/cdc-deltaai/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt
```

## Completed Sanity Test

Single-image reconstruction profiling completed successfully on DeltaAI.

Command shape:

```bash
python profile_reconstruction.py \
    --ckpt /projects/bfod/$USER/cdc-deltaai/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt \
    --img_dir /projects/bfod/$USER/cdc-deltaai/data/imgs \
    --out_dir /projects/bfod/$USER/cdc-deltaai/output/test_profile \
    --n_denoise_step 20 \
    --lpips_weight 0.9 \
    --n_images 1
```

Observed result for 20 denoising steps, fp32, one image:

| Metric | Value |
|--------|-------|
| Model load time | 8.58 s |
| Data load / preprocessing | 0.22 s |
| Inference time | 45.33 s |
| Post-processing time | 7.26 s |
| Total per image | 52.82 s |
| Inference share | 85.8% |
| Peak GPU memory | 52013.4 MB |
| PSNR | 30.62 dB |
| SSIM | 0.8954 |
| BPP | 0.3299 |
| Compression ratio | 72.75x vs uncompressed RGB |

Interpretation: reconstruction is dominated by diffusion inference.

## Completed Batch-Size Pilot

Batch-size pilot completed for 20 denoising steps, fp32, two images.

Observed result:

| Steps | Precision | Batch size | Status | Avg inference time | Throughput | Peak GPU memory | PSNR | SSIM |
|-------|-----------|------------|--------|--------------------|------------|-----------------|------|------|
| 20 | fp32 | 1 | Success | 44.54 s/image | 80.8 images/hour | 52127.4 MB | 30.54 dB | 0.8942 |
| 20 | fp32 | 2 | CUDA OOM | Not available | Not available | Not available | Not available | Not available |

Decision for this cycle:

```text
Use batch_size=1 for full-resolution reconstruction experiments.
```

Message for Jacob:

```text
For full-resolution reconstruction, I tested batch sizes 1 and 2 on DeltaAI GH200. Batch size 2 ran out of GPU memory, while batch size 1 completed successfully at about 44.5s inference time per image for 20 denoising steps. I will use batch_size=1 for the reconstruction experiments.
```

## Active Batch Job

The full reconstruction profiling sweep has been submitted.

- SLURM job id: `2195446`
- Partition: `ghx4`
- Job name: `cdc_profile_sweep`
- Node: `gh087`
- Status when recorded: running
- Log file:

```bash
xparam/logs/profiling_2195446.log
```

The job is currently running the baseline 65-step fp32 profile. Early log output showed normal progress:

```text
Processing 10 images (index 0 to 9)
sampling loop time step: 5%|...| 3/65
```

Warnings observed so far are non-blocking:

- `torchvision` VGG weight argument deprecation warning
- `torch.cuda.amp.autocast` future deprecation warning

## Expected Outputs for May 1 Meeting

If the batch job completes, collect these files:

- `/projects/bfod/$USER/cdc-deltaai/output/profiling/baseline_65steps_fp32/profile_report.txt`
- `/projects/bfod/$USER/cdc-deltaai/output/profiling/baseline_65steps_fp16/profile_report.txt`
- `/projects/bfod/$USER/cdc-deltaai/output/sweep/batch_pilot/sweep_summary.csv`
- `/projects/bfod/$USER/cdc-deltaai/output/sweep/step_sweep/sweep_summary.csv`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_time_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_psnr_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_ssim_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_quality_vs_speed.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_memory_vs_steps.png`

## Next Actions

Before the 2026-05-01 meeting:

- Monitor job `2195446` until completion.
- If it completes, summarize the elbow point in the quality-speed curve.
- If it fails, inspect `xparam/logs/profiling_2195446.log` and rerun only the failed step.
- Prepare one reconstruction slide with time per image, throughput, GPU memory, and quality-speed trade-off.
- Keep the conclusion focused on reconstruction latency: diffusion inference is the bottleneck, and reducing denoising steps is the primary optimization lever.
