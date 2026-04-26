# SC26 CDC Reconstruction Progress Log

Recorded: 2026-04-25 23:34:18 CDT
Last updated: 2026-04-26 00:24:53 CDT
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
- `d6c062f` Record reconstruction progress for May meeting
- `5c5cead` Expand reconstruction progress timeline

## Process Timeline

This section records the workflow so the May 1 meeting can focus on what was done during this cycle.

1. The SC26 reconstruction task was separated from Jacob's compression task.
   Yifan's question is: "How fast can we use the compressed data again?"

2. The local repository was cloned and updated from GitHub.
   The working repository is:

```bash
/projects/bfod/$USER/cdc-deltaai/code
```

3. The original expected path `/projects/bfod/$USER/cdc-deltaai/code` did not exist on DeltaAI.
   The project folder initially contained only `data`, `logs`, `output`, and `weights`.
   The repository was then cloned into the missing `code` directory.

4. The latest reconstruction workflow commit was confirmed on DeltaAI:

```text
1913f5c Add reconstruction profiling experiment workflow
```

5. The interactive GPU session was started successfully:

```bash
srun --account=bfod-dtai-gh --partition=ghx4-interactive \
     --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G \
     --time=00:30:00 --pty bash
```

6. The first environment assumption failed because `module load anaconda3` is not available on this DeltaAI system.
   The correct PyTorch module is:

```bash
module load python/miniforge3_pytorch/2.10.0
conda activate base
```

7. The shared `base` environment had CUDA PyTorch but did not include all CDC dependencies.
   Required packages were installed into the user site:

```bash
python -m pip install --user scikit-image compressai einops lpips ema-pytorch tqdm matplotlib pandas
```

8. The shared conda environment did not automatically expose user-site packages.
   The import path fix is:

```bash
export PYTHONPATH=/u/yyang48/.local/lib/python3.12/site-packages:$PYTHONPATH
```

9. The initial checkpoint path in the README did not match the actual DeltaAI weight directory.
   The working checkpoint path is:

```bash
/projects/bfod/$USER/cdc-deltaai/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt
```

10. The first profiling run hit `RuntimeError: Invalid device argument`.
    The scripts were patched to select the CUDA device explicitly and pass the integer device index into CUDA memory accounting functions.

11. The environment, dependency path, checkpoint path, and device fix were pushed to GitHub in commit `9ea1303`.

12. After the fix, the single-image reconstruction test completed successfully.

13. The batch-size pilot completed and showed that `batch_size=2` causes CUDA OOM.

14. The full repeated profiling sweep was submitted as SLURM job `2195446`.

15. The full job completed baseline 65-step fp32 profiling.

16. The full job completed baseline 65-step fp16 profiling.

17. The full job entered STEP 2, the batch-size pilot over steps `[20, 65]` and batch sizes `[1, 2]`.

## Done vs Pending

Completed in this cycle:

- Local and DeltaAI repository setup.
- DeltaAI PyTorch environment discovery.
- CDC dependency installation and import path fix.
- Correct checkpoint path discovery.
- GPU memory accounting bug fix in the profiling scripts.
- Single-image reconstruction sanity test.
- Batch-size pilot.
- Batch-size decision for reconstruction: `batch_size=1`.
- Full profiling sweep submitted to SLURM.
- Full job baseline 65-step fp32 profiling completed.
- Full job baseline 65-step fp16 profiling completed.
- Progress recorded and pushed to GitHub.

Still pending:

- Wait for SLURM job `2195446` to finish STEP 2 and STEP 3.
- Collect final repeated-run averages from `output/sweep/step_sweep/sweep_summary.csv`.
- Collect final figures from `output/plots`.
- Determine the sampling-step elbow point for the 2026-05-01 meeting.
- Prepare the final slide figure format for Jacob to match.

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

## Full Job Baseline Results

The submitted SLURM job `2195446` completed the 65-step baseline profiles for both fp32 and fp16.

| Setting | Inference time | Total time | Inference share | Peak GPU memory | PSNR | SSIM | BPP |
|---------|----------------|------------|-----------------|-----------------|------|------|-----|
| 65 steps, fp32 | 143.80 s/image | 151.18 s/image | 95.1% | 52422.4 MB | 29.83 dB | 0.8822 | 0.3317 |
| 65 steps, fp16 | 133.73 s/image | 140.94 s/image | 94.9% | 34406.6 MB | 29.92 dB | 0.8817 | `inf` |

Interpretation:

- fp16 reduced inference time from 143.80 to 133.73 s/image, about a 7.0% speedup.
- fp16 reduced peak GPU memory from 52.4 GB to 34.4 GB, about a 34% reduction.
- PSNR and SSIM stayed almost unchanged.
- fp16 produced `BPP = inf`, so fp16 BPP and compression ratio should not be used for final bitrate conclusions.
- For bitrate and compression-ratio reporting, use fp32 results.

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

Updated 2026-04-26 00:24 CDT:

- Baseline 65-step fp32 profiling completed.
- Baseline 65-step fp16 profiling completed.
- STEP 2 batch-size pilot started.
- Common cropped image size reported by the job: `5440 x 3648 pixels`.
- STEP 2 started with `steps=20`, `fp32`, `batch=1`.
- First observed STEP 2 row:

```text
repeat=01 | steps=20 | fp32 | batch=1 | infer 44.59s (44.59s/img, 80.7 img/hr) | PSNR 30.64 dB | SSIM 0.8954 | mem 52014 MB
```

The STEP 2 summary is still pending.

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
