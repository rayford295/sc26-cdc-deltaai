# SC26 CDC Reconstruction Progress Log

Recorded: 2026-04-25 23:34:18 CDT
Last updated: 2026-04-26 07:18:04 CDT
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
- `0a7a228` Record baseline profiling results
- `d0ed767` Record batch pilot summary

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

18. The full job completed STEP 2 batch-size pilot.

19. The full job entered STEP 3, the repeated step sweep over `[5, 10, 20, 30, 50, 65, 100]`.

20. The full job completed STEP 3, generated all plots, and saved all outputs.

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
- Full job STEP 2 batch-size pilot completed.
- Full job STEP 3 repeated step sweep started.
- Full job STEP 3 repeated step sweep completed.
- All five reconstruction plots generated.
- Progress recorded and pushed to GitHub.

Current follow-up status:

- No required reconstruction code runs remain for this week's task.
- Final CSV, PNG, report, and visual comparison outputs have been copied into `results/2026-04-26-reconstruction/` and pushed to GitHub.
- The first visual comparison check has been completed and looked acceptable.
- If the group later asks for more evidence, optional follow-up work is to add one or two more visual comparisons from different images.

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

Hardware clarification recorded on 2026-04-26:

- The current reconstruction results are DeltaAI GH200 results.
- DeltaAI is separate from Delta. The official DeltaAI documentation describes the system as NVIDIA GH200 Grace Hopper: <https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/index.html>
- Delta documentation lists an `8-way H200` GPU node type. Treat Delta H200 as possible future comparison work, not as hardware used in this run: <https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html>

ACCESS project resource snapshot from the portal:

| Project | Resource | Status | Balance | End date | Username |
|---------|----------|--------|---------|----------|----------|
| `CIV250023: Upscaling for Flood Resilience: A Benchmarking Study` | NCSA Delta GPU | Active | 1.08K of 2.04K GPU hours remaining (53%) | 2026-08-07 | `yyang48` |
| `CIV250023: Upscaling for Flood Resilience: A Benchmarking Study` | NCSA DeltaAI | Active | 93 of 141 GPU hours remaining (66%) | 2026-08-07 | `yyang48` |

Future Delta comparison checklist:

- DeltaAI login used in this cycle: `ssh yyang48@dtai-login.delta.ncsa.illinois.edu`
- Delta login for possible H200 comparison: `ssh yyang48@login.delta.ncsa.illinois.edu` or `ssh yyang48@dt-login.delta.ncsa.illinois.edu`
- After logging into Delta, confirm partition and allocation before running code:

```bash
sinfo -o "%P %G %D %N"
accounts
module avail 2>&1 | grep -i -E "python|conda|cuda|pytorch"
```

- If Delta H200 is available, rerun the same experiment structure: single-image sanity test, batch-size pilot, repeated step sweep, and matching plots/reports.

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

The initial interactive batch-size pilot completed for 20 denoising steps, fp32, two images.

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

## Full Job Batch-Size Pilot Results

The submitted SLURM job `2195446` completed the repeated STEP 2 batch-size pilot.

Configuration:

- Steps tested: `20`, `65`
- Precision: `fp32`
- Batch sizes tested: `1`, `2`
- Repeats: `2`
- Images per configuration: `4`
- Common cropped size: `5440 x 3648 pixels`

Summary:

| Steps | Precision | Batch size | Status | Avg inference time | Throughput | Peak GPU memory | PSNR | SSIM | BPP | N |
|-------|-----------|------------|--------|--------------------|------------|-----------------|------|------|-----|---|
| 20 | fp32 | 1 | Success | 44.39 s/image | 81.1 images/hour | 52212.6 MB | 30.50 dB | 0.8961 | 0.3295 | 8 |
| 20 | fp32 | 2 | CUDA OOM | Not available | Not available | Not available | Not available | Not available | Not available | 0 |
| 65 | fp32 | 1 | Success | 143.65 s/image | 25.1 images/hour | 52210.8 MB | 29.95 dB | 0.8846 | 0.3295 | 8 |
| 65 | fp32 | 2 | CUDA OOM | Not available | Not available | Not available | Not available | Not available | Not available | 0 |

Decision:

```text
The repeated batch-size pilot confirms batch_size=1 for full-resolution reconstruction.
```

Interpretation:

- `batch_size=2` failed for both 20-step and 65-step reconstruction.
- GPU memory is the hard limit for full-image reconstruction.
- Reducing denoising steps from 65 to 20 reduced inference time from 143.65 to 44.39 s/image, a 69.1% reduction.
- 20 steps also had slightly higher PSNR and SSIM in this pilot, but the final conclusion should use the STEP 3 repeated sweep.

## Full Job Step-Sweep Results

The submitted SLURM job `2195446` completed STEP 3 and generated plots.

Finished at:

```text
Sun Apr 26 06:28:36 AM CDT 2026
```

Configuration:

- Steps tested: `5`, `10`, `20`, `30`, `50`, `65`, `100`
- Precisions tested: `fp32`, `fp16`
- Batch size: `1`
- Repeats: `3`
- Images per configuration: `5`
- Rows per configuration: `15`
- Common cropped size: `5440 x 3648 pixels`

Final repeated step-sweep summary:

| Steps | Precision | Batch | Inference time | Throughput | Peak GPU memory | PSNR | SSIM | BPP | N |
|-------|-----------|-------|----------------|------------|-----------------|------|------|-----|---|
| 5 | fp16 | 1 | 10.47 s/image | 343.8 images/hour | 34207.5 MB | 31.63 dB | 0.9063 | `inf` | 15 |
| 10 | fp16 | 1 | 20.73 s/image | 173.7 images/hour | 34207.5 MB | 31.05 dB | 0.9017 | `inf` | 15 |
| 20 | fp16 | 1 | 41.26 s/image | 87.2 images/hour | 34207.5 MB | 30.55 dB | 0.8958 | `inf` | 15 |
| 30 | fp16 | 1 | 61.80 s/image | 58.3 images/hour | 34207.5 MB | 30.37 dB | 0.8917 | `inf` | 15 |
| 50 | fp16 | 1 | 102.87 s/image | 35.0 images/hour | 34207.5 MB | 30.15 dB | 0.8867 | `inf` | 15 |
| 65 | fp16 | 1 | 133.68 s/image | 26.9 images/hour | 34207.5 MB | 30.03 dB | 0.8839 | `inf` | 15 |
| 100 | fp16 | 1 | 205.54 s/image | 17.5 images/hour | 34207.5 MB | 29.87 dB | 0.8800 | `inf` | 15 |
| 5 | fp32 | 1 | 11.27 s/image | 319.3 images/hour | 52225.8 MB | 31.57 dB | 0.9066 | 0.3300 | 15 |
| 10 | fp32 | 1 | 22.29 s/image | 161.5 images/hour | 52225.9 MB | 31.00 dB | 0.9019 | 0.3300 | 15 |
| 20 | fp32 | 1 | 44.36 s/image | 81.1 images/hour | 52225.9 MB | 30.47 dB | 0.8961 | 0.3300 | 15 |
| 30 | fp32 | 1 | 67.16 s/image | 53.7 images/hour | 52225.9 MB | 30.25 dB | 0.8921 | 0.3300 | 15 |
| 50 | fp32 | 1 | 110.56 s/image | 32.6 images/hour | 52225.9 MB | 30.03 dB | 0.8873 | 0.3300 | 15 |
| 65 | fp32 | 1 | 143.67 s/image | 25.1 images/hour | 52225.8 MB | 29.92 dB | 0.8845 | 0.3300 | 15 |
| 100 | fp32 | 1 | 220.85 s/image | 16.3 images/hour | 52225.9 MB | 29.77 dB | 0.8809 | 0.3300 | 15 |

Generated plots:

- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_time_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_psnr_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_ssim_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_quality_vs_speed.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_memory_vs_steps.png`

Key findings from STEP 3:

- Time scales strongly with denoising steps. In fp32, 65 steps took 143.67 s/image, while 5 steps took 11.27 s/image. This is a 92.2% reduction.
- Throughput increased from 25.1 images/hour at 65 fp32 steps to 319.3 images/hour at 5 fp32 steps.
- fp16 was consistently faster than fp32 and used much less memory.
- fp16 peak memory was about 34.2 GB; fp32 peak memory was about 52.2 GB. This is about a 34.5% memory reduction.
- PSNR and SSIM did not degrade at lower step counts in this run. The 5-step results had the highest PSNR/SSIM in both fp32 and fp16.
- fp16 reported `BPP = inf` across all step counts, so fp16 BPP and compression ratio should not be used for bitrate conclusions.
- fp32 BPP was stable at about `0.3300` across all step counts.

Initial recommendation for the May 1 meeting:

```text
For reconstruction speed, 5 denoising steps is the best measured setting so far.
Use batch_size=1. Use fp16 if the slide focuses on speed and memory. Use fp32 BPP for bitrate/compression-ratio reporting.
```

## Batch Job Record

The full reconstruction profiling sweep completed successfully.

- SLURM job id: `2195446`
- Partition: `ghx4`
- Job name: `cdc_profile_sweep`
- Node: `gh087`
- Status when recorded: completed
- Completed at: `Sun Apr 26 06:28:36 AM CDT 2026`
- Log file:

```bash
xparam/logs/profiling_2195446.log
```

Early log output showed normal progress:

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

Updated 2026-04-26 00:51 CDT:

- STEP 2 batch-size pilot completed.
- `batch_size=2` caused CUDA OOM for both 20 and 65 steps.
- STEP 3 repeated step sweep started.
- STEP 3 configuration: steps `[5, 10, 20, 30, 50, 65, 100]`, precisions `fp32 + fp16`, batch size `1`, repeats `3`, images per configuration `5`.
- STEP 3 model loaded successfully and preloaded 5 images.
- STEP 3 common cropped image size: `5440 x 3648 pixels`.

Updated 2026-04-26 06:28 CDT:

- STEP 3 repeated step sweep completed.
- Plot generation completed.
- All outputs were saved to:

```bash
/projects/bfod/$USER/cdc-deltaai/output/profiling
/projects/bfod/$USER/cdc-deltaai/output/sweep/batch_pilot
/projects/bfod/$USER/cdc-deltaai/output/sweep/step_sweep
/projects/bfod/$USER/cdc-deltaai/output/plots
```

Updated 2026-04-26 07:45 CDT:

- A visual comparison example was created from image `100_0005_0001`.
- The comparison includes original, fp32 reconstructions at 5, 10, 20, 65, and 100 steps, plus fp16 reconstructions at 5, 20, and 65 steps.
- The visual check looked good: low-step reconstructions did not show obvious artifacts in the reviewed example.
- Full-resolution visual examples were kept on DeltaAI under `/projects/bfod/$USER/cdc-deltaai/output/visual_examples`.
- GitHub stores compressed visual examples under `results/2026-04-26-reconstruction/visual_examples_small`.
- The main visual comparison file for slides is `results/2026-04-26-reconstruction/visual_examples_small/comparison_100_0005_0001.jpg`.

## Generated Outputs for May 1 Meeting

The batch job completed and produced these files:

- `/projects/bfod/$USER/cdc-deltaai/output/profiling/baseline_65steps_fp32/profile_report.txt`
- `/projects/bfod/$USER/cdc-deltaai/output/profiling/baseline_65steps_fp16/profile_report.txt`
- `/projects/bfod/$USER/cdc-deltaai/output/sweep/batch_pilot/sweep_summary.csv`
- `/projects/bfod/$USER/cdc-deltaai/output/sweep/step_sweep/sweep_summary.csv`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_time_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_psnr_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_ssim_vs_steps.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_quality_vs_speed.png`
- `/projects/bfod/$USER/cdc-deltaai/output/plots/plot_memory_vs_steps.png`
- `results/2026-04-26-reconstruction/visual_examples_small/comparison_100_0005_0001.jpg`

## Next Actions

Before the 2026-05-01 meeting:

- Prepare one reconstruction slide with time per image, throughput, GPU memory, and quality-speed trade-off.
- Keep the conclusion focused on reconstruction latency: diffusion inference is the bottleneck, and reducing denoising steps is the primary optimization lever.
- Use `batch_size=1` as the final realistic workload batch size for full-resolution reconstruction.
- Use the visual comparison figure to support the low-step recommendation.
