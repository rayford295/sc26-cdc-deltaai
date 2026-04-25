# Yifan Reconstruction Experiment Plan

## Your Task

Yifan owns the reconstruction side of the SC26 CDC experiment:

- Main question: How fast can we use the compressed data again?
- Pipeline: decoding / diffusion reconstruction.
- Main bottleneck to test: diffusion inference time, especially the number of denoising steps.
- Cluster: NSF ACCESS DeltaAI only.

Jacob owns compression / encoding. Use the same experiment structure and figure format on both sides.

## Experiment Structure

Use this structure for every reconstruction result:

1. Single-image sanity test.
2. Realistic batch-size workload.
3. Repeated runs, then report averaged results.

For the current full-resolution drone images, the raw image size is 5472 x 3648 pixels. The scripts crop each image to 5440 x 3648 so the compressor and hyperprior stay aligned. This image size is large, so the safe starting batch size is:

```text
batch_size = 1
```

Run the batch-size pilot before finalizing this. If `batch_size=2` completes without CUDA OOM and improves images/hour, use `batch_size=2`. If it fails or gives little throughput benefit, keep `batch_size=1`.

## Runs to Submit

First run the quick interactive sanity test:

```bash
srun --account=bfod-dtai-gh --partition=ghx4-interactive \
     --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G \
     --time=00:30:00 --pty bash

module load anaconda3
conda activate exp_pytorch
cd /projects/bfod/$USER/cdc-deltaai/code/xparam

python profile_reconstruction.py \
    --ckpt /projects/bfod/$USER/cdc-deltaai/weights/xparam/b0.2048.pt \
    --img_dir /projects/bfod/$USER/cdc-deltaai/data/imgs \
    --out_dir /projects/bfod/$USER/cdc-deltaai/output/test_profile \
    --n_denoise_step 20 \
    --lpips_weight 0.9 \
    --n_images 1
```

Then submit the full profiling job:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code
mkdir -p xparam/logs
sbatch xparam/run_profiling_sweep.sh
```

The SLURM script now runs:

- baseline profile: 65 steps, fp32, 10 images
- baseline profile: 65 steps, fp16, 10 images
- batch pilot: steps 20 and 65, batch sizes 1 and 2, 2 repeats
- final step sweep: steps 5, 10, 20, 30, 50, 65, 100, fp32 and fp16, 3 repeats
- plot generation

## Outputs to Collect

Use these files in the slides:

- `output/profiling/baseline_65steps_fp32/profile_report.txt`
- `output/profiling/baseline_65steps_fp16/profile_report.txt`
- `output/sweep/batch_pilot/sweep_summary.csv`
- `output/sweep/step_sweep/sweep_summary.csv`
- `output/plots/plot_time_vs_steps.png`
- `output/plots/plot_psnr_vs_steps.png`
- `output/plots/plot_ssim_vs_steps.png`
- `output/plots/plot_quality_vs_speed.png`
- `output/plots/plot_memory_vs_steps.png`

Core metrics for the slide table:

- reconstruction time per image
- throughput, in images/hour
- GPU memory usage
- PSNR and SSIM
- quality vs speed trade-off
- bottleneck summary

## What to Highlight

Expected conclusion pattern:

- Diffusion inference dominates reconstruction time.
- Lower denoising steps should reduce time almost directly.
- The key figure is quality vs speed. Pick the smallest step count where PSNR and SSIM plateau.
- fp16 is useful only if it improves images/hour without reducing PSNR/SSIM meaningfully.
- Model loading should be excluded from per-image throughput once the model is reused.

## Message to Jacob

Use this message after the batch pilot completes:

```text
Jacob, for the shared SC26 experiment format I am using full-resolution drone images cropped to 5440 x 3648. I ran reconstruction with repeated runs and will report sec/image plus images/hour. The safe starting batch size is 1 for full-image reconstruction; I am testing batch size 2 on DeltaAI and will use it only if it is stable and improves throughput. Please use the same reporting format for compression: single-image sanity test, realistic workload batch, repeated runs, averaged results.
```
