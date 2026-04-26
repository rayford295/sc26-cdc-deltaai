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

module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
cd /projects/bfod/$USER/cdc-deltaai/code/xparam

python profile_reconstruction.py \
    --ckpt /projects/bfod/$USER/cdc-deltaai/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt \
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

Updated conclusion pattern after the 2026-04-26 sweep:

- Diffusion inference dominates reconstruction time.
- Reducing denoising steps is the strongest speed lever.
- The measured realistic batch size is `1` for full-resolution reconstruction.
- The 5-step setting is the best speed result so far and passed the first visual check on image `100_0005_0001`.
- fp16 reduces memory and improves speed, but fp16 BPP is invalid in the current output, so use fp32 for bitrate and compression-ratio discussion.
- Model loading should be excluded from per-image throughput once the model is reused.

## Message to Jacob

Use this updated message:

```text
Jacob, for the shared SC26 experiment format I am using full-resolution drone images cropped to 5440 x 3648. I ran reconstruction with repeated runs and will report sec/image plus images/hour. The realistic batch size for full-image reconstruction is 1; batch size 2 caused CUDA OOM on DeltaAI. Please use the same reporting format for compression: single-image sanity test, realistic workload batch, repeated runs, averaged results.
```
