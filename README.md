# CDC Image Compression on DeltaAI

Lossy image compression with Conditional Diffusion Models, adapted for running on [NSF DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/).

Based on: [Lossy Image Compression with Conditional Diffusion Models](https://arxiv.org/pdf/2209.06950.pdf)

## Current Evaluation Goal

The current evaluation task is to:

1. Apply the compression model on 100 drone images.
2. Report the overall compression rate from the evaluation output.
3. Compare the total size of the original images and the reconstructed images.
4. Run the same evaluation workflow on both GPU targets when available:
- GH200
- H100

For each run, collect and summarize the following outputs:
- `compression_report.txt`
- `compression_results.csv`
- average BPP
- overall compression ratio
- total original image size
- total reconstructed image size
- file size ratio between original and reconstructed images

## Current Status

The DeltaAI x-parameterization workflow has been debugged and completed successfully on GH200 for the full 100-image evaluation sweep across all 6 checkpoints.

Validated progress so far:
- The DeltaAI environment issues were resolved, including Python user site-packages visibility and missing runtime dependencies.
- The x-param evaluation script was updated to crop each input image to multiples of 64 so the compressor and hyperprior remain shape-aligned during inference.
- A single-image GH200 validation run completed successfully and produced both `compression_report.txt` and `compression_results.csv`.
- The full 100-image GH200 sweep was executed across all 6 checkpoints.
- The final checkpoint (`b0.2048`) hit the walltime limit during the first batch run, so the remaining images were completed with a targeted resume job and then merged into a full 100-image result set.

## Final GH200 Results for 100 Images

| Checkpoint | Average BPP | Compression Ratio | Total Original Size | Total Reconstructed Size | File Size Ratio |
|-----------|-------------|-------------------|---------------------|--------------------------|-----------------|
| `b0.0032` | 0.4394 bits/pixel | 54.62x | 834.7 MB | 2.4 GB | 0.33x |
| `b0.0064` | 0.2872 bits/pixel | 83.58x | 834.7 MB | 2.3 GB | 0.36x |
| `b0.0128` | 0.1632 bits/pixel | 147.09x | 834.7 MB | 2.1 GB | 0.38x |
| `b0.0512` | 0.7444 bits/pixel | 32.24x | 834.7 MB | 3.0 GB | 0.27x |
| `b0.1024` | 0.5388 bits/pixel | 44.54x | 834.7 MB | 2.9 GB | 0.28x |
| `b0.2048` | 0.3438 bits/pixel | 69.82x | 834.6 MB | 2.9 GB | 0.29x |

Interpretation:
- `b0.0128` produced the lowest average bitrate and the highest compression ratio relative to uncompressed 24-bit RGB.
- Across all checkpoints, the reconstructed PNG outputs were still larger than the original JPEG files, so the BPP-based compression result and the stored-file-size comparison should be interpreted as different metrics.
- The GH200 runtime was consistently about 143.4 seconds per image with 65 denoising steps.

Remaining work:
- Run the same evaluation on H100 once the partition name and access are confirmed.
- Compare GH200 and H100 runtime and output statistics side by side.

## GPU Partition Summary

The currently visible GPU-backed partitions from `sinfo -o "%P %G %D %N"` are:

- `full` — NVIDIA GH200 120GB, 4 GPUs per node
- `ghx4` — NVIDIA GH200 120GB, 4 GPUs per node
- `ghx4-interactive` — NVIDIA GH200 120GB, 4 GPUs per node
- `test` — no GPU nodes reported

Practical interpretation:
- `ghx4` is the confirmed batch partition currently used for the GH200 evaluation jobs.
- `ghx4-interactive` is the confirmed interactive partition currently used for debugging and validation runs.
- `full` also reports GH200 nodes, but the current workflow has been validated on `ghx4` and `ghx4-interactive`.
- No H100-backed partition is currently visible in `sinfo`, so an H100 run will require either a different partition name or additional access confirmation.

In other words, the currently confirmed and usable GPU target in this environment is GH200.

## Repository Structure

```
.
├── epsilonparam/                   # epsilon-parameterization model
├── xparam/                         # x-parameterization model
│   ├── evaluate_compression.py     # evaluation script (compression rate + size report)
│   ├── run_evaluation.sh           # Slurm job script for DeltaAI (100-image sweep)
│   └── run_b02048_resume.sh        # targeted resume script for unfinished b0.2048 images
├── imgs/                           # sample Kodak test images
└── environment.yml                 # conda environment
```

## Model Weights

HuggingFace: [rhyang/CDC_params](https://huggingface.co/rhyang/CDC_params)
- `epsilon_lpips0.9.pt` — use with `--lpips_weight 0.9`
- `epsilon_lpips0.0.pt` — use with `--lpips_weight 0.0`
- x-param weights are ~2x larger (EMA + latest model saved; only EMA is loaded)

---

## Running on NSF DeltaAI (UIUC)

### Step 1: Log in to DeltaAI

```bash
ssh yyang48@dtai-login.delta.ncsa.illinois.edu
```

Use DUO two-factor authentication when prompted.

---

### Step 2: Set up your working directory

| Path | Use |
|------|-----|
| `$HOME` | Code, small files |
| `/scratch/<allocation>/$USER/` | Large data, weights, outputs (faster I/O) |

```bash
mkdir -p /projects/bfod/$USER/cdc-deltaai
cd /projects/bfod/$USER/cdc-deltaai
```

---

### Step 3: Clone this repo on DeltaAI

```bash
git clone https://github.com/rayford295/sc26-cdc-deltaai.git code
cd code
```

---

### Step 4: Transfer your image data

From your **local machine**:

```bash
# Transfer drone images (e.g., 100_0005/ dataset)
rsync -avz /path/to/your/images/ \
  yyang48@dtai-login.delta.ncsa.illinois.edu:/projects/bfod/$USER/cdc-deltaai/data/imgs/
```

---

### Step 5: Download model weights

```bash
# On DeltaAI
cd /projects/bfod/$USER/cdc-deltaai
mkdir -p weights

module load anaconda3
conda activate exp_pytorch
pip install huggingface_hub --quiet

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='rhyang/CDC_params', local_dir='./weights')
"
```

---

### Step 6: Set up the conda environment

```bash
module load anaconda3

# First time only (~10 min)
conda env create -f code/environment.yml
conda activate exp_pytorch
```

> If `environment.yml` fails due to CUDA version conflicts, install manually:
> ```bash
> conda create -n exp_pytorch python=3.9
> conda activate exp_pytorch
> conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
> pip install compressai einops ema-pytorch lpips opencv-python scikit-image timm tqdm
> ```

---

### Step 7: Submit the evaluation job

The default sweep script now processes 100 images across all 6 x-param checkpoints:

```bash
cd /projects/bfod/$USER/cdc-deltaai/code
mkdir -p xparam/logs output/evaluation

sbatch xparam/run_evaluation.sh
```

If the final checkpoint sweep is interrupted and only the unfinished `b0.2048` images need to be completed, use:

```bash
sbatch xparam/run_b02048_resume.sh
```

Monitor the job:

```bash
squeue -u $USER                          # check status
tail -f xparam/logs/eval_<job_id>.log   # live output
scancel <job_id>                         # cancel if needed
```

---

### Step 8: Retrieve results

From your **local machine**:

```bash
rsync -avz \
  yyang48@dtai-login.delta.ncsa.illinois.edu:/projects/bfod/$USER/cdc-deltaai/output/ \
  ./output/
```

The output folder will contain:
- `compression_report.txt` — overall compression rate summary
- `compression_results.csv` — per-image BPP and file size data
- `*_recon.png` — reconstructed images

---

### Step 9: Interactive session (for debugging)

```bash
srun --account=bfod-dtai-gh --partition=ghx4-interactive \
     --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G \
     --time=01:00:00 --pty bash
```

---

## DeltaAI GPU Partitions

| Partition | GPU | Notes |
|-----------|-----|-------|
| `ghx4` | 4x NVIDIA GH200 120GB | Current confirmed batch partition |
| `ghx4-interactive` | 4x NVIDIA GH200 120GB | Current confirmed interactive partition |
| `full` | 4x NVIDIA GH200 120GB | Visible in `sinfo`, but not the current validated workflow target |
| `gpuH100x8` | 8x H100 | Planned target once partition and access are confirmed |

---

## Evaluation Script

`xparam/evaluate_compression.py` processes N drone images through the CDC model and reports:

1. **Overall compression rate** — average BPP vs uncompressed RGB (24 bpp)
2. **File size comparison** — total original JPEG size vs reconstructed PNG size

It also supports `--start_index` for partial resume runs when only a subset of the sorted image list needs to be processed.

```bash
python xparam/evaluate_compression.py \
  --ckpt        /path/to/checkpoint.pt \
  --img_dir     /path/to/images \
  --out_dir     /path/to/output \
  --n_images    100 \
  --start_index 0 \
  --lpips_weight 0.9
```

---

## DeltaAI Documentation

- Official docs: https://docs.ncsa.illinois.edu/systems/deltaai/
- Help: help@ncsa.illinois.edu
