# CDC Image Compression on DeltaAI

Lossy image compression with Conditional Diffusion Models, adapted for running on [NSF DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/).

Based on: [Lossy Image Compression with Conditional Diffusion Models](https://arxiv.org/pdf/2209.06950.pdf)

## Repository Structure

```
.
├── epsilonparam/                   # epsilon-parameterization model
├── xparam/                         # x-parameterization model
│   ├── evaluate_compression.py     # evaluation script (compression rate + size report)
│   └── run_evaluation.sh           # Slurm job script for DeltaAI
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
ssh YOUR_NETID@dtai-login.delta.ncsa.illinois.edu
```

Use DUO two-factor authentication when prompted.

---

### Step 2: Set up your working directory

| Path | Use |
|------|-----|
| `$HOME` | Code, small files |
| `/scratch/<allocation>/$USER/` | Large data, weights, outputs (faster I/O) |

```bash
mkdir -p /scratch/<your_allocation>/$USER/cdc-deltaai
cd /scratch/<your_allocation>/$USER/cdc-deltaai
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
  YOUR_NETID@dtai-login.delta.ncsa.illinois.edu:/scratch/<your_allocation>/$USER/cdc-deltaai/data/imgs/
```

---

### Step 5: Download model weights

```bash
# On DeltaAI
cd /scratch/<your_allocation>/$USER/cdc-deltaai
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

Edit `xparam/run_evaluation.sh` to fill in your allocation ID, email, and checkpoint path, then:

```bash
cd /scratch/<your_allocation>/$USER/cdc-deltaai/code
mkdir -p xparam/logs output/evaluation

sbatch xparam/run_evaluation.sh
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
  YOUR_NETID@dtai-login.delta.ncsa.illinois.edu:/scratch/<your_allocation>/$USER/cdc-deltaai/output/ \
  ./output/
```

The output folder will contain:
- `compression_report.txt` — overall compression rate summary
- `compression_results.csv` — per-image BPP and file size data
- `*_recon.png` — reconstructed images

---

### Step 9: Interactive session (for debugging)

```bash
srun --account=<your_allocation> --partition=gpuA100x4 \
     --nodes=1 --ntasks=1 --gres=gpu:1 --mem=16G \
     --time=01:00:00 --pty bash
```

---

## DeltaAI GPU Partitions

| Partition | GPU | Recommended for |
|-----------|-----|-----------------|
| `gpuA100x4` | 4x A100 40GB | General inference (use this) |
| `gpuA100x8` | 8x A100 80GB | Large memory jobs |
| `gpuH100x8` | 8x H100 | Largest jobs |

For 100-image inference, `gpuA100x4` with `--gres=gpu:1` is sufficient.

---

## Evaluation Script

`xparam/evaluate_compression.py` processes N drone images through the CDC model and reports:

1. **Overall compression rate** — average BPP vs uncompressed RGB (24 bpp)
2. **File size comparison** — total original JPEG size vs reconstructed PNG size

```bash
python xparam/evaluate_compression.py \
  --ckpt     /path/to/checkpoint.pt \
  --img_dir  /path/to/images \
  --out_dir  /path/to/output \
  --n_images 100 \
  --lpips_weight 0.9
```

---

## DeltaAI Documentation

- Official docs: https://docs.ncsa.illinois.edu/systems/deltaai/
- Help: help@ncsa.illinois.edu
