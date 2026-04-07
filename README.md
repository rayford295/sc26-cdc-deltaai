# CDC Image Compression on DeltaAI

Lossy image compression with Conditional Diffusion Models, adapted for running on [NSF DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/).

Based on: [Lossy Image Compression with Conditional Diffusion Models](https://arxiv.org/pdf/2209.06950.pdf)

## Repository Structure

```
.
├── epsilonparam/       # epsilon-parameterization model
├── xparam/             # x-parameterization model
├── imgs/               # sample Kodak test images
├── scripts/
│   └── run_deltaai.sh  # Slurm job script for DeltaAI
├── data/
│   └── README.md       # instructions for uploading data & weights
└── environment.yml     # conda environment
```

## Quick Start on DeltaAI

### 1. Clone this repo on DeltaAI

```bash
ssh YOUR_NETID@login.deltaai.ncsa.illinois.edu
git clone https://github.com/rayford295/sc26-cdc-deltaai.git
cd sc26-cdc-deltaai
```

### 2. Set up conda environment

```bash
module load anaconda3
conda env create -f environment.yml
conda activate cdc_env
```

### 3. Upload data and weights — see data/README.md

### 4. Submit the Slurm job

```bash
nano scripts/run_deltaai.sh   # fill in YOUR_ALLOCATION and paths
sbatch scripts/run_deltaai.sh
squeue -u $USER
```

## Model Weights

HuggingFace: [rhyang/CDC_params](https://huggingface.co/rhyang/CDC_params)
- `epsilon_lpips0.9.pt` — use with `--lpips_weight 0.9`
- `epsilon_lpips0.0.pt` — use with `--lpips_weight 0.0`
