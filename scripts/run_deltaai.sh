#!/bin/bash
#SBATCH --job-name=cdc_compress
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ALLOCATION   # <-- replace with your DeltaAI allocation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3

conda activate cdc_env

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR=$SLURM_SUBMIT_DIR
DATA_DIR=/projects/YOUR_PROJECT/data/100_0005   # <-- replace with your DeltaAI data path
CKPT_DIR=/projects/YOUR_PROJECT/weights         # <-- replace with your DeltaAI weights path
OUT_DIR=$REPO_DIR/outputs/${SLURM_JOB_ID}

mkdir -p $OUT_DIR logs

# ── Run ───────────────────────────────────────────────────────────────────────
cd $REPO_DIR/epsilonparam

python test_epsilonparam.py \
    --ckpt      $CKPT_DIR/epsilon_lpips0.9.pt \
    --img_dir   $DATA_DIR \
    --out_dir   $OUT_DIR \
    --gamma     0.8 \
    --n_denoise_step 200 \
    --lpips_weight 0.9 \
    --device    0

echo "Done. Output at: $OUT_DIR"
