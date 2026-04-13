#!/bin/bash
#SBATCH --job-name=cdc_compress
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yyang48@illinois.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3_gpu

conda activate exp_pytorch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/projects/bfod/yyang48/cdc-deltaai
DATA_DIR=$BASE/data/100_0005
CKPT_DIR=$BASE/weights
OUT_DIR=$BASE/output/${SLURM_JOB_ID}

mkdir -p $OUT_DIR logs

# ── Run ───────────────────────────────────────────────────────────────────────
cd /u/yyang48/code/epsilonparam

python test_epsilonparam.py \
    --ckpt           $CKPT_DIR/epsilon_lpips0.9.pt \
    --img_dir        $DATA_DIR \
    --out_dir        $OUT_DIR \
    --gamma          0.8 \
    --n_denoise_step 200 \
    --lpips_weight   0.9 \
    --device         0

echo "Done. Output at: $OUT_DIR"
