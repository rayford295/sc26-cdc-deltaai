#!/bin/bash
#SBATCH --job-name=cdc-evaluate
#SBATCH --account=CIV25002
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yyang48@illinois.edu

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/scratch/CIV25002/$USER/cdc-deltaai
CKPT=$BASE/weights/xparam_ema.pt
IMG_DIR=$BASE/data/imgs
OUT_DIR=$BASE/output/evaluation
CODE_DIR=$BASE/code/xparam

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda3_gpu
conda activate exp_pytorch

mkdir -p $OUT_DIR
mkdir -p $CODE_DIR/logs

# ── Run evaluation on drone images ────────────────────────────────────────────
echo "Starting CDC compression evaluation..."
echo "Images:     $IMG_DIR"
echo "Output:     $OUT_DIR"
echo "Checkpoint: $CKPT"

cd $CODE_DIR

python evaluate_compression.py \
    --ckpt           $CKPT \
    --img_dir        $IMG_DIR \
    --out_dir        $OUT_DIR \
    --n_images       100 \
    --gamma          0.8 \
    --n_denoise_step 65 \
    --device         0 \
    --lpips_weight   0.9

echo "Done. Results in: $OUT_DIR"
