#!/bin/bash
#SBATCH --job-name=cdc-evaluate
#SBATCH --account=<your_allocation>        # TODO: replace with your allocation ID
#SBATCH --partition=gpuA100x4             # DeltaAI A100 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>           # TODO: replace with your email

set -e

# ── Paths (edit these) ────────────────────────────────────────────────────────
BASE=/scratch/<your_allocation>/$USER/cdc-deltaai    # TODO: replace allocation
CKPT=$BASE/weights/<checkpoint_name>.pt              # TODO: replace checkpoint filename
IMG_DIR=$BASE/data/imgs
OUT_DIR=$BASE/output/evaluation
CODE_DIR=$BASE/code/xparam

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda3_gpu
conda activate exp_pytorch

mkdir -p $OUT_DIR
mkdir -p $CODE_DIR/logs

# ── Run evaluation on 100 drone images ───────────────────────────────────────
echo "Starting CDC compression evaluation..."
echo "Images: $IMG_DIR"
echo "Output: $OUT_DIR"
echo "Checkpoint: $CKPT"

cd $CODE_DIR

python evaluate_compression.py \
    --ckpt        $CKPT \
    --img_dir     $IMG_DIR \
    --out_dir     $OUT_DIR \
    --n_images    100 \
    --gamma       0.8 \
    --n_denoise_step 65 \
    --device      0 \
    --lpips_weight 0.9

echo "Done. Results in: $OUT_DIR"
