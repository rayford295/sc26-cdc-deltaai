#!/bin/bash
#SBATCH --job-name=cdc-epsilon-eval
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/epsilon_eval_%j.out
#SBATCH --error=logs/epsilon_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yyang48@illinois.edu

set -e

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load default
module load gcc/14.2.0
module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH=/u/yyang48/.local/lib/python3.12/site-packages:$PYTHONPATH
python -m pip install --user ema-pytorch lpips --quiet

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/projects/bfod/yyang48/cdc-deltaai
CKPT_DIR=$BASE/weights/epsilon_param
IMG_DIR=$BASE/data/imgs
OUT_BASE=$BASE/output/epsilon_eval_${SLURM_JOB_ID}
CODE_DIR=/u/yyang48/code/epsilonparam

mkdir -p $OUT_BASE logs

echo "Starting CDC epsilon-param compression evaluation..."
echo "Images:   $IMG_DIR"
echo "Output:   $OUT_BASE"

cd $CODE_DIR

# ── Loop over all epsilon_param checkpoints ───────────────────────────────────
for CKPT in \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.0128-vbrFalse-noise-linear-aux-1.0_0_ckpt.pt" \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.0256-vbrFalse-noise-linear-aux-1.0_0_ckpt.pt" \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.0512-vbrFalse-noise-linear-aux-1.0_0_ckpt.pt" \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.1024-vbrFalse-noise-linear-aux0.9lpips_0.pt" \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.2048-vbrFalse-noise-linear-aux0.9lpips_0.pt" \
    "$CKPT_DIR/big-l1-vimeo-d64-t20000-b0.3072-vbrFalse-noise-linear-aux0.9lpips_0.pt"
do
    BNAME=$(basename "$CKPT" .pt)
    BRATE=$(echo "$BNAME" | grep -oP 'b[0-9]+\.[0-9]+')

    if echo "$BNAME" | grep -q "lpips"; then
        LPIPS=0.9
    else
        LPIPS=0.0
    fi

    OUT_DIR=$OUT_BASE/$BRATE
    mkdir -p $OUT_DIR

    echo ""
    echo "=== Running: $BRATE (lpips_weight=$LPIPS) ==="
    python test_epsilonparam.py \
        --ckpt           "$CKPT" \
        --img_dir        "$IMG_DIR" \
        --out_dir        "$OUT_DIR" \
        --gamma          0.8 \
        --n_denoise_step 200 \
        --lpips_weight   $LPIPS \
        --device         0

done

echo ""
echo "Done. All results in: $OUT_BASE"
