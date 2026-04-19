#!/bin/bash
# =============================================================================
# run_profiling_sweep.sh
# ----------------------
# SLURM job script to run the CDC reconstruction profiling sweep on DeltaAI.
#
# What this does:
#   1. Runs profile_reconstruction.py on 10 images with the baseline config
#      (65 steps, fp32) to get a detailed timing breakdown.
#   2. Runs sweep_steps.py across 7 step counts (5, 10, 20, 30, 50, 65, 100)
#      in both fp32 and fp16 to map the speed/quality trade-off.
#   3. Runs plot_results.py to generate all five plots from the sweep CSV.
#
# Submit with:
#   sbatch xparam/run_profiling_sweep.sh
#
# Monitor with:
#   squeue -u $USER
#   tail -f xparam/logs/profiling_<jobid>.log
# =============================================================================

#SBATCH --job-name=cdc_profile_sweep
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=xparam/logs/profiling_%j.log
#SBATCH --error=xparam/logs/profiling_%j.log

# ── Environment setup ─────────────────────────────────────────────────────────
module load anaconda3
conda activate exp_pytorch

# Install skimage and matplotlib if not already present
pip install scikit-image matplotlib --quiet

# ── Paths -- edit these to match your DeltaAI setup ──────────────────────────
REPO_DIR="/projects/bfod/$USER/cdc-deltaai/code"
IMG_DIR="/projects/bfod/$USER/cdc-deltaai/data/imgs"
WEIGHT_DIR="/projects/bfod/$USER/cdc-deltaai/weights"

# Checkpoint to use for profiling (choose any one checkpoint for step sweep)
CKPT="${WEIGHT_DIR}/xparam/b0.2048.pt"
LPIPS_WEIGHT=0.9

# Output directories
PROFILE_OUT="/projects/bfod/$USER/cdc-deltaai/output/profiling"
SWEEP_OUT="/projects/bfod/$USER/cdc-deltaai/output/sweep"
PLOT_OUT="/projects/bfod/$USER/cdc-deltaai/output/plots"

mkdir -p xparam/logs "${PROFILE_OUT}" "${SWEEP_OUT}" "${PLOT_OUT}"

cd "${REPO_DIR}/xparam"

echo "=========================================="
echo "  CDC Reconstruction Profiling Sweep"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date   : $(date)"
echo "=========================================="

# ── Step 1: Detailed profiling (baseline = 65 steps, fp32) ───────────────────
echo ""
echo ">>> STEP 1: Profiling baseline (65 steps, fp32, 10 images)"
echo ""

python profile_reconstruction.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${PROFILE_OUT}/baseline_65steps_fp32" \
    --n_denoise_step 65 \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 10 \
    --device 0

echo ""
echo ">>> STEP 1 (fp16): Profiling baseline (65 steps, fp16, 10 images)"
echo ""

python profile_reconstruction.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${PROFILE_OUT}/baseline_65steps_fp16" \
    --n_denoise_step 65 \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 10 \
    --device 0 \
    --fp16

# ── Step 2: Parameter sweep over step counts ──────────────────────────────────
echo ""
echo ">>> STEP 2: Sweeping step counts [5 10 20 30 50 65 100] x [fp32, fp16]"
echo ""

python sweep_steps.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${SWEEP_OUT}" \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 5 \
    --device 0 \
    --test_fp16 \
    --steps 5 10 20 30 50 65 100

# ── Step 3: Generate plots ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Generating plots"
echo ""

python plot_results.py \
    --sweep_csv "${SWEEP_OUT}/sweep_results.csv" \
    --out_dir "${PLOT_OUT}"

echo ""
echo "=========================================="
echo "  All done. Results saved to:"
echo "    Profiling : ${PROFILE_OUT}"
echo "    Sweep     : ${SWEEP_OUT}"
echo "    Plots     : ${PLOT_OUT}"
echo "  $(date)"
echo "=========================================="
