#!/bin/bash
#SBATCH --job-name=sc26_cdc_base
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/compression/slurm/logs/baseline_%j.log
#SBATCH --error=experiments/compression/slurm/logs/baseline_%j.log

set -euo pipefail

if [[ -n "${REPO_DIR:-}" && -f "${REPO_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${REPO_DIR}/experiments/compression/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/experiments/compression/slurm"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_deltaai.sh"

RUN_ROOT="${OUT_BASE}/01_baseline_resolution_batch"
mkdir -p "${RUN_ROOT}"

echo ">>> Baseline: native/full-resolution batch-size pilot"
for BATCH_SIZE in 1 2 4; do
  run_sc26_compression \
    --ckpt "${CKPT_BASELINE}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${RUN_ROOT}/baseline_native_batch${BATCH_SIZE}" \
    --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
    --experiment_name "baseline_native_batch${BATCH_SIZE}" \
    --compression_setting "baseline_b02048" \
    --resolution_label "native_full_resolution" \
    --max_edge 0 \
    --batch_size "${BATCH_SIZE}" \
    --n_images "${N_IMAGES}" \
    --start_index "${START_INDEX}" \
    --n_denoise_step "${N_DENOISE_STEP}" \
    --gamma "${GAMMA}" \
    --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
    --storage_label "shared" \
    --notes "Baseline pretrained checkpoint with full native image resolution."
done

echo ">>> Resolution sweep: 4K, 2K, 1K at batch size 1"
for SPEC in "4k 4096" "2k 2048" "1k 1024"; do
  read -r RES_LABEL MAX_EDGE <<< "${SPEC}"
  run_sc26_compression \
    --ckpt "${CKPT_BASELINE}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${RUN_ROOT}/resolution_${RES_LABEL}" \
    --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
    --experiment_name "resolution_${RES_LABEL}" \
    --compression_setting "baseline_b02048" \
    --resolution_label "${RES_LABEL}" \
    --max_edge "${MAX_EDGE}" \
    --batch_size 1 \
    --n_images "${N_IMAGES}" \
    --start_index "${START_INDEX}" \
    --n_denoise_step "${N_DENOISE_STEP}" \
    --gamma "${GAMMA}" \
    --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
    --storage_label "shared" \
    --notes "Resolution sweep with longest image edge resized to ${MAX_EDGE}px."
done

summarize_sc26_compression "${RUN_ROOT}"
