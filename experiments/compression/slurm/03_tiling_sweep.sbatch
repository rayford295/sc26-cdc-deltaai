#!/bin/bash
#SBATCH --job-name=sc26_cdc_tile
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/compression/slurm/logs/tiling_%j.log
#SBATCH --error=experiments/compression/slurm/logs/tiling_%j.log

set -euo pipefail

if [[ -n "${REPO_DIR:-}" && -f "${REPO_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${REPO_DIR}/experiments/compression/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/experiments/compression/slurm"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_deltaai.sh"

RUN_ROOT="${OUT_BASE}/03_tiling_sweep"
mkdir -p "${RUN_ROOT}"

echo ">>> No-tiling reference for the same image subset"
run_sc26_compression \
  --ckpt "${CKPT_BASELINE}" \
  --img_dir "${IMG_DIR}" \
  --out_dir "${RUN_ROOT}/tile_none_reference" \
  --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
  --experiment_name "tile_none_reference" \
  --compression_setting "baseline_b02048" \
  --resolution_label "native_full_resolution" \
  --max_edge 0 \
  --batch_size 1 \
  --n_images "${N_IMAGES}" \
  --start_index "${START_INDEX}" \
  --n_denoise_step "${N_DENOISE_STEP}" \
  --gamma "${GAMMA}" \
  --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
  --comparison_max_edge "${COMPARISON_MAX_EDGE}" \
  --storage_label "shared" \
  --notes "Reference run for tiling comparison."

echo ">>> Tiling sweep"
for TILE_SIZE in ${TILING_SIZES}; do
  CURRENT_TILE_BATCH_SIZE="${TILE_BATCH_SIZE}"
  if [[ "${TILE_SIZE}" == "256" ]]; then
    CURRENT_TILE_BATCH_SIZE="${TILE_BATCH_SIZE_256}"
  fi

  run_sc26_compression \
    --ckpt "${CKPT_BASELINE}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${RUN_ROOT}/tile_${TILE_SIZE}" \
    --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
    --experiment_name "tile_${TILE_SIZE}" \
    --compression_setting "baseline_b02048" \
    --resolution_label "native_full_resolution" \
    --max_edge 0 \
    --batch_size "${CURRENT_TILE_BATCH_SIZE}" \
    --tile_size "${TILE_SIZE}" \
    --tile_batch_size "${CURRENT_TILE_BATCH_SIZE}" \
    --n_images "${N_IMAGES}" \
    --start_index "${START_INDEX}" \
    --n_denoise_step "${N_DENOISE_STEP}" \
    --gamma "${GAMMA}" \
    --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
    --comparison_max_edge "${COMPARISON_MAX_EDGE}" \
    --storage_label "shared" \
    --notes "Independent ${TILE_SIZE} x ${TILE_SIZE} tile compression followed by direct stitching and heatmap-based visual comparison."
done

summarize_sc26_compression "${RUN_ROOT}"
