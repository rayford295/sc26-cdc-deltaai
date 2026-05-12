#!/bin/bash
#SBATCH --job-name=sc26_cdc_scale
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=experiments/compression/slurm/logs/scaling_%A_%a.log
#SBATCH --error=experiments/compression/slurm/logs/scaling_%A_%a.log

set -euo pipefail

if [[ -n "${REPO_DIR:-}" && -f "${REPO_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${REPO_DIR}/experiments/compression/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/experiments/compression/slurm/common_deltaai.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/experiments/compression/slurm"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_deltaai.sh"

RUN_ROOT="${OUT_BASE}/04_hpc_scaling_array"
SHARD_IMAGES="${SHARD_IMAGES:-${N_IMAGES}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_START=$(( START_INDEX + TASK_ID * SHARD_IMAGES ))
mkdir -p "${RUN_ROOT}"

echo ">>> Parallel scaling shard ${TASK_ID}: start=${SHARD_START}, images=${SHARD_IMAGES}"

run_sc26_compression \
  --ckpt "${CKPT_BASELINE}" \
  --img_dir "${IMG_DIR}" \
  --out_dir "${RUN_ROOT}/parallel_shard_${TASK_ID}" \
  --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
  --experiment_name "parallel_shard_${TASK_ID}" \
  --compression_setting "baseline_b02048" \
  --resolution_label "native_full_resolution" \
  --max_edge 0 \
  --batch_size 1 \
  --n_images "${SHARD_IMAGES}" \
  --start_index "${SHARD_START}" \
  --n_denoise_step "${N_DENOISE_STEP}" \
  --gamma "${GAMMA}" \
  --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
  --storage_label "shared" \
  --notes "HPC scaling shard from a SLURM job array."

summarize_sc26_compression "${RUN_ROOT}"
