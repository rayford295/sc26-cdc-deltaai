#!/bin/bash
#SBATCH --job-name=sc26_cdc_store
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/compression/slurm/logs/storage_%j.log
#SBATCH --error=experiments/compression/slurm/logs/storage_%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_deltaai.sh"

RUN_ROOT="${OUT_BASE}/05_storage_compare"
mkdir -p "${RUN_ROOT}"

echo ">>> Shared storage run"
run_sc26_compression \
  --ckpt "${CKPT_BASELINE}" \
  --img_dir "${IMG_DIR}" \
  --out_dir "${RUN_ROOT}/shared_storage" \
  --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
  --experiment_name "shared_storage" \
  --compression_setting "baseline_b02048" \
  --resolution_label "native_full_resolution" \
  --max_edge 0 \
  --batch_size 1 \
  --n_images "${N_IMAGES}" \
  --start_index "${START_INDEX}" \
  --n_denoise_step "${N_DENOISE_STEP}" \
  --gamma "${GAMMA}" \
  --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
  --storage_label "shared" \
  --notes "Reads images and checkpoint from project/shared filesystem."

echo ">>> Local storage staging run"
LOCAL_STAGE="${SLURM_TMPDIR:-/tmp/${USER}/sc26_cdc_storage_${SLURM_JOB_ID:-local}}"
LOCAL_IMG_DIR="${LOCAL_STAGE}/imgs"
LOCAL_WEIGHT_DIR="${LOCAL_STAGE}/weights"
mkdir -p "${LOCAL_IMG_DIR}" "${LOCAL_WEIGHT_DIR}"

echo "Staging images to ${LOCAL_IMG_DIR}"
rsync -a "${IMG_DIR}/" "${LOCAL_IMG_DIR}/"
echo "Staging checkpoint to ${LOCAL_WEIGHT_DIR}"
cp "${CKPT_BASELINE}" "${LOCAL_WEIGHT_DIR}/$(basename "${CKPT_BASELINE}")"
LOCAL_CKPT="${LOCAL_WEIGHT_DIR}/$(basename "${CKPT_BASELINE}")"

run_sc26_compression \
  --ckpt "${LOCAL_CKPT}" \
  --img_dir "${LOCAL_IMG_DIR}" \
  --out_dir "${RUN_ROOT}/local_storage" \
  --lpips_weight "${LPIPS_WEIGHT_BASELINE}" \
  --experiment_name "local_storage" \
  --compression_setting "baseline_b02048" \
  --resolution_label "native_full_resolution" \
  --max_edge 0 \
  --batch_size 1 \
  --n_images "${N_IMAGES}" \
  --start_index "${START_INDEX}" \
  --n_denoise_step "${N_DENOISE_STEP}" \
  --gamma "${GAMMA}" \
  --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
  --storage_label "local" \
  --notes "Reads images and checkpoint from node-local staged storage."

summarize_sc26_compression "${RUN_ROOT}"
