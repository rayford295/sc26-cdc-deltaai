#!/bin/bash
#SBATCH --job-name=sc26_cdc_ckpt
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/compression/slurm/logs/checkpoints_%j.log
#SBATCH --error=experiments/compression/slurm/logs/checkpoints_%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_deltaai.sh"

RUN_ROOT="${OUT_BASE}/02_checkpoint_level_sweep"
mkdir -p "${RUN_ROOT}"

python experiments/compression/inspect_compression_controls.py

echo ">>> Checkpoint sweep for compression-level candidates"
for SPEC in \
  "b00032 ${CKPT_B00032} ${LPIPS_B00032}" \
  "b00064 ${CKPT_B00064} ${LPIPS_B00064}" \
  "b00128 ${CKPT_B00128} ${LPIPS_B00128}" \
  "b00512 ${CKPT_B00512} ${LPIPS_B00512}" \
  "b01024 ${CKPT_B01024} ${LPIPS_B01024}" \
  "b02048 ${CKPT_B02048} ${LPIPS_B02048}"
do
  read -r CKPT_LABEL CKPT_PATH LPIPS_VALUE <<< "${SPEC}"
  run_sc26_compression \
    --ckpt "${CKPT_PATH}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${RUN_ROOT}/checkpoint_${CKPT_LABEL}" \
    --lpips_weight "${LPIPS_VALUE}" \
    --experiment_name "checkpoint_${CKPT_LABEL}" \
    --compression_setting "checkpoint_${CKPT_LABEL}" \
    --resolution_label "native_full_resolution" \
    --max_edge 0 \
    --batch_size 1 \
    --n_images "${N_IMAGES}" \
    --start_index "${START_INDEX}" \
    --n_denoise_step "${N_DENOISE_STEP}" \
    --gamma "${GAMMA}" \
    --save_visual_limit "${SAVE_VISUAL_LIMIT}" \
    --storage_label "shared" \
    --notes "Checkpoint sweep. Assign low/medium/high compression from measured BPP and compression ratio."
done

summarize_sc26_compression "${RUN_ROOT}"
