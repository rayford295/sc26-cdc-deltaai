#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_FROM_SCRIPT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

source "${EXPERIMENT_DIR}/configs/deltaai_paths.env"

module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
python -m pip install --user scikit-image compressai einops lpips ema-pytorch tqdm matplotlib pandas --quiet

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
else
  cd "${REPO_FROM_SCRIPT}"
fi

mkdir -p "${OUT_BASE}" experiments/compression/slurm/logs

echo "=========================================="
echo "SC26 compression experiment"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Array task : ${SLURM_ARRAY_TASK_ID:-none}"
echo "Run stamp  : ${RUN_STAMP}"
echo "Host       : $(hostname)"
echo "Repo       : $(pwd)"
echo "Images     : ${IMG_DIR}"
echo "Output     : ${OUT_BASE}"
echo "Date       : $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
fi
echo "=========================================="

run_sc26_compression() {
  python experiments/compression/run_compression_experiment.py "$@"
}

summarize_sc26_compression() {
  python experiments/compression/summarize_results.py --root "$1"
}
