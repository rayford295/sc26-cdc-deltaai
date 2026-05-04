#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_STAMP

echo "Submitting SC26 compression suite."
echo "Suite timestamp: ${RUN_STAMP}"
echo "Override defaults like this:"
echo "  N_IMAGES=100 ${SCRIPT_DIR}/run_all_suite.sh"
echo ""

jid1=$(sbatch --parsable --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/01_baseline_resolution_batch.sbatch")
echo "Submitted baseline/resolution/batch job: ${jid1}"

jid2=$(sbatch --parsable --dependency=afterok:${jid1} --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/02_checkpoint_level_sweep.sbatch")
echo "Submitted checkpoint-level sweep job: ${jid2}"

jid3=$(sbatch --parsable --dependency=afterok:${jid1} --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/03_tiling_sweep.sbatch")
echo "Submitted tiling sweep job: ${jid3}"

jid4=$(sbatch --parsable --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/04_hpc_scaling_array.sbatch")
echo "Submitted HPC scaling array job: ${jid4}"

jid5=$(sbatch --parsable --dependency=afterok:${jid1} --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/05_storage_compare.sbatch")
echo "Submitted storage compare job: ${jid5}"

jid6=$(sbatch --parsable --dependency=afterany:${jid2}:${jid3}:${jid4}:${jid5} --export=ALL,RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/06_summarize_all.sbatch")
echo "Submitted final summary job: ${jid6}"

echo ""
echo "Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f experiments/compression/slurm/logs/*.log"
