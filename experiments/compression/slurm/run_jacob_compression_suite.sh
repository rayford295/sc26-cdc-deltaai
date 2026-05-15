#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S_jacob_compression)}"
N_IMAGES="${N_IMAGES:-8}"
SHARD_IMAGES="${SHARD_IMAGES:-8}"
SAVE_VISUAL_LIMIT="${SAVE_VISUAL_LIMIT:-2}"

export REPO_DIR RUN_STAMP N_IMAGES SHARD_IMAGES SAVE_VISUAL_LIMIT

echo "Submitting Jacob compression-side SC26 suite."
echo "Repo        : ${REPO_DIR}"
echo "Run stamp   : ${RUN_STAMP}"
echo "N_IMAGES    : ${N_IMAGES}"
echo "SHARD_IMAGES: ${SHARD_IMAGES}"
echo ""
echo "This suite submits:"
echo "  01 baseline + resolution + batch-size sweep"
echo "  02 checkpoint sweep for practical low/medium/high compression"
echo "  04 HPC scaling array"
echo "  05 shared-vs-local storage comparison"
echo "  06 final summary"
echo ""

jid1=$(sbatch --parsable --export=ALL,REPO_DIR="${REPO_DIR}",RUN_STAMP="${RUN_STAMP}",N_IMAGES="${N_IMAGES}",SAVE_VISUAL_LIMIT="${SAVE_VISUAL_LIMIT}" "${SCRIPT_DIR}/01_baseline_resolution_batch.sbatch")
echo "Submitted baseline/resolution/batch job: ${jid1}"

jid2=$(sbatch --parsable --dependency=afterok:${jid1} --export=ALL,REPO_DIR="${REPO_DIR}",RUN_STAMP="${RUN_STAMP}",N_IMAGES="${N_IMAGES}",SAVE_VISUAL_LIMIT="${SAVE_VISUAL_LIMIT}" "${SCRIPT_DIR}/02_checkpoint_level_sweep.sbatch")
echo "Submitted checkpoint-level sweep job: ${jid2}"

jid4=$(sbatch --parsable --export=ALL,REPO_DIR="${REPO_DIR}",RUN_STAMP="${RUN_STAMP}",SHARD_IMAGES="${SHARD_IMAGES}",SAVE_VISUAL_LIMIT="${SAVE_VISUAL_LIMIT}" "${SCRIPT_DIR}/04_hpc_scaling_array.sbatch")
echo "Submitted HPC scaling array job: ${jid4}"

jid5=$(sbatch --parsable --dependency=afterok:${jid1} --export=ALL,REPO_DIR="${REPO_DIR}",RUN_STAMP="${RUN_STAMP}",N_IMAGES="${N_IMAGES}",SAVE_VISUAL_LIMIT="${SAVE_VISUAL_LIMIT}" "${SCRIPT_DIR}/05_storage_compare.sbatch")
echo "Submitted storage compare job: ${jid5}"

jid6=$(sbatch --parsable --dependency=afterany:${jid2}:${jid4}:${jid5} --export=ALL,REPO_DIR="${REPO_DIR}",RUN_STAMP="${RUN_STAMP}" "${SCRIPT_DIR}/06_summarize_all.sbatch")
echo "Submitted final summary job: ${jid6}"

echo ""
echo "Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f ${REPO_DIR}/experiments/compression/slurm/logs/*.log"
echo ""
echo "Result root:"
echo "  /projects/bfod/${USER}/cdc-deltaai/output/sc26_compression/${RUN_STAMP}/"
