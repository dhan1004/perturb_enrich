#!/usr/bin/env bash
#SBATCH --job-name=perturb_enrich
#SBATCH --partition=day
#SBATCH --cpus-per-task=8               # matches THREADS=8 in config.sh (HOMER -p)
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Submit from the repo root (the directory holding config.sh):
#   mkdir -p logs
#   N=$(bash -c 'source config.sh; ls -d "$PERTURB_ROOT_SCAN"/*/results/Astrocyte/regions.tsv | wc -l')
#   sbatch --array=1-${N}%20 run_array.sbatch
#
# Optional: FORCE=1 to rerun genes that already have final_regions.tsv
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?must be run under sbatch}"
source ./config.sh

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb
export PATH="${HOMER_HOME}/bin:${PATH}"

CELLTYPE_DIR=Astrocyte

# Same sorted list in every task, so array index N always maps to the same gene
mapfile -t REGION_FILES < <(ls -d "${PERTURB_ROOT_SCAN}"/*/results/${CELLTYPE_DIR}/regions.tsv | sort)
REGIONS="${REGION_FILES[$((SLURM_ARRAY_TASK_ID - 1))]:-}"
[[ -n "$REGIONS" ]] || { echo "No gene for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

# <root>/<gene_id>/results/Astrocyte/regions.tsv -> <gene_id>
GENE=$(basename "$(dirname "$(dirname "$(dirname "$REGIONS")")")")

if [[ -s "${OUT_ROOT}/genes/${GENE}/final_regions.tsv" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date '+%F %T')] ${GENE}: already done, skipping (FORCE=1 to rerun)"
    exit 0
fi

echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID}: ${GENE} on $(hostname)"
python scripts/run_gene.py --gene "$GENE" --regions "$REGIONS" --config "$PWD/config.sh"
echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID}: ${GENE} finished"