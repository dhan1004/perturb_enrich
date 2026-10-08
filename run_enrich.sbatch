#!/usr/bin/env bash
#SBATCH --job-name=enrich_gene
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


# Enrichment stage only (use for genes already run through run_array.sbatch; no HOMER rerun).
# Needs thresholds from calibrate_thresholds.sbatch. Then aggregate with a dependent job:
#   N=$(bash -c 'source config.sh; ls -d "$PERTURB_ROOT_SCAN"/*/results/Astrocyte/regions.tsv | wc -l')
#   JID=$(sbatch --parsable --array=1-${N}%50 enrich_array.sbatch)
#   sbatch --dependency=afterok:${JID} aggregate_enrich.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?must be run under sbatch}"
source ./config.sh
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb

mapfile -t REGION_FILES < <(ls -d "${PERTURB_ROOT_SCAN}"/*/results/Astrocyte/regions.tsv | sort)
REGIONS="${REGION_FILES[$((SLURM_ARRAY_TASK_ID - 1))]:-}"
[[ -n "$REGIONS" ]] || { echo "No gene for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
GENE=$(basename "$(dirname "$(dirname "$(dirname "$REGIONS")")")")

ARGS=()
[[ "${ENRICH_EXCLUDE_PERTURBED:-0}" == "1" ]] && ARGS+=(--exclude-perturbed)
python scripts/enrich_gene.py --gene "$GENE" --regions "$REGIONS" \
    --tested "$(dirname "$REGIONS")/tested_regions.tsv" --outdir "${OUT_ROOT}/genes/${GENE}" \
    --tracks "$TRACKS_MANIFEST" --thresholds "$THRESHOLDS_TSV" \
    --nperm "$ENRICH_NPERM" --strata "$ENRICH_STRATA" "${ARGS[@]}"