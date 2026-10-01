#!/usr/bin/env bash
#SBATCH --job-name=perturb_enrich
#SBATCH --array=1-1000%50            # EDIT: 1-N where N = number of lines in genes.txt; %50 = max concurrent tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Submit from the project directory (the one holding config.sh and genes.txt):
#   mkdir -p logs
#   sbatch --array=1-$(wc -l < genes.txt)%50 run_array.sbatch
#
# Optional environment overrides:
#   GENES_FILE   file with one gene symbol per line   (default: genes.txt)
#   CONFIG       path to config.sh                    (default: ./config.sh)
#   RUNNER       command prefix, e.g. "apptainer exec pipeline.sif"
#   EXTRA_ARGS   extra run_gene.py flags, e.g. "--enrichment" or "--force"
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?must be run under sbatch}"
GENES_FILE=${GENES_FILE:-genes.txt}
CONFIG=${CONFIG:-$PWD/config.sh}
RUNNER=${RUNNER:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
SCRIPT_DIR=${SCRIPT_DIR:-$PWD}        

GENE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GENES_FILE" | tr -d '[:space:]')
[[ -n "$GENE" ]] || { echo "No gene on line ${SLURM_ARRAY_TASK_ID} of $GENES_FILE" >&2; exit 1; }

# --- environment: edit for your cluster ---
# module load bedtools homer
# source activate perturb_enrich

echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID}: $GENE on $(hostname)"

# shellcheck disable=SC2086
$RUNNER python3 "$SCRIPT_DIR/run_gene.py" --gene "$GENE" --config "$CONFIG" $EXTRA_ARGS

echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID}: $GENE finished"
