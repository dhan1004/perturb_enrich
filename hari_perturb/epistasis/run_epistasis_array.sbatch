#!/usr/bin/env bash
#SBATCH --job-name=hari_conditional
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one gene (row COND_GENES_TSV[$SLURM_ARRAY_TASK_ID]). Submit with
# submit_conditional_array.sh, which freezes the gene list and sets COND_ROOT/COND_GENES_TSV.
# Every condition is checkpointed per window: if a task hits the time limit, just resubmit the
# same genes and it resumes.
set -Eeuo pipefail
trap 'printf "ERROR at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR

: "${COND_ROOT:?Submit through submit_conditional_array.sh (sets COND_ROOT)}"
: "${COND_GENES_TSV:?Submit through submit_conditional_array.sh (sets COND_GENES_TSV)}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
source "${CONDITIONAL_CONFIG:-$COND_ROOT/conditional_config.sh}"
: "${REF_SCAN_ROOT:?Set REF_SCAN_ROOT (original HARI scan output root) in conditional_config.sh}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

printf 'Node: %s\nPython: %s\n' "$(hostname)" "$(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" "$COND_ROOT/epistasis_scan.py" run \
    --ref-root "$REF_SCAN_ROOT" --out-root "$COND_OUT_ROOT" \
    --fasta "$FASTA" --suite-root "$SUITE_ROOT" \
    --genes-tsv "$COND_GENES_TSV" --task-index "$SLURM_ARRAY_TASK_ID" "$@"