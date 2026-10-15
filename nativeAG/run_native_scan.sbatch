#!/usr/bin/env bash
#SBATCH --job-name=NativeAG_scan
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=NativeAG_scan_%A_%a.out
#SBATCH --error=NativeAG_scan_%A_%a.err

set -Eeuo pipefail
trap 'printf "ERROR at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
# Set NATIVE_SCAN_ROOT before sbatch if submitting from a different directory.
NATIVE_SCAN_ROOT="${NATIVE_SCAN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -n "${CONDA_SH:-}" ]]; then
    : "${CONDA_ENV:?Set CONDA_ENV when using CONDA_SH}"
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -f "$NATIVE_SCAN_ROOT/scan_native_alphagenome.py" ]] || { printf 'Set NATIVE_SCAN_ROOT to the package directory\n' >&2; exit 2; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
printf 'Node: %s\n' "$(hostname)"
printf 'Python: %s\n' "$(command -v "$PYTHON_BIN")"
if [[ -n "${NATIVE_ARRAY_TARGETS:-}" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
    array_gene="$("$PYTHON_BIN" - "$NATIVE_ARRAY_TARGETS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
i = int(sys.argv[2])
if not 0 <= i < len(rows):
    raise SystemExit('Array task index outside target snapshot')
print(rows[i]['gene_id'])
PY
    )"
    "$PYTHON_BIN" "$NATIVE_SCAN_ROOT/scan_native_alphagenome.py" scan \
        --targets "$NATIVE_ARRAY_TARGETS" --genes "$array_gene" "$@"
else
    "$PYTHON_BIN" "$NATIVE_SCAN_ROOT/scan_native_alphagenome.py" scan "$@"
fi
