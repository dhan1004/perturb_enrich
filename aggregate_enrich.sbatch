#!/usr/bin/env bash
#SBATCH --job-name=aggregate_enrich
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Pool per-gene counts -> $OUT_ROOT/enrichment/*.tsv. Run once after all genes finished
# (sbatch --dependency=afterok:<array_jobid> aggregate_enrich.sbatch).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?must be run under sbatch}"
source ./config.sh
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb
python scripts/aggregate_enrich.py --out-root "$OUT_ROOT"
