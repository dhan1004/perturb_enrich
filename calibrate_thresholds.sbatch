#!/usr/bin/env bash
#SBATCH --job-name=calibrate_thr
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One-time, genome-wide: bigwig -> q-value cutoff. Rerun if tracks.tsv or ENRICH_QVALUE changes.
#   mkdir -p logs; sbatch calibrate_thresholds.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?must be run under sbatch}"
source ./config.sh
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb
python scripts/calibrate_thresholds.py --manifest "$TRACKS_MANIFEST" --out "$THRESHOLDS_TSV" --alpha "$ENRICH_QVALUE"
