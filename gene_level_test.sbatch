#!/usr/bin/env bash
#SBATCH --job-name=gene_level_test
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Gene-level robustness tests (gene bootstrap CI, per-gene signed-rank, stratum contrasts, null
# distributions, region-length summary). Run after aggregate_enrich.sbatch:
#   sbatch --dependency=afterok:<aggregate_jobid> gene_level_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source ./config.sh
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb
python scripts/gene_level_test.py --out-root "$OUT_ROOT"
