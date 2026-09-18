#!/usr/bin/env bash
#SBATCH --job-name=homer_enrich
#SBATCH --partition=day      # sinfo -s to list; a CPU partition
#SBATCH --cpus-per-task=8               # matches THREADS=8 in config.sh
#SBATCH --mem=16G                       # preparse of masked hg38 is memory-hungry
#SBATCH --time=02:00:00                 # generous for the one-time genome preparse
#SBATCH --output=logs/homer_%j.log      # %j = job ID; stderr merges here too

set -euo pipefail

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hari_env

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${HERE}/config.sh"
export PATH="${HOMER_HOME}/bin:${PATH}"

bash "${HERE}/scripts/run_homer.sh"