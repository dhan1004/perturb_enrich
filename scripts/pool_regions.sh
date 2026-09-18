#!/usr/bin/env bash

# Pool HARI perturbation regions to be foreground/background BEDs for HOMER

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../config.sh"
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hari_env

: "${PERTURB_ROOT_SCAN:?set PERTURB_ROOT_SCAN (source config.sh)}"
: "${OUT_ROOT:?set OUT_ROOT (source config.sh)}"

POOLED="${OUT_ROOT}/pooled"
mkdir -p "${POOLED}"

# merge helper 
merge_bed3() {
    sort -k1,1 -k2,2n | bedtools merge -i -
}

# find cell type directories under each gene's results
mapfile -t CELLS < <(find "${PERTURB_ROOT_SCAN}"/*/results -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -u)
if [ "${#CELLS[@]}" -eq 0 ]; then
    echo "ERROR: no cell types found under ${PERTURB_ROOT_SCAN}/*/results/" >&2
    exit 1
fi

# foreground (regions.bed)
echo "cell types: ${CELLS[*]}"
for cell in "${CELLS[@]}"; do
    out="${POOLED}/${cell}.fg.bed"
    cat "${PERTURB_ROOT_SCAN}"/*/results/"${cell}"/regions.bed 2>/dev/null \
        | sort -k1,1 -k2,2n > "${out}" || true
    echo "  ${cell}: $(wc -l < "${out}") regions -> ${out}"
done

# background (unmerged where backgrounds are NOT overlapped vs merged where they are)
UNMERGED="${POOLED}/bg.unmerged.bed"
MERGED="${POOLED}/bg.merged.bed"

cat "${PERTURB_ROOT_SCAN}"/*/results/tested_space.bed \
    | sort -k1,1 -k2,2n \
    | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,($4==""?"bg":$4),0,"."}' > "${UNMERGED}"

cat "${PERTURB_ROOT_SCAN}"/*/results/tested_space.bed \
    | cut -f1-3 \
    | merge_bed3 \
    | awk 'BEGIN{OFS="\t"}{print $1,$2,$3, "bg_"NR,0,"."}' > "${MERGED}"

nu=$(wc -l < "${UNMERGED}")
nm=$(wc -l < "${MERGED}")
echo "background: ${nu} rows (unmerged) -> ${nm} rows (merged)"
if [ "${nu}" -eq "${nm}" ]; then
    echo "  (no overlap collapsed; both files have the same intervals)"
fi
echo "  ${UNMERGED}"
echo "  ${MERGED}"
