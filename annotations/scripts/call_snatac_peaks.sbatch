#!/usr/bin/env bash
#SBATCH --job-name=snatac_peaks
#SBATCH --partition=day
#SBATCH --cpus-per-task=2               # MACS3 callpeak is effectively single-threaded
#SBATCH --mem=64G                       # fragment loading is memory-hungry; lower it after a first run (sacct -j <id> -o MaxRSS)
#SBATCH --time=08:00:00
#SBATCH --output=logs/snatac_peaks_%j.log

# Call pseudobulk peaks from an ENCODE snATAC fragments file.
#
# Submit (mkdir logs first, or the job fails immediately). Arguments go AFTER the script name:
#   mkdir -p logs
#   sbatch call_snatac_peaks.sbatch ENCSR986YGY \
#     annot/encode_astro/raw/ENCSR986YGY.fragments.tsv.gz annot/encode_astro 1000 /path/to/hg38-blacklist.v2.bed
#
# Args: ACC FRAG OUT [MIN_FRAGS=1000] [BLACKLIST]. BLACKLIST is optional. MIN_FRAGS is the
# minimum fragments per barcode; tune it against your fragments-per-barcode distribution.
# Use absolute paths if you submit from a different directory than your data.

set -euo pipefail

# Positional args win; environment variables (--export) are the fallback.
ACC="${1:-${ACC:-}}"
FRAG="${2:-${FRAG:-}}"
OUT="${3:-${OUT:-}}"
MIN_FRAGS="${4:-${MIN_FRAGS:-1000}}"
BLACKLIST="${5:-${BLACKLIST:-}}"
if [ -z "${ACC}" ] || [ -z "${FRAG}" ] || [ -z "${OUT}" ]; then
  echo "usage: sbatch call_snatac_peaks.sbatch ACC FRAG OUT [MIN_FRAGS] [BLACKLIST]" >&2
  echo "got: ACC='${ACC}' FRAG='${FRAG}' OUT='${OUT}'" >&2
  exit 2
fi

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hari_env
command -v macs3 >/dev/null || { echo "ERROR: macs3 not in hari_env (conda install -c bioconda macs3)" >&2; exit 1; }
command -v bedtools >/dev/null || { echo "ERROR: bedtools not on PATH" >&2; exit 1; }
[ -s "${FRAG}" ] || { echo "ERROR: ${FRAG} missing/empty" >&2; exit 1; }

WORK="${OUT}/${ACC}_work"
mkdir -p "${WORK}"
export LC_ALL=C

echo "[1/4] count fragments per barcode"
zcat "${FRAG}" | grep -v '^#' | awk '{n[$4]++} END{for(b in n) print b"\t"n[b]}' > "${WORK}/barcode_counts.tsv"
awk -v m="${MIN_FRAGS}" '$2>=m{print $1}' "${WORK}/barcode_counts.tsv" > "${WORK}/keep_barcodes.txt"
echo "  barcodes total: $(wc -l < "${WORK}/barcode_counts.tsv")  kept (>=${MIN_FRAGS}): $(wc -l < "${WORK}/keep_barcodes.txt")"
[ -s "${WORK}/keep_barcodes.txt" ] || { echo "ERROR: no barcodes pass MIN_FRAGS=${MIN_FRAGS}" >&2; exit 1; }

echo "[2/4] filter fragments to kept barcodes (BED3)"
zcat "${FRAG}" | grep -v '^#' \
  | awk 'BEGIN{OFS="\t"} NR==FNR{k[$1];next} ($4 in k){print $1,$2,$3}' "${WORK}/keep_barcodes.txt" - \
  > "${WORK}/fragments.kept.bed"
echo "  fragments kept: $(wc -l < "${WORK}/fragments.kept.bed")"

echo "[3/4] MACS3 pseudobulk peak calling"
macs3 callpeak -t "${WORK}/fragments.kept.bed" -f BEDPE -g hs -n "${ACC}" \
  --nomodel --keep-dup all -q 0.01 --outdir "${WORK}/macs3"

echo "[4/4] post-process peaks -> ${OUT}/ENCODE_snATAC_${ACC}.hg38.bed"
PEAKS="${WORK}/macs3/${ACC}_peaks.narrowPeak"
FINAL="${OUT}/ENCODE_snATAC_${ACC}.hg38.bed"
cut -f1-3 "${PEAKS}" \
  | awk 'BEGIN{OFS="\t"} $1 ~ /^chr([0-9]+|X|Y)$/' \
  | sort -k1,1 -k2,2n > "${WORK}/peaks.std.bed"
if [ -n "${BLACKLIST}" ]; then
  bedtools intersect -v -a "${WORK}/peaks.std.bed" -b "${BLACKLIST}" | bedtools merge -i - > "${FINAL}"
else
  bedtools merge -i "${WORK}/peaks.std.bed" > "${FINAL}"
fi
echo "  peaks (raw narrowPeak): $(wc -l < "${PEAKS}")   final merged: $(wc -l < "${FINAL}")"

# fragments.kept.bed is large and reproducible; delete it once you're happy with the peaks
echo "done: ${FINAL}"
