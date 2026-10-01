#!/usr/bin/env bash
# Per-gene HOMER step.
#   usage: run_homer.sh <gene_id> <outdir>
# <outdir> is the gene's output directory from annotate_regions.py (all_regions.bed, distal_enh.bed).
#
# Writes into <outdir>:
#   homer_annot.tsv   annotatePeaks.pl on ALL regions (read by merge_annotations.py)
#   annStats.txt      annotatePeaks genomic-annotation summary for all regions
#   motifs/           findMotifsGenome.pl on distal_enh.bed vs. HOMER's genome background
#   go/               annotatePeaks.pl -go on distal_enh.bed
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../config.sh"
export PATH="${HOMER_HOME}/bin:${PATH}"

GENE="${1:?usage: run_homer.sh <gene_id> <outdir>}"
OUTDIR="${2:?usage: run_homer.sh <gene_id> <outdir>}"
: "${GENOME:?set GENOME in config.sh (installed HOMER genome e.g. hg38, or /path/to/genome.fa)}"
: "${PREPARSED_DIR:?set PREPARSED_DIR in config.sh}"
THREADS="${THREADS:-8}"
MOTIF_SIZE="${MOTIF_SIZE:-given}"
DENOVO="${DENOVO:-0}" # if 1, also run de novo motif discovery

command -v findMotifsGenome.pl >/dev/null || { echo "ERROR: HOMER not on PATH" >&2; exit 1; }

ALL="${OUTDIR}/all_regions.bed"
FG="${OUTDIR}/distal_enh.bed"
[ -s "${ALL}" ] || { echo "ERROR: ${ALL} missing/empty; run annotate_regions.py first" >&2; exit 1; }
mkdir -p "${PREPARSED_DIR}"

nomotif=(); [ "${DENOVO}" = "1" ] || nomotif=(-nomotif)

echo "[homer] ${GENE}: annotate all regions"
annotatePeaks.pl "${ALL}" "${GENOME}" -annStats "${OUTDIR}/annStats.txt" \
    > "${OUTDIR}/homer_annot.tsv" 2> "${OUTDIR}/annotatePeaks.log"

if [ ! -s "${FG}" ]; then
    echo "[homer] ${GENE}: no distal enhancers, skipping motifs and GO"
    exit 0
fi

# no -bg: HOMER picks GC-matched random genomic sequence as background
echo "[homer] ${GENE}: motifs on distal enhancers"
findMotifsGenome.pl "${FG}" "${GENOME}" "${OUTDIR}/motifs" \
    -size "${MOTIF_SIZE}" -mask -p "${THREADS}" \
    -preparsedDir "${PREPARSED_DIR}" ${nomotif[@]+"${nomotif[@]}"}

echo "[homer] ${GENE}: GO on distal enhancers"
mkdir -p "${OUTDIR}/go"
annotatePeaks.pl "${FG}" "${GENOME}" -go "${OUTDIR}/go" \
    > /dev/null 2> "${OUTDIR}/annotatePeaks_distal.log"

echo "[homer] ${GENE}: done"