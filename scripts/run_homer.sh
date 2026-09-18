#!/usr/bin/env bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"

: "${OUT_ROOT:?}"
: "${GENOME:?set GENOME in config.sh (installed HOMER genome e.g. hg38, or /path/to/genome.fa)}"
: "${PREPARSED_DIR:?set PREPARSED_DIR in config.sh}"
THREADS="${THREADS:-8}"
MOTIF_SIZE="${MOTIF_SIZE:-given}"
DENOVO="${DENOVO:-0}" # if this was 1 it would also run de novo motif 

command -v findMotifsGenome.pl >/dev/null || { echo "ERROR: HOMER not on PATH" >&2; exit 1; }
 
POOLED="${OUT_ROOT}/pooled"
BG="${POOLED}/bg.merged.bed"
mkdir -p "${PREPARSED_DIR}" "${OUT_ROOT}/motifs" "${OUT_ROOT}/anno" "${OUT_ROOT}/summary"

[ -s "${BG}" ] || { echo "ERROR: ${BG} missing/empty; run pool_regions.sh first" >&2; exit 1; }
mapfile -t FG < <(find "${POOLED}" -maxdepth 1 -name '*.fg.bed' | sort)
[ "${#FG[@]}" -gt 0 ] || { echo "ERROR: no *.fg.bed under ${POOLED}" >&2; exit 1; }

nomotif=(); [ "${DENOVO}" = "1" ] || nomotif=(-nomotif)

for fg in "${FG[@]}"; do
    cell="$(basename "${fg}" .fg.bed)"
    if [ ! -s "${fg}" ]; then echo "  skip ${cell}: empty foreground"; continue; fi
 
    echo "[2] motifs  : ${cell}"
    findMotifsGenome.pl "${fg}" "${GENOME}" "${OUT_ROOT}/motifs/${cell}" \
        -bg "${BG}" -size "${MOTIF_SIZE}" -mask -p "${THREADS}" \
        -preparsedDir "${PREPARSED_DIR}" "${nomotif[@]}"
 
    echo "[3] annotate: ${cell}"
    adir="${OUT_ROOT}/anno/${cell}"; mkdir -p "${adir}/go"
    annotatePeaks.pl "${fg}" "${GENOME}" \
        -annStats "${adir}/annStats.txt" -go "${adir}/go" \
        > "${adir}/annotated.txt" 2> "${adir}/annotatePeaks.log"
done