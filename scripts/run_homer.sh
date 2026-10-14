#!/usr/bin/env bash
# Per-gene HOMER step.
#   usage: run_homer.sh <gene_id> <outdir>
# <outdir> is the gene's output directory from annotate_regions.py (all_regions.bed, distal_enh.bed).
#
# Writes into <outdir>:
#   homer_annot.tsv   annotatePeaks.pl on ALL regions (read by merge_annotations.py)
#   annStats.txt      annotatePeaks genomic-annotation summary for all regions
#   motifs[_<fg>]/    findMotifsGenome.pl on each foreground in HOMER_FOREGROUNDS vs. HOMER's genome background
#   go[_<fg>]/        annotatePeaks.pl -go on each foreground (default foreground: distal_enh.bed -> motifs/, go/)
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
[ -s "${ALL}" ] || { echo "ERROR: ${ALL} missing/empty; run annotate_regions.py first" >&2; exit 1; }
mkdir -p "${PREPARSED_DIR}"

nomotif=(); [ "${DENOVO}" = "1" ] || nomotif=(-nomotif)

echo "[homer] ${GENE}: annotate all regions"
annotatePeaks.pl "${ALL}" "${GENOME}" -annStats "${OUTDIR}/annStats.txt" \
    > "${OUTDIR}/homer_annot.tsv" 2> "${OUTDIR}/annotatePeaks.log"

# Foregrounds for motif / GO analysis (space-separated HOMER_FOREGROUNDS, default: distal_enh).
#   distal_enh        distal_enh.bed                    -> motifs/        go/            (as before)
#   all               every perturbed region            -> motifs_all/    go_all/
#   <HARI class>      class_<class>.bed, i.e. distal, TSS_proximal, TSS_boundary
#                                                       -> motifs_<class>/ go_<class>/
# Every region keeps its HARI class in homer_annot.tsv / annotated.tsv whichever foregrounds are run.
FOREGROUNDS="${HOMER_FOREGROUNDS:-distal_enh}"

for NAME in ${FOREGROUNDS}; do
    case "${NAME}" in
        distal_enh) FG="${OUTDIR}/distal_enh.bed";       SUF="";          GOLOG="annotatePeaks_distal.log" ;;
        all)        FG="${OUTDIR}/all_regions.bed";      SUF="_all";      GOLOG="annotatePeaks_go_all.log" ;;
        *)          FG="${OUTDIR}/class_${NAME}.bed";    SUF="_${NAME}";  GOLOG="annotatePeaks_go_${NAME}.log" ;;
    esac
    if [ ! -s "${FG}" ]; then
        echo "[homer] ${GENE}: no regions for foreground '${NAME}' (${FG##*/}), skipping"
        continue
    fi

    # no -bg: HOMER picks GC-matched random genomic sequence as background
    echo "[homer] ${GENE}: motifs on '${NAME}'"
    findMotifsGenome.pl "${FG}" "${GENOME}" "${OUTDIR}/motifs${SUF}" \
        -size "${MOTIF_SIZE}" -mask -p "${THREADS}" \
        -preparsedDir "${PREPARSED_DIR}" ${nomotif[@]+"${nomotif[@]}"}

    echo "[homer] ${GENE}: GO on '${NAME}'"
    mkdir -p "${OUTDIR}/go${SUF}"
    annotatePeaks.pl "${FG}" "${GENOME}" -go "${OUTDIR}/go${SUF}" \
        > /dev/null 2> "${OUTDIR}/${GOLOG}"
done

echo "[homer] ${GENE}: done"
