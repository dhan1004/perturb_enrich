#!/usr/bin/env bash

# inputs
# export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_stride250_full"
# path to all 1000 ish gene list output

export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_window5000_stride1000_full"
# export PERTURB_ROOT_SCAN="/home/dh2226/perturb_enrich/nativeAG/native_astrocyte_pilot"
export TRACKS_MANIFEST="/home/dh2226/perturb_enrich/annotations/tracks.tsv"

# outputs
export OUT_ROOT="/nfs/roberts/pi/pi_cs3222/dh2226/epilepsy_annot"
# export OUT_ROOT="/nfs/roberts/pi/pi_cs3222/dh2226/AGnaive"

# genome
export GENOME="hg38"
export HOMER_HOME="/home/dh2226/homer"

# motif finding
export MOTIF_SIZE="given"
export DENOVO=0 
export PREPARSED_DIR="${OUT_ROOT}/preparsed"

# resources
export THREADS=8

# One-base TSS BED (BED6; col 4 = Ensembl gene_id; one MANE/canonical TSS per gene, see
# gene_list/make_tss_bed.py). The single TSS source for the pipeline: dist_to_target_tss and tss_class are
# computed from it in annotate_regions.py (step 1, before HOMER). Do not also list it in tracks.tsv.
export TSS_BED="/home/dh2226/perturb_enrich/gene_list/tss_one_base.bed"
# tss_class (annotated.tsv / final_regions.tsv): TSS_proximal = region covers the TSS; TSS_boundary = nearest
# region edge is within PROMOTER_HALF_BP of the TSS; distal = otherwise. Keep equal to the HARI scan's
# --promoter-half-bp. This is separate from the scan's own `annotation` column, which is left untouched.
export PROMOTER_HALF_BP=5000
export GENE_TRACK="/home/dh2226/gencode.v38.annotation.gtf.gz" 

# concordance / enrichment
export ENRICH_QVALUE=0.05      # BH q cutoff used to call a bigwig bp "active" (p-value tracks)
# export THRESHOLDS_TSV="/home/dh2226/perturb_enrich/annotations/thresholds_q${ENRICH_QVALUE/./}.tsv"
export THRESHOLDS_TSV="/home/dh2226/perturb_enrich/annotations/thresholds.tsv"
export ENRICH_NPERM=1000
export ENRICH_STRATA="direction annotation tss_class len_bin"   # annotation = HARI scan call; tss_class = recomputed
export ENRICH_EXCLUDE_PERTURBED=0   # 1 = null drawn only from tested sequence outside perturbed regions

# HOMER motif / GO foregrounds (space-separated), run per gene by run_homer.sh. Per-region annotation of ALL
# regions (homer_annot.tsv) always runs and can be filtered by tss_class afterwards; motif / GO enrichment is
# a test of a SET of regions, so it needs one foreground per class:
#   all -> every perturbed region -> motifs_all/ go_all/
#   TSS_proximal | TSS_boundary | distal -> that tss_class -> motifs_<class>/ go_<class>/
export HOMER_FOREGROUNDS="all distal TSS_proximal TSS_boundary"