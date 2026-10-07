#!/usr/bin/env bash

# inputs
# export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_stride250_full"
# path to all 1000 ish gene list output

export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_window5000_stride1000_full"
export TRACKS_MANIFEST="/home/dh2226/perturb_enrich/annotations/tracks.tsv"
export CHROM_SIZES="/home/dh2226/hg38.chrom.sizes"

# outputs
export OUT_ROOT="/home/dh2226/scratch_pi_cs3222/dh2226/20261001"

# genome
export GENOME="hg38"
export HOMER_HOME="/home/dh2226/homer"

# motif finding
export MOTIF_SIZE="given"
export DENOVO=0 
export PREPARSED_DIR="${OUT_ROOT}/preparsed"

# resources
export THREADS=8

export TSS_TARGET_BED="/home/dh2226/perturb_enrich/gene_list/tss_one_base.bed" 
export GENE_TRACK="/home/dh2226/gencode.v38.annotation.gtf.gz" 

# concordance / enrichment (scripts/calibrate_thresholds.py, enrich_gene.py, aggregate_enrich.py)
export ENRICH_QVALUE=0.05      # BH q cutoff used to call a bigwig bp "active" (p-value tracks)
export THRESHOLDS_TSV="/home/dh2226/perturb_enrich/annotations/thresholds_q${ENRICH_QVALUE/./}.tsv"
export ENRICH_NPERM=1000
export ENRICH_STRATA="direction annotation is_distal_enh"
export ENRICH_EXCLUDE_PERTURBED=0   # 1 = null drawn only from tested sequence outside perturbed regions
