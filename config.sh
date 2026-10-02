#!/usr/bin/env bash

# inputs
# export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_stride250_full"
# path to all 1000 ish gene list output

export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_window5000_stride1000_full"
export TRACKS_MANIFEST="/home/dh2226/perturb_enrich/annotations/tracks.tsv"
export CHROM_SIZES="/home/dh2226/hg38.chrom.sizes"

# outputs
export OUT_ROOT="/home/dh2226/scratch_pi_cs3222/dh2226/homer_enrich_run"

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