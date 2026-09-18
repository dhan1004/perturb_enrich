#!/usr/bin/env bash

# inputs
export PERTURB_ROOT_SCAN="/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_stride250_full"

# outputs
export OUT_ROOT="${SCRATCH:-$HOME/scratch}/homer_enrich_run"

# genome
export GENOME="hg38"
export HOMER_HOME="/home/dh2226/homer"

# motif finding
export MOTIF_SIZE="given"
export MOTIF_LEN="8,10,12"
export N_MOTIFS=25
export MASK_REPEATS=1
export DENOVO=0 
export PREPARSED_DIR="${OUT_ROOT}/preparsed"

# resources
export THREADS=8