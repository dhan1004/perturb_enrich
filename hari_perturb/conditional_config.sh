#!/usr/bin/env bash
# Settings for conditional_scan.py batch runs. Edit REF_SCAN_ROOT; the rest are filled in from
# nativeAG/config.sh and the HARI README and may need adjusting.

export PYTHON_BIN="${PYTHON_BIN:-/nfs/roberts/pi/pi_cs3222/fq37/conda_envs/AlphaGenome/bin/python}"
export FASTA="${FASTA:-/nfs/roberts/pi/pi_cs3222/fq37/AMP_PD/wgs-BR-DSNWGS/gatk/vcf/GRCh38.primary_assembly.genome.fa}"
export SUITE_ROOT="${SUITE_ROOT:-/nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/alphagenome_expression_suite}"

# Output root of the ORIGINAL (unmasked) HARI scan: holds <gene_id>/scan_metadata.json and
# <gene_id>/results/Astrocyte/regions.tsv. Left blank on purpose. Candidates seen (commented out)
# in perturb_enrich/config.sh:
#   /nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_stride250_full
#   /nfs/roberts/pi/pi_cs3222/fq37/ARCHIVE_Alpha_pre/scData_v/AlphaGenome_ALL_Multicell/RevisedAlphaGenomeModel/PerturbedRegions_1mb_window5000_stride1000_full
export REF_SCAN_ROOT="${REF_SCAN_ROOT:-}"

# Where conditional results go (one directory per gene).
export COND_OUT_ROOT="${COND_OUT_ROOT:-/nfs/roberts/pi/pi_cs3222/dh2226/HARI_conditional}"
