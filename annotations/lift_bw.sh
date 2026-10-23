#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH -o lift_%j.log
set -euo pipefail

IN=$1      # hg19 bigWig
OUT=$2     # output .bw path
CHAIN=/home/dh2226/hg19ToHg38.over.chain
SIZES=/home/dh2226/hg38.chrom.sizes    # main chroms only (chr1-22, X, Y, M)

SCR=/home/dh2226/scratch_pi_cs3222/dh2226/lift_$SLURM_JOB_ID
mkdir -p $SCR/chunks && export TMPDIR=$SCR

bigWigToBedGraph "$IN" $SCR/in.bg
CrossMap bed $CHAIN $SCR/in.bg $SCR/lifted.bg

# keep only chroms in the sizes file, split by chrom
awk -v d=$SCR/chunks 'NR==FNR{ok[$1]=1; next} ($1 in ok){print > d"/"$1".bg"}' $SIZES $SCR/lifted.bg

# sort each chrom, clip overlaps, append
: > $SCR/final.bg
for c in $(cut -f1 $SIZES | LC_ALL=C sort); do
  f=$SCR/chunks/$c.bg; [ -s "$f" ] || continue
  LC_ALL=C sort -S 2G -T $SCR -k2,2n -k3,3n "$f" | awk -v OFS='\t' \
    '{s=$2; if (s<pe) s=pe; if (s<$3) {print $1,s,$3,$4; pe=$3}}' >> $SCR/final.bg
done

bedGraphToBigWig $SCR/final.bg $SIZES "$OUT"
bigWigInfo "$OUT" && echo DONE
# rm -r $SCR   # only after you've checked the output