#!/usr/bin/env python3
"""Step 1 (per gene): annotate perturbation regions with BED overlaps and bigwig signal.

Inputs
  --regions   regions.tsv (header row; may hold many genes / cell types). Required columns:
              gene_id, chrom, start0, end0. Optional: cell_type, annotation, ... (all kept).
  --gene      gene_id to process (Ensembl ID; version suffix like .12 is ignored)
  --tracks    tracks.tsv with columns: name, path, type (bigwig|bed), role (call|mask), assay, celltype
  --tss-bed   one-base TSS BED (the single TSS source; do not also list it in tracks.tsv)
Outputs (in --outdir)
  annotated.tsv     one row per region: ALL original regions.tsv columns, then per-track
                    signal / overlap counts, then in_promoter_mask / has_enh_call / is_distal_enh
                    (in_promoter_mask = overlaps --tss-bed padded by --tss-pad;
                    has_enh_call = overlaps a BED peak of an enhancer assay [--enh-assays] from an allowed
                    cell type [--enh-celltypes]; is_distal_enh = has_enh_call and not in_promoter_mask)
  all_regions.bed   BED6 of every region (region_id in column 4) -> HOMER annotatePeaks
  distal_enh.bed    BED6 of regions flagged as distal enhancers -> HOMER findMotifsGenome

region_id = <gene_id>|<cell_type>|<chrom>|<start0>|<end0>   (cell_type omitted if absent)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pybedtools



def log(msg):
    print(f"[annotate] {msg}", file=sys.stderr)

def read_regions(path, gene):
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str, "gene_id": str})
    if df.empty:
        sys.exit(f"[annotate] no rows for gene_id {gene} in {path}")
    df.insert(0, "region_id", df["gene_id"] + "|" + df["chrom"]
              + "|" + df["start0"].astype(str) + "|" + df["end0"].astype(str))
    return df.drop_duplicates("region_id").reset_index(drop=True)

def overlap_counts(region, bedtool):
    """Number of intervals in `bedtool` overlapping each region (keyed on region_id)."""
    a = pybedtools.BedTool.from_dataframe(region[["chrom", "start0", "end0", "region_id"]])
    counts = {}
    for interval in a.intersect(bedtool, c=True):
        counts[interval.fields[3]] = int(interval.fields[-1])
    return region["region_id"].map(counts).fillna(0).astype(int)

def bigwig_signal(reg, path, stat):
    bw = pyBigWig.open(str(path))
    chroms = bw.chroms()
    missing, vals = set(), []
    for c, s, e in zip(reg.chrom, reg.start0, reg.end0):
        if c not in chroms:
            missing.add(c)
            vals.append(np.nan)
            continue
        e2 = min(e, chroms[c])
        if e2 <= s:
            vals.append(np.nan)
            continue
        v = bw.stats(c, s, e2, type=stat)[0]
        vals.append(np.nan if v is None else v)
    bw.close()
    if missing:
        log(f"WARNING {path}: chromosomes not in bigwig (chr naming mismatch?): {sorted(missing)[:5]}")
    return vals

def write_bed6(df, path):
    out = df[["chrom", "start0", "end0", "gene_id", "mean_signed_effect", "gene_strand"]].copy()
    out.to_csv(path, sep="\t", header=False, index=False)

def add_track_column(region, name, values):
    if name in region.columns:
        sys.exit(f"[annotate] manifest track name '{name}' collides with a regions.tsv column; rename it")
    region[name] = values

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="gene_id (Ensembl)") # should be getting this from directory name
    p.add_argument("--regions", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--chrom-sizes", required=True, help="hg38 chrom.sizes (for padding the TSS mask)")
    p.add_argument("--tss-bed", default="",
                   help="TSS BED (e.g. TSS_BED in config.sh), padded by --tss-pad into the promoter mask; "
                        "empty = no promoter mask")
    p.add_argument("--tss-pad", type=int, default=5000)
    p.add_argument("--enh-assays", default="h3k27ac,atac",
                   help="comma-separated assays whose BED peaks define an enhancer (case-insensitive; "
                        "any manifest role, including 'mask')")
    p.add_argument("--enh-celltypes", default="",
                   help="comma-separated manifest celltypes allowed to contribute enhancer peaks "
                        "(case-insensitive; empty = any), e.g. 'astrocyte'")
    p.add_argument("--distal-mode", default="enhancer", choices=["enhancer", "hari"],
                   help="how is_distal_enh (and distal_enh.bed for HOMER) is defined. enhancer: enhancer peak "
                        "and outside the TSS mask. hari: the HARI scan's own `annotation` column == 'distal' "
                        "(no overlap with the target gene's TSS +/- promoter_half_bp; no enhancer evidence)")
    p.add_argument("--bw-stat", default="mean", choices=["mean", "max", "min"])
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    regions = read_regions(args.regions, args.gene)
    log(f"{args.gene}: {len(regions)} regions")

    tracks_tsv = pd.read_csv(args.tracks, sep=r"\s+", dtype=str).fillna("")

    def split_set(s):
        return {x.strip().lower() for x in s.split(",") if x.strip()}

    enh_assays = split_set(args.enh_assays)
    enh_celltypes = split_set(args.enh_celltypes)
    enh_cols = []

    for _, t in tracks_tsv.iterrows():
        assay, celltype = t["assay"].strip().lower(), t["celltype"].strip().lower()
        if t["type"] == "bigwig":
            add_track_column(regions, t["name"], bigwig_signal(regions, t["path"], args.bw_stat))
        elif t["type"] == "bed":
            if assay == "tss":
                log(f"WARNING manifest row '{t['name']}' (assay=tss) ignored: the TSS BED now comes from "
                    "--tss-bed (TSS_BED in config.sh); delete this row from tracks.tsv")
                continue
            # every BED (any role) becomes a peak-count column; enhancer-assay peaks from an
            # allowed cell type also count as an enhancer call
            add_track_column(regions, t["name"], overlap_counts(regions, pybedtools.BedTool(t["path"])))
            if assay in enh_assays and (not enh_celltypes or celltype in enh_celltypes):
                enh_cols.append(t["name"])

    if not enh_cols:
        log(f"WARNING {args.gene}: no BED rows match --enh-assays {sorted(enh_assays)} "
            f"/ --enh-celltypes {sorted(enh_celltypes) or 'any'}; no region can be a distal enhancer")

    # promoter mask: the TSS BED padded by --tss-pad
    if args.tss_bed:
        tss_mask = pybedtools.BedTool(args.tss_bed).slop(b=args.tss_pad, g=args.chrom_sizes)
        mask_total = overlap_counts(regions, tss_mask).to_numpy()
    else:
        log(f"WARNING {args.gene}: no --tss-bed given; promoter mask is empty")
        mask_total = np.zeros(len(regions), dtype=int)

    regions["in_promoter_mask"] = mask_total > 0
    regions["has_enh_call"] = (regions[enh_cols].sum(axis=1) > 0) if enh_cols else False
    if args.distal_mode == "hari":
        if "annotation" not in regions.columns:
            sys.exit("[annotate] --distal-mode hari needs an `annotation` column in regions.tsv")
        distal = regions["annotation"] == "distal"
    else:
        distal = (~regions["in_promoter_mask"]) & regions["has_enh_call"]
    regions["is_distal_enh"] = distal

    regions.to_csv(out / "annotated.tsv", sep="\t", index=False)
    write_bed6(regions, out / "all_regions.bed")
    write_bed6(regions[regions["is_distal_enh"]], out / "distal_enh.bed")
    # one BED per HARI annotation class (TSS_proximal / TSS_boundary / distal) -> optional HOMER foregrounds
    # (HOMER_FOREGROUNDS in config.sh); stale files from a previous run are removed first
    for old in out.glob("class_*.bed"):
        old.unlink()
    if "annotation" in regions.columns:
        for cls, sub in regions.groupby("annotation"):
            write_bed6(sub, out / f"class_{cls}.bed")
    log(f"{args.gene}: {int(regions.is_distal_enh.sum())} distal enhancer regions "
        f"({int(regions.has_enh_call.sum())} with an enhancer peak, "
        f"{int(regions.in_promoter_mask.sum())} in promoter mask, of {len(regions)})")

if __name__ == "__main__":
    main()