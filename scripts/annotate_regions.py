#!/usr/bin/env python3
"""Step 1 (per gene): annotate perturbation regions with BED overlaps and bigwig signal, classifies into 
TSS-relative distance bins

Inputs
  --regions   regions.tsv (header row; may hold many genes / cell types). Required columns:
              gene_id, chrom, start0, end0. Optional: cell_type, annotation, ... (all kept).
  --gene      gene_id to process (Ensembl ID; version suffix like .12 is ignored)
  --manifest  tracks.tsv with columns: name, path, type (bigwig|bed), role (signal|call|mask),
              assay, celltype

--tss-bed   one-base TSS BED6, one TSS per gene (TSS_BED in config.sh); column 4 is matched to each
              region's gene_id (version suffixes ignored on both sides)
Outputs (in --outdir)
  annotated.tsv     one row per region: ALL original regions.tsv columns (including the HARI scan's own
                    `annotation`), then per-track signal / overlap counts, then
                      dist_to_target_tss, abs_dist_to_target_tss
                          region midpoint to the gene's TSS, bp; signed in gene orientation
                          (positive = downstream of the TSS)
                      tss_class, from the region's edges [a, b) and the TSS position tss0 (0-based start
                      of the 1-bp BED record):
                          TSS_proximal  a <= tss0 < b                            region covers the TSS
                          TSS_boundary  min(|a - tss0|, |b - tss0|) <= half      nearest edge within
                                                                                 --promoter-half-bp
                          distal        otherwise
                      The distance columns and tss_class are empty if the gene has no TSS on the
                      region's chromosome.
  all_regions.bed   BED6 of every region (region_id in column 4) -> HOMER annotatePeaks
  class_<tss_class>.bed
                    BED6 of the regions in each tss_class (class_TSS_proximal.bed, class_TSS_boundary.bed,
                    class_distal.bed; written only if non-empty) -> HOMER motif / GO foregrounds

region_id = <gene_id>|<chrom>|<start0>|<end0>
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
    """
    Number of intervals in bedtools overlapping each region
    """
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

def tss_class(a, b, tss0, promoter_half_bp):
    """
    Class of the region relative to a the gene TSS
    """
    if a <= tss0 < b:                                             
        return "TSS_proximal"
    if min(abs(a - tss0), abs(b - tss0)) <= promoter_half_bp:
        return "TSS_boundary"                                
    return "distal"

def add_tss_columns(regions, tss_path, promoter_half_bp):
    """
    Add dist_to_target_tss, abs_dist_to_target_tss and tss_class columns using each region's own gene's TSS
    """
    t = pd.read_csv(tss_path, sep="\t", header=None, comment="#", dtype={0: str, 3: str})
    by_gene = {g: d for g, d in t.groupby(t[3].str.split(".").str[0])}
    dist = np.full(len(regions), np.nan)
    cls = np.full(len(regions), np.nan, dtype=object)
    genes = regions["gene_id"].str.split(".").str[0]
    mids = ((regions["start0"] + regions["end0"]) // 2).to_numpy()
    absent = set()
    for i, (gene, chrom, a, b, mid) in enumerate(zip(genes, regions["chrom"], regions["start0"],
                                                      regions["end0"], mids)):
        g = by_gene.get(gene)
        if g is None:
            absent.add(gene)
            continue
        cand = g[g[0] == chrom]
        if cand.empty:
            continue
        tss = cand[1].to_numpy()          
        signed = np.where(cand[5].to_numpy() == "-", tss - mid, mid - tss)
        dist[i] = signed[np.argmin(np.abs(signed))]
        # if a gene has several TSS records, classify against the one closest to the region's edges
        edge = np.where((a <= tss) & (tss < b), 0, np.minimum(np.abs(a - tss), np.abs(b - tss)))
        cls[i] = tss_class(int(a), int(b), int(tss[np.argmin(edge)]), promoter_half_bp)
    if absent:
        log(f"WARNING: {sorted(absent)[:5]} not found in column 4 of {tss_path} (needs the Ensembl gene_id); "
            "their distance and tss_class are empty")
    regions["dist_to_target_tss"] = dist
    regions["abs_dist_to_target_tss"] = np.abs(dist)
    regions["tss_class"] = cls

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
    p.add_argument("--tss-bed", required=True)
    p.add_argument("--promoter-half-bp", type=int, default=5000,
                   help="tss_class: TSS_boundary if the nearest region edge is within this many bp of the TSS "
                        "(keep equal to the HARI scan's --promoter-half-bp)")
    p.add_argument("--bw-stat", default="mean", choices=["mean", "max", "min"])
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    regions = read_regions(args.regions, args.gene)
    log(f"{args.gene}: {len(regions)} regions")

    tracks_tsv = pd.read_csv(args.tracks, sep=r"\s+", dtype=str).fillna("")
    for _, t in tracks_tsv.iterrows():
        if t["type"] == "bigwig":
            add_track_column(regions, t["name"], bigwig_signal(regions, t["path"], args.bw_stat))
        elif t["type"] == "bed":
            add_track_column(regions, t["name"], overlap_counts(regions, pybedtools.BedTool(t["path"])))

    add_tss_columns(regions, args.tss_bed, args.promoter_half_bp)

    regions.to_csv(out / "annotated.tsv", sep="\t", index=False)
    write_bed6(regions, out / "all_regions.bed")
    # one BED per TSS annotation class (TSS_proximal / TSS_boundary / distal)
    # (HOMER_FOREGROUNDS in config.sh)
    for old in out.glob("class_*.bed"):
        old.unlink()
    for cls, sub in regions.groupby("tss_class"):
        write_bed6(sub, out / f"class_{cls}.bed")
    log(f"{args.gene}: {len(regions)} regions; tss_class {regions['tss_class'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()