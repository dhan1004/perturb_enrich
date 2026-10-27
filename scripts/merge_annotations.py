#!/usr/bin/env python3
"""Step 3 (per gene): join HOMER annotation onto annotated.tsv and add the target gene's TSS-relative
columns. Writes OUTDIR/final_regions.tsv.

--tss-bed is a BED6 of one TSS per gene (e.g. MANE Select; TSS_BED in config.sh). Column 4 is matched
against the region's gene_id (Ensembl ID, version suffix ignored on both sides).

Columns added:
  dist_to_target_tss, abs_dist_to_target_tss
      region midpoint to the TSS, bp; signed in gene orientation (positive = downstream of the TSS)
  tss_class   recomputed from the region's edges [a, b) and the TSS position tss0 (0-based start of the
              1-bp BED record), independent of the HARI scan's own `annotation` column:
                TSS_proximal  a <= tss0 < b                             region covers the TSS
                TSS_boundary  min(|a - tss0|, |b - tss0|) <= half       nearest edge within --promoter-half-bp
                distal        otherwise
              Empty if the gene has no TSS on the region's chromosome.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg):
    print(f"[merge] {msg}", file=sys.stderr)

def read_homer(path):
    h = pd.read_csv(path, sep="\t", dtype=str)
    h = h.rename(columns={h.columns[0]: "region_id"})
    h.columns = ["region_id"] + ["homer_" + c.strip().replace(" ", "_") for c in h.columns[1:]]
    drop = [c for c in ("homer_Chr", "homer_Start", "homer_End", "homer_Strand") if c in h.columns]
    return h.drop(columns=drop)

def load_target_tss(tss_path, gene):
    """TSS record(s) of `gene` from the TSS BED (column 4 = Ensembl gene_id, version ignored)."""
    t = pd.read_csv(tss_path, sep="\t", header=None, comment="#", dtype={0: str, 3: str})
    t = t[t[3].str.split(".").str[0] == gene.split(".")[0]]
    if t.empty:
        log(f"WARNING: {gene} not found in column 4 of {tss_path} (needs the Ensembl gene_id); "
            "distance and tss_class columns left empty")
    return t

def tss_class(a, b, tss0, promoter_half_bp):
    """Class of the region [a, b) relative to a TSS at 0-based position tss0."""
    if a <= tss0 < b:                                              # region covers the TSS
        return "TSS_proximal"
    if min(abs(a - tss0), abs(b - tss0)) <= promoter_half_bp:      # nearest edge within the promoter window
        return "TSS_boundary"                                      # near the promoter but not on the TSS
    return "distal"

def add_target_tss_columns(df, t, promoter_half_bp):
    dist = np.full(len(df), np.nan)
    cls = np.full(len(df), np.nan, dtype=object)
    mids = ((df["start0"] + df["end0"]) // 2).to_numpy()
    for i, (chrom, a, b, mid) in enumerate(zip(df["chrom"], df["start0"], df["end0"], mids)):
        cand = t[t[0] == chrom]
        if cand.empty:
            continue
        tss = cand[1].to_numpy()          # TSS position is the 0-based start of the 1-bp BED record
        signed = np.where(cand[5].to_numpy() == "-", tss - mid, mid - tss)
        dist[i] = signed[np.argmin(np.abs(signed))]
        # if the gene has several TSS records, classify against the one closest to the region's edges
        edge = np.where((a <= tss) & (tss < b), 0, np.minimum(np.abs(a - tss), np.abs(b - tss)))
        cls[i] = tss_class(int(a), int(b), int(tss[np.argmin(edge)]), promoter_half_bp)
    df["dist_to_target_tss"] = dist
    df["abs_dist_to_target_tss"] = np.abs(dist)
    df["tss_class"] = cls
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="gene_id (Ensembl)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--tss-bed", "--target-tss", dest="tss_bed", required=True,
                   help="one-base TSS BED6, one TSS per gene (--target-tss is the old name)")
    p.add_argument("--promoter-half-bp", type=int, default=5000,
                   help="tss_class: TSS_boundary if the nearest region edge is within this many bp of the TSS "
                        "(keep equal to the HARI scan's --promoter-half-bp)")
    args = p.parse_args()
    if args.promoter_half_bp < 0:
        sys.exit("[merge] --promoter-half-bp must be >= 0")

    out = Path(args.outdir)
    annotations = pd.read_csv(out / "annotated.tsv", sep="\t", dtype={"chrom": str, "gene_id": str})
    homer = read_homer(out / "homer_annot.tsv")

    merged = annotations.merge(homer, on="region_id", how="left", validate="one_to_one")
    homer_cols = [c for c in merged.columns if c.startswith("homer_")]
    n_matched = merged[homer_cols[0]].notna().sum() if homer_cols else 0
    if n_matched < len(annotations):
        log(f"WARNING: only {n_matched}/{len(annotations)} regions matched in HOMER output")

    merged = add_target_tss_columns(merged, load_target_tss(args.tss_bed, args.gene), args.promoter_half_bp)
    merged = merged.sort_values(["chrom", "start0"]).reset_index(drop=True)
    merged.to_csv(out / "final_regions.tsv", sep="\t", index=False)
    counts = merged["tss_class"].value_counts().to_dict()
    log(f"{args.gene}: wrote final_regions.tsv ({len(merged)} regions; tss_class {counts})")


if __name__ == "__main__":
    main()
