#!/usr/bin/env python3
"""Step 3 (per gene): join HOMER annotation onto annotated.tsv and add distance to the
target gene's TSS. Writes OUTDIR/final_regions.tsv.

--target-tss is a BED of one TSS per gene (e.g. MANE Select). Column 4 is matched against the
region's gene_id (Ensembl ID, version suffix ignored) or, failing that, against a gene symbol.
Signed distance is in gene orientation: positive = downstream of the TSS.
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

def add_target_tss_distance(df, tss_path, gene):
    t = pd.read_csv(tss_path, sep="\t", header=None, comment="#", dtype={0: str})
    df["dist_to_target_tss"] = np.nan
    df["abs_dist_to_target_tss"] = np.nan
    if t.empty:
        log(f"WARNING: {gene} not found in column 4 of {tss_path} (needs the Ensembl gene_id); "
            "distance columns left empty")
        return df
    mids = (df["start0"] + df["end0"]) // 2
    best = []
    t = t[t[3] == gene.split(".")[0]]
    for chrom, mid in zip(df["chrom"], mids):
        cand = t[t[0] == chrom]
        if cand.empty:
            best.append(np.nan)
            continue
        # TSS position is the 0-based start of the 1-bp BED record
        signed = np.where(cand[5].to_numpy() == "-", cand[1].to_numpy() - mid, mid - cand[1].to_numpy())
        best.append(signed[np.argmin(np.abs(signed))])
    df["dist_to_target_tss"] = best
    df["abs_dist_to_target_tss"] = df["dist_to_target_tss"].abs()
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="gene_id (Ensembl)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--target-tss", required=True)
    args = p.parse_args()

    out = Path(args.outdir)
    annotations = pd.read_csv(out / "annotated.tsv", sep="\t", dtype={"chrom": str, "gene_id": str})
    homer = read_homer(out / "homer_annot.tsv")

    merged = annotations.merge(homer, on="region_id", how="left", validate="one_to_one")
    homer_cols = [c for c in merged.columns if c.startswith("homer_")]
    n_matched = merged[homer_cols[0]].notna().sum() if homer_cols else 0
    if n_matched < len(annotations):
        log(f"WARNING: only {n_matched}/{len(annotations)} regions matched in HOMER output")

    merged = add_target_tss_distance(merged, args.target_tss, args.gene)
    merged = merged.sort_values(["chrom", "start0"]).reset_index(drop=True)
    merged.to_csv(out / "final_regions.tsv", sep="\t", index=False)
    log(f"{args.gene}: wrote final_regions.tsv ({len(merged)} regions)")


if __name__ == "__main__":
    main()