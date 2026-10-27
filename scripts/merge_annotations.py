#!/usr/bin/env python3
"""Step 3 (per gene): join the HOMER per-region annotation (homer_annot.tsv) onto annotated.tsv.
Writes OUTDIR/final_regions.tsv.

annotated.tsv (from annotate_regions.py) already carries dist_to_target_tss, abs_dist_to_target_tss and
tss_class (TSS_proximal / TSS_boundary / distal), so they are in final_regions.tsv unchanged.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def log(msg):
    print(f"[merge] {msg}", file=sys.stderr)

def read_homer(path):
    h = pd.read_csv(path, sep="\t", dtype=str)
    h = h.rename(columns={h.columns[0]: "region_id"})
    h.columns = ["region_id"] + ["homer_" + c.strip().replace(" ", "_") for c in h.columns[1:]]
    drop = [c for c in ("homer_Chr", "homer_Start", "homer_End", "homer_Strand") if c in h.columns]
    return h.drop(columns=drop)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="gene_id (Ensembl)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    out = Path(args.outdir)
    annotations = pd.read_csv(out / "annotated.tsv", sep="\t", dtype={"chrom": str, "gene_id": str})
    if "tss_class" not in annotations.columns:
        sys.exit(f"[merge] {out / 'annotated.tsv'} has no tss_class column: rerun annotate_regions.py for {args.gene}")
    homer = read_homer(out / "homer_annot.tsv")

    merged = annotations.merge(homer, on="region_id", how="left", validate="one_to_one")
    homer_cols = [c for c in merged.columns if c.startswith("homer_")]
    n_matched = merged[homer_cols[0]].notna().sum() if homer_cols else 0
    if n_matched < len(annotations):
        log(f"WARNING: only {n_matched}/{len(annotations)} regions matched in HOMER output")

    merged = merged.sort_values(["chrom", "start0"]).reset_index(drop=True)
    merged.to_csv(out / "final_regions.tsv", sep="\t", index=False)
    log(f"{args.gene}: wrote final_regions.tsv ({len(merged)} regions)")


if __name__ == "__main__":
    main()
