#!/usr/bin/env python3
"""One-base BED6 TSS per gene from a GENCODE GTF.

Picks the MANE_Select transcript per gene; falls back to Ensembl_canonical
(e.g. lncRNAs without MANE). Output is 0-based half-open, chr-style names,
Ensembl gene IDs without version suffix -- the format prepare_targets.py expects.

Usage:
    python make_tss_bed.py gencode.v38.annotation.gtf.gz tss_one_base.bed
"""
import gzip
import re
import sys

import pandas as pd


def main(gtf_path, out_path):
    opener = gzip.open if gtf_path.endswith(".gz") else open
    rows = []
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "transcript" or c[0] == "chrM":
                continue
            a = c[8]
            gene_full = re.search(r'gene_id "([^"]+)"', a).group(1)
            if gene_full.endswith("_PAR_Y"):
                continue  # keep the chrX copy of pseudoautosomal genes
            tags = re.findall(r'tag "([^"]+)"', a)
            if "MANE_Select" in tags:
                rank = 0
            elif "Ensembl_canonical" in tags:
                rank = 1
            else:
                continue
            start1, end1, strand = int(c[3]), int(c[4]), c[6]
            tss0 = start1 - 1 if strand == "+" else end1 - 1  # 5' end, 0-based
            rows.append((c[0], tss0, tss0 + 1, gene_full.split(".")[0], rank, strand))

    df = pd.DataFrame(rows, columns=["chrom", "start0", "end0", "gene_id", "rank", "strand"])
    if df.empty:
        sys.exit("No MANE_Select or Ensembl_canonical transcripts found; check the GTF release")
    df = df.sort_values("rank", kind="stable").drop_duplicates("gene_id")
    assert not df.gene_id.duplicated().any()
    df["score"] = 0
    df = df.sort_values(["chrom", "start0"], kind="stable")
    df[["chrom", "start0", "end0", "gene_id", "score", "strand"]].to_csv(
        out_path, sep="\t", header=False, index=False)

    n_mane = int((df["rank"] == 0).sum())
    print(f"Wrote {len(df)} genes to {out_path}: {n_mane} MANE_Select, "
          f"{len(df) - n_mane} Ensembl_canonical fallback")
    if n_mane == len(df):
        print("Note: no Ensembl_canonical fallbacks -- this release may lack that tag")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])