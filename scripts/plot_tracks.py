#!/usr/bin/env python3
"""Step 4 (per gene): build a pyGenomeTracks .ini from the manifest and render OUTDIR/{gene}.tracks.pdf.

Tracks, top to bottom: perturbation effect (mean_signed_effect from annotated.tsv, red = increase,
blue = decrease), optional external perturbation track, manifest bigwigs, manifest peak/cCRE BEDs,
all perturbation regions, distal enhancers, gene models.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]


def log(msg):
    print(f"[plot] {msg}", file=sys.stderr)


def window(all_regions_bed, flank):
    r = pd.read_csv(all_regions_bed, sep="\t", header=None, dtype={0: str})
    chroms = r[0].unique()
    if len(chroms) > 1:
        log(f"WARNING: regions span {len(chroms)} chromosomes; plotting {chroms[0]} only")
    r = r[r[0] == chroms[0]]
    return chroms[0], max(0, int(r[1].min()) - flank), int(r[2].max()) + flank


def write_effect_bedgraph(annotated_tsv, path):
    """Per-region signed effect as a bedgraph; returns the effect units label (or '')."""
    df = pd.read_csv(annotated_tsv, sep="\t", dtype={"chrom": str})
    if "mean_signed_effect" not in df.columns:
        return None
    bg = df[["chrom", "start0", "end0", "mean_signed_effect"]].sort_values(["chrom", "start0"])
    bg.to_csv(path, sep="\t", header=False, index=False)
    units = ""
    if "effect_units" in df.columns and df["effect_units"].notna().any():
        units = str(df["effect_units"].dropna().iloc[0])
    return units


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--gene-track", required=True, help="GENCODE basic GTF (sorted/indexed) or BED12")
    p.add_argument("--perturb-track", default="", help="optional extra bigwig/bedgraph of perturbation effect")
    p.add_argument("--celltype", default="astrocyte")
    p.add_argument("--flank", type=int, default=20000)
    a = p.parse_args()

    out = Path(a.outdir)
    all_bed, distal_bed = out / "all_regions.bed", out / "distal_enh.bed"
    chrom, start, end = window(all_bed, a.flank)

    man = pd.read_csv(a.manifest, sep=r"\s+", comment="#", dtype=str).fillna("")
    if "role" not in man.columns:
        man["role"] = ""
    if a.celltype and "celltype" in man.columns:
        man = man[man["celltype"].isin([a.celltype, "all", ""])]

    blocks, i = [], 0

    def color():
        nonlocal i
        c = COLORS[i % len(COLORS)]
        i += 1
        return c

    effect_bg = out / f"{a.gene}.effect.bedgraph"
    units = write_effect_bedgraph(out / "annotated.tsv", effect_bg)
    if units is not None:
        title = "mean signed effect" + (f" ({units})" if units else "")
        blocks.append(f"[effect]\nfile = {effect_bg}\ntitle = {title}\nheight = 3\n"
                      f"color = #b2182b\nnegative_color = #2166ac\n")
    if a.perturb_track:
        blocks.append(f"[perturbation]\nfile = {a.perturb_track}\ntitle = perturbation effect\n"
                      f"height = 3\ncolor = #000000\n")
    for _, t in man[man["type"] == "bigwig"].iterrows():
        blocks.append(f"[{t['name']}]\nfile = {t['path']}\ntitle = {t['name']}\nheight = 2.5\n"
                      f"color = {color()}\nmin_value = 0\n")
    for _, t in man[(man["type"] == "bed") & (man["role"] != "mask")].iterrows():
        blocks.append(f"[{t['name']}]\nfile = {t['path']}\ntitle = {t['name']}\n"
                      f"display = collapsed\nheight = 0.7\ncolor = {color()}\n")
    blocks.append(f"[perturbed regions]\nfile = {all_bed}\ntitle = perturbed regions\n"
                  f"display = collapsed\nheight = 0.7\ncolor = #999999\n")
    blocks.append(f"[distal enhancers]\nfile = {distal_bed}\ntitle = distal enhancers\n"
                  f"display = collapsed\nheight = 0.7\ncolor = #000000\n")
    genes = f"[genes]\nfile = {a.gene_track}\ntitle = GENCODE\nheight = 4\nfontsize = 8\n"
    if a.gene_track.endswith((".gtf", ".gtf.gz")):
        genes += "prefered_name = gene_name\nmerge_transcripts = true\n"
    blocks.append(genes)

    ini = out / f"{a.gene}.tracks.ini"
    ini.write_text("\n".join(blocks))

    pdf = out / f"{a.gene}.tracks.pdf"
    cmd = ["pyGenomeTracks", "--tracks", str(ini), "--region", f"{chrom}:{start}-{end}",
           "--outFileName", str(pdf), "--title", f"{a.gene} ({chrom}:{start:,}-{end:,})"]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()