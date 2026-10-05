#!/usr/bin/env python3
"""Step 4 (per gene): build a pyGenomeTracks .ini from the manifest and render OUTDIR/{gene}.tracks.pdf.

Tracks, top to bottom: perturbation effect (mean_signed_effect from annotated.tsv, red = increase,
blue = decrease), optional external perturbation track, manifest bigwigs, manifest peak/cCRE BEDs,
all perturbation regions, distal enhancers, TSS of every gene in the window, TSS of the target gene,
gene models. Only the gene models carry per-feature labels. Tracks are separated by spacers; heights
and spacer size are the H_* / SPACER constants below.

The target gene is shown as "SYMBOL (ENSG...)": the symbol is looked up in --gene-track when that is a
GENCODE GTF (override with --gene-symbol).
"""
import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path
import math

import pandas as pd

# GENCODE transcript tags used to pick the canonical transcript, in order of preference.
CANONICAL_TAGS = ("MANE_Select", "Ensembl_canonical")

COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]

# pyGenomeTracks height units per track type, and the blank gap inserted between tracks.
H_EFFECT = 4.5
H_BIGWIG = 3.5
H_BED = 1.2
H_GENES = 6.0
SPACER = 0.6


def log(msg):
    print(f"[plot] {msg}", file=sys.stderr)


def window(all_regions_bed, flank):
    r = pd.read_csv(all_regions_bed, sep="\t", header=None, dtype={0: str})
    chroms = r[0].unique()
    if len(chroms) > 1:
        log(f"WARNING: regions span {len(chroms)} chromosomes; plotting {chroms[0]} only")
    r = r[r[0] == chroms[0]]
    return chroms[0], max(0, int(r[1].min()) - flank), int(r[2].max()) + flank

def nice_step(span, target_ticks=8):
    """Round tick spacing (1/2/5 x 10^k bp) giving roughly target_ticks across the window."""
    raw = span / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return int(m * mag)


def write_ruler(chrom, start, end, step, ruler_bed, vlines_bed):
    """Ticks at every multiple of `step`, labelled with the full coordinate (e.g. 12,345,000)."""
    first = -(-start // step) * step  # ceil to a multiple of step
    ticks = list(range(first, end + 1, step))
    pad = max(1, (end - start) // 1500)  # widen the 1-bp tick so it is visible
    with open(ruler_bed, "w") as f:
        for t in ticks:
            f.write(f"{chrom}\t{max(0, t - pad)}\t{t + pad}\t{t:,}\t0\t.\n")
    with open(vlines_bed, "w") as f:
        for t in ticks:
            f.write(f"{chrom}\t{t}\t{t + 1}\n")
    return len(ticks)


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


def gene_symbol(gene_track, gene):
    """gene_name for `gene` (Ensembl ID, version ignored; a symbol also matches) from a GENCODE GTF.

    Returns '' if the track is not a GTF or the gene is not in it. Stops at the first hit.
    """
    gtf = str(gene_track)
    if not gtf.endswith((".gtf", ".gtf.gz")):
        return ""
    gid = gene.split(".")[0]
    opener = gzip.open if gtf.endswith(".gz") else open
    with opener(gtf, "rt") as f:
        for line in f:
            if line[0] == "#" or gid not in line:
                continue
            c = line.split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            m = re.search(r'gene_id "([^"]+)"', c[8])
            n = re.search(r'gene_name "([^"]+)"', c[8])
            if m and n and (m.group(1).split(".")[0] == gid or n.group(1) == gene):
                return n.group(1)
    return ""


def write_canonical_gtf(gtf, chrom, start, end, out_path):
    """Write a small GTF of the window with one canonical transcript per gene.

    Keeps every `gene` line plus, for each gene, only the transcript(s) tagged MANE_Select; genes with
    no MANE_Select transcript fall back to Ensembl_canonical, and genes with neither keep all their
    transcripts (counted in the return value so the caller can warn). Streams the whole GTF once.

    Returns (n_genes, n_fallback_all_transcripts).
    """
    opener = gzip.open if str(gtf).endswith(".gz") else open
    lines = []  # (gene_id, transcript_id or None, raw line) for features overlapping the window
    tags = {}   # gene_id -> {transcript_id: set of tags}
    with opener(gtf, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if len(c) < 9 or c[0] != chrom or int(c[4]) < start or int(c[3]) > end:
                continue
            g = re.search(r'gene_id "([^"]+)"', c[8])
            if not g:
                continue
            t = re.search(r'transcript_id "([^"]+)"', c[8])
            tid = t.group(1) if t and c[2] != "gene" else None
            if tid:
                tags.setdefault(g.group(1), {}).setdefault(tid, set()).update(
                    re.findall(r'tag "([^"]+)"', c[8]))
            lines.append((g.group(1), tid, line))

    keep, n_fallback = {}, 0
    for gid, txs in tags.items():
        for tag in CANONICAL_TAGS:
            sel = {t for t, s in txs.items() if tag in s}
            if sel:
                break
        else:
            sel = set(txs)
            n_fallback += 1
        keep[gid] = sel

    with open(out_path, "w") as o:
        for gid, tid, line in lines:
            if tid is None or tid in keep.get(gid, ()):
                o.write(line)
    return len({g for g, _, _ in lines}), n_fallback


def write_tss_beds(tss_path, gene, chrom, start, end, all_bed, target_bed):
    """Write the in-window TSS (all genes, and the target gene) as BED6 for plotting.

    Each 1-bp TSS is widened to ~1/750 of the window per side so it is visible as a tick at
    plot scale. Returns (n_tss_in_window, target_found).
    """
    t = pd.read_csv(tss_path, sep="\t", header=None, comment="#", dtype={0: str}, usecols=[0, 1, 2, 3])
    t = t[(t[0] == chrom) & (t[2] > start) & (t[1] < end)].copy()
    pad = max(1, (end - start) // 1500)
    t["s"] = (t[1] - pad).clip(lower=0)
    t["e"] = t[2] + pad
    t["score"] = 0
    t["strand"] = "."
    cols = [0, "s", "e", 3, "score", "strand"]
    t[cols].to_csv(all_bed, sep="\t", header=False, index=False)
    tgt = t[t[3].astype(str).str.split(".").str[0] == gene.split(".")[0]]
    tgt[cols].to_csv(target_bed, sep="\t", header=False, index=False)
    return len(t), len(tgt) > 0


def bed_block(name, path, color, title=None):
    """A collapsed BED track with no per-feature labels."""
    return (f"[{name}]\nfile = {path}\ntitle = {title or name}\n"
            f"display = collapsed\nlabels = false\nheight = {H_BED}\ncolor = {color}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True)
    p.add_argument("--gene-symbol", default="", help="gene symbol for the title (default: look up in --gene-track)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--gene-track", required=True, help="GENCODE basic GTF (sorted/indexed) or BED12")
    p.add_argument("--perturb-track", default="", help="optional extra bigwig/bedgraph of perturbation effect")
    p.add_argument("--all-transcripts", action="store_true",
                   help="plot every transcript from a GTF --gene-track (default: canonical transcript only)")
    p.add_argument("--celltype", default="astrocyte")
    p.add_argument("--tick-step", type=int, default=0, help="ruler tick spacing in bp (default: auto, ~8 ticks)")
    p.add_argument("--flank", type=int, default=20000)
    a = p.parse_args()

    out = Path(a.outdir)
    all_bed, distal_bed = out / "all_regions.bed", out / "distal_enh.bed"
    chrom, start, end = window(all_bed, a.flank)

    symbol = a.gene_symbol or gene_symbol(a.gene_track, a.gene)
    if not symbol:
        log(f"WARNING: no gene symbol found for {a.gene}; titling with the ID only "
            f"(pass --gene-symbol, or a GTF as --gene-track)")
    label = f"{symbol} ({a.gene})" if symbol and symbol != a.gene else a.gene

    # Gene models: for a GTF, subset the window to the canonical transcript of each gene.
    gene_file = a.gene_track
    if a.gene_track.endswith((".gtf", ".gtf.gz")) and not a.all_transcripts:
        gene_file = str(out / f"{a.gene}.canonical.gtf")
        n_genes, n_fallback = write_canonical_gtf(a.gene_track, chrom, start, end, gene_file)
        log(f"canonical-only gene track: {n_genes} genes in window -> {gene_file}")
        if n_fallback:
            log(f"WARNING: {n_fallback} gene(s) had no MANE_Select/Ensembl_canonical transcript; "
                f"all their transcripts are plotted")
        if not n_genes:
            log(f"WARNING: no genes found in {chrom}:{start}-{end}; check chromosome naming in {a.gene_track}")

    man_all = pd.read_csv(a.manifest, sep=r"\s+", comment="#", dtype=str).fillna("")
    if "role" not in man_all.columns:
        man_all["role"] = ""
    man = man_all
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

    step = a.tick_step or nice_step(end - start)
    ruler_bed, vlines_bed = out / f"{a.gene}.ruler.bed", out / f"{a.gene}.vlines.bed"
    write_ruler(chrom, start, end, step, ruler_bed, vlines_bed)
    blocks.append(f"[ruler]\nfile = {ruler_bed}\ntitle = {chrom} (every {step:,} bp)\n"
                f"display = collapsed\nlabels = true\nfontsize = 8\nheight = 0.5\ncolor = #000000\n")
    
    if units is not None:
        title = "mean signed effect" + (f" ({units})" if units else "")
        blocks.append(f"[effect]\nfile = {effect_bg}\ntitle = {title}\nheight = {H_EFFECT}\n"
                      f"color = #b2182b\nnegative_color = #2166ac\n")
    if a.perturb_track:
        blocks.append(f"[perturbation]\nfile = {a.perturb_track}\ntitle = perturbation effect\n"
                      f"height = {H_EFFECT}\ncolor = #000000\n")
    for _, t in man[man["type"] == "bigwig"].iterrows():
        blocks.append(f"[{t['name']}]\nfile = {t['path']}\ntitle = {t['name']}\nheight = {H_BIGWIG}\n"
                      f"color = {color()}\nmin_value = 0\n")
    for _, t in man[(man["type"] == "bed") & (man["role"] != "mask")].iterrows():
        blocks.append(bed_block(t["name"], t["path"], color()))
    # blocks.append(bed_block("perturbed regions", all_bed, "#999999"))
    blocks.append(bed_block("distal enhancers", distal_bed, "#000000"))

    # TSS: the manifest's TSS BED is role=mask (used to exclude promoters in annotate_regions.py), so
    # it is skipped by the generic BED loop above. Plot it explicitly. Not cell-type specific, hence man_all.
    # tss = man_all[(man_all["type"] == "bed") & (man_all["assay"] == "tss")]
    # if tss.empty:
    #     log("no assay=tss BED in the manifest; skipping TSS tracks")
    # else:
    #     tss_all, tss_target = out / f"{a.gene}.tss.bed", out / f"{a.gene}.target_tss.bed"
    #     n, found = write_tss_beds(tss.iloc[0]["path"], a.gene, chrom, start, end, tss_all, tss_target)
    #     if n:
    #         blocks.append(bed_block("TSS", tss_all, "#555555", "TSS (all genes)"))
    #     else:
    #         log(f"no TSS in {chrom}:{start}-{end}; skipping TSS track")
    #     if found:
    #         blocks.append(bed_block("target TSS", tss_target, "#d62728", f"{symbol or a.gene} TSS"))
    #     else:
    #         log(f"WARNING: {a.gene} not found in {tss.iloc[0]['path']} (column 4 must be the Ensembl gene_id)")

    genes = f"[genes]\nfile = {gene_file}\ntitle = GENCODE\nheight = {H_GENES}\nfontsize = 8\n"
    if str(gene_file).endswith((".gtf", ".gtf.gz")):
        genes += "prefered_name = gene_name\nmerge_transcripts = true\n"
    blocks.append(genes)

    parts = []
    for k, b in enumerate(blocks):
        if k:
            parts.append(f"[spacer]\nheight = {SPACER}\n")  # pyGenomeTracks needs the name to be exactly [spacer]; repeats are allowed
        parts.append(b)
    
    parts.append(f"[vlines]\nfile = {vlines_bed}\ntype = vlines\nline_width = 0.4\n"
             f"color = #999999\nalpha = 0.5\nline_style = dotted\nzorder = -1\n")

    ini = out / f"{a.gene}.tracks.ini"
    ini.write_text("\n".join(parts))

    pdf = out / f"{a.gene}.tracks.pdf"
    cmd = ["pyGenomeTracks", "--tracks", str(ini), "--region", f"{chrom}:{start}-{end}",
           "--outFileName", str(pdf), "--title", f"{label}  {chrom}:{start:,}-{end:,}"]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()