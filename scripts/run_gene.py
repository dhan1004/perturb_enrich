#!/usr/bin/env python3
"""Run the per-gene pipeline: annotate_regions.py -> run_homer.sh -> merge_annotations.py -> plot_tracks.py.

usage: python run_gene.py --gene ENSG00000002549 --regions regions.tsv [--config config.sh] [--outdir DIR]

Reads config.sh for TRACKS_MANIFEST, CHROM_SIZES, TSS_TARGET_BED, GENE_TRACK and OUT_ROOT
(optional: TSS_PAD, ENH_ASSAYS, CELLTYPE).
Annotation outputs go to $OUT_ROOT/genes/<gene>/ unless --outdir is given.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config(path):
    """Source a bash config file and return the resulting environment as a dict."""
    res = subprocess.run(["bash", "-c", f'set -a; source "{path}"; env -0'],
                         capture_output=True, check=True)
    return dict(item.split("=", 1) for item in res.stdout.decode().split("\0") if "=" in item)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="Ensembl gene_id")
    p.add_argument("--regions", required=True, help="regions.tsv for this gene")
    p.add_argument("--config", default="config.sh")
    p.add_argument("--outdir", default="")
    args = p.parse_args()

    cfg = load_config(Path(args.config).resolve())
    outdir = Path(args.outdir) if args.outdir else Path(cfg["OUT_ROOT"]) / "genes" / args.gene
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run_gene] {args.gene}: annotate", flush=True)
    subprocess.run([sys.executable, str(HERE / "annotate_regions.py"),
                    "--gene", args.gene, "--regions", args.regions, "--outdir", str(outdir),
                    "--tracks", cfg["TRACKS_MANIFEST"], "--chrom-sizes", cfg["CHROM_SIZES"],
                    "--tss-pad", cfg.get("TSS_PAD", "5000"),
                    "--enh-assays", cfg.get("ENH_ASSAYS", "H3K27ac,ATAC")],
                   check=True)

    print(f"[run_gene] {args.gene}: homer", flush=True)
    subprocess.run(["bash", str(HERE / "run_homer.sh"), args.gene, str(outdir)], check=True)

    print(f"[run_gene] {args.gene}: merge", flush=True)
    subprocess.run([sys.executable, str(HERE / "merge_annotations.py"),
                    "--gene", args.gene, "--outdir", str(outdir),
                    "--target-tss", cfg["TSS_TARGET_BED"]],
                   check=True)

    print(f"[run_gene] {args.gene}: plot", flush=True)
    subprocess.run([sys.executable, str(HERE / "plot_tracks.py"),
                    "--gene", args.gene, "--outdir", str(outdir),
                    "--manifest", cfg["TRACKS_MANIFEST"], "--gene-track", cfg["GENE_TRACK"],
                    "--celltype", cfg.get("CELLTYPE", "astrocyte")],
                   check=True)

    print(f"[run_gene] {args.gene}: finished", flush=True)


if __name__ == "__main__":
    main()
