#!/usr/bin/env python3
"""
Runa the per-gene pipeline: annotate_regions.py -> run_homer.sh -> merge_annotations.py -> enrich_gene.py -> plot_tracks.py

usage: python run_gene.py --gene ENSG00000002549 --regions regions.tsv [--config config.sh] [--outdir DIR]

Reads config.sh for TRACKS_MANIFEST, CHROM_SIZES, TSS_TARGET_BED, GENE_TRACK and OUT_ROOT
(optional: TSS_PAD, ENH_ASSAYS, ENH_CELLTYPES, MASK_ASSAYS, DISTAL_MODE, CELLTYPE).
Annotation outputs go to $OUT_ROOT/genes/<gene>/ unless --outdir is given.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config(path):
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

    config = load_config(Path(args.config).resolve())
    outdir = Path(args.outdir) if args.outdir else Path(config["OUT_ROOT"]) / "genes" / args.gene
    outdir.mkdir(parents=True, exist_ok=True)

    # step 1: annotate genes with tracks
    print(f"[run_gene] {args.gene}: annotate", flush=True)
    subprocess.run([sys.executable, str(HERE / "annotate_regions.py"),
                    "--gene", args.gene, 
                    "--regions", args.regions, 
                    "--outdir", str(outdir),
                    "--tracks", config["TRACKS_MANIFEST"],
                    "--tss-bed", config["TSS_BED"],
                    "--promoter-half-bp", config.get("PROMOTER_HALF_BP", "5000")],
                   check=True)

    # step 2: HOMER enrichment
    print(f"[run_gene] {args.gene}: homer", flush=True)
    subprocess.run(["bash", str(HERE / "run_homer.sh"), args.gene, str(outdir)], check=True)

    # step 3: merge annotations
    print(f"[run_gene] {args.gene}: merge", flush=True)
    subprocess.run([sys.executable, str(HERE / "merge_annotations.py"),
                    "--gene", args.gene, "--outdir", str(outdir)],
                   check=True)

    # step 4: enrichment calculations across genes
    print(f"[run_gene] {args.gene}: enrich", flush=True)
    enrich_cmd = [sys.executable, str(HERE / "enrich_gene.py"),
                  "--gene", args.gene, "--regions", args.regions,
                  "--tested", str(Path(args.regions).resolve().parents[1] / "tested_space.bed"),
                  "--outdir", str(outdir),
                  "--tracks", config["TRACKS_MANIFEST"], "--thresholds", config["THRESHOLDS_TSV"],
                  "--nperm", config.get("ENRICH_NPERM", "1000"),
                  "--strata", config.get("ENRICH_STRATA", "direction annotation is_distal_enh len_bin")]
    if config.get("ENRICH_EXCLUDE_PERTURBED", "0") == "1":
        enrich_cmd.append("--exclude-perturbed")
    subprocess.run(enrich_cmd, check=True)

    # step 5: plot tracks per gene         
    print(f"[run_gene] {args.gene}: plot", flush=True)
    subprocess.run([sys.executable, str(HERE / "plot_tracks.py"),
                    "--gene", args.gene, "--outdir", str(outdir),
                    "--manifest", config["TRACKS_MANIFEST"], "--gene-track", config["GENE_TRACK"],
                    "--celltype", config.get("CELLTYPE", "astrocyte")],
                   check=True)

    print(f"[run_gene] {args.gene}: finished", flush=True)


if __name__ == "__main__":
    main()