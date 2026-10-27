#!/usr/bin/env python3
"""Run the per-gene pipeline: annotate_regions.py -> run_homer.sh -> merge_annotations.py -> plot_tracks.py.

usage: python run_gene.py --gene ENSG00000002549 --regions regions.tsv [--config config.sh] [--outdir DIR]

Reads --config (also handed to run_homer.sh through PERTURB_CONFIG, so every step sees the same file):
  required: TRACKS_MANIFEST, CHROM_SIZES, TSS_BED, THRESHOLDS_TSV, GENE_TRACK, OUT_ROOT
  optional: TSS_PAD (5000), PROMOTER_HALF_BP (5000), ENH_ASSAYS, ENH_CELLTYPES, DISTAL_MODE, CELLTYPE,
            ENRICH_NPERM, ENRICH_STRATA, ENRICH_EXCLUDE_PERTURBED
TSS_BED is the single TSS source (promoter mask, distance to target TSS, tss_class); it is not a tracks.tsv row.
Annotation outputs go to $OUT_ROOT/genes/<gene>/ unless --outdir is given.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ("TRACKS_MANIFEST", "CHROM_SIZES", "TSS_BED", "THRESHOLDS_TSV", "GENE_TRACK", "OUT_ROOT")


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

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    missing = [k for k in REQUIRED if not config.get(k)]
    if missing:
        hint = " (TSS_TARGET_BED was renamed TSS_BED)" if "TSS_BED" in missing and config.get("TSS_TARGET_BED") else ""
        sys.exit(f"[run_gene] {config_path} does not set: {', '.join(missing)}{hint}")
    outdir = Path(args.outdir) if args.outdir else Path(config["OUT_ROOT"]) / "genes" / args.gene
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run_gene] {args.gene}: annotate", flush=True)
    subprocess.run([sys.executable, str(HERE / "annotate_regions.py"),
                    "--gene", args.gene, "--regions", args.regions, "--outdir", str(outdir),
                    "--tracks", config["TRACKS_MANIFEST"], "--chrom-sizes", config["CHROM_SIZES"],
                    "--tss-bed", config["TSS_BED"], "--tss-pad", config.get("TSS_PAD", "5000"),
                    "--enh-assays", config.get("ENH_ASSAYS", "h3k27ac,atac"),
                    "--enh-celltypes", config.get("ENH_CELLTYPES", "astrocyte"),
                    "--distal-mode", config.get("DISTAL_MODE", "enhancer")],
                   check=True)

    print(f"[run_gene] {args.gene}: homer", flush=True)
    # run_homer.sh re-sources a config itself (GENOME, HOMER_*, THREADS, ...): point it at the one we were given
    subprocess.run(["bash", str(HERE / "run_homer.sh"), args.gene, str(outdir)],
                   env={**os.environ, "PERTURB_CONFIG": str(config_path)}, check=True)

    print(f"[run_gene] {args.gene}: merge", flush=True)
    subprocess.run([sys.executable, str(HERE / "merge_annotations.py"),
                    "--gene", args.gene, "--outdir", str(outdir),
                    "--tss-bed", config["TSS_BED"],
                    "--promoter-half-bp", config.get("PROMOTER_HALF_BP", "5000")],
                   check=True)

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

    print(f"[run_gene] {args.gene}: plot", flush=True)
    subprocess.run([sys.executable, str(HERE / "plot_tracks.py"),
                    "--gene", args.gene, "--outdir", str(outdir),
                    "--manifest", config["TRACKS_MANIFEST"], "--gene-track", config["GENE_TRACK"],
                    "--celltype", config.get("CELLTYPE", "astrocyte")],
                   check=True)

    print(f"[run_gene] {args.gene}: finished", flush=True)


if __name__ == "__main__":
    main()