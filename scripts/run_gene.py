#!/usr/bin/env python3
"""Run the whole per-gene pipeline: annotate -> HOMER -> merge -> plot.

usage: python run_gene.py --gene ENSG00000002549 [--config config.sh] [--regions regions.tsv] [--force]

--gene is the Ensembl gene_id used in regions.tsv (version suffix ignored).

Settings are read from config.sh (sourced in bash, so it can use shell variables). Variables:
  REGIONS_FILE          regions.tsv with all genes (filtered by gene_id)  -- or --
  REGIONS_DIR           per-gene files at $REGIONS_DIR/<gene_id>.tsv      (or pass --regions)
  RESULTS_DIR           outputs go to $RESULTS_DIR/<gene_id>/
  TRACKS_MANIFEST       tracks.tsv
  CHROM_SIZES           hg38.chrom.sizes
  TSS_TARGET_BED        one TSS per gene (MANE Select), Ensembl gene_id in column 4
  GENE_TRACK            GENCODE basic GTF (sorted + indexed) or BED12 for the plots
  TSS_PAD (2000)  CELLTYPE (astrocyte; filters the manifest AND, case-insensitively, regions.tsv cell_type)
  REGION_CELLTYPE       override the regions.tsv cell_type filter ("all" = keep every cell type)
  ENH_ASSAYS (H3K27ac,ATAC,cCRE)
  REQUIRE_INPUT_DISTAL  1 = also require annotation=="distal" and tss_region_overlap_bp==0 from regions.tsv
  PERTURB_TRACK_PATTERN optional extra track, e.g. /path/perturb/{gene}.bw
  HOMER_* / RUN_ENRICHMENT / MIN_ENRICH_REGIONS   see run_homer.sh

Each step is skipped if its outputs already exist (use --force to redo), so a failed
array task can simply be resubmitted.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config(path):
    """Source a bash config file and return the resulting environment as a dict."""
    res = subprocess.run(["bash", "-c", f'set -a; source "{path}"; env -0'],
                         capture_output=True, check=True)
    env = {}
    for item in res.stdout.decode().split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env


def need(cfg, key):
    if not cfg.get(key):
        sys.exit(f"[run_gene] {key} is not set in the config")
    return cfg[key]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="Ensembl gene_id as in regions.tsv")
    p.add_argument("--config", default="config.sh")
    p.add_argument("--regions", default="")
    p.add_argument("--outdir", default="")
    p.add_argument("--force", action="store_true", help="rerun every step")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--enrichment", action="store_true", help="also run per-gene findMotifsGenome")
    a = p.parse_args()

    config = str(Path(a.config).resolve())
    if not Path(config).exists():
        sys.exit(f"[run_gene] config not found: {config}")
    cfg = load_config(config)

    gene = a.gene
    regions = a.regions or cfg.get("REGIONS_FILE") or f"{need(cfg, 'REGIONS_DIR')}/{gene}.tsv"
    if not Path(regions).exists():
        sys.exit(f"[run_gene] regions file not found: {regions}")
    outdir = Path(a.outdir or f"{need(cfg, 'RESULTS_DIR')}/{gene}")
    outdir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, **cfg, "CONFIG": config}
    if a.enrichment:
        env["RUN_ENRICHMENT"] = "1"
    py = sys.executable

    def step(name, outputs, cmd):
        if not a.force and all((outdir / o).exists() for o in outputs):
            print(f"[run_gene] {gene}: {name} - outputs exist, skipping", flush=True)
            return
        t0 = time.time()
        print(f"[run_gene] {gene}: {name}", flush=True)
        subprocess.run(cmd, check=True, env=env)
        print(f"[run_gene] {gene}: {name} done in {time.time() - t0:.0f}s", flush=True)

    annotate_cmd = [py, str(HERE / "annotate_regions.py"),
                    "--gene", gene, "--regions", regions, "--outdir", str(outdir),
                    "--manifest", need(cfg, "TRACKS_MANIFEST"), "--chrom-sizes", need(cfg, "CHROM_SIZES"),
                    "--tss-pad", cfg.get("TSS_PAD", "2000"), "--celltype", cfg.get("CELLTYPE", "astrocyte"),
                    "--enh-assays", cfg.get("ENH_ASSAYS", "H3K27ac,ATAC,cCRE")]
    region_ct = cfg.get("REGION_CELLTYPE", "")
    if region_ct:
        annotate_cmd += ["--region-celltype", "" if region_ct.lower() == "all" else region_ct]
    if cfg.get("REQUIRE_INPUT_DISTAL", "0") == "1":
        annotate_cmd.append("--require-input-distal")
    step("annotate", ["annotated.tsv", "all_regions.bed", "distal_enh.bed"], annotate_cmd)

    # resume check is on the annotation output only; use --force to rerun enrichment
    step("homer", ["homer_annot.tsv"], ["bash", str(HERE / "run_homer.sh"), gene, str(outdir)])

    step("merge", ["final_regions.tsv"],
         [py, str(HERE / "merge_annotations.py"),
          "--gene", gene, "--outdir", str(outdir), "--target-tss", need(cfg, "TSS_TARGET_BED")])

    if not a.no_plot:
        cmd = [py, str(HERE / "plot_tracks.py"),
               "--gene", gene, "--manifest", need(cfg, "TRACKS_MANIFEST"), "--outdir", str(outdir),
               "--gene-track", need(cfg, "GENE_TRACK"), "--celltype", cfg.get("CELLTYPE", "astrocyte")]
        pattern = cfg.get("PERTURB_TRACK_PATTERN", "")
        if pattern and Path(pattern.format(gene=gene)).exists():
            cmd += ["--perturb-track", pattern.format(gene=gene)]
        step("plot", [f"{gene}.tracks.pdf"], cmd)

    (outdir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[run_gene] {gene}: finished", flush=True)


if __name__ == "__main__":
    main()