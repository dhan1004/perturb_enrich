#!/usr/bin/env python3
"""Final step: pool the per-gene enrichment counts (genes/*/enrich_counts.npz) and compute
enrichment of region/annotation concordance against the length-matched null.

Per-gene overlap counts are additive, so summing observed counts and, permutation by
permutation, null counts across genes is exactly the pooled analysis. Run once after all genes.

Outputs in --outdir (default <out-root>/enrichment):
    enrichment_by_set.tsv     observed vs null overlap fraction per annotation set and stratum
    enrichment_by_file.tsv    same per file (bigwig/bed) within sets
    set_pairwise.tsv          co-occurrence of two sets (observed joint overlap vs null joint)
    region_concordance.tsv    every perturbed region with its 0/1 values and covered bp (all genes)
    per_gene_counts.tsv       per-gene region counts and set overlap counts
  Base-pair coverage (needs enrich_counts.npz written by the coverage-aware enrich_gene.py):
    enrichment_cov_by_set.tsv fraction of perturbed bp covered by active signal vs the permutation
                              null and vs the tested-territory background, per set and stratum
    cov_pairwise.tsv          does set A show more enrichment than set B? (paired permutation test
                              on the difference in enrichment; same placements for every set)
    per_gene_coverage.tsv     the same enrichment per gene (stratum 'all') for gene-level tests
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_lib import bh, cov_pairwise_rows, cov_rows, enrich_rows, pairwise_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="$OUT_ROOT (contains genes/<gene>/)")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--min-n", type=int, default=20, help="min pooled regions for a stratum [20]")
    a = ap.parse_args()

    root = Path(a.out_root)
    outdir = Path(a.outdir) if a.outdir else root / "enrichment"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.glob("genes/*/enrich_counts.npz"))
    if not paths:
        sys.exit(f"[aggregate] no enrich_counts.npz under {root}/genes")
    n_final = len(list(root.glob("genes/*/final_regions.tsv")))
    print(f"[aggregate] {len(paths)} genes with enrichment counts ({n_final} with final_regions.tsv)",
          file=sys.stderr)

    data = [np.load(p, allow_pickle=False) for p in paths]
    ref = data[0]
    for p, d in zip(paths, data):
        for key in ("set_names", "file_names"):
            if list(d[key]) != list(ref[key]):
                sys.exit(f"[aggregate] {p}: {key} differ from first gene (manifest changed mid-run?)")
        if int(d["nperm"]) != int(ref["nperm"]):
            sys.exit(f"[aggregate] {p}: nperm differs")
    sets, files = list(ref["set_names"]), list(ref["file_names"])
    S, F, P = len(sets), len(files), int(ref["nperm"])

    allnames = {"all"}
    for d in data:
        allnames.update(d["strata_names"])
    names = ["all"] + sorted(allnames - {"all"})
    kidx = {k: i for i, k in enumerate(names)}
    K = len(names)

    n = np.zeros(K, dtype=np.int64)
    n_genes = np.zeros(K, dtype=np.int64)
    obs_set = np.zeros((K, S), dtype=np.int64)
    obs_file = np.zeros((K, F), dtype=np.int64)
    obs_joint = np.zeros((K, S, S), dtype=np.int64)
    null_set = np.zeros((P, K, S), dtype=np.int64)
    null_file = np.zeros((P, K, F), dtype=np.int64)
    null_joint = np.zeros((P, K, S, S), dtype=np.int64)
    missing_cov = [str(d["gene"]) for d in data if "obs_cov" not in d.files]
    has_cov = not missing_cov
    if not has_cov:
        print(f"[aggregate] {len(missing_cov)} gene(s) lack coverage counts (enrich_counts.npz predates "
              f"the coverage statistic, e.g. {missing_cov[:3]}); rerun run_enrich.sbatch to get "
              "enrichment_cov_by_set.tsv / cov_pairwise.tsv / per_gene_coverage.tsv", file=sys.stderr)
    obs_cov = np.zeros((K, S), dtype=np.int64)
    obs_len = np.zeros(K, dtype=np.int64)
    null_cov = np.zeros((P, K, S), dtype=np.int64)
    bg_cov = np.zeros(S, dtype=np.int64)
    bg_len = 0
    per_gene, per_gene_cov = [], []
    for d in data:
        for j, nm in enumerate(d["strata_names"]):
            k = kidx[nm]
            n[k] += d["n"][j]
            n_genes[k] += int(d["n"][j] > 0)
            obs_set[k] += d["obs_set"][j]
            obs_file[k] += d["obs_file"][j]
            obs_joint[k] += d["obs_joint"][j]
            null_set[:, k] += d["null_set"][:, j]
            null_file[:, k] += d["null_file"][:, j]
            null_joint[:, k] += d["null_joint"][:, j]
            if has_cov:
                obs_cov[k] += d["obs_cov"][j]
                obs_len[k] += d["obs_len"][j]
                null_cov[:, k] += d["null_cov"][:, j]
        j0 = list(d["strata_names"]).index("all")
        row = {"gene_id": str(d["gene"]), "n_regions": int(d["n"][j0])}
        row.update({f"n_{s}": int(v) for s, v in zip(sets, d["obs_set"][j0])})
        per_gene.append(row)
        if has_cov:
            bg_cov += d["bg_cov"]
            bg_len += int(d["bg_len"])
            if d["obs_len"][j0] > 0 and int(d["bg_len"]) > 0:
                g = cov_rows("all", int(d["n"][j0]), int(d["obs_len"][j0]), sets, d["obs_cov"][j0],
                             d["null_cov"][:, j0, :], d["bg_cov"] / int(d["bg_len"]))
                g.insert(0, "gene_id", str(d["gene"]))
                per_gene_cov.append(g)
    pd.DataFrame(per_gene).to_csv(outdir / "per_gene_counts.tsv", sep="\t", index=False)

    meta = pd.DataFrame(dict(annotation=sets, source=list(ref["set_source"]),
                             assay=list(ref["set_assay"]), celltype=list(ref["set_celltype"])))
    tabS, tabF, tabP = [], [], []
    tabC, tabCP = [], []
    bg_frac = bg_cov / bg_len if (has_cov and bg_len > 0) else np.full(S, np.nan)
    for k, nm in enumerate(names):
        if n[k] < a.min_n:
            continue
        if has_cov and obs_len[k] > 0:
            t = cov_rows(nm, int(n[k]), int(obs_len[k]), sets, obs_cov[k], null_cov[:, k, :], bg_frac)
            t.insert(3, "n_genes", int(n_genes[k]))
            tabC.append(t.merge(meta, on="annotation", how="left"))
            tabCP.append(cov_pairwise_rows(nm, int(n[k]), int(obs_len[k]), sets,
                                           obs_cov[k], null_cov[:, k, :]))
        t = enrich_rows(nm, int(n[k]), sets, obs_set[k], null_set[:, k, :])
        t.insert(3, "n_genes", int(n_genes[k]))
        tabS.append(t.merge(meta, on="annotation", how="left"))
        t = enrich_rows(nm, int(n[k]), files, obs_file[k], null_file[:, k, :])
        t.insert(3, "n_genes", int(n_genes[k]))
        tabF.append(t)
        tabP.append(pairwise_rows(nm, int(n[k]), sets, obs_set[k], obs_joint[k], null_joint[:, k]))
    for nm, tabs in (("enrichment_by_set", tabS), ("enrichment_by_file", tabF), ("set_pairwise", tabP)):
        if not tabs:
            print(f"[aggregate] no stratum reached --min-n {a.min_n}; {nm}.tsv not written", file=sys.stderr)
            continue
        df = pd.concat(tabs, ignore_index=True)
        if nm == "set_pairwise":
            df["q_enrich"] = bh(df.p_enrich.values)
        df.to_csv(outdir / f"{nm}.tsv", sep="\t", index=False)
    if has_cov:
        for nm, tabs in (("enrichment_cov_by_set", tabC), ("cov_pairwise", tabCP),
                         ("per_gene_coverage", per_gene_cov)):
            if not tabs:
                print(f"[aggregate] nothing to write for {nm}.tsv", file=sys.stderr)
                continue
            pd.concat(tabs, ignore_index=True).to_csv(outdir / f"{nm}.tsv", sep="\t", index=False)

    regs =[pd.read_csv(p.with_name("enrich_regions.tsv"), sep="\t", dtype={"chrom": str})
            for p in paths if p.with_name("enrich_regions.tsv").exists()]
    if regs:
        pd.concat(regs, ignore_index=True).to_csv(outdir / "region_concordance.tsv", sep="\t", index=False)
    print(f"[aggregate] wrote {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()