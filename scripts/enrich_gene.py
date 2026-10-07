#!/usr/bin/env python3
"""Stage 5 (per gene): binary concordance of perturbed regions with annotation sets, plus the
permutation-null overlap counts needed for enrichment. Pooling across genes happens later in
aggregate_enrich.py (counts are additive, so the pooled result is exact).

Annotation sets (from tracks.tsv): one set per manifest `name`; role=mask rows are ignored;
a set uses its bigwig rows if it has any (a bp is "active" when signal >= the cutoff from
calibrate_thresholds.py), otherwise its BED rows. A region scores 1 for a set if it overlaps
>= 1 active bp / peak in ANY of the set's files, else 0. Each perturbed region is one block.

Null: each region is re-placed --nperm times at a uniformly random position inside THIS gene's
tested windows (tested_regions.tsv), same length, and re-scored.  --exclude-perturbed restricts
the draw to tested sequence not covered by perturbed regions.

Outputs in --outdir (default $OUT_ROOT/genes/<gene>):
    enrich_regions.tsv   region_id + 0/1 per file and per set (+ strata columns)
    enrich_counts.npz    observed and null overlap counts per stratum (input to aggregation)
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_lib import (G, build_sampler, choose_sources, merge, norm_path,  # noqa: E402
                        overlaps, read_manifest, resolve_chrom)


def log(msg):
    print(f"[enrich] {msg}", file=sys.stderr, flush=True)


def read_regions(path):
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str, "gene_id": str})
    df.insert(0, "region_id", df["gene_id"] + "|" + df["chrom"] + "|"
              + df["start0"].astype(str) + "|" + df["end0"].astype(str))
    df = df.drop_duplicates("region_id").reset_index(drop=True)
    df["length"] = df["end0"] - df["start0"]
    return df


def load_active_bigwig(path, territory, chrom_id, thr):
    """Merged global-coordinate intervals where signal >= thr, restricted to the territory."""
    if not np.isfinite(thr):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    bw = pyBigWig.open(path)
    bwc = bw.chroms()
    S, E, C, matched = [], [], [], False
    for chrom, (ts, te) in territory.items():
        name = resolve_chrom(chrom, bwc)
        if name is None:
            continue
        matched = True
        clen = bwc[name]
        for s, e in zip(ts, te):
            s, e = int(min(s, clen)), int(min(e, clen))
            if e <= s:
                continue
            ivs = bw.intervals(name, s, e)
            if not ivs:
                continue
            a = np.asarray(ivs, dtype=np.float64)
            a = a[a[:, 2] >= thr]
            if len(a):
                S.append(np.maximum(a[:, 0], s).astype(np.int64))
                E.append(np.minimum(a[:, 1], e).astype(np.int64))
                C.append(np.full(len(a), chrom_id[chrom], dtype=np.int64))
    bw.close()
    if not matched:
        log(f"WARNING {path}: none of the region chromosomes found in bigwig (naming mismatch?)")
    if not S:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    S, E, C = map(np.concatenate, (S, E, C))
    return merge(C * G + S, C * G + E)


def load_bed(path, chrom_id):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"],
                     dtype={"c": str}, comment="#")
    df["s"] = pd.to_numeric(df["s"], errors="coerce")
    df["e"] = pd.to_numeric(df["e"], errors="coerce")
    df = df.dropna()
    lookup = {}
    for c in chrom_id:
        lookup[c] = c
        lookup.setdefault(c[3:] if c.startswith("chr") else "chr" + c, c)
    df["c"] = df["c"].map(lookup)
    df = df.dropna(subset=["c"])
    if df.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cid = df["c"].map(chrom_id).values.astype(np.int64)
    return merge(cid * G + df["s"].values.astype(np.int64), cid * G + df["e"].values.astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", required=True)
    ap.add_argument("--regions", required=True, help="regions.tsv for this gene")
    ap.add_argument("--tested", default="", help="tested_regions.tsv (default: next to --regions)")
    ap.add_argument("--tracks", required=True, help="tracks.tsv manifest")
    ap.add_argument("--thresholds", required=True, help="thresholds.tsv from calibrate_thresholds.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--strata", default="direction annotation is_distal_enh",
                    help="space-separated columns (regions.tsv / annotated.tsv) to also analyse per level")
    ap.add_argument("--exclude-perturbed", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tested_path = Path(a.tested) if a.tested else Path(a.regions).with_name("tested_regions.tsv")
    if not tested_path.exists():
        sys.exit(f"[enrich] tested windows not found: {tested_path} (pass --tested)")

    regions = read_regions(a.regions)
    tested = pd.read_csv(tested_path, sep="\t", usecols=["gene_id", "chrom", "start0", "end0"],
                         dtype={"chrom": str, "gene_id": str})

    # extra columns produced by stage 1 (is_distal_enh etc.) for stratification
    ann = out / "annotated.tsv"
    if ann.exists():
        extra = pd.read_csv(ann, sep="\t", dtype={"chrom": str}, usecols=lambda c: c in (
            "region_id", "in_promoter_mask", "has_enh_call", "is_distal_enh"))
        extra = extra[[c for c in extra.columns if c == "region_id" or c not in regions.columns]]
        regions = regions.merge(extra, on="region_id", how="left")

    chrom_id = {c: i for i, c in enumerate(sorted(set(regions.chrom) | set(tested.chrom)))}

    # ---- annotation sets ------------------------------------------------ #
    sets = choose_sources(read_manifest(a.tracks))
    thr_tab = pd.read_csv(a.thresholds, sep="\t", dtype={"path": str})
    thr_by_path = {norm_path(p): t for p, t in zip(thr_tab["path"], thr_tab["threshold"])}

    pr = pd.concat([tested[["chrom", "start0", "end0"]], regions[["chrom", "start0", "end0"]]])
    territory = {c: merge(d.start0.values, d.end0.values) for c, d in pr.groupby("chrom")}

    file_names, file_set, peaks = [], [], []
    for si, s in enumerate(sets):
        for p in s["paths"]:
            if s["source"] == "bigwig":
                if p not in thr_by_path:
                    sys.exit(f"[enrich] no calibrated threshold for {p}; run calibrate_thresholds.py")
                pk = load_active_bigwig(p, territory, chrom_id, float(thr_by_path[p]))
            else:
                pk = load_bed(p, chrom_id)
            peaks.append(pk)
            file_names.append(f"{s['set']}|{Path(p).name}")
            file_set.append(si)
    file_set = np.array(file_set)
    set_names = [s["set"] for s in sets]
    F, S = len(file_names), len(sets)
    set_files = [np.flatnonzero(file_set == k) for k in range(S)]

    def binarize(qs, qe):
        X = np.zeros((len(qs), F), dtype=bool)
        for j, (ps, pe) in enumerate(peaks):
            X[:, j] = overlaps(qs, qe, ps, pe)
        Xs = np.zeros((len(qs), S), dtype=bool)
        for k, idx in enumerate(set_files):
            Xs[:, k] = X[:, idx].any(axis=1)
        return X, Xs

    # ---- null placement ------------------------------------------------- #
    keep, sample = build_sampler(regions, tested, chrom_id, a.exclude_perturbed)
    if (~keep).any():
        log(f"{a.gene}: {(~keep).sum()} region(s) cannot be placed in the tested windows; excluded")
    regions = regions[keep].reset_index(drop=True)
    n = len(regions)

    # ---- strata --------------------------------------------------------- #
    strata = {"all": np.ones(n, dtype=bool)}
    for col in a.strata.split():
        if col not in regions.columns:
            continue
        vals = regions[col].astype(str)
        for v in vals.dropna().unique():
            strata[f"{col}={v}"] = (vals == v).values
    names = list(strata)
    masks = [strata[k] for k in names]
    K = len(names)

    P = a.nperm
    obs_set = np.zeros((K, S), dtype=np.int32)
    obs_file = np.zeros((K, F), dtype=np.int32)
    obs_joint = np.zeros((K, S, S), dtype=np.int32)
    null_set = np.zeros((P, K, S), dtype=np.int32)
    null_file = np.zeros((P, K, F), dtype=np.int32)
    null_joint = np.zeros((P, K, S, S), dtype=np.int32)

    if n:
        cid = regions.chrom.map(chrom_id).values.astype(np.int64)
        L = regions.length.values.astype(np.int64)
        Xf, Xs = binarize(cid * G + regions.start0.values, cid * G + regions.end0.values)

        rng = np.random.default_rng([a.seed, zlib.crc32(a.gene.encode())])
        starts = sample(rng, P).ravel()
        Nf, Ns = binarize(starts, starts + np.tile(L, P))
        Nf, Ns = Nf.reshape(P, n, F), Ns.reshape(P, n, S)

        for k, m in enumerate(masks):
            x = Xs[m].astype(np.int32)
            obs_set[k], obs_file[k] = x.sum(0), Xf[m].sum(0)
            obs_joint[k] = x.T @ x
            xn = Ns[:, m, :].astype(np.int32)
            null_set[:, k] = xn.sum(1)
            null_file[:, k] = Nf[:, m, :].sum(1)
            null_joint[:, k] = np.einsum("pns,pnt->pst", xn, xn)

        tab = regions[["region_id", "gene_id", "chrom", "start0", "end0"]
                      + [c for c in a.strata.split() if c in regions.columns]].copy()
        for j, f in enumerate(file_names):
            tab["file:" + f] = Xf[:, j].astype(int)
        for k, s in enumerate(set_names):
            tab["set:" + s] = Xs[:, k].astype(int)
        tab.to_csv(out / "enrich_regions.tsv", sep="\t", index=False)

    np.savez_compressed(
        out / "enrich_counts.npz", gene=a.gene, nperm=P, seed=a.seed,
        set_names=np.array(set_names), set_source=np.array([s["source"] for s in sets]),
        set_assay=np.array([s["assay"] for s in sets]), set_celltype=np.array([s["celltype"] for s in sets]),
        file_names=np.array(file_names), file_set=file_set,
        strata_names=np.array(names), n=np.array([int(m.sum()) for m in masks]),
        obs_set=obs_set, obs_file=obs_file, obs_joint=obs_joint,
        null_set=null_set, null_file=null_file, null_joint=null_joint)
    log(f"{a.gene}: {n} regions, {S} sets ({', '.join(f'{s['set']}:{s['source']}' for s in sets)})")


if __name__ == "__main__":
    main()
