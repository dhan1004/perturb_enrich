#!/usr/bin/env python3
"""
Concordance + enrichment of perturbed regions with bigwig-derived annotations.

For every perturbed region (regions.tsv, treated as ONE continuous interval) we
record a binary value per annotation file / annotation set:
    1 = the region overlaps >= 1 "active" bp, 0 = it does not.
A bigwig is binarised by calling a bp "active" when signal >= threshold
(per-file absolute threshold from the manifest, or a quantile of the signal across
the tested territory). An annotation SET (e.g. all ATAC bigwigs) is concordant if
ANY file in the set is.

Enrichment is computed against a length-matched null: each perturbed region is
re-placed at a random position (same length, same gene) inside that gene's tested
windows (tested_regions.tsv), the same binary measure is recomputed, and this is
repeated --nperm times. This controls for region length, local accessibility and
the fact that tested windows are not a random sample of the genome.

Outputs (in --outdir)
    region_concordance.tsv   one row per region, 0/1 per file and per set
    enrichment_by_set.tsv    obs vs null overlap fraction per annotation set
    enrichment_by_file.tsv   same, per bigwig
    set_pairwise.tsv         co-occurrence of two sets among perturbed regions
                             (obs joint overlap vs null joint overlap, + Fisher OR)

Annotation manifest (TSV, header required):
    set    name    path    [threshold]
    ATAC   ast_1   /path/a.bw   2.5
    H3K27ac ast_k  /path/b.bw
(`threshold` optional; blank -> use --quantile)

Requires: numpy pandas scipy pyBigWig
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import fisher_exact

G = 1 << 34  # per-chromosome offset for global coordinates (> any chrom length)


# --------------------------------------------------------------------------- #
# interval helpers
# --------------------------------------------------------------------------- #
def merge(starts, ends):
    """Merge half-open intervals; returns sorted, non-overlapping arrays."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if len(starts) == 0:
        return starts, ends
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    cm = np.maximum.accumulate(e)
    new = np.r_[True, s[1:] > cm[:-1]]
    idx = np.flatnonzero(new)
    return s[idx], np.maximum.reduceat(e, idx)


def subtract(bs, be, rs, re_):
    """Blocks (bs,be) minus regions (rs,re_); all sorted & merged."""
    out_s, out_e = [], []
    for s, e in zip(bs, be):
        cur = s
        m = (rs < e) & (re_ > s)
        for a, b in zip(rs[m], re_[m]):
            if a > cur:
                out_s.append(cur)
                out_e.append(a)
            cur = max(cur, b)
        if cur < e:
            out_s.append(cur)
            out_e.append(e)
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


def overlaps(qs, qe, ps, pe):
    """Binary: does [qs,qe) overlap any merged interval in (ps,pe)? (global coords)"""
    res = np.zeros(len(qs), dtype=bool)
    if len(ps) == 0:
        return res
    i = np.searchsorted(pe, qs, side="right")  # first interval ending after qs
    ok = i < len(ps)
    res[ok] = ps[i[ok]] < qe[ok]
    return res


# --------------------------------------------------------------------------- #
# bigwig -> binary "peaks" restricted to the tested territory
# --------------------------------------------------------------------------- #
def resolve_chrom(chrom, bw_chroms):
    if chrom in bw_chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    return alt if alt in bw_chroms else None


def weighted_quantile(values, lengths, zero_len, q):
    values = np.append(values, 0.0)
    lengths = np.append(lengths, max(zero_len, 0))
    o = np.argsort(values)
    values, lengths = values[o], lengths[o]
    cum = np.cumsum(lengths) / lengths.sum()
    return values[min(np.searchsorted(cum, q), len(values) - 1)]


def load_peaks(path, territory, chrom_id, quantile, thr):
    bw = pyBigWig.open(path)
    bwc = bw.chroms()
    C, S, E, V = [], [], [], []
    total_len = 0
    for chrom, (ts, te) in territory.items():
        name = resolve_chrom(chrom, bwc)
        if name is None:
            continue
        clen = bwc[name]
        for s, e in zip(ts, te):
            s, e = int(min(s, clen)), int(min(e, clen))
            if e <= s:
                continue
            total_len += e - s
            ivs = bw.intervals(name, s, e)
            if not ivs:
                continue
            a = np.array(ivs, dtype=float)
            a[:, 0] = np.maximum(a[:, 0], s)
            a[:, 1] = np.minimum(a[:, 1], e)
            S.append(a[:, 0].astype(np.int64))
            E.append(a[:, 1].astype(np.int64))
            V.append(a[:, 2])
            C.append(np.full(len(a), chrom_id[chrom], dtype=np.int64))
    bw.close()
    if not S:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.nan
    S, E, V, C = map(np.concatenate, (S, E, V, C))
    fin = np.isfinite(V)
    S, E, V, C = S[fin], E[fin], V[fin], C[fin]
    if thr is None:
        thr = weighted_quantile(V, E - S, total_len - (E - S).sum(), quantile)
    keep = V >= thr
    ps, pe = merge(C[keep] * G + S[keep], C[keep] * G + E[keep])
    return ps, pe, float(thr)


# --------------------------------------------------------------------------- #
# length-matched sampler
# --------------------------------------------------------------------------- #
def build_sampler(regions, tested, chrom_id, exclude_perturbed):
    """Returns (kept_mask, sample_fn). sample_fn(rng) -> global start per kept region."""
    blocks = {}
    reg_groups = regions.groupby(["gene_id", "chrom"])
    for (g, c), df in tested.groupby(["gene_id", "chrom"]):
        bs, be = merge(df.start0.values, df.end0.values)
        if exclude_perturbed and (g, c) in reg_groups.groups:
            r = reg_groups.get_group((g, c))
            rs, re_ = merge(r.start0.values, r.end0.values)
            bs, be = subtract(bs, be, rs, re_)
        blocks[(g, c)] = (bs, be)

    n = len(regions)
    keep = np.zeros(n, dtype=bool)
    ent_r, ent_s, ent_c = [], [], []
    empty = (np.array([], dtype=np.int64),) * 2
    for i, (g, c, L) in enumerate(zip(regions.gene_id, regions.chrom, regions.length)):
        bs, be = blocks.get((g, c), empty)
        cnt = be - bs - L + 1  # number of valid start positions per block
        ok = cnt > 0
        if ok.any():
            keep[i] = True
            ent_r.extend([i] * int(ok.sum()))
            ent_s.append(chrom_id[c] * G + bs[ok])
            ent_c.append(cnt[ok])
    ent_r = np.array(ent_r, dtype=np.int64)
    ent_s = np.concatenate(ent_s)
    ent_c = np.concatenate(ent_c)
    newidx = np.cumsum(keep) - 1
    ent_r = newidx[ent_r]
    n_kept = int(keep.sum())
    cum = np.cumsum(ent_c)
    cum_before = cum - ent_c
    first = np.searchsorted(ent_r, np.arange(n_kept))
    base = cum_before[first]
    total = np.add.reduceat(ent_c, first)

    def sample(rng):
        u = base + (rng.random(n_kept) * total).astype(np.int64)
        idx = np.searchsorted(cum, u, side="right")
        return ent_s[idx] + (u - cum_before[idx])

    return keep, sample


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    if len(pv) == 0:
        return out
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty_like(pv)
    q[o] = np.minimum(r, 1)
    out[m] = q
    return out


def emp_p(obs, null):
    P = len(null)
    return (1 + (null >= obs).sum()) / (1 + P), (1 + (null <= obs).sum()) / (1 + P)


def enrich_rows(stratum, n, names, obs, null):
    rows = []
    for j, nm in enumerate(names):
        nf = null[:, j] / n
        pe, pd_ = emp_p(obs[j], null[:, j])
        m, sd = nf.mean(), nf.std(ddof=1)
        rows.append(dict(
            stratum=stratum, annotation=nm, n_regions=n, n_overlap=int(obs[j]),
            frac_overlap=obs[j] / n, null_mean_frac=m, null_sd_frac=sd,
            fold_enrichment=(obs[j] / n) / m if m > 0 else np.nan,
            z=((obs[j] / n) - m) / sd if sd > 0 else np.nan,
            p_enrich=pe, p_deplete=pd_))
    df = pd.DataFrame(rows)
    df["q_enrich"] = bh(df.p_enrich.values)
    df["q_deplete"] = bh(df.p_deplete.values)
    return df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", required=True, help="perturbed regions TSV (regions.tsv; concatenate genes first)")
    ap.add_argument("--tested", required=True, help="tested windows TSV (tested_regions.tsv; concatenate genes first)")
    ap.add_argument("--annotations", required=True, help="manifest TSV: set, name, path, [threshold]")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantile", type=float, default=0.95,
                    help="signal quantile (over tested territory, gaps = 0) used as 'active' cutoff when no per-file threshold is given [0.95]")
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stratify", nargs="*", default=[],
                    help="regions.tsv columns to also analyse per level, e.g. direction annotation")
    ap.add_argument("--min-n", type=int, default=20, help="min regions for a stratum level [20]")
    ap.add_argument("--exclude-perturbed", action="store_true",
                    help="draw null positions only from tested territory NOT covered by perturbed regions "
                         "(perturbed vs. unperturbed contrast). Default: whole tested territory.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    # ---- inputs --------------------------------------------------------- #
    regions = pd.read_csv(a.regions, sep="\t")
    tested = pd.read_csv(a.tested, sep="\t", usecols=["gene_id", "chrom", "start0", "end0"])
    man = pd.read_csv(a.annotations, sep="\t")
    for col in ("set", "name", "path"):
        if col not in man.columns:
            sys.exit(f"manifest needs column '{col}'")
    if "threshold" not in man.columns:
        man["threshold"] = np.nan

    regions["length"] = regions.end0 - regions.start0
    miss = ~regions.gene_id.isin(tested.gene_id)
    if miss.any():
        print(f"[warn] dropping {miss.sum()} regions whose gene has no tested windows", file=sys.stderr)
        regions = regions[~miss].reset_index(drop=True)

    chroms = sorted(set(regions.chrom) | set(tested.chrom))
    chrom_id = {c: i for i, c in enumerate(chroms)}

    keep, sample = build_sampler(regions, tested, chrom_id, a.exclude_perturbed)
    if (~keep).any():
        print(f"[warn] {(~keep).sum()} regions are longer than any available block in their gene; excluded "
              "from observed and null", file=sys.stderr)
    regions = regions[keep].reset_index(drop=True)
    n = len(regions)
    print(f"[info] {n} perturbed regions, {regions.gene_id.nunique()} genes", file=sys.stderr)

    # ---- annotations ---------------------------------------------------- #
    pr = pd.concat([tested[["chrom", "start0", "end0"]], regions[["chrom", "start0", "end0"]]])
    territory = {c: merge(d.start0.values, d.end0.values) for c, d in pr.groupby("chrom")}

    peaks, thr_used = [], []
    for _, r in man.iterrows():
        thr = None if pd.isna(r.threshold) else float(r.threshold)
        ps, pe, t = load_peaks(r.path, territory, chrom_id, a.quantile, thr)
        peaks.append((ps, pe))
        thr_used.append(t)
        print(f"[info] {r['set']}/{r['name']}: threshold={t:.4g}, {len(ps)} active blocks", file=sys.stderr)
    man["threshold_used"] = thr_used
    man.to_csv(os.path.join(a.outdir, "annotation_thresholds.tsv"), sep="\t", index=False)

    files = (man["set"] + "|" + man["name"]).tolist()
    sets = list(dict.fromkeys(man["set"]))
    set_files = [np.flatnonzero(man["set"].values == s) for s in sets]
    F, S = len(files), len(sets)

    def binarize(qs, qe):
        X = np.zeros((len(qs), F), dtype=bool)
        for j, (ps, pe) in enumerate(peaks):
            X[:, j] = overlaps(qs, qe, ps, pe)
        Xs = np.zeros((len(qs), S), dtype=bool)
        for k, idx in enumerate(set_files):
            Xs[:, k] = X[:, idx].any(axis=1)
        return X, Xs

    # ---- observed ------------------------------------------------------- #
    cid = regions.chrom.map(chrom_id).values.astype(np.int64)
    qs = cid * G + regions.start0.values
    qe = cid * G + regions.end0.values
    L = regions.length.values
    Xf, Xs = binarize(qs, qe)

    out = regions.drop(columns="length").copy()
    for j, f in enumerate(files):
        out["file:" + f] = Xf[:, j].astype(int)
    for k, s in enumerate(sets):
        out["set:" + s] = Xs[:, k].astype(int)
    out.to_csv(os.path.join(a.outdir, "region_concordance.tsv"), sep="\t", index=False)

    # ---- strata --------------------------------------------------------- #
    strata = {"all": np.ones(n, dtype=bool)}
    for col in a.stratify:
        if col not in regions.columns:
            print(f"[warn] stratify column '{col}' not in regions; skipped", file=sys.stderr)
            continue
        for v, cnt in regions[col].value_counts().items():
            if cnt >= a.min_n:
                strata[f"{col}={v}"] = (regions[col] == v).values
    names = list(strata)
    masks = [strata[k] for k in names]
    K = len(names)

    # ---- permutations --------------------------------------------------- #
    rng = np.random.default_rng(a.seed)
    nullF = np.zeros((a.nperm, K, F), dtype=np.int32)
    nullS = np.zeros((a.nperm, K, S), dtype=np.int32)
    nullJ = np.zeros((a.nperm, K, S, S), dtype=np.int32)
    for p in range(a.nperm):
        st = sample(rng)
        pf, ps_ = binarize(st, st + L)
        for k, m in enumerate(masks):
            nullF[p, k] = pf[m].sum(axis=0)
            x = ps_[m].astype(np.int32)
            nullS[p, k] = x.sum(axis=0)
            nullJ[p, k] = x.T @ x
        if (p + 1) % max(1, a.nperm // 10) == 0:
            print(f"[perm] {p + 1}/{a.nperm}  ({time.time() - t0:.0f}s)", file=sys.stderr)

    # ---- tables --------------------------------------------------------- #
    tabS, tabF, tabP = [], [], []
    for k, (nm, m) in enumerate(zip(names, masks)):
        nk = int(m.sum())
        tabS.append(enrich_rows(nm, nk, sets, Xs[m].sum(axis=0), nullS[:, k, :]))
        tabF.append(enrich_rows(nm, nk, files, Xf[m].sum(axis=0), nullF[:, k, :]))
        x = Xs[m].astype(int)
        J = x.T @ x
        rows = []
        for i in range(S):
            for j in range(i + 1, S):
                nj = nullJ[:, k, i, j]
                pe, pd_ = emp_p(J[i, j], nj)
                a_, b_ = J[i, j], x[:, i].sum() - J[i, j]
                c_, d_ = x[:, j].sum() - J[i, j], nk - x[:, i].sum() - x[:, j].sum() + J[i, j]
                orr, pf = fisher_exact([[a_, b_], [c_, d_]])
                null_f = nj / nk
                rows.append(dict(
                    stratum=nm, set_a=sets[i], set_b=sets[j], n_regions=nk,
                    n_both=int(J[i, j]), frac_both=J[i, j] / nk,
                    jaccard=J[i, j] / max(1, x[:, i].sum() + x[:, j].sum() - J[i, j]),
                    null_mean_frac_both=null_f.mean(),
                    fold_vs_null=(J[i, j] / nk) / null_f.mean() if null_f.mean() > 0 else np.nan,
                    p_enrich=pe, p_deplete=pd_, fisher_or_within_regions=orr, fisher_p_within_regions=pf))
        tabP.append(pd.DataFrame(rows))
    for nm, tabs in (("enrichment_by_set", tabS), ("enrichment_by_file", tabF), ("set_pairwise", tabP)):
        df = pd.concat(tabs, ignore_index=True)
        if nm == "set_pairwise" and len(df):
            df["q_enrich"] = bh(df.p_enrich.values)
        df.to_csv(os.path.join(a.outdir, nm + ".tsv"), sep="\t", index=False)
    print(f"[done] {time.time() - t0:.0f}s -> {a.outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()