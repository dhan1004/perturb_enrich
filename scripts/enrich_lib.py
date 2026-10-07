"""Shared helpers for the concordance / enrichment stage (enrich_gene.py, aggregate_enrich.py,
calibrate_thresholds.py). Not a command-line script."""
import os

import numpy as np
import pandas as pd

G = 1 << 34  # per-chromosome offset for global coordinates (larger than any chromosome)


# --------------------------------------------------------------------------- #
# manifest / paths
# --------------------------------------------------------------------------- #
def norm_path(p):
    return os.path.normpath(str(p).strip())


def read_manifest(path):
    """tracks.tsv is whitespace-delimited (same parsing as annotate_regions.py)."""
    return pd.read_csv(path, sep=r"\s+", dtype=str).fillna("")


def choose_sources(manifest):
    """One annotation SET per manifest `name`. Role 'mask' rows are not annotations.
    A set uses its bigwig rows when it has any; otherwise its BED rows.
    Returns list of dicts: set, source ('bigwig'|'bed'), assay, celltype, paths (list)."""
    man = manifest[manifest["role"] != "mask"]
    sets = []
    for name in dict.fromkeys(man["name"]):
        rows = man[man["name"] == name]
        bw = rows[rows["type"] == "bigwig"]
        use, source = (bw, "bigwig") if len(bw) else (rows[rows["type"] == "bed"], "bed")
        if use.empty:
            continue
        sets.append(dict(set=name, source=source, assay=use["assay"].iloc[0],
                         celltype=use["celltype"].iloc[0],
                         paths=[norm_path(p) for p in use["path"]]))
    return sets


# --------------------------------------------------------------------------- #
# interval helpers (half-open, global coordinates = chrom_id * G + pos)
# --------------------------------------------------------------------------- #
def merge(starts, ends):
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
    """Binary: does [qs,qe) overlap any interval of the merged set (ps,pe)?"""
    res = np.zeros(len(qs), dtype=bool)
    if len(ps) == 0:
        return res
    i = np.searchsorted(pe, qs, side="right")
    ok = i < len(ps)
    res[ok] = ps[i[ok]] < qe[ok]
    return res


def resolve_chrom(chrom, available):
    if chrom in available:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    return alt if alt in available else None


# --------------------------------------------------------------------------- #
# length-matched null sampler
# --------------------------------------------------------------------------- #
def build_sampler(regions, tested, chrom_id, exclude_perturbed):
    """regions needs gene_id, chrom, start0, end0, length. Returns (kept_mask, sample)
    where sample(rng, reps) -> int64 array (reps, n_kept) of global start positions: each
    region re-placed uniformly inside its gene's tested windows, same length."""
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
        cnt = be - bs - L + 1
        ok = cnt > 0
        if ok.any():
            keep[i] = True
            ent_r.extend([i] * int(ok.sum()))
            ent_s.append(chrom_id[c] * G + bs[ok])
            ent_c.append(cnt[ok])
    n_kept = int(keep.sum())
    if n_kept == 0:
        return keep, None
    ent_r = np.array(ent_r, dtype=np.int64)
    ent_s = np.concatenate(ent_s)
    ent_c = np.concatenate(ent_c)
    ent_r = (np.cumsum(keep) - 1)[ent_r]
    cum = np.cumsum(ent_c)
    cum_before = cum - ent_c
    first = np.searchsorted(ent_r, np.arange(n_kept))
    base = cum_before[first]
    total = np.add.reduceat(ent_c, first)

    def sample(rng, reps):
        u = base[None, :] + (rng.random((reps, n_kept)) * total[None, :]).astype(np.int64)
        idx = np.searchsorted(cum, u.ravel(), side="right").reshape(u.shape)
        return ent_s[idx] + (u - cum_before[idx])

    return keep, sample


# --------------------------------------------------------------------------- #
# statistics
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
    """obs: (X,) overlap counts; null: (P, X) overlap counts from the permutations."""
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
    if len(df):
        df["q_enrich"] = bh(df.p_enrich.values)
        df["q_deplete"] = bh(df.p_deplete.values)
    return df


def pairwise_rows(stratum, n, sets, obs_set, obs_joint, null_joint):
    """obs_set (S,), obs_joint (S,S), null_joint (P,S,S) -- all overlap counts."""
    from scipy.stats import fisher_exact
    rows = []
    S = len(sets)
    for i in range(S):
        for j in range(i + 1, S):
            both = obs_joint[i, j]
            nj = null_joint[:, i, j]
            pe, pd_ = emp_p(both, nj)
            only_a, only_b = obs_set[i] - both, obs_set[j] - both
            neither = n - obs_set[i] - obs_set[j] + both
            orr, pf = fisher_exact([[both, only_a], [only_b, neither]])
            null_f = nj / n
            rows.append(dict(
                stratum=stratum, set_a=sets[i], set_b=sets[j], n_regions=n,
                n_both=int(both), frac_both=both / n,
                jaccard=both / max(1, obs_set[i] + obs_set[j] - both),
                null_mean_frac_both=null_f.mean(),
                fold_vs_null=(both / n) / null_f.mean() if null_f.mean() > 0 else np.nan,
                p_enrich=pe, p_deplete=pd_,
                fisher_or_within_regions=orr, fisher_p_within_regions=pf))
    return pd.DataFrame(rows)
