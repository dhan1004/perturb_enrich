#!/usr/bin/env python3
"""Robustness statistics on top of the pooled permutation enrichment (run after aggregate_enrich.py).

The pooled test in aggregate_enrich.py treats every perturbed region as an independent draw and
weights regions by length. Two things can make it look more convincing than it is: regions that
cluster within genes (the permutation null places them independently), and a few very long regions
dominating the base-pair totals. This script asks the same question with the GENE as the replicate.

Reads $OUT_ROOT/genes/*/enrich_counts.npz (coverage-aware enrich_gene.py output) and, for the length
and region-weighted tables, enrichment/region_concordance.tsv.

Outputs in --outdir (default <out-root>/enrichment):
  gene_level_enrichment.tsv   per stratum x annotation set:
        pooled bp-weighted fold enrichment (same number as enrichment_cov_by_set.tsv) with a GENE
        BOOTSTRAP CI (genes resampled with replacement), and a Wilcoxon signed-rank test of the
        per-gene log2 enrichment against 0 (one value per gene; genes with zero difference drop out).
        q_* = BH across the annotation sets within a stratum.
  stratum_contrasts.tsv       is enrichment different between two levels of the same factor (e.g.
        direction=increase vs direction=decrease)? Difference in log2 enrichment tested against the
        permutation distribution of that difference (placements are paired by permutation index).
        q = BH across all contrast rows.
  null_distributions.tsv      the pooled null: covered-bp fraction for every permutation, per stratum x
        set (histograms with the observed value marked)
  region_length_summary.tsv   region-length distribution and share of perturbed bp per stratum
  region_weighted_enrichment.tsv
        the same enrichment with every REGION weighted equally (mean per-region covered fraction vs the
        mean of its own expected fraction), with a gene-bootstrap CI. Needs exp_cov_bp columns in
        region_concordance.tsv (genes run with the per-region test).

Caveats: per-gene values are noisy for genes with one short region; the signed-rank test weights every
gene equally. The bootstrap resamples genes, so it also reflects between-gene heterogeneity.
"""
import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_lib import bh  # noqa: E402

PSEUDO = 0.5   # bp pseudo-count, same as cov_rows() in enrich_lib.py
STRATUM_COLS = ("direction", "annotation", "is_distal_enh", "len_bin")


def log(msg):
    print(f"[gene_level] {msg}", file=sys.stderr, flush=True)


def l2(obs, exp):
    return np.log2((obs + PSEUDO) / (exp + PSEUDO))


def signed_rank(x, alternative):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if (x != 0).sum() < 5:
        return np.nan
    try:
        return float(wilcoxon(x, zero_method="wilcox", alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def boot_ratio(num, den, idx):
    """pooled ratio sum(num)/sum(den) for each bootstrap resample (idx: B x G gene indices)."""
    n, d = num[idx].sum(axis=1), den[idx].sum(axis=1)
    out = np.full(len(n), np.nan)
    ok = d > 0
    out[ok] = n[ok] / d[ok]
    return out


def ci_cols(b, point):
    b = b[np.isfinite(b)]
    if len(b) < 20:
        return dict(boot_ci_lo=np.nan, boot_ci_hi=np.nan, boot_p_le1=np.nan)
    return dict(boot_ci_lo=float(np.percentile(b, 2.5)), boot_ci_hi=float(np.percentile(b, 97.5)),
                boot_p_le1=float((1 + (b <= 1).sum()) / (1 + len(b))))


def load_counts(root):
    paths = sorted(root.glob("genes/*/enrich_counts.npz"))
    if not paths:
        sys.exit(f"[gene_level] no enrich_counts.npz under {root}/genes")
    genes = defaultdict(list)       # stratum -> [(gene, n_regions, obs_len, obs_cov[S], exp_cov[S])]
    pooled = {}                     # stratum -> dict(obs, null, len, n)
    ref, skipped = None, []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if "obs_cov" not in d.files:
            skipped.append(str(d["gene"]))
            continue
        if ref is None:
            ref = dict(sets=list(d["set_names"]), nperm=int(d["nperm"]))
        elif list(d["set_names"]) != ref["sets"] or int(d["nperm"]) != ref["nperm"]:
            sys.exit(f"[gene_level] {p}: set_names / nperm differ from first gene")
        gene = str(d["gene"])
        for j, nm in enumerate(d["strata_names"]):
            nj = int(d["n"][j])
            if nj == 0:
                continue
            nm = str(nm)
            oc = d["obs_cov"][j].astype(np.int64)
            nc = d["null_cov"][:, j, :].astype(np.int64)
            genes[nm].append((gene, nj, int(d["obs_len"][j]), oc, nc.mean(axis=0)))
            if nm not in pooled:
                pooled[nm] = dict(obs=np.zeros_like(oc), null=np.zeros_like(nc), len=0, n=0)
            pooled[nm]["obs"] += oc
            pooled[nm]["null"] += nc
            pooled[nm]["len"] += int(d["obs_len"][j])
            pooled[nm]["n"] += nj
    if ref is None:
        sys.exit("[gene_level] no gene has coverage counts; rerun enrich_gene.py (coverage-aware version)")
    if skipped:
        log(f"{len(skipped)} gene(s) lack coverage counts and are ignored, e.g. {skipped[:3]}")
    log(f"{len(paths) - len(skipped)} genes, {len(genes)} strata, {len(ref['sets'])} sets, nperm={ref['nperm']}")
    return ref["sets"], ref["nperm"], genes, pooled


def gene_level_table(sets, genes, pooled, a, rng):
    rows = []
    for nm in sorted(genes, key=lambda s: (s != "all", s)):
        pool = pooled[nm]
        gl = genes[nm]
        if pool["n"] < a.min_n or len(gl) < a.min_genes:
            continue
        obs = np.array([g[3] for g in gl], dtype=float)    # G x S
        exp = np.array([g[4] for g in gl], dtype=float)    # G x S
        G = len(gl)
        idx = rng.integers(0, G, size=(a.nboot, G))
        block = []
        for s, sname in enumerate(sets):
            gl2 = l2(obs[:, s], exp[:, s])
            point = obs[:, s].sum() / exp[:, s].sum() if exp[:, s].sum() > 0 else np.nan
            row = dict(stratum=nm, annotation=sname, n_regions=pool["n"], n_genes=G,
                       perturbed_bp=pool["len"], fold_enrichment=point,
                       log2_enrichment=float(np.log2(point)) if point and point > 0 else np.nan)
            row.update(ci_cols(boot_ratio(obs[:, s], exp[:, s], idx), point))
            row.update(median_gene_log2=float(np.median(gl2)),
                       frac_genes_enriched=float((gl2 > 0).mean()),
                       n_genes_nonzero=int((gl2 != 0).sum()),
                       p_signed_rank_greater=signed_rank(gl2, "greater"),
                       p_signed_rank_less=signed_rank(gl2, "less"),
                       p_signed_rank_two_sided=signed_rank(gl2, "two-sided"))
            block.append(row)
        df = pd.DataFrame(block)
        for c in ("p_signed_rank_greater", "p_signed_rank_less", "p_signed_rank_two_sided"):
            df["q" + c[1:]] = bh(df[c].values)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def contrast_table(sets, pooled, a):
    by_factor = defaultdict(list)
    for nm, pool in pooled.items():
        if "=" in nm and pool["n"] >= a.min_n:
            by_factor[nm.split("=", 1)[0]].append(nm)
    rows = []
    for factor, levels in by_factor.items():
        for A, B in combinations(sorted(levels), 2):
            pa, pb = pooled[A], pooled[B]
            P = pa["null"].shape[0]
            for s, sname in enumerate(sets):
                ma, mb = pa["null"][:, s].mean(), pb["null"][:, s].mean()
                ea, eb = l2(pa["obs"][s], ma), l2(pb["obs"][s], mb)
                dn = l2(pa["null"][:, s], ma) - l2(pb["null"][:, s], mb)   # paired by permutation
                c_null = dn - dn.mean()
                c_obs = (ea - eb) - dn.mean()
                rows.append(dict(
                    factor=factor, stratum_a=A, stratum_b=B, annotation=sname,
                    n_regions_a=pa["n"], n_regions_b=pb["n"],
                    log2_enrichment_a=float(ea), log2_enrichment_b=float(eb),
                    delta_log2_enrichment=float(ea - eb),
                    p_a_greater=float((1 + (c_null >= c_obs).sum()) / (1 + P)),
                    p_two_sided=float((1 + (np.abs(c_null) >= abs(c_obs)).sum()) / (1 + P))))
    df = pd.DataFrame(rows)
    if len(df):
        df["q_two_sided"] = bh(df.p_two_sided.values)
    return df


def null_dump(sets, pooled, a):
    parts = []
    for nm, pool in sorted(pooled.items()):
        if pool["n"] < a.min_n or pool["len"] <= 0:
            continue
        P = pool["null"].shape[0]
        for s, sname in enumerate(sets):
            parts.append(pd.DataFrame(dict(
                stratum=nm, annotation=sname, perm=np.arange(P),
                null_frac=pool["null"][:, s] / pool["len"],
                obs_frac=pool["obs"][s] / pool["len"])))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def region_tables(sets, root, a, rng):
    path = root / "enrichment" / "region_concordance.tsv" if not a.regions else Path(a.regions)
    if not path.exists():
        log(f"{path} not found; skipping length and region-weighted tables")
        return pd.DataFrame(), pd.DataFrame()
    rc = pd.read_csv(path, sep="\t", dtype={"chrom": str, "gene_id": str}, low_memory=False)
    masks = {"all": np.ones(len(rc), dtype=bool)}
    for col in STRATUM_COLS:
        if col in rc.columns:
            vals = rc[col].astype(str)
            for v in vals.unique():
                masks[f"{col}={v}"] = (vals == v).values
    total_bp = rc["length"].sum()
    lrows, wrows = [], []
    have_exp = all(f"exp_cov_bp:{s}" in rc.columns for s in sets)
    if not have_exp:
        log("exp_cov_bp columns missing in region_concordance.tsv (genes predate the per-region test); "
            "region-weighted table skipped")
    for nm in sorted(masks, key=lambda s: (s != "all", s)):
        sub = rc[masks[nm]]
        if len(sub) < a.min_n:
            continue
        L = sub["length"]
        lrows.append(dict(stratum=nm, n_regions=len(sub), n_genes=sub["gene_id"].nunique(),
                          total_bp=int(L.sum()), share_of_perturbed_bp=float(L.sum() / total_bp),
                          mean_bp=float(L.mean()), median_bp=float(L.median()),
                          q10_bp=float(L.quantile(.1)), q90_bp=float(L.quantile(.9)), max_bp=int(L.max())))
        if not have_exp:
            continue
        sub = sub[np.isfinite(sub[[f"exp_cov_bp:{s}" for s in sets]]).all(axis=1)]
        if len(sub) < a.min_n:
            continue
        gid = sub["gene_id"].values
        genes, inv = np.unique(gid, return_inverse=True)
        G = len(genes)
        if G < a.min_genes:
            continue
        idx = rng.integers(0, G, size=(a.nboot, G))
        Ls = sub["length"].values.astype(float)
        for s, sname in enumerate(sets):
            o = np.bincount(inv, weights=sub[f"cov_bp:{sname}"].values / Ls, minlength=G)
            e = np.bincount(inv, weights=sub[f"exp_cov_bp:{sname}"].values / Ls, minlength=G)
            point = o.sum() / e.sum() if e.sum() > 0 else np.nan
            row = dict(stratum=nm, annotation=sname, n_regions=len(sub), n_genes=G,
                       mean_frac_covered=float((sub[f"cov_bp:{sname}"].values / Ls).mean()),
                       mean_frac_expected=float((sub[f"exp_cov_bp:{sname}"].values / Ls).mean()),
                       fold_enrichment_region_weighted=point)
            row.update(ci_cols(boot_ratio(o, e, idx), point))
            wrows.append(row)
    return pd.DataFrame(lrows), pd.DataFrame(wrows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="$OUT_ROOT (contains genes/<gene>/)")
    ap.add_argument("--outdir", default="", help="default <out-root>/enrichment")
    ap.add_argument("--regions", default="", help="region_concordance.tsv (default <outdir>/region_concordance.tsv)")
    ap.add_argument("--min-n", type=int, default=20, help="min pooled regions for a stratum [20]")
    ap.add_argument("--min-genes", type=int, default=10, help="min genes with regions in a stratum [10]")
    ap.add_argument("--nboot", type=int, default=2000, help="gene-bootstrap resamples [2000]")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    root = Path(a.out_root)
    outdir = Path(a.outdir) if a.outdir else root / "enrichment"
    outdir.mkdir(parents=True, exist_ok=True)
    if not a.regions:
        a.regions = str(outdir / "region_concordance.tsv")
    rng = np.random.default_rng(a.seed)

    sets, P, genes, pooled = load_counts(root)

    gl = gene_level_table(sets, genes, pooled, a, rng)
    if len(gl):
        gl.to_csv(outdir / "gene_level_enrichment.tsv", sep="\t", index=False)
    else:
        log(f"no stratum reached --min-n {a.min_n} and --min-genes {a.min_genes}")

    ct = contrast_table(sets, pooled, a)
    if len(ct):
        ct.to_csv(outdir / "stratum_contrasts.tsv", sep="\t", index=False)
    else:
        log("no factor with >= 2 levels reaching --min-n; stratum_contrasts.tsv not written")

    nd = null_dump(sets, pooled, a)
    if len(nd):
        nd.to_csv(outdir / "null_distributions.tsv", sep="\t", index=False)

    lens, rw = region_tables(sets, root, a, rng)
    if len(lens):
        lens.to_csv(outdir / "region_length_summary.tsv", sep="\t", index=False)
    if len(rw):
        rw.to_csv(outdir / "region_weighted_enrichment.tsv", sep="\t", index=False)
    log(f"wrote {outdir}")


if __name__ == "__main__":
    main()
