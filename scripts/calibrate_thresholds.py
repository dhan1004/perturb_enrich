#!/usr/bin/env python3
"""One-time (genome-wide) calibration of an 'active' cutoff for every bigwig in tracks.tsv.

ENCODE "signal p-value" bigwigs store -log10(p) per position. For those, the cutoff is the
smallest signal whose Benjamini-Hochberg q-value is <= --alpha, with every covered bp one test
(the same procedure MACS uses to build q-value tracks):

    sort positions by decreasing signal; rank_i = bp with signal >= value_i;
    q_i = p_i * N / rank_i  (monotone-corrected);  cutoff = lowest signal with q <= alpha.

For bigwigs that are NOT p-value tracks (e.g. fold-change), set signal_type to anything other
than 'pvalue' in the manifest (extra column `signal_type`); the cutoff is then the genome-wide
--quantile of covered bp. Default signal_type for rows without that column: --default-signal-type.

The signal is histogrammed (--nbins bins over [0, maxVal]) so memory stays flat; bin lower
edges are used, which is slightly conservative.

Output (TSV, one row per bigwig, keyed by normalised path) is read by enrich_gene.py.

usage: calibrate_thresholds.py --manifest tracks.tsv --out thresholds.tsv [--alpha 0.05]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_lib import norm_path, read_manifest  # noqa: E402

MAIN_CHROM = re.compile(r"^(chr)?([0-9]+|X|Y)$")


def log(msg):
    print(f"[calibrate] {msg}", file=sys.stderr, flush=True)


def histogram(path, nbins):
    bw = pyBigWig.open(path)
    vmax = float(bw.header()["maxVal"])
    if not np.isfinite(vmax) or vmax <= 0:
        raise SystemExit(f"{path}: bigwig has no positive signal (maxVal={vmax})")
    width = vmax / nbins
    hist = np.zeros(nbins, dtype=np.float64)
    for chrom in bw.chroms():
        if not MAIN_CHROM.match(chrom):
            continue
        ivs = bw.intervals(chrom)
        if not ivs:
            continue
        a = np.asarray(ivs, dtype=np.float64)
        v = np.clip(a[:, 2], 0, None)
        idx = np.minimum((v / width).astype(np.int64), nbins - 1)
        hist += np.bincount(idx, weights=a[:, 1] - a[:, 0], minlength=nbins)
        log(f"  {chrom}: {len(a):,} intervals")
    bw.close()
    return hist, width, vmax


def cutoff_qvalue(hist, width, alpha):
    edges = np.arange(len(hist)) * width              # lower edge of each bin = -log10(p)
    order = np.arange(len(hist))[::-1]                # most significant first
    cum = np.cumsum(hist[order])                      # bp with signal >= edge
    N = cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cum > 0, (10.0 ** (-edges[order])) * N / cum, np.inf)
    q = np.minimum.accumulate(q[::-1])[::-1]          # monotone BH
    ok = np.flatnonzero((q <= alpha) & (hist[order] > 0))
    if len(ok) == 0:
        return np.inf, 0.0, N
    last = ok.max()                                    # least significant bin still passing
    return edges[order][last], cum[last], N


def cutoff_quantile(hist, width, quantile):
    cum = np.cumsum(hist) / hist.sum()
    i = int(np.searchsorted(cum, quantile))
    thr = i * width
    return thr, hist[i:].sum(), hist.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05, help="BH q-value cutoff for p-value tracks [0.05]")
    ap.add_argument("--quantile", type=float, default=0.99, help="genome-wide quantile for non-p-value tracks [0.99]")
    ap.add_argument("--default-signal-type", default="pvalue")
    ap.add_argument("--nbins", type=int, default=200000)
    args = ap.parse_args()

    man = read_manifest(args.manifest)
    if "signal_type" not in man.columns:
        man["signal_type"] = args.default_signal_type
    man["signal_type"] = man["signal_type"].replace("", args.default_signal_type)
    bws = man[(man["type"] == "bigwig") & (man["role"] != "mask")]
    rows, seen = [], set()
    for _, r in bws.iterrows():
        path = norm_path(r["path"])
        if path in seen:
            continue
        seen.add(path)
        log(f"{r['name']}: {path} ({r['signal_type']})")
        hist, width, vmax = histogram(path, args.nbins)
        if r["signal_type"] == "pvalue":
            thr, active, total = cutoff_qvalue(hist, width, args.alpha)
            method, param = "BH_q", args.alpha
        else:
            thr, active, total = cutoff_quantile(hist, width, args.quantile)
            method, param = "quantile", args.quantile
        log(f"  threshold={thr:.4g} ({method}={param}); {100 * active / total:.2f}% of covered bp active")
        if not np.isfinite(thr):
            log("  WARNING: nothing reaches the cutoff; every region will be 0 for this track")
        rows.append(dict(name=r["name"], path=path, signal_type=r["signal_type"], method=method,
                         param=param, threshold=thr, bp_total=int(total), bp_active=int(active),
                         frac_active=active / total, track_max=vmax))
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    log(f"wrote {args.out} ({len(rows)} bigwigs)")


if __name__ == "__main__":
    main()
