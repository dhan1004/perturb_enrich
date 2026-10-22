#!/usr/bin/env python3
"""Compare an unmasked HARI scan with a conditional (masked-background) scan of the same gene.

For window w and mask A, per scramble r (scrambles are paired because window seeds are
coordinate-derived and identical across the two scans):

    delta_ref[w,r]    = f(w scrambled)      - f(intact)
    delta_masked[w,r] = f(A and w scrambled) - f(A scrambled)
    epsilon[w,r]      = delta_masked - delta_ref
                      = f(A,w) - f(A) - f(w) + f(intact)        (deviation from additivity)

f is the model's output scale (study-adjusted expression, ~log scale), so epsilon = 0 means
*multiplicative* on raw expression. Windows overlapping the mask are reported but flagged
(`overlaps_mask`): the mask already scrambled them, so they are not interpretable as epistasis.

usage:
  python epistasis_compare.py --ref-dir RUN/ENSG... --masked-dir RUN_maskA/ENSG... \
      --cell-type Astrocyte --threshold 0.10 --out epi_ENSG....tsv [--regions-tsv .../regions.tsv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STAT = {"target-sd": 1, "adjusted": 0}   # axis 2 of scan_state.npz["effects"]
MUST_MATCH = ["gene_id", "chrom", "tss0", "seed", "shuffle", "window_bp", "stride_bp", "repeats",
              "bundle_sha256", "covariate_mode", "context_population", "max_contexts", "cell_types",
              "input_start0", "embedding_dtype"]


def load(gene_dir):
    gene_dir = Path(gene_dir)
    meta = json.loads((gene_dir / "scan_metadata.json").read_text())
    with np.load(gene_dir / "scan_state.npz", allow_pickle=False) as z:
        state = {k: z[k].copy() for k in z.files}
    if not state["done"].all():
        raise ValueError(f"{gene_dir}: scan incomplete")
    return meta, state


def compare(ref_meta, ref, msk_meta, msk, cell, units="target-sd", threshold=0.10):
    for k in MUST_MATCH:
        if ref_meta.get(k) != msk_meta.get(k):
            raise ValueError(f"Scans differ in {k!r}: {ref_meta.get(k)!r} vs {msk_meta.get(k)!r}. "
                             "Epistasis needs identical model, seed, window grid, shuffle and contexts.")
    if ref_meta.get("mask_intervals"):
        raise ValueError("--ref-dir must be the unmasked scan")
    if not msk_meta.get("mask_intervals"):
        raise ValueError("--masked-dir was not run with --mask-bed")
    c = ref_meta["cell_types"].index(cell)
    k = STAT[units]
    base = int(ref_meta["input_start0"])
    ref_key = {(int(a), int(b)): i for i, (a, b) in enumerate(ref["windows"])}
    rows = []
    mask = msk_meta["mask_intervals"]
    for j, (a, b) in enumerate(msk["windows"]):
        i = ref_key.get((int(a), int(b)))
        if i is None:
            continue   # window not in the unmasked grid (cannot be paired)
        g0, g1 = base + int(a), base + int(b)
        overlaps = any(g0 < e and g1 > s for s, e in mask)
        ok = ref["status"][i] == "tested" and msk["status"][j] == "tested"
        row = dict(gene_id=ref_meta["gene_id"], cell_type=cell, chrom=ref_meta["chrom"], start0=g0, end0=g1,
                   overlaps_mask=overlaps, both_tested=bool(ok), effect_units=units)
        if ok:
            d_ref, d_msk = ref["effects"][i, :, k, c], msk["effects"][j, :, k, c]
            eps = d_msk - d_ref
            # Pairing sanity: outside the mask the scramble must be bit-identical in both scans.
            row["pairing_ok"] = bool(overlaps or np.array_equal(ref["changed_bp"][i], msk["changed_bp"][j]))
            row.update(delta_ref=d_ref.mean(), delta_masked=d_msk.mean(), epsilon=eps.mean(),
                       epsilon_sd_scrambles=eps.std(ddof=1) if len(eps) > 1 else np.nan,
                       abs_ref=np.abs(d_ref).mean(), abs_masked=np.abs(d_msk).mean(),
                       n_scrambles=len(eps))
            row["crosses_up"] = bool(row["abs_ref"] <= threshold < row["abs_masked"])
            row["crosses_down"] = bool(row["abs_masked"] <= threshold < row["abs_ref"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No overlapping windows between the two scans")
    return frame


def summarize_regions(frame, regions):
    """Window means per called region. Mean of window epsilons is NOT a region-level knockout."""
    out = []
    for r in regions.itertuples(index=False):
        w = frame[(frame.start0 < r.end0) & (frame.end0 > r.start0) & frame.both_tested & ~frame.overlaps_mask]
        out.append(dict(region=f"{r.chrom}:{r.start0}-{r.end0}", direction=r.direction, n_windows=len(w),
                        mean_delta_ref=w.delta_ref.mean() if len(w) else np.nan,
                        mean_delta_masked=w.delta_masked.mean() if len(w) else np.nan,
                        mean_epsilon=w.epsilon.mean() if len(w) else np.nan,
                        n_overlapping_mask=int(((frame.start0 < r.end0) & (frame.end0 > r.start0) & frame.overlaps_mask).sum())))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-dir", required=True, help="gene directory of the unmasked scan")
    p.add_argument("--masked-dir", required=True, help="gene directory of the --mask-bed scan")
    p.add_argument("--cell-type", required=True)
    p.add_argument("--units", choices=list(STAT), default="target-sd")
    p.add_argument("--threshold", type=float, default=0.10, help="same effect cutoff used to call regions")
    p.add_argument("--regions-tsv", help="regions.tsv from the unmasked scan, for a per-region summary")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rm, rs = load(a.ref_dir)
    mm, ms = load(a.masked_dir)
    frame = compare(rm, rs, mm, ms, a.cell_type, a.units, a.threshold)
    frame.to_csv(a.out, sep="\t", index=False, na_rep="NA")
    tested = frame[frame.both_tested & ~frame.overlaps_mask]
    print(f"{len(frame)} windows, {len(tested)} pairable outside the mask; pairing failures: "
          f"{int((~tested.pairing_ok).sum()) if len(tested) else 0}")
    if len(tested):
        print(f"gained (|effect| crosses up): {int(tested.crosses_up.sum())}   lost: {int(tested.crosses_down.sum())}")
    ke = Path(a.masked_dir) / "mask_effect.json"
    if ke.exists():
        d = json.loads(ke.read_text())
        i = d["cell_types"].index(a.cell_type)
        key = "knockout_effect_target_sd" if a.units == "target-sd" else "knockout_effect_adjusted"
        print(f"mask alone changes predicted expression by {d[key][i]:+.4f} ({a.units}) in {a.cell_type}")
    if a.regions_tsv:
        regions = pd.read_csv(a.regions_tsv, sep="\t")
        summary = summarize_regions(frame, regions)
        summary.to_csv(Path(a.out).with_suffix(".regions.tsv"), sep="\t", index=False, na_rep="NA")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
