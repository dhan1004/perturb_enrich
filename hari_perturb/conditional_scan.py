#!/usr/bin/env python3
"""Region-knockout (epistasis) scans on top of a completed HARI scan.

For gene G and every hit region A in <ref-root>/G/results/<cell>/regions.tsv:

  1. scramble A, independently --n-shuffles times (so one shuffle that happens to create a
     motif cannot drive a result),
  2. rerun the HARI window scan on that masked background, using the model, seeds, window grid,
     shuffle method and covariate contexts recorded by the ORIGINAL scan (read from its
     scan_metadata.json, so the two scans cannot silently disagree),
  3. epsilon[w] = delta_masked[w] - delta_ref[w], per window scramble (paired: identical window
     scrambles in both scans), i.e. f(A,w) - f(A) - f(w) + f(intact).

The model is loaded once per gene and every condition runs in-process. Each condition is
checkpointed per window, so a killed job resumes where it stopped (re-run the same command).

Default scope scans only windows overlapping hit regions plus --n-control-windows random
non-hit windows (cheap). --scope all rescans every tested window (about the cost of the original
scan per condition) and is the only way to see regions that APPEAR after masking.
--n-null N adds N size/GC-matched scrambled non-hit regions per hit as a null for epsilon.

  python conditional_scan.py run --ref-root RUN --out-root OUT --fasta GRCh38.fa --suite-root SUITE \
      --gene ENSG00000142599 --trust-pickle [--dry-run]
  python conditional_scan.py run ... --genes-tsv genes.tsv [--task-index 3]
  python conditional_scan.py collate --out-root OUT

Units follow the original scan's regions call (log.json effect_units); epsilon is on the model's
output scale (log-like), so epsilon = 0 means multiplicative effects on raw expression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core import (IndexedFasta, safe_name, canonical_hash, sha256_file, seed_for, onehot,
                  shuffle_window, mask_sequence)
from scan_hari import atomic_json, atomic_npz, read_state

LOG = logging.getLogger("hari_conditional")
VERSION = "1.0.0"
STAT = {"adjusted": 0, "target-sd": 1}          # index on the stat axis, in the original and in our cache
EFFECT_COL = {"adjusted": "knockout_effect_adjusted", "target-sd": "knockout_effect_target_sd"}
CONTROL = "CONTROL_WINDOWS"
_ENGINES = {}


# ----------------------------------------------------------------------------- inputs
def read_gene_ids(path):
    """gene_id column (if a header names one) or first column; blank/# lines ignored; order kept."""
    ids, col = [], None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = [c.strip() for c in line.rstrip("\n").split("\t")]
            if col is None:
                col = cells.index("gene_id") if "gene_id" in cells else 0
                if "gene_id" in cells:
                    continue          # header row
            if col < len(cells) and cells[col] and cells[col] not in ids:
                ids.append(cells[col])
    if not ids:
        raise ValueError(f"No gene IDs in {path}")
    return ids


def load_reference(ref_root, gene, cell):
    gdir = Path(ref_root) / safe_name(gene)
    if not (gdir / "scan_metadata.json").exists():
        raise FileNotFoundError(f"No original scan for {gene!r} under {ref_root}. Gene IDs must match the scan directory "
                                "names (Ensembl IDs as in targets.tsv, including any version suffix), not symbols.")
    meta = json.loads((gdir / "scan_metadata.json").read_text())
    if meta["gene_id"] != gene:
        raise ValueError(f"{gdir} holds {meta['gene_id']}, not {gene}")
    if meta.get("mask_intervals"):
        raise ValueError("The reference scan is itself masked; point --ref-root at an unmasked scan")
    if cell not in meta["cell_types"]:
        raise ValueError(f"{cell!r} not in scan cell types {meta['cell_types']}")
    if not (gdir / "completion.json").exists():
        raise ValueError(f"{gdir}: original scan did not complete")
    state = read_state(gdir / "scan_state.npz")
    if not state["done"].all():
        raise ValueError(f"{gdir}: original scan is incomplete")
    cdir = gdir / "results" / safe_name(cell)
    regions = pd.read_csv(cdir / "regions.tsv", sep="\t")
    log = json.loads((cdir / "log.json").read_text())
    return gdir, meta, state, regions, log


def hit_table(regions):
    h = regions.sort_values("start0").reset_index(drop=True).copy()
    h.insert(0, "region_id", [f"R{i:02d}" for i in range(1, len(h) + 1)])
    return h


def select_windows(meta, state, hits_all, hits_sel, scope, n_controls, flank):
    """Indices (into the original scan's windows) to rescan, and a control flag for each."""
    base = int(meta["input_start0"])
    w = state["windows"] + base
    ok = state["status"] == "tested"

    def near(table):
        m = np.zeros(len(w), bool)
        for r in table.itertuples():
            m |= (w[:, 0] < r.end0 + flank) & (w[:, 1] > r.start0 - flank)
        return m
    near_all, near_sel = near(hits_all), near(hits_sel)
    if scope == "all":
        sel = np.where(ok)[0]
        return sel, ~near_all[sel]
    pool = np.where(ok & ~near_all)[0]
    rng = np.random.default_rng(seed_for(meta["seed"], meta["gene_id"], 0, 0, "controls"))
    ctrl = np.sort(rng.choice(pool, min(n_controls, len(pool)), replace=False)) if n_controls and len(pool) else np.array([], int)
    sel = np.union1d(np.where(ok & near_sel)[0], ctrl).astype(int)
    return sel, np.isin(sel, ctrl)


def draw_null_masks(seq, input_start, forbidden, length, gc, n, rng, tol, max_tries=5000):
    """n same-length, ACGT-only intervals with GC within `tol` of `gc`, avoiding `forbidden` and each other."""
    out, tries, total = [], 0, len(seq)
    while len(out) < n and length < total and tries < max_tries:
        tries += 1
        a = int(rng.integers(0, total - length + 1))
        s, e = input_start + a, input_start + a + length
        if any(s < fe and e > fs for fs, fe in list(forbidden) + out):
            continue
        block = seq[a:a + length]
        if set(block) - set(b"ACGT") or abs((block.count(b"G") + block.count(b"C")) / length - gc) > tol:
            continue
        out.append((s, e))
    return out


def plan_conditions(gene, hits, seq, input_start, forbidden, args):
    conds = []
    for h in hits.itertuples():
        start0, end0 = int(h.start0), int(h.end0)
        for k in range(args.n_shuffles):
            conds.append(dict(cond_id=f"{h.region_id}_rep{k}", kind="hit", mask_id=h.region_id,
                              matched_to=h.region_id, chrom=h.chrom, start0=start0, end0=end0, rep=k))
        if args.n_null:
            a, b = max(start0 - input_start, 0), min(end0 - input_start, len(seq))
            block = seq[a:b]
            gc = (block.count(b"G") + block.count(b"C")) / max(len(block), 1)
            rng = np.random.default_rng(seed_for(args.mask_seed, gene, start0, end0, "null"))
            nulls = draw_null_masks(seq, input_start, forbidden, end0 - start0, gc, args.n_null, rng, args.null_gc_tol)
            if len(nulls) < args.n_null:
                LOG.warning("%s %s: only %d/%d GC-matched null regions found", gene, h.region_id, len(nulls), args.n_null)
            for j, (s, e) in enumerate(nulls, 1):
                for k in range(args.n_shuffles):
                    conds.append(dict(cond_id=f"{h.region_id}_null{j}_rep{k}", kind="null", mask_id=f"{h.region_id}_null{j}",
                                      matched_to=h.region_id, chrom=h.chrom, start0=s, end0=e, rep=k))
    return conds


# ----------------------------------------------------------------------------- scanning
def condition_fingerprint(meta, cond, args, mask_shuffle, n_rep, windows):
    return canonical_hash(dict(
        version=VERSION, ref=meta["fingerprint"], cell=args.cell_type, mask_seed=args.mask_seed,
        mask_shuffle=mask_shuffle, n_rep=n_rep, windows=np.asarray(windows).tolist(),
        cond={k: cond[k] for k in ("cond_id", "kind", "mask_id", "start0", "end0", "rep")}))


def run_condition(ctx, cond, bg_seq, state_path, fingerprint):
    """Window scan on `bg_seq` (masked reference). Returns the (checkpointed) state dict.

    effects[w, r] = (adjusted, target-sd) delta of window scramble r, paired against the masked
    background at identical covariate profiles. mask_effect = f(masked) - f(intact).
    """
    nsel, n_rep = len(ctx.windows), ctx.n_rep
    state = None
    if state_path.exists():
        state = read_state(state_path)
        if str(state["fingerprint"]) != fingerprint or not np.array_equal(state["windows"], ctx.windows):
            raise ValueError(f"{state_path}: cached condition does not match this run; use a new --out-root")
        if state["done"].all():
            return state
    bg_emb = ctx.embed(bg_seq)
    ctx.ensemble.set_reference(ctx.intact_emb)
    mask_effect = np.asarray(ctx.ensemble.delta(bg_emb))[:2, 0].astype(float)
    ctx.ensemble.set_reference(bg_emb)
    if not np.allclose(ctx.ensemble.delta(bg_emb), 0, atol=1e-8, rtol=0):
        raise ValueError("No-op check failed on the masked background: inference is not deterministic")
    if state is None:
        state = dict(fingerprint=np.array(fingerprint), windows=ctx.windows,
                     effects=np.full((nsel, n_rep, 2), np.nan), done=np.zeros(nsel, bool),
                     status=np.full(nsel, "pending", dtype="U32"),
                     changed_bp=np.zeros((nsel, n_rep), dtype=np.int32), mask_effect=mask_effect)
        atomic_npz(state_path, **state)
    for i, (a, b) in enumerate(ctx.windows):
        if state["done"][i]:
            continue
        a, b = int(a), int(b)
        block = bg_seq[a:b]
        if set(block) - set(b"ACGT"):
            state["status"][i] = "skipped_ambiguous_or_padding"
        else:
            for r in range(n_rep):
                rng = np.random.default_rng(seed_for(ctx.seed, ctx.gene, ctx.input_start + a, ctx.input_start + b, r))
                shuffled = shuffle_window(block, rng, ctx.shuffle)
                changes = sum(x != y for x, y in zip(block, shuffled))
                state["changed_bp"][i, r] = changes
                vector = ctx.embed(bg_seq[:a] + shuffled + bg_seq[b:]) if changes else bg_emb
                d = np.asarray(ctx.ensemble.delta(vector))
                state["effects"][i, r] = (d[0, 0], d[1, 0])
            state["status"][i] = "tested" if np.any(state["changed_bp"][i]) else "unchanged_all_scrambles"
        state["done"][i] = True
        atomic_npz(state_path, **state)
        if (i + 1) % 20 == 0 or i + 1 == nsel:
            LOG.info("  %s window %d/%d", cond["cond_id"], i + 1, nsel)
    return state


# ----------------------------------------------------------------------------- analysis
def build_window_table(gene, cell, meta, ref, c, sel, is_control, n_rep, units, threshold, conds, states):
    k, base = STAT[units], int(meta["input_start0"])
    ref_eff = ref["effects"][sel][:, :n_rep, k, c]
    ref_changed = ref["changed_bp"][sel][:, :n_rep]
    ref_ok = ref["status"][sel] == "tested"
    frames = []
    for cond in conds:
        st = states[cond["cond_id"]]
        g0, g1 = base + st["windows"][:, 0], base + st["windows"][:, 1]
        over = np.zeros(len(g0), bool)
        for s, e in cond["applied"]:
            over |= (g0 < e) & (g1 > s)
        eff = st["effects"][:, :, k]
        ok = ref_ok & (st["status"] == "tested")
        eps = eff - ref_eff
        nan = np.full(len(g0), np.nan)
        sd = eps.std(axis=1, ddof=1) if n_rep > 1 else nan
        abs_ref, abs_msk = np.abs(ref_eff).mean(1), np.abs(eff).mean(1)
        frames.append(pd.DataFrame(dict(
            gene_id=gene, cell_type=cell, mask_id=cond["mask_id"], kind=cond["kind"], matched_to=cond["matched_to"],
            rep=cond["rep"], chrom=meta["chrom"], start0=g0, end0=g1, is_control=is_control, overlaps_mask=over,
            valid=ok, pairing_ok=np.where(ok, over | (st["changed_bp"] == ref_changed).all(axis=1), True),
            delta_ref=np.where(ok, ref_eff.mean(1), nan), delta_masked=np.where(ok, eff.mean(1), nan),
            epsilon=np.where(ok, eps.mean(1), nan), epsilon_sd_scrambles=np.where(ok, sd, nan),
            crosses_up=ok & (abs_ref <= threshold) & (abs_msk > threshold),
            crosses_down=ok & (abs_msk <= threshold) & (abs_ref > threshold), effect_units=units)))
    return pd.concat(frames, ignore_index=True)


def summarize_windows(wdf):
    use = wdf[wdf.valid & ~wdf.overlaps_mask].copy()
    cols = ["gene_id", "cell_type", "mask_id", "kind", "matched_to", "chrom", "start0", "end0", "is_control", "n_reps",
            "delta_ref", "delta_masked", "epsilon_mean", "epsilon_sd_reps", "sign_consistency",
            "frac_reps_up", "frac_reps_down", "effect_units"]
    if use.empty:
        return pd.DataFrame(columns=cols)
    key = ["gene_id", "cell_type", "mask_id", "chrom", "start0", "end0"]
    use["_m"] = use.groupby(key, sort=False).epsilon.transform("mean")
    use["_agree"] = np.sign(use.epsilon) == np.sign(use._m)
    out = use.groupby(key, sort=False).agg(
        kind=("kind", "first"), matched_to=("matched_to", "first"), is_control=("is_control", "first"),
        n_reps=("rep", "nunique"), delta_ref=("delta_ref", "mean"), delta_masked=("delta_masked", "mean"),
        epsilon_mean=("epsilon", "mean"), epsilon_sd_reps=("epsilon", "std"), sign_consistency=("_agree", "mean"),
        frac_reps_up=("crosses_up", "mean"), frac_reps_down=("crosses_down", "mean"),
        effect_units=("effect_units", "first")).reset_index()
    return out[cols]


def mask_effect_table(gene, cell, conds, states, units):
    rows = []
    for cond in conds:
        me = states[cond["cond_id"]]["mask_effect"]
        rows.append(dict(gene_id=gene, cell_type=cell, mask_id=cond["mask_id"], kind=cond["kind"], matched_to=cond["matched_to"],
                         rep=cond["rep"], chrom=cond["chrom"], mask_start0=cond["applied"][0][0] if cond["applied"] else None,
                         mask_end0=cond["applied"][-1][1] if cond["applied"] else None,
                         knockout_effect_adjusted=me[0], knockout_effect_target_sd=me[1]))
    return pd.DataFrame(rows)


def summarize_regions(wdf, targets, conds, mask_effects, units):
    """Per (mask, target region): mean epsilon over the target's windows, then over mask reshuffles."""
    use = wdf[wdf.valid & ~wdf.overlaps_mask]
    eff_col = EFFECT_COL[units]
    spans = {}
    for cond in conds:
        spans.setdefault(cond["mask_id"], cond)
    tlist = [(t.region_id, t.start0, t.end0) for t in targets.itertuples()] + [(CONTROL, None, None)]
    rows = []
    for mid, cond in spans.items():
        sub = use[use.mask_id == mid]
        me = mask_effects[mask_effects.mask_id == mid][eff_col]
        for tid, ts, te in tlist:
            if tid == cond["matched_to"] and cond["kind"] == "hit":
                continue
            t = sub[sub.is_control] if tid == CONTROL else sub[(sub.start0 < te) & (sub.end0 > ts)]
            if t.empty:
                continue
            per = t.groupby("rep").agg(eps=("epsilon", "mean"), dref=("delta_ref", "mean"), dmsk=("delta_masked", "mean"))
            m = per.eps.mean()
            rows.append(dict(
                gene_id=cond_gene(t), cell_type=t.cell_type.iloc[0], mask_id=mid, kind=cond["kind"], matched_to=cond["matched_to"],
                mask_chrom=cond["chrom"], mask_start0=cond["applied"][0][0], mask_end0=cond["applied"][-1][1],
                target_id=tid, target_start0=ts, target_end0=te, n_windows=t[["start0", "end0"]].drop_duplicates().shape[0],
                n_reps=len(per), delta_ref=per.dref.mean(), delta_masked=per.dmsk.mean(), epsilon_mean=m,
                epsilon_sd_reps=per.eps.std() if len(per) > 1 else np.nan,
                sign_consistency=float((np.sign(per.eps) == np.sign(m)).mean()),
                mask_effect_mean=me.mean(), mask_effect_sd_reps=me.std() if len(me) > 1 else np.nan, effect_units=units))
    table = pd.DataFrame(rows)
    if table.empty or not (table.kind == "null").any():
        return table
    nl = table[table.kind == "null"].groupby(["matched_to", "target_id"]).agg(
        n_null_masks=("mask_id", "nunique"), null_epsilon_mean=("epsilon_mean", "mean"),
        null_abs_epsilon_max=("epsilon_mean", lambda s: s.abs().max())).reset_index().rename(columns={"matched_to": "_hit"})
    table = table.merge(nl, left_on=["mask_id", "target_id"], right_on=["_hit", "target_id"], how="left").drop(columns="_hit")
    table["exceeds_all_nulls"] = np.where(table.kind == "hit", table.epsilon_mean.abs() > table.null_abs_epsilon_max, np.nan)
    return table


def cond_gene(frame):
    return frame.gene_id.iloc[0]


# ----------------------------------------------------------------------------- model
def make_model(meta, args, out_dir):
    """Load the SAME bundle/embedder/contexts the original scan used (all recorded in its metadata)."""
    if not args.trust_pickle:
        raise ValueError("Use --trust-pickle only for your own/trusted inference bundles; pickle can execute code")
    if not args.suite_root:
        raise ValueError("--suite-root is required to run the model")
    sys.path.insert(0, str(Path(args.suite_root).resolve()))
    from agsuite_adapter import load_bundle, SuiteEmbedder, Ensemble
    bundle_path = Path(meta["bundle_path"])
    if sha256_file(bundle_path) != meta["bundle_sha256"]:
        raise ValueError("Bundle file changed since the original scan; paired comparison would be invalid")
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_json = out_dir / "embedding_schema.json"
    atomic_json(schema_json, meta["embedding_schema"])
    bundle, schema, _ = load_bundle(bundle_path, meta["gene_id"], str(schema_json))
    context = None
    if meta.get("contexts_sha256"):
        if not args.contexts_tsv or sha256_file(args.contexts_tsv) != meta["contexts_sha256"]:
            raise ValueError("The original scan used an external --contexts-tsv; pass the identical file")
        context = pd.read_csv(args.contexts_tsv, sep="\t")[bundle["covariate_names"]].apply(pd.to_numeric, errors="raise")
    ensemble = Ensemble(bundle, [args.cell_type], meta["covariate_mode"], context, meta["max_contexts"], meta["seed"],
                        meta["head_batch_size"], meta["head_device"])
    key = canonical_hash(schema)
    if key not in _ENGINES:
        _ENGINES[key] = SuiteEmbedder(schema, meta["embedder_class"], args.allow_cpu_trunk)
    engine, dtype = _ENGINES[key], meta["embedding_dtype"]

    def embed(sequence):   # same precision round-trip as the original scan
        return engine.embed(onehot(sequence)).astype(dtype).astype(np.float32)
    return SimpleNamespace(embed=embed, ensemble=ensemble)


# ----------------------------------------------------------------------------- per gene
def run_gene(args, gene, model_factory=None):
    t0 = time.time()
    gdir, meta, ref, regions, log = load_reference(args.ref_root, gene, args.cell_type)
    c = meta["cell_types"].index(args.cell_type)
    n_rep = args.window_repeats or int(meta["repeats"])
    if not 1 <= n_rep <= int(meta["repeats"]):
        raise ValueError(f"--window-repeats must be 1..{meta['repeats']} (the original scan's repeat count)")
    units = log["effect_units"]
    threshold = float(args.threshold if args.threshold is not None else log["threshold"])
    out = Path(args.out_root) / safe_name(gene)
    out.mkdir(parents=True, exist_ok=True)
    hits_all = hit_table(regions)
    if hits_all.empty:
        atomic_json(out / "no_hits.json", dict(gene_id=gene, cell_type=args.cell_type, note="original scan called no regions"))
        LOG.info("%s: no hit regions in the original scan; nothing to do", gene)
        return None
    hits = hits_all.nlargest(args.max_regions, "mean_abs_effect").sort_values("start0") if args.max_regions else hits_all
    sel, is_control = select_windows(meta, ref, hits_all, hits, args.scope, args.n_control_windows, args.flank_bp)
    if len(sel) == 0:
        raise ValueError(f"{gene}: no testable windows selected")

    input_start = int(meta["input_start0"])
    length = int(meta["input_end0"]) - input_start
    seq, _ = IndexedFasta(args.fasta).fetch(meta["chrom"], input_start, input_start + length)
    if hashlib.sha256(seq).hexdigest() != meta["sequence_sha256"]:
        raise ValueError("FASTA sequence differs from the original scan's input; use the same reference file")
    tss0 = int(meta["tss0"])
    forbidden = [(int(h.start0), int(h.end0)) for h in hits_all.itertuples()] + [(tss0 - args.null_tss_pad, tss0 + args.null_tss_pad)]
    conds = plan_conditions(gene, hits, seq, input_start, forbidden, args)
    mask_shuffle = meta["shuffle"] if args.mask_shuffle == "same" else args.mask_shuffle
    windows = ref["windows"][sel]
    passes = len(conds) * (1 + len(sel) * n_rep) + 1
    atomic_json(out / "plan.json", dict(
        version=VERSION, gene_id=gene, cell_type=args.cell_type, effect_units=units, threshold=threshold, scope=args.scope,
        n_hits=len(hits), n_conditions=len(conds), n_windows=int(len(sel)), n_control_windows=int(is_control.sum()),
        window_repeats=n_rep, n_shuffles=args.n_shuffles, n_null=args.n_null, mask_shuffle=mask_shuffle,
        mask_seed=args.mask_seed, estimated_trunk_passes=passes, reference_scan=str(gdir),
        hits=hits[["region_id", "chrom", "start0", "end0"]].to_dict("records"),
        conditions=[{k: v for k, v in cd.items() if k != "applied"} for cd in conds]))
    LOG.info("%s: %d hit regions, %d conditions (%d reshuffles%s), %d windows x %d scrambles -> ~%d trunk passes",
             gene, len(hits), len(conds), args.n_shuffles, f" + {args.n_null} null/hit" if args.n_null else "",
             len(sel), n_rep, passes)
    if args.dry_run:
        return None

    cache = {}

    def model():
        if "m" not in cache:
            m = (model_factory or make_model)(meta, args, out)
            intact = m.embed(seq)
            m.ensemble.set_reference(intact)
            got, want = float(m.ensemble.reference[0]), float(ref["reference_predictions"][c])
            if not np.isclose(got, want, atol=args.ref_tol, rtol=args.ref_tol):
                raise ValueError(f"Cannot reproduce the original scan's intact prediction ({got!r} vs {want!r}); "
                                 "same bundle, suite source, GPU/runtime? Paired epsilons would be biased.")
            m.ctx = SimpleNamespace(embed=m.embed, ensemble=m.ensemble, intact_emb=intact, seed=int(meta["seed"]), gene=gene,
                                    input_start=input_start, shuffle=meta["shuffle"], windows=windows, n_rep=n_rep)
            cache["m"] = m
        return cache["m"].ctx

    (out / "conditions").mkdir(exist_ok=True)
    states = {}
    for n, cond in enumerate(conds, 1):
        bg, applied = mask_sequence(seq, input_start, [(cond["start0"], cond["end0"])], args.mask_seed, gene, mask_shuffle, cond["rep"])
        cond["applied"] = [[int(a), int(b)] for a, b in applied]
        path = out / "conditions" / f"{cond['cond_id']}.npz"
        fp = condition_fingerprint(meta, cond, args, mask_shuffle, n_rep, windows)
        if path.exists():
            state = read_state(path)
            if str(state["fingerprint"]) != fp or not np.array_equal(state["windows"], windows):
                raise ValueError(f"{path}: cached condition does not match this run; use a new --out-root")
            if state["done"].all():
                states[cond["cond_id"]] = state
                continue
        LOG.info("%s condition %d/%d %s (%s:%d-%d)", gene, n, len(conds), cond["cond_id"], cond["chrom"], cond["start0"], cond["end0"])
        states[cond["cond_id"]] = run_condition(model(), cond, bg, path, fp)

    wdf = build_window_table(gene, args.cell_type, meta, ref, c, sel, is_control, n_rep, units, threshold, conds, states)
    mask_effects = mask_effect_table(gene, args.cell_type, conds, states, units)
    targets = hits_all if args.scope == "all" else hits
    regions_tbl = summarize_regions(wdf, targets, conds, mask_effects, units)
    wdf.to_csv(out / "windows_by_condition.tsv.gz", sep="\t", index=False, na_rep="NA")
    summarize_windows(wdf).to_csv(out / "window_epsilon_summary.tsv.gz", sep="\t", index=False, na_rep="NA")
    mask_effects.to_csv(out / "mask_effects.tsv", sep="\t", index=False, na_rep="NA")
    regions_tbl.to_csv(out / "region_epistasis.tsv", sep="\t", index=False, na_rep="NA")
    bad = wdf[wdf.valid & ~wdf.overlaps_mask & ~wdf.pairing_ok]
    if len(bad):
        LOG.warning("%s: %d window rows are not paired with the original scan's scrambles (check seeds/grid); "
                    "epsilon is NOT interpretable for them", gene, len(bad))
    atomic_json(out / "completion.json", dict(version=VERSION, complete=True, elapsed_seconds=time.time() - t0,
                                              unpaired_window_rows=int(len(bad)), estimated_trunk_passes=passes))
    LOG.info("%s: done in %.1f min -> %s", gene, (time.time() - t0) / 60, out)
    return regions_tbl


def collate(out_root):
    out_root = Path(out_root)
    for name in ("region_epistasis.tsv", "mask_effects.tsv"):
        frames = [pd.read_csv(p, sep="\t") for p in sorted(out_root.glob(f"*/{name}"))]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_root / f"all_genes_{name}", sep="\t", index=False, na_rep="NA")
            LOG.info("wrote %s (%d genes)", out_root / f"all_genes_{name}", len(frames))


# ----------------------------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Mask every hit region of each gene, rescan, compute epsilon")
    col = sub.add_parser("collate", help="Concatenate per-gene tables under --out-root")
    col.add_argument("--out-root", required=True)
    run.add_argument("--ref-root", required=True, help="Output root of the ORIGINAL HARI scan (<gene>/scan_metadata.json, results/<cell>/regions.tsv)")
    run.add_argument("--out-root", required=True)
    run.add_argument("--fasta", required=True, help="Same uncompressed, .fai-indexed FASTA as the original scan")
    run.add_argument("--suite-root", help="agsuite root (required unless --dry-run)")
    g = run.add_mutually_exclusive_group(required=True)
    g.add_argument("--gene", help="One gene_id, exactly as the scan directory is named")
    g.add_argument("--genes-tsv", help="TSV with a gene_id column, or one gene_id per line")
    run.add_argument("--task-index", type=int, help="With --genes-tsv: 0-based row to process (SLURM array index)")
    run.add_argument("--cell-type", default="Astrocyte")
    run.add_argument("--n-shuffles", type=int, default=5, help="Independent reshuffles of each masked region")
    run.add_argument("--mask-seed", type=int, default=7)
    run.add_argument("--mask-shuffle", choices=["same", "mononucleotide", "dinucleotide"], default="same",
                     help="How the masked region is scrambled (same = the original scan's method)")
    run.add_argument("--window-repeats", type=int, help="Use only the first N window scrambles (paired with the original's first N); default all")
    run.add_argument("--scope", choices=["hits", "all"], default="hits",
                     help="hits: only windows overlapping hit regions + control windows (cheap). all: every tested window (can reveal regions that appear after masking)")
    run.add_argument("--n-control-windows", type=int, default=20, help="Random non-hit windows to detect global shifts (scope hits)")
    run.add_argument("--flank-bp", type=int, default=0, help="Also scan windows within this distance of a hit region")
    run.add_argument("--max-regions", type=int, help="Only the top-N hit regions by mean_abs_effect")
    run.add_argument("--n-null", type=int, default=0, help="Matched null masks per hit region (same length, GC within tolerance)")
    run.add_argument("--null-gc-tol", type=float, default=0.05)
    run.add_argument("--null-tss-pad", type=int, default=5000, help="Null masks avoid TSS +/- this many bp")
    run.add_argument("--threshold", type=float, help="Cutoff for crosses_up/down flags; default = the original region-calling cutoff")
    run.add_argument("--ref-tol", type=float, default=1e-4, help="Allowed difference when re-deriving the original intact prediction")
    run.add_argument("--contexts-tsv", help="Only if the original scan used one; must be the identical file")
    run.add_argument("--allow-cpu-trunk", action="store_true")
    run.add_argument("--trust-pickle", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Plan and print the cost; no model is loaded")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("agsuite.alphagenome").setLevel(logging.WARNING)
    args = build_parser().parse_args()
    if args.command == "collate":
        return collate(args.out_root)
    if args.n_shuffles < 1 or args.n_null < 0 or args.n_control_windows < 0:
        raise SystemExit("--n-shuffles must be >= 1; --n-null and --n-control-windows >= 0")
    genes = [args.gene] if args.gene else read_gene_ids(args.genes_tsv)
    if args.task_index is not None:
        if not args.genes_tsv or not 0 <= args.task_index < len(genes):
            raise SystemExit("--task-index needs --genes-tsv and an index inside it")
        genes = [genes[args.task_index]]
    failed = []
    for gene in genes:
        try:
            run_gene(args, gene)
        except Exception:
            LOG.exception("%s failed; completed conditions stay cached and resume on rerun", gene)
            failed.append(gene)
    if len(genes) > 1:
        collate(args.out_root)
    if failed:
        LOG.error("failed genes: %s", ",".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
