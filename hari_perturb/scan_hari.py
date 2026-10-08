#!/usr/bin/env python3
"""Scan reference DNA with frozen AlphaGenome + exported cell-type HARI models.

Use the existing AlphaGenome/agsuite environment. Run --help, or see README.md.
The `recall` command only needs NumPy, pandas, and matplotlib; no GPU/checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

# Must precede JAX imports, without overriding an intentional user setting.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from core import (VERSION, IndexedFasta, safe_name, sha256_file, canonical_hash, seed_for,
                  onehot, shuffle_window, make_windows, smooth_windows, project_windows,
                  call_regions, direction)

LOG = logging.getLogger("hari_scan")
REGION_FIELDS = ["gene_id", "cell_type", "chrom", "start0", "end0", "length_bp", "mean_abs_effect",
                 "max_abs_effect", "mean_signed_effect", "direction", "positive_bp_fraction",
                 "negative_bp_fraction", "tss_region_overlap_bp", "annotation", "gene_strand", "effect_units"]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_state(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def make_plot(path, frame, segments, regions, gene, cell, tss0, strand, threshold, units, half_bp, accepted, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True, facecolor="white",
                             gridspec_kw={"height_ratios": [1.35, 1]})
    x = ((frame.start0 + frame.end0) / 2 - tss0) / 1000
    axes[0].plot(x, frame.mean_abs_effect, ".", color="#9AADB7", ms=3, label="Window mean |Δ|")
    axes[1].plot(x, frame.mean_signed_effect, ".", color="#9AADB7", ms=3)
    for i, s in enumerate(segments):
        left, right = (s["start0"]-tss0)/1000, (s["end0"]-tss0)/1000
        axes[0].plot([left, right], [s["abs_effect"]]*2, color="#198F93", lw=2,
                     label="Calling track" if i == 0 else None)
        color = "#DB815E" if s["signed_effect"] > 0 else "#5D78B1"
        axes[1].plot([left, right], [s["signed_effect"]]*2, color=color, lw=2)
    axes[0].axhline(threshold, color="#AA5474", ls="--", lw=1.3, label=f"Effect cutoff = {threshold:g}")
    axes[1].axhline(0, color="#A8B3BC", lw=0.8)
    for ax in axes:
        ax.set_facecolor("white")
        ax.axvspan(-half_bp/1000, half_bp/1000, color="#ECE9F5", alpha=0.65, lw=0)
        ax.axvline(0, color="#3B4652", ls=":", lw=1)
        ax.grid(False)
        ax.tick_params(labelsize=11)
        for r in regions:
            ax.axvspan((r["start0"]-tss0)/1000, (r["end0"]-tss0)/1000,
                       color="#81C5BE", alpha=0.12, lw=0)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(f"Mean |Δ expression|\n({units})", fontsize=13)
    axes[1].set_ylabel(f"Signed Δ expression\n({units})", fontsize=13)
    axes[1].set_xlabel(f"Genomic position relative to TSS (kb; increasing genomic coordinate; gene strand {strand})", fontsize=12)
    axes[0].legend(frameon=False, fontsize=10, loc="best")
    fig.suptitle(f"{gene}  |  {cell}", fontsize=17, fontweight="bold", y=0.98)
    note = (f"Accepted sequence models: {accepted}/{total} folds | {len(regions)} effect-threshold regions | "
            "Δ = perturbed − intact; not an enrichment P value")
    if accepted == 0:
        note = "BASELINE-ONLY ENSEMBLE: no accepted sequence model; zero effect is not evidence of biological inactivity"
    fig.text(0.5, 0.018, note, ha="center", fontsize=9, color="#5D6B78")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def render_outputs(meta, state, output, threshold, min_region_bp, smooth, units, promoter_half_bp):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    windows = state["windows"] + int(meta["input_start0"])
    gene, chrom, tss0 = meta["gene_id"], meta["chrom"], int(meta["tss0"])
    n_repeats = int(meta["repeats"])
    valid = state["status"] == "tested"
    if not state["done"].all():
        raise ValueError("Scan is incomplete; resume it before region calling")
    for c, cell in enumerate(meta["cell_types"]):
        cell_dir = output / safe_name(cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        raw, z = state["effects"][:, :, 0, c], state["effects"][:, :, 1, c]
        selected = raw if units == "adjusted" else z
        # Invalid windows are explicitly masked, not converted to zero.
        frame = pd.DataFrame({"gene_id": gene, "cell_type": cell, "chrom": chrom,
                              "start0": windows[:, 0], "end0": windows[:, 1],
                              "window_length_bp": windows[:, 1]-windows[:, 0],
                              "status": state["status"], "n_scrambles": n_repeats,
                              "n_changed_scrambles": np.sum(state["changed_bp"] > 0, axis=1),
                              "mean_changed_bp": state["changed_bp"].mean(axis=1),
                              "baseline_prediction_adjusted": float(state["reference_predictions"][c]),
                              "mean_signed_delta_adjusted": raw.mean(axis=1),
                              "mean_abs_delta_adjusted": np.abs(raw).mean(axis=1),
                              "mean_signed_delta_target_sd": z.mean(axis=1),
                              "mean_abs_delta_target_sd": np.abs(z).mean(axis=1),
                              "mean_signed_effect": selected.mean(axis=1),
                              "mean_abs_effect": np.abs(selected).mean(axis=1),
                              "mean_context_abs_delta_adjusted": state["effects"][:, :, 2, c].mean(axis=1),
                              "mean_fold_sd_delta_adjusted": state["effects"][:, :, 3, c].mean(axis=1),
                              "scramble_sd_signed_effect": selected.std(axis=1, ddof=1) if n_repeats > 1 else np.full(len(windows), np.nan),
                              "positive_scramble_fraction": np.mean(selected > 1e-12, axis=1),
                              "negative_scramble_fraction": np.mean(selected < -1e-12, axis=1),
                              "effect_units": units})
        effect_cols = [k for k in frame if ("effect" in k or "delta" in k or "scramble_fraction" in k) and k != "effect_units"]
        frame.loc[~valid, effect_cols] = np.nan
        frame["mean_perturbed_prediction_adjusted"] = frame.baseline_prediction_adjusted + frame.mean_signed_delta_adjusted
        frame["direction"] = [direction(row) if good else "not_tested" for row, good in zip(selected, valid)]
        values = frame[["mean_abs_effect", "mean_signed_effect"]].to_numpy()
        smoothed = smooth_windows(windows, values, smooth)
        frame["smoothed_abs_effect"] = smoothed[:, 0]
        frame["smoothed_signed_effect"] = smoothed[:, 1]
        segments = project_windows(windows, smoothed)
        regions = call_regions(segments, threshold, min_region_bp, tss0, promoter_half_bp)
        for r in regions:
            r.update(gene_id=gene, cell_type=cell, chrom=chrom, gene_strand=meta["strand"], effect_units=units)
        frame.to_csv(cell_dir / "windows.tsv.gz", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(regions, columns=REGION_FIELDS).to_csv(cell_dir / "regions.tsv", sep="\t", index=False)
        for label, field in [("absolute", "abs_effect"), ("signed", "signed_effect")]:
            with (cell_dir / f"{label}.bedGraph").open("w") as f:
                for s in segments:
                    f.write(f"{chrom}\t{s['start0']}\t{s['end0']}\t{s[field]:.9g}\n")
        with (cell_dir / "regions.bed").open("w") as f:
            for i, r in enumerate(regions, 1):
                name = f"{safe_name(gene)}.{safe_name(cell)}.R{i}.{r['direction']}"
                f.write(f"{chrom}\t{r['start0']}\t{r['end0']}\t{name}\t0\t{meta['strand']}\n")
        accepted = int(meta["accepted_folds"][cell])
        total = int(meta["total_folds"][cell])
        atomic_json(cell_dir / "log.json", dict(gene_id=gene, cell_type=cell, effect_units=units,
            threshold=threshold, minimum_region_bp=min_region_bp, smooth_windows=smooth,
            promoter_half_bp=promoter_half_bp, region_count=len(regions), tested_windows=int(valid.sum()),
            accepted_sequence_folds=accepted, total_folds=total, status="baseline_only" if accepted == 0 else "sequence_model",
            effect_definition="mean over scrambles of absolute paired marginal ensemble delta",
            coordinate_system="0-based half-open, forward genomic", cutoff_is_statistical_significance=False))
        make_plot(cell_dir / "perturbation_track.png", frame, segments, regions, gene, cell, tss0,
                  meta["strand"], threshold, units, promoter_half_bp, accepted, total)
    # Per-gene tested space for downstream background design, independent of calls.
    support = project_windows(windows, np.where(valid[:, None], np.zeros((len(windows), 2)), np.nan))
    merged = []
    for s in support:
        if merged and merged[-1][1] == s["start0"]:
            merged[-1][1] = s["end0"]
        else:
            merged.append([s["start0"], s["end0"]])
    with (output / "tested_space.bed").open("w") as f:
        for start, stop in merged:
            f.write(f"{chrom}\t{start}\t{stop}\t{gene}\n")


def validate_args(args):
    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.min_region_bp < 1:
        raise ValueError("Set an explicit positive --threshold and --min-region-bp")
    if args.smooth_windows < 1 or args.smooth_windows % 2 != 1 or args.promoter_half_bp < 1:
        raise ValueError("Smoothing must be odd/positive; promoter-half-bp must be positive")


def run_scan(args):
    from agsuite_adapter import (load_bundle, source_provenance, SuiteEmbedder, Ensemble, validate_embedder_class_length)
    if not args.trust_pickle:
        raise ValueError("Use --trust-pickle only for your own/trusted inference bundles; pickle can execute code")
    if args.repeats < 1 or args.max_contexts < 0 or args.head_batch_size < 1 or args.seed < 0:
        raise ValueError("Invalid repeat count, context count, batch size, or seed")
    if args.covariate_mode == "mean" and args.contexts_tsv:
        raise ValueError("Do not combine mean-profile mode with external contexts")
    sys.path.insert(0, str(Path(args.suite_root).resolve()))
    provenance = source_provenance(args.embedder_class)
    versions = {}
    for name in ["numpy", "jax", "flax", "alphagenome", "alphagenome-research", "dm-haiku", "jmp"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    manifest_path = Path(args.targets).resolve()
    targets = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"gene_id", "chrom", "tss0", "strand", "bundle_path"}
    if required - set(targets.columns):
        raise ValueError(f"Target TSV must contain {sorted(required)}")
    if targets.empty or targets.gene_id.duplicated().any():
        raise ValueError("Target table is empty or has duplicate genes/TSS choices")
    if args.genes:
        genes = set(args.genes.split(","))
        absent = genes - set(targets.gene_id)
        if absent:
            raise ValueError(f"Genes absent from targets: {sorted(absent)}")
        targets = targets[targets.gene_id.isin(genes)]
    fasta = IndexedFasta(args.fasta)
    external = pd.read_csv(args.contexts_tsv, sep="\t") if args.contexts_tsv else None
    engines = {}
    preflight = []
    for row in targets.itertuples(index=False):
        start_time = time.time()
        if row.strand not in {"+", "-"}:
            raise ValueError(f"Invalid gene strand for {row.gene_id}")
        tss0 = int(row.tss0)
        bundle_path = Path(row.bundle_path).expanduser()
        if not bundle_path.is_absolute():
            bundle_path = manifest_path.parent / bundle_path
        bundle, schema, saved_dtype = load_bundle(bundle_path, row.gene_id, args.schema_json)
        validate_embedder_class_length(schema, args.embedder_class)
        length = int(schema["required_sequence_length"])
        if args.expected_input_length is not None and length != args.expected_input_length:
            raise ValueError(f"{row.gene_id}: expected {args.expected_input_length} bp but training used {length} bp")
        dtype = saved_dtype if args.embedding_dtype == "auto" else args.embedding_dtype
        if dtype not in {"float16", "float32"}:
            raise ValueError("Training embedding output_dtype is missing; set --embedding-dtype from the actual shard (not a guess)")
        if saved_dtype is not None and dtype != saved_dtype:
            raise ValueError("Requested embedding dtype disagrees with training shard")
        cells = list(bundle["cell_types"]) if args.cell_types == "auto" else args.cell_types.split(",")
        if not cells or len(set(cells)) != len(cells) or set(cells) - set(bundle["cell_types"]):
            raise ValueError("Requested cell types are missing or duplicated")
        if external is not None:
            if set(external.columns) != set(bundle["covariate_names"]):
                raise ValueError("--contexts-tsv must contain exactly the saved covariate_names, numeric values")
            context = external[bundle["covariate_names"]].apply(pd.to_numeric, errors="raise")
        else:
            context = None
        input_start = tss0 - length // 2
        seq, contig_length = fasta.fetch(row.chrom, input_start, input_start + length)
        if not (0 <= tss0 < contig_length):
            raise ValueError("TSS falls outside the declared contig")
        scan_start, scan_stop = 0, length
        if args.scan_half_window_bp is not None:
            h = args.scan_half_window_bp
            scan_start, scan_stop = length // 2 - h, length // 2 + h
            if scan_start < 0 or scan_stop > length:
                raise ValueError("Requested scan extends outside sequence seen by the pretrained model")
        windows = make_windows(length, args.window_bp, args.stride_bp, scan_start, scan_stop)
        accepted = {c: sum(bool(f["residual_accepted"]) for f in bundle["cell_types"][c]["fold_models"]) for c in cells}
        total = {c: len(bundle["cell_types"][c]["fold_models"]) for c in cells}
        meta = dict(version=VERSION, gene_id=row.gene_id, chrom=row.chrom, tss0=tss0, strand=row.strand,
            assembly=args.assembly, input_start0=input_start, input_end0=input_start+length,
            reference_orientation="forward_genomic", reference_haplotype_mode="same sequence on both homologues",
            sequence_sha256=hashlib.sha256(seq).hexdigest(), contig_length=contig_length,
            fasta_path=str(Path(args.fasta).resolve()), fasta_index_sha256=sha256_file(str(args.fasta)+".fai"),
            bundle_path=str(bundle_path.resolve()), bundle_sha256=sha256_file(bundle_path),
            embedding_schema=schema, embedding_dtype=dtype, embedder_class=args.embedder_class,
            cell_types=cells, accepted_folds=accepted, total_folds=total, repeats=args.repeats,
            target_sd_scale={c:float(np.mean([float(f["target_transform"]["stds"][0]) for f in bundle["cell_types"][c]["fold_models"]])) for c in cells},
            seed=args.seed, shuffle=args.shuffle, window_bp=args.window_bp, stride_bp=args.stride_bp,
            scan_start_offset0=scan_start, scan_stop_offset0=scan_stop,
            covariate_mode=args.covariate_mode, context_population="external_common" if context is not None else "fold_training_donors",
            contexts_sha256=sha256_file(args.contexts_tsv) if args.contexts_tsv else None,
            max_contexts=args.max_contexts, head_device=args.head_device, head_batch_size=args.head_batch_size,
            covariate_names=bundle["covariate_names"], source_modules=provenance, runtime_versions=versions,
            implementation_sha256={p:sha256_file(Path(__file__).parent/p) for p in ["core.py", "agsuite_adapter.py", "scan_hari.py"]})
        fingerprint = canonical_hash(meta)
        meta["fingerprint"] = fingerprint
        gene_dir = Path(args.out_dir) / safe_name(row.gene_id)
        if not args.preflight:
            if gene_dir.exists() and not args.resume:
                raise FileExistsError(f"Output exists: {gene_dir}; use --resume with identical scan settings")
            if gene_dir.exists() and args.resume:
                mpath = gene_dir / "scan_metadata.json"
                if not mpath.exists():
                    raise ValueError(f"Cannot resume unrecognized output directory {gene_dir}")
                old = json.loads(mpath.read_text())
                if old["fingerprint"] != fingerprint:
                    raise ValueError("Resume fingerprint mismatch (sequence, weights, code, schema, context, or scan settings changed). Use a new output directory. For cutoff changes use recall.")
            gene_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(gene_dir / "scan_metadata.json", meta)
        LOG.info("%s: %s:%d-%d, model=%d bp, windows=%d x %d scrambles; accepted folds=%s",
                 row.gene_id, row.chrom, input_start, input_start+length, length, len(windows), args.repeats, accepted)
        ensemble = Ensemble(bundle, cells, args.covariate_mode, context, args.max_contexts,
                            args.seed, args.head_batch_size, args.head_device)
        if args.preflight:
            ensemble.set_reference(np.zeros(int(schema["pooled_feature_dim"]), dtype=np.float32))
            preflight.append(dict(gene_id=row.gene_id, input_length=length, n_windows=len(windows),
                                  accepted_folds=accepted, schema=schema, structural_weight_check="passed",
                                  note="Dummy-embedding structural inference only; AlphaGenome has NOT run"))
            continue
        state_path = gene_dir / "scan_state.npz"
        state = read_state(state_path) if args.resume and state_path.exists() else None
        if state is not None:
            if str(state["fingerprint"]) != fingerprint or not np.array_equal(state["windows"], windows):
                raise ValueError("Scan cache does not match metadata")
            expected = (len(windows), args.repeats, 4, len(cells))
            if state["effects"].shape != expected or state["done"].shape != (len(windows),):
                raise ValueError("Invalid scan cache shape")

        def embed(sequence):
            key = canonical_hash(schema)
            if key not in engines:
                engines[key] = SuiteEmbedder(schema, args.embedder_class, args.allow_cpu_trunk)
            vector = engines[key].embed(onehot(sequence))
            # Reproduce the precision of cached training features before scaling.
            return vector.astype(dtype).astype(np.float32)

        reference_embedding = state["reference_embedding"] if state is not None else embed(seq)
        ensemble.set_reference(reference_embedding)
        if not np.allclose(ensemble.delta(reference_embedding), 0, atol=1e-8, rtol=0):
            raise ValueError("No-op reference check failed: inference is not deterministic")
        if state is None:
            state = dict(fingerprint=np.array(fingerprint), windows=windows,
                         effects=np.full((len(windows), args.repeats, 4, len(cells)), np.nan, dtype=np.float64),
                         done=np.zeros(len(windows), dtype=bool), status=np.full(len(windows), "pending", dtype="U32"),
                         changed_bp=np.zeros((len(windows), args.repeats), dtype=np.int32),
                         reference_embedding=reference_embedding, reference_predictions=ensemble.reference)
            atomic_npz(state_path, **state)
        elif not np.allclose(state["reference_predictions"], ensemble.reference, atol=1e-6, rtol=1e-6):
            raise ValueError("Cached reference prediction is not reproducible")
        atomic_json(gene_dir / "baseline_contexts.json", ensemble.baseline_rows)
        pd.DataFrame([{k:v for k,v in r.items() if k != "context_ids"} for r in ensemble.baseline_rows]).to_csv(
            gene_dir / "baselines.tsv", sep="\t", index=False)
        for i, (a, b) in enumerate(windows):
            if state["done"][i]:
                continue
            block = seq[a:b]
            if set(block) - set(b"ACGT"):
                state["status"][i] = "skipped_ambiguous_or_padding"
            else:
                for repeat in range(args.repeats):
                    rng = np.random.default_rng(seed_for(args.seed, row.gene_id, input_start+int(a), input_start+int(b), repeat))
                    shuffled = shuffle_window(block, rng, args.shuffle)
                    changes = sum(x != y for x, y in zip(block, shuffled))
                    state["changed_bp"][i, repeat] = changes
                    vector = embed(seq[:a] + shuffled + seq[b:]) if changes else reference_embedding
                    state["effects"][i, repeat] = ensemble.delta(vector)
                state["status"][i] = "tested" if np.any(state["changed_bp"][i]) else "unchanged_all_scrambles"
            state["done"][i] = True
            atomic_npz(state_path, **state)
            LOG.info("%s window %d/%d %s:%d-%d %s elapsed=%.1f min", row.gene_id, i+1, len(windows),
                     row.chrom, input_start+a, input_start+b, state["status"][i], (time.time()-start_time)/60)
        render_outputs(meta, state, gene_dir / "results", args.threshold, args.min_region_bp,
                       args.smooth_windows, args.effect_units, args.promoter_half_bp)
        atomic_json(gene_dir / "completion.json", dict(fingerprint=fingerprint, complete=True,
                    elapsed_seconds=time.time()-start_time, threshold=args.threshold, effect_units=args.effect_units))
        del ensemble, bundle, state
        gc.collect()
    if args.preflight:
        print(json.dumps(preflight, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Preflight or execute checkpoint-compatible sequence scan")
    recall = sub.add_parser("recall", help="New thresholds/plots from an existing completed scan; no GPU")
    for parser in [scan, recall]:
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--threshold", type=float, required=True, help="Effect-size cutoff, NOT a P value")
        parser.add_argument("--effect-units", choices=["target-sd", "adjusted"], default="target-sd")
        parser.add_argument("--min-region-bp", type=int, default=1000)
        parser.add_argument("--smooth-windows", type=int, default=1)
        parser.add_argument("--promoter-half-bp", type=int, default=5000, help="Annotation only: default 10-kb total TSS region")
    recall.add_argument("--gene-dir", required=True)
    scan.add_argument("--targets", required=True, help="TSV: gene_id chrom tss0 strand bundle_path")
    scan.add_argument("--fasta", required=True, help="Reference FASTA, uncompressed and .fai-indexed")
    scan.add_argument("--assembly", required=True, help="Explicit genome-build label, e.g. GRCh38")
    scan.add_argument("--suite-root", required=True)
    scan.add_argument("--genes", help="Optional comma-separated subset of target rows")
    scan.add_argument("--cell-types", default="auto", help="Exact bundle labels, comma separated, or auto")
    scan.add_argument("--window-bp", type=int, default=1000)
    scan.add_argument("--stride-bp", type=int, default=1000)
    scan.add_argument("--repeats", type=int, default=5)
    scan.add_argument("--seed", type=int, default=42)
    scan.add_argument("--shuffle", choices=["mononucleotide", "dinucleotide"], default="mononucleotide")
    scan.add_argument("--covariate-mode", choices=["marginal", "mean", "fixed"], default="marginal")
    scan.add_argument("--contexts-tsv", help="Optional common panel, numeric columns exactly equal to covariate_names")
    scan.add_argument("--max-contexts", type=int, default=0, help="0 = all saved training donors per fold")
    scan.add_argument("--head-device", choices=["cpu", "gpu"], default="cpu")
    scan.add_argument("--head-batch-size", type=int, default=128)
    scan.add_argument("--scan-half-window-bp", type=int, help="Optional smaller symmetric scan; input remains unchanged")
    scan.add_argument("--expected-input-length", type=int, help="Fail unless training used exactly this length")
    scan.add_argument("--schema-json", help="Authoritative training shard schema only if absent from legacy bundle")
    scan.add_argument("--embedding-dtype", choices=["auto", "float16", "float32"], default="auto")
    scan.add_argument("--embedder-class", default="agsuite.alphagenome:AlphaGenomePooledEmbedder")
    scan.add_argument("--allow-cpu-trunk", action="store_true")
    scan.add_argument("--trust-pickle", action="store_true")
    scan.add_argument("--preflight", action="store_true", help="Validate model structure and metadata without loading AlphaGenome")
    scan.add_argument("--resume", action="store_true")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("agsuite.alphagenome").setLevel(logging.WARNING)
    args = build_parser().parse_args()
    validate_args(args)
    if args.command == "recall":
        out = Path(args.out_dir)
        if out.exists():
            raise FileExistsError("Use a new --out-dir for re-calling so earlier results remain intact")
        source = Path(args.gene_dir)
        meta = json.loads((source / "scan_metadata.json").read_text())
        state = read_state(source / "scan_state.npz")
        if str(state["fingerprint"]) != meta["fingerprint"]:
            raise ValueError("Scan metadata/cache fingerprint mismatch")
        render_outputs(meta, state, out, args.threshold, args.min_region_bp,
                       args.smooth_windows, args.effect_units, args.promoter_half_bp)
    else:
        run_scan(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Scan failed; completed windows remain resumable where a cache was written")
        sys.exit(1)
