#!/usr/bin/env python3
"""Reference-genome scrambling -> native astrocyte RNA effect tracks.

No HARI bundle, learned head, embeddings, personalized NPZ, or covariates.
All BED/TSV genomic coordinates are zero-based half-open; --region is 1-based.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from native_rna_backend import NativeAstrocyte, Region, build_exon_mask, load_gene_exons, parse_region

SCHEMA_VERSION = 1
CELL = "Astrocyte"
UNITS = ("log1p", "raw", "fractional")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def state_write(path, state):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    native_effects = effects(state["predictions"], float(state["baseline"]))
    state["effects"] = np.stack([native_effects[u] for u in ("raw", "log1p", "fractional")], axis=-1)[..., None]
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **state)
    os.replace(tmp, path)


@contextmanager
def writer_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".scan.lock", "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another writer holds the scan lock for {folder}") from None
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class IndexedFasta:
    """Uncompressed samtools .fai reader; no pysam/pyfaidx dependency."""

    def __init__(self, path):
        self.path = Path(path)
        self.index_path = Path(str(path) + ".fai")
        if self.path.suffix in (".gz", ".bgz", ".bgzf"):
            raise ValueError("Use an uncompressed FASTA and its samtools .fai index.")
        self.records = {}
        with open(self.index_path) as f:
            for line in f:
                fields = line.rstrip().split("\t")
                if len(fields) < 5:
                    raise ValueError("Invalid FASTA index")
                name = fields[0]
                if name in self.records:
                    raise ValueError(f"Duplicate FASTA contig {name}")
                self.records[name] = tuple(map(int, fields[1:5]))

    def resolve(self, name):
        if name in self.records:
            return name
        alternate = name[3:] if name.startswith("chr") else "chr" + name
        if alternate in self.records:
            print(f"FASTA contig alias: {name} -> {alternate}", flush=True)
            return alternate
        raise KeyError(f"Contig {name!r} and chr-prefix alias are absent from FASTA")

    def fetch(self, chrom, start, end):
        length, offset, bases, width = self.records[chrom]
        if end <= start or start < 0 or end > length or bases <= 0 or width < bases:
            raise ValueError(f"Invalid FASTA interval/index: {chrom}:{start}-{end}")
        begin = offset + (start // bases) * width + start % bases
        finish = offset + ((end - 1) // bases) * width + (end - 1) % bases + 1
        with open(self.path, "rb") as f:
            f.seek(begin)
            raw = f.read(finish - begin).replace(b"\n", b"").replace(b"\r", b"")
        if len(raw) != end - start or b">" in raw or any(c in raw for c in (b" ", b"\t")):
            raise ValueError("FASTA .fai appears stale or incompatible; rebuild using samtools faidx.")
        return raw.decode("ascii").upper()

    def padded(self, chrom, start, end):
        length = self.records[chrom][0]
        left, right = max(0, start), min(end, length)
        if right <= left:
            raise ValueError("Input interval does not intersect the chromosome")
        return "N" * (left - start) + self.fetch(chrom, left, right) + "N" * (end - right)


def base_gene(gene):
    return str(gene).split(".", 1)[0]


def validate_target(row):
    gene = str(row["gene_id"])
    if not gene or gene in (".", "..") or "/" in gene or "\\" in gene:
        raise ValueError(f"Unsafe/empty gene identifier: {gene!r}")
    strand = str(row["strand"])
    if strand not in ("+", "-"):
        raise ValueError(f"Target {gene}: strand must be + or -")
    tss = int(str(row["tss0"]))
    if tss < 0:
        raise ValueError("tss0 must be zero-based and nonnegative")
    return {"gene_id": gene, "chrom": str(row["chrom"]), "tss0": tss, "strand": strand}


def read_targets(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"gene_id", "chrom", "tss0", "strand"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Targets TSV needs columns {sorted(required)}; extra HARI columns are ignored")
        rows = [validate_target(r) for r in reader]
    if len({base_gene(r["gene_id"]) for r in rows}) != len(rows):
        raise ValueError("Targets contain duplicate/version-alias genes; choose one TSS per gene")
    if not rows:
        raise ValueError("Empty targets TSV")
    return rows


def bed_targets(path, requested=None):
    """Only unique one-base BED6 TSS entries; never silently select a transcript."""
    candidates = {}
    requested_keys = None if requested is None else {base_gene(g) for g in requested}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            p = line.split()
            if p[:3] == ["chrom", "start", "end"]:
                continue
            if len(p) < 6:
                raise ValueError("TSS BED must be BED6 with gene ID in column 4 and strand in column 6")
            key = base_gene(p[3])
            if requested_keys is not None and key not in requested_keys:
                continue
            if int(p[2]) != int(p[1]) + 1:
                raise ValueError(f"TSS BED entry for {p[3]} is not one base; supply an explicit targets.tsv")
            candidates.setdefault(key, set()).add((p[0], int(p[1]), p[5]))
    genes = requested if requested is not None else sorted(candidates)
    rows, skipped = [], []
    for gene in genes:
        hits = candidates.get(base_gene(gene), set())
        if len(hits) != 1:
            skipped.append({"gene_id": gene, "reason": "missing_tss" if not hits else "ambiguous_tss"})
            continue
        chrom, tss, strand = next(iter(hits))
        rows.append(validate_target(dict(gene_id=gene, chrom=chrom, tss0=tss, strand=strand)))
    return rows, skipped


def windows_for(start, end, width, stride):
    if min(width, stride) <= 0 or end - start < width:
        raise ValueError("Scan span must fit at least one full perturbation window")
    starts = list(range(start, end - width + 1, stride))
    if starts[-1] != end - width:
        starts.append(end - width)
    return np.array([(s, s + width) for s in starts], dtype=np.int64)


def window_seed(seed, gene, chrom, start, end, repeat):
    data = json.dumps([int(seed), gene, chrom, int(start), int(end), int(repeat)]).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def dinucleotide_shuffle(codes, rng):
    """Random Euler trail preserving endpoints and internal dinucleotide counts.

    Random adjacency order does not guarantee uniform sampling of possible strings.
    """
    adjacency = [[] for _ in range(4)]
    for a, b in zip(codes[:-1], codes[1:]):
        adjacency[int(a)].append(int(b))
    for edges in adjacency:
        rng.shuffle(edges)
    stack, trail = [int(codes[0])], []
    while stack:
        edges = adjacency[stack[-1]]
        if edges:
            stack.append(edges.pop())
        else:
            trail.append(stack.pop())
    out = np.asarray(trail[::-1], dtype=codes.dtype)
    if len(out) != len(codes) or out[0] != codes[0] or out[-1] != codes[-1]:
        raise RuntimeError("Invalid dinucleotide Euler trail")
    return out


def prepare_gene(args, target, fasta):
    target = dict(target)
    chrom = fasta.resolve(target["chrom"])
    target["chrom"] = chrom
    chrom_len = fasta.records[chrom][0]
    tss = target["tss0"]
    if tss >= chrom_len:
        raise ValueError(f"TSS for {target['gene_id']} exceeds chromosome length")
    if args.region:
        region = parse_region(args.region)
        if fasta.resolve(region.chrom) != chrom:
            raise ValueError("Explicit interval chromosome disagrees with target TSS")
        region = Region(chrom, region.start, region.end)
        if region.width != args.sequence_length:
            raise ValueError("--region width must equal --sequence-length")
    else:
        start0 = tss - args.sequence_length // 2
        region = Region(chrom, start0 + 1, start0 + args.sequence_length)
    start0, end0 = region.start - 1, region.end
    if not start0 <= tss < end0:
        raise ValueError("Input interval must contain the target TSS")
    seq = fasta.padded(chrom, start0, end0)
    ann = load_gene_exons(args.gtf, target["gene_id"])
    if ann.strand != target["strand"]:
        raise ValueError("Target TSV/TSS BED strand disagrees with GTF")
    mask, exon_bp, fraction = build_exon_mask(ann, region, len(seq))
    if fraction < args.min_exon_fraction:
        raise ValueError(f"Only {fraction:.1%} of target exon union is in the model input; minimum {args.min_exon_fraction:.1%}")
    if fraction < 1:
        print(f"{target['gene_id']}: only {fraction:.1%} of annotated exon union is inside input; scoring fixed in-window exon union.", flush=True)
    lo, hi = max(0, start0), min(chrom_len, end0)
    if args.scan_half_window_bp is not None:
        lo, hi = max(lo, tss - args.scan_half_window_bp), min(hi, tss + args.scan_half_window_bp)
    windows = windows_for(lo, hi, args.window_bp, args.stride_bp)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    ref_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    canonical = np.isin(ref_bytes, np.frombuffer(b"ACGT", dtype=np.uint8))
    contract = {
        "schema_version": SCHEMA_VERSION,
        "target": target, "assembly": args.assembly,
        "input_interval": {"chrom": chrom, "start0": start0, "end0": end0},
        "sequence_length": len(seq), "reference_sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
        "fasta_path": str(Path(args.fasta).resolve()), "fai_sha256": sha_file(fasta.index_path),
        "gtf_path": str(Path(args.gtf).resolve()), "gtf_sha256": args._gtf_sha256,
        "exon_union_1based_inclusive": ann.exon_intervals, "exonic_bp_in_window": exon_bp,
        "exon_fraction": fraction, "exon_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "window_bp": args.window_bp, "stride_bp": args.stride_bp, "repeats": args.repeats,
        "seed": args.seed, "shuffle": args.shuffle, "max_windows": args.max_windows,
        "scan_half_window_bp": args.scan_half_window_bp, "batch_size": args.batch_size,
        "validate_public": not args.skip_public_check,
        "expression_definition": "mean native astrocyte RNA coverage over fixed in-window exon union on target gene strand",
        "cell_types": [CELL], "track_name": "CL:0000127 total RNA-seq",
        "state_predictions_axes": ["window", "repeat"],
        "state_effects_axes": ["window", "repeat", "statistic", "cell_type"],
        "effect_statistics": ["signed_delta_raw", "signed_delta_log1p", "signed_delta_fractional"],
        "coordinates": "zero-based half-open", "effect_definition": "perturbed minus intact reference",
        "code_sha256": {p: sha_file(Path(__file__).parent / p) for p in ("scan_native_alphagenome.py", "native_rna_backend.py")},
    }
    return target, seq, mask, canonical, windows, contract


def effects(predictions, baseline):
    raw = predictions - baseline
    log = np.log1p(predictions) - np.log1p(baseline)
    fractional = raw / baseline if baseline > 0 else np.full_like(raw, np.nan)
    return {"raw": raw, "log1p": log, "fractional": fractional}


def smooth_valid(values, valid, windows, n):
    """Centered unweighted window mean, never crossing missing support or gaps."""
    out = np.full_like(values, np.nan)
    if n == 1:
        out[valid] = values[valid]
        return out
    i = 0
    while i < len(values):
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < len(values) and valid[j] and windows[j, 0] <= windows[j - 1, 1]:
            j += 1
        for k in range(i, j):
            out[k] = np.mean(values[max(i, k - n // 2):min(j, k + n // 2 + 1)])
        i = j
    return out


def disjoint_segments(windows, valid, absolute, signed):
    """Overlap-weighted calling track. Any missing window masks its entire support."""
    events = {}
    for i, (a, b) in enumerate(windows):
        events.setdefault(int(a), []).append((i, 1))
        events.setdefault(int(b), []).append((i, -1))
    active, rows = set(), []
    points = sorted(events)
    for k, point in enumerate(points[:-1]):
        for i, direction in events[point]:
            if direction == 1:
                active.add(i)
            else:
                active.remove(i)
        if not active or any(not valid[i] for i in active):
            continue
        indices = sorted(active)
        rows.append((point, points[k + 1], float(np.mean(absolute[indices])), float(np.mean(signed[indices]))))
    return rows


# gene_id / annotation / gene_strand are required by perturb_enrich/scripts/annotate_regions.py
# (and gene_id by enrich_gene.py). "region_id" is renamed scan_region_id because both of those
# scripts call df.insert(0, "region_id", ...), which raises if the column already exists.
REGION_COLS = ["gene_id", "chrom", "start0", "end0", "scan_region_id", "length_bp", "mean_absolute_effect",
               "max_absolute_effect", "mean_signed_effect", "direction", "positive_bp_fraction",
               "negative_bp_fraction", "tss_overlap", "promoter_overlap_bp", "exon_overlap_bp",
               "annotation", "gene_strand", "effect_units"]


def call_regions(segments, threshold, min_bp, target, meta, promoter_half, units):
    runs, current = [], []
    for seg in segments:
        if seg[2] > threshold:
            if current and current[-1][1] != seg[0]:
                runs.append(current)
                current = []
            current.append(seg)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    rows = []
    for run in runs:
        a, b = run[0][0], run[-1][1]
        if b - a < min_bp:
            continue
        w = np.array([x[1] - x[0] for x in run])
        absolute, signed = np.array([x[2] for x in run]), np.array([x[3] for x in run])
        pos, neg = float(w[signed > 0].sum() / w.sum()), float(w[signed < 0].sum() / w.sum())
        direction = "mixed" if pos and neg else "increase" if pos else "decrease" if neg else "zero"
        tss = target["tss0"]
        exon_overlap = sum(max(0, min(b, e) - max(a, s - 1)) for s, e in meta["exon_union_1based_inclusive"])
        promoter_overlap = max(0, min(b, tss + promoter_half) - max(a, tss - promoter_half))
        # Same rule as hari_perturb/core.py call_regions: fully inside the +/- promoter span,
        # straddling its edge, or entirely outside it.
        annotation = "TSS_proximal" if promoter_overlap == b - a else "TSS_boundary" if promoter_overlap else "distal"
        rows.append({
            "gene_id": target["gene_id"], "chrom": target["chrom"], "start0": a, "end0": b,
            "scan_region_id": f"{target['gene_id']}_nativeAG_{len(rows)+1}", "length_bp": b - a,
            "mean_absolute_effect": float(np.average(absolute, weights=w)),
            "max_absolute_effect": float(absolute.max()),
            "mean_signed_effect": float(np.average(signed, weights=w)),
            "direction": direction, "positive_bp_fraction": pos, "negative_bp_fraction": neg,
            "tss_overlap": a <= tss < b, "promoter_overlap_bp": promoter_overlap,
            "exon_overlap_bp": exon_overlap, "annotation": annotation,
            "gene_strand": target["strand"], "effect_units": units})
    return pd.DataFrame(rows, columns=REGION_COLS)


def output_results(folder, meta, state, args):
    windows, status = state["windows"], state["status"].astype(str)
    pred, baseline = state["predictions"], float(state["baseline"])
    units = args.effect_units
    if units == "fractional" and baseline <= 0:
        raise ValueError("Reference coverage is zero; fractional effects are undefined. Use --effect-units log1p or raw.")
    deltas = effects(pred, baseline)
    valid = (status == "tested") & np.isfinite(pred).all(axis=1)
    columns = {
        "window_id": np.arange(len(windows)), "chrom": meta["target"]["chrom"],
        "start0": windows[:, 0], "end0": windows[:, 1],
        "center_relative_tss_bp": windows.mean(axis=1) - meta["target"]["tss0"],
        "status": status, "n_changed_scrambles": state["changed"].sum(axis=1),
        "n_completed_repeats": np.isfinite(pred).sum(axis=1),
        "baseline_mean_exon_coverage": baseline, "effect_units": units,
    }
    for label, delta in deltas.items():
        # Retain no-op repeat zeros; skipped windows stay NA.
        present = np.isfinite(delta).all(axis=1)
        for name, func in [("mean_signed_delta", np.mean), ("mean_absolute_delta", lambda x, axis: np.mean(np.abs(x), axis=axis)),
                           ("sd_signed_delta", lambda x, axis: np.std(x, axis=axis, ddof=1) if x.shape[axis] > 1 else np.zeros(x.shape[0]))]:
            values = np.full(len(windows), np.nan)
            if present.any():
                values[present] = func(delta[present], axis=1)
            columns[name + "_" + label] = values
    selected = deltas[units]
    columns["positive_scramble_fraction"] = np.where(np.isfinite(selected).all(axis=1), (selected > 0).mean(axis=1), np.nan)
    columns["negative_scramble_fraction"] = np.where(np.isfinite(selected).all(axis=1), (selected < 0).mean(axis=1), np.nan)
    columns["mean_perturbed_exon_coverage"] = np.array([x.mean() if np.isfinite(x).all() else np.nan for x in pred])
    absolute = columns["mean_absolute_delta_" + units]
    signed = columns["mean_signed_delta_" + units]
    absolute = smooth_valid(absolute, valid, windows, args.smooth_windows)
    signed = smooth_valid(signed, valid, windows, args.smooth_windows)
    columns["calling_absolute_effect"] = absolute
    columns["calling_signed_effect"] = signed
    exon_pairs = meta["exon_union_1based_inclusive"]
    columns["exon_overlap_bp"] = [sum(max(0, min(int(b), e) - max(int(a), s - 1)) for s, e in exon_pairs) for a,b in windows]
    table = pd.DataFrame(columns)
    out = folder / "results" / CELL
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "windows.tsv.gz", sep="\t", index=False, na_rep="NA")
    segments = disjoint_segments(windows, valid, absolute, signed)
    chrom = meta["target"]["chrom"]
    for name, idx in (("absolute", 2), ("signed", 3)):
        with open(out / f"{name}.bedGraph", "w") as f:
            for seg in segments:
                f.write(f"{chrom}\t{seg[0]}\t{seg[1]}\t{seg[idx]:.10g}\n")
    # Union of genuinely tested support, after conservative masking of skipped windows.
    union = []
    for a,b,_,_ in segments:
        if union and union[-1][1] == a:
            union[-1][1] = b
        else:
            union.append([a,b])
    with open(folder / "results" / "tested_space.bed", "w") as f:
        for a,b in union:
            f.write(f"{chrom}\t{a}\t{b}\t{meta['target']['gene_id']}\n")  # enrich_gene.py reads 4 columns
    regions = call_regions(segments, args.threshold, args.min_region_bp, meta["target"], meta, args.promoter_half_bp, units)
    regions.to_csv(out / "regions.tsv", sep="\t", index=False)
    with open(out / "regions.bed", "w") as f:
        for row in regions.itertuples():
            f.write(f"{row.chrom}\t{row.start0}\t{row.end0}\t{row.scan_region_id}\t0\t{meta['target']['strand']}\n")
    log = {"threshold": args.threshold, "effect_units": units, "min_region_bp": args.min_region_bp,
           "smooth_windows": args.smooth_windows, "promoter_half_bp": args.promoter_half_bp,
           "n_windows": len(windows), "n_tested": int(valid.sum()), "n_regions": len(regions),
           "status_counts": dict(zip(*[a.tolist() for a in np.unique(status, return_counts=True)])),
           "calling_rule": "mean(abs(repeat delta)); overlap-average; segments strictly above threshold; genomic union length",
           "region_interpretation": "effect-threshold regions; no statistical significance claim"}
    json_write(out / "log.json", log)
    plot_track(out, table, segments, regions, meta, args)
    completion = {"complete": bool(np.all(status != "pending")), "n_tested": int(valid.sum()),
                  "n_windows": len(windows), "scan_contract_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                  "call_settings": log}
    json_write(folder / "completion.json", completion)
    print(f"{meta['target']['gene_id']}: {valid.sum()}/{len(windows)} tested windows; {len(regions)} effect regions ({units}).", flush=True)


def plot_track(out, table, segments, regions, meta, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tss = meta["target"]["tss0"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.7), sharex=True, constrained_layout=True)
    x, av, sv = [], [], []
    previous = None
    for a,b,ab,si in segments:
        if previous is not None and a != previous:
            x.append(np.nan); av.append(np.nan); sv.append(np.nan)
        x.extend([(a-tss)/1000, (b-tss)/1000]); av.extend([ab, ab]); sv.extend([si, si])
        previous = b
    axes[0].plot(x, av, color="steelblue", lw=1)
    axes[0].axhline(args.threshold, color="grey", ls="--", lw=0.8)
    axes[1].plot(x, sv, color="#39a7a3", lw=1)
    axes[1].axhline(0, color="grey", lw=0.6)
    for ax in axes:
        ax.axvline(0, color="black", lw=0.8)
        ax.axvspan(-args.promoter_half_bp/1000, args.promoter_half_bp/1000, color="grey", alpha=0.09)
        for r in regions.itertuples():
            ax.axvspan((r.start0-tss)/1000, (r.end0-tss)/1000, color="steelblue", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"Mean absolute effect\n({args.effect_units})")
    axes[1].set_ylabel(f"Mean signed effect\n({args.effect_units})")
    axes[1].set_xlabel("Increasing genomic coordinate relative to TSS (kb)")
    if len(table):
        axes[1].set_xlim((table.start0.min()-tss)/1000, (table.end0.max()-tss)/1000)
    axes[0].set_title(f"{meta['target']['gene_id']} | native AlphaGenome Astrocyte | strand {meta['target']['strand']}")
    fig.savefig(out / "perturbation_track.png", dpi=220)
    plt.close(fig)


def run_gene(args, prepared, backend):
    target, seq, mask, canonical, windows, contract = prepared
    contract["native_model"] = backend.provenance
    # Device ordinals are execution details; weights, versions and model source are the resume contract.
    contract["native_model"] = {k:v for k,v in contract["native_model"].items() if k != "devices"}
    folder = Path(args.out_dir).resolve() / target["gene_id"]
    with writer_lock(folder):
        metadata_path, state_path = folder / "scan_metadata.json", folder / "scan_state.npz"
        occupied = [p for p in folder.iterdir() if p.name != ".scan.lock"]
        if args.resume:
            if not metadata_path.exists() or not state_path.exists():
                raise ValueError("Resume needs both scan_metadata.json and scan_state.npz; use a new directory for a fresh scan")
            previous = json.loads(metadata_path.read_text())
            # Normalize tuples to the JSON representation before comparing.
            if previous != json.loads(json.dumps(contract)):
                keys = sorted(k for k in set(previous) | set(contract) if previous.get(k) != json.loads(json.dumps(contract.get(k))))
                raise ValueError(f"Resume contract mismatch: {keys}; use a separate output directory")
            with np.load(state_path, allow_pickle=False) as f:
                state = {k:f[k] for k in f.files}
            if not np.array_equal(state["windows"], windows) or state["predictions"].shape != (len(windows), args.repeats):
                raise ValueError("Corrupt or incompatible scan state")
            if np.all(state["status"] != "pending"):
                output_results(folder, previous, state, args)
                return
        elif occupied:
            raise ValueError(f"Output exists at {folder}. Use --resume or a new --out-dir")
        onehot = backend.encode(seq)
        if not args.resume:
            baseline, baseline_tracks = backend.baseline(seq, onehot, mask, target["strand"], not args.skip_public_check)
            state = {"windows": windows, "predictions": np.full((len(windows), args.repeats), np.nan),
                     "changed": np.zeros((len(windows), args.repeats), dtype=bool),
                     "status": np.full(len(windows), "pending", dtype="U32"), "baseline": np.array(baseline)}
            if args.save_baseline_tracks:
                np.save(folder / "reference.native_rnaseq_tracks.npy", baseline_tracks)
            pd.DataFrame([{"gene_id": target["gene_id"], "cell_type": CELL, "strand": target["strand"],
                           "mean_exon_coverage": baseline, "log1p_mean_exon_coverage": float(np.log1p(baseline)),
                           "exonic_bp_in_window": int(mask.sum()), "exon_fraction": contract["exon_fraction"]}]).to_csv(folder / "baselines.tsv", sep="\t", index=False)
            backend.track_metadata.to_csv(folder / "rnaseq_track_metadata.tsv", sep="\t", index=False)
            json_write(metadata_path, contract)
            state_write(state_path, state)
        baseline = float(state["baseline"])
        if args.effect_units == "fractional" and baseline <= 0:
            raise ValueError("Reference coverage is zero; choose --effect-units raw or log1p")
        input_start = contract["input_interval"]["start0"]
        for i, (a,b) in enumerate(windows):
            if state["status"][i] != "pending":
                continue
            left, right = int(a-input_start), int(b-input_start)
            if not canonical[left:right].all():
                state["status"][i] = "ambiguous_bases"
            else:
                source = onehot[left:right]
                codes = source.argmax(axis=1)
                mutations, repeat_ids = [], []
                for repeat in range(args.repeats):
                    rng = np.random.default_rng(window_seed(args.seed, target["gene_id"], target["chrom"], a, b, repeat))
                    if args.shuffle == "mononucleotide":
                        shuffled = source[rng.permutation(len(source))]
                    else:
                        shuffled = source[0:1].repeat(len(source), axis=0)
                        # Recover the actual encoder row for each base, preserving channel order.
                        new_codes = dinucleotide_shuffle(codes, rng)
                        rows = {int(c): source[np.flatnonzero(codes == c)[0]] for c in np.unique(codes)}
                        shuffled[:] = np.stack([rows[int(c)] for c in new_codes])
                    changed = not np.array_equal(source, shuffled)
                    state["changed"][i, repeat] = changed
                    if not changed:
                        state["predictions"][i, repeat] = baseline
                        continue
                    mutations.append(shuffled)
                    repeat_ids.append(repeat)
                for j in range(0, len(mutations), args.batch_size):
                    subset = mutations[j:j+args.batch_size]
                    # Keep the compiled batch shape constant, including the final partial batch.
                    batch = np.repeat(onehot[None], args.batch_size, axis=0)
                    for k, mutation in enumerate(subset):
                        batch[k, left:right] = mutation
                    values = backend.scores(batch, mask, target["strand"])
                    if values.shape != (args.batch_size,) or not np.isfinite(values).all() or (values < 0).any():
                        raise RuntimeError("Invalid native scalar prediction")
                    for k, value in enumerate(values[:len(subset)]):
                        state["predictions"][i, repeat_ids[j+k]] = value
                state["status"][i] = "tested" if state["changed"][i].any() else "unchanged_all_scrambles"
            state_write(state_path, state)
            print(f"{target['gene_id']} window {i+1}/{len(windows)} {target['chrom']}:{a}-{b} {state['status'][i]}", flush=True)
        output_results(folder, contract, state, args)


def add_call_args(parser):
    parser.add_argument("--threshold", type=float, required=True, help="Exploratory effect threshold, in --effect-units")
    parser.add_argument("--effect-units", choices=UNITS, default="log1p")
    parser.add_argument("--min-region-bp", type=int, default=1000)
    parser.add_argument("--smooth-windows", type=int, default=1, help="Positive odd centered window count; default no smoothing")
    parser.add_argument("--promoter-half-bp", type=int, default=5000)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Reference FASTA -> native astrocyte perturbation scan")
    scan.add_argument("--targets", help="Existing HARI target TSV accepted; bundle_path is ignored")
    scan.add_argument("--gene-id", help="One target gene; use with --tss-bed")
    scan.add_argument("--tss-bed", help="One-base BED6 with gene ID in column 4")
    scan.add_argument("--genes", help="Comma-separated subset of --targets")
    scan.add_argument("--fasta", required=True)
    scan.add_argument("--gtf", required=True, help="Same-build GTF, needed for fixed exon union and strand")
    scan.add_argument("--assembly", required=True)
    scan.add_argument("--out-dir", required=True)
    scan.add_argument("--checkpoint", default="all_folds")
    scan.add_argument("--sequence-length", type=int, default=1048576)
    scan.add_argument("--region", help="Optional exact 1-based inclusive model input interval; one gene only")
    scan.add_argument("--window-bp", type=int, default=5000)
    scan.add_argument("--stride-bp", type=int, default=1000)
    scan.add_argument("--repeats", type=int, default=5)
    scan.add_argument("--shuffle", choices=("mononucleotide", "dinucleotide"), default="mononucleotide")
    scan.add_argument("--seed", type=int, default=42)
    scan.add_argument("--batch-size", type=int, default=1)
    scan.add_argument("--scan-half-window-bp", type=int)
    scan.add_argument("--max-windows", type=int, help="Pilot cap; use a different output directory for a full scan")
    scan.add_argument("--min-exon-fraction", type=float, default=0.5)
    scan.add_argument("--preflight", action="store_true", help="Validate FASTA/TSS/GTF/coordinates without loading AlphaGenome or writing results")
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--save-baseline-tracks", action="store_true", help="Save reference 1 bp +/- astrocyte RNA tracks only")
    scan.add_argument("--skip-public-check", action="store_true", help="Skip default intact direct/public native wrapper parity check")
    add_call_args(scan)
    recalc = sub.add_parser("recalculate", aliases=["recall"], help="Rewrite calls/plots from saved repeat scalars; no GPU")
    recalc.add_argument("--gene-dir", required=True)
    add_call_args(recalc)
    prep = sub.add_parser("prepare-targets", help="BED6 + gene list -> clean native target TSV, without bundle filtering")
    prep.add_argument("--tss-bed", required=True)
    prep.add_argument("--genes-file", required=True, help="One gene ID per line; blank lines and # comments allowed")
    prep.add_argument("--output", required=True)
    return ap


def main(argv=None, backend_factory=NativeAstrocyte):
    args = parser().parse_args(argv)
    if args.command == "prepare-targets":
        genes = [line.split()[0] for line in Path(args.genes_file).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if len({base_gene(g) for g in genes}) != len(genes):
            raise ValueError("Gene list contains duplicates/version aliases")
        rows, skipped = bed_targets(args.tss_bed, genes)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["gene_id", "chrom", "tss0", "strand"]).to_csv(output, sep="\t", index=False)
        pd.DataFrame(skipped, columns=["gene_id", "reason"]).to_csv(str(output) + ".skipped.tsv", sep="\t", index=False)
        print(f"Prepared {len(rows)} targets; skipped {len(skipped)}")
        return
    if not np.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("--threshold must be finite and nonnegative")
    if args.min_region_bp < 1 or args.smooth_windows < 1 or args.smooth_windows % 2 != 1 or args.promoter_half_bp < 0:
        raise ValueError("Invalid region length, smoothing (positive odd), or promoter span")
    if args.command in ("recalculate", "recall"):
        folder = Path(args.gene_dir).resolve()
        if not folder.is_dir():
            raise ValueError("--gene-dir must be an existing scan directory")
        with writer_lock(folder):
            meta = json.loads((folder / "scan_metadata.json").read_text())
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported scan state format")
            with np.load(folder / "scan_state.npz", allow_pickle=False) as f:
                state = {k:f[k] for k in f.files}
            output_results(folder, meta, state, args)
        return
    if args.sequence_length not in (16384, 131072, 524288, 1048576):
        raise ValueError("Use a standard native AlphaGenome length: 16384, 131072, 524288, or 1048576; legacy 102400 embedding windows are not supported here")
    if min(args.window_bp, args.stride_bp, args.repeats, args.batch_size) <= 0:
        raise ValueError("Window, stride, repeats and batch size must be positive")
    if not 0 < args.min_exon_fraction <= 1:
        raise ValueError("--min-exon-fraction must be in (0,1]")
    if args.scan_half_window_bp is not None and args.scan_half_window_bp <= 0:
        raise ValueError("--scan-half-window-bp must be positive")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")
    if bool(args.targets) == bool(args.gene_id):
        raise ValueError("Use either --targets [--genes ...] or --gene-id ... --tss-bed ...")
    if args.targets:
        if args.tss_bed:
            raise ValueError("Use --targets or --tss-bed; not both")
        targets = read_targets(args.targets)
        if args.genes:
            requested = {base_gene(x.strip()) for x in args.genes.split(",") if x.strip()}
            absent = requested - {base_gene(t["gene_id"]) for t in targets}
            if absent:
                raise ValueError(f"Genes absent from targets: {sorted(absent)}")
            targets = [t for t in targets if base_gene(t["gene_id"]) in requested]
    else:
        if not args.tss_bed or args.genes:
            raise ValueError("--gene-id requires --tss-bed and cannot use --genes")
        targets, skipped = bed_targets(args.tss_bed, [args.gene_id])
        if skipped:
            raise ValueError(f"Cannot select unique TSS: {skipped}")
    if args.region and len(targets) != 1:
        raise ValueError("--region requires exactly one selected gene")
    if not targets:
        raise ValueError("No selected targets")
    fasta = IndexedFasta(args.fasta)
    args._gtf_sha256 = sha_file(args.gtf)
    backend = None
    for target in targets:
        prepared = prepare_gene(args, target, fasta)
        print(f"{target['gene_id']}: {len(prepared[1]):,} bp input; {prepared[2].sum():,} exonic bp; {len(prepared[4]):,} windows; at most {len(prepared[4])*args.repeats:,} perturbation passes plus baseline/check.", flush=True)
        if args.preflight:
            continue
        if backend is None:
            backend = backend_factory(args.checkpoint)
        run_gene(args, prepared, backend)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)