#!/usr/bin/env python3
"""
NativeAlphaGenomeRNAExpression.py

Zero-shot gene-expression extraction from the *native* AlphaGenome RNA-seq
prediction head using pre-existing one-hot encoded personalized sequences.

This intentionally does NOT train an expression head and does NOT use observed
expression or covariates.  It:

  1. loads one-hot sequence matrices (N, L, 4), or phased (N, 2, L, 4),
  2. runs the frozen AlphaGenome "all_folds" model directly on those matrices,
  3. retains only the native RNA-seq tracks for:
       - Astrocyte              CL:0000127 total RNA-seq
       - Glutamatergic neuron   CL:0000679 total RNA-seq
     (both + and - strand tracks are retained),
  4. selects the track matching the target gene strand,
  5. reduces the 1-bp RNA-seq prediction to a gene-level score by taking the
     mean predicted coverage across the union of the gene's exonic bases,
  6. writes both the raw mean-exon-coverage score and log1p(score).

The mean-over-exons aggregation mirrors the gene-resolution RNA-seq evaluation
reported for AlphaGenome ("average exon coverage").  The resulting values are
AlphaGenome-native expression scores; they are NOT TPM.

Input orientation
-----------------
The one-hot sequence MUST be in forward genomic orientation.  Do not
reverse-complement minus-strand genes before supplying the NPZ.  Gene strand is
handled by selecting the corresponding native +/- RNA-seq output track.

Coordinate handling
-------------------
The script needs to know which genomic interval the one-hot matrix represents.
Use either:

  --region chrN:START-END
      Explicit 1-based inclusive interval. Recommended when available.

or

  --tss-bed FILE
      Reconstruct the interval exactly as a TSS-centred window whose width is
      the sequence length L. This is compatible with the older workflow that
      created 102,400-bp TSS-centred hottools matrices.

A GTF is required to define the union of target-gene exons and its strand.
GENCODE v46 is a sensible choice for matching the AlphaGenome-era annotation.

Outputs
-------
<out-dir>/<gene-id>/
  <gene>.alphagenome_native_expression.tsv.gz
      One row per sequence unit x cell type. Contains:
      mean_exon_coverage and log1p_mean_exon_coverage.

  <gene>.alphagenome_native_expression_wide.tsv.gz
      Convenient one-row-per-sequence-unit representation.

  <gene>.rnaseq_track_metadata.tsv
      Metadata and column order of selected native RNA-seq tracks.

  <gene>.sequence_units.tsv
      Mapping from prediction row to sample/haplotype.

  <gene>.run_metadata.json
      Coordinates, exon coverage, model/checkpoint and output details.

  <gene>.native_rnaseq_tracks.npy       [only with --save-tracks]
      Float32 memory-mapped NPY with shape (n_units, L, n_selected_tracks).
      This keeps the original 1-bp AlphaGenome RNA-seq predictions for both
      requested cell types and both strands without collecting all tracks in
      RAM first.

Example
-------
python NativeAlphaGenomeRNAExpression.py \
  --npz-path /path/ENSG00000182240.onehot.npz \
  --gene-id ENSG00000182240 \
  --gtf /path/gencode.v46.annotation.gtf.gz \
  --tss-bed /path/GRCh38_TSS.bed \
  --out-dir /path/native_alphagenome_rna \
  --batch-size 2 \
  --save-tracks
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from alphagenome_research.model import dna_model as ag_dna_model
from alphagenome.models import dna_model as core_dna_model
from alphagenome.models import dna_output


# -----------------------------------------------------------------------------
# Exact native RNA-seq tracks requested for this benchmark.
# Metadata contains a + and - instance of each track name.
# -----------------------------------------------------------------------------
TARGET_TRACKS = {
    "Astrocyte": {
        "ontology_curie": "CL:0000127",
        "track_name": "CL:0000127 total RNA-seq",
    },
    "Glutamatergic_neuron": {
        "ontology_curie": "CL:0000679",
        "track_name": "CL:0000679 total RNA-seq",
    },
}

RNA_OUTPUT = dna_output.OutputType.RNA_SEQ


# -----------------------------------------------------------------------------
# NPZ loading
# -----------------------------------------------------------------------------

def _find_onehot_array(npz: np.lib.npyio.NpzFile) -> Tuple[str, np.ndarray]:
    preferred = ["onehot", "one_hot", "onehot_Lx4", "sequences", "seqs", "X"]
    for key in preferred:
        if key in npz.files:
            return key, npz[key]

    best_key: Optional[str] = None
    best_size = -1
    for key in npz.files:
        arr = npz[key]
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
            if arr.size > best_size:
                best_key, best_size = key, arr.size

    if best_key is None:
        raise ValueError(f"No numeric one-hot-like array found. NPZ keys: {list(npz.files)}")
    return best_key, npz[best_key]


def _find_sample_ids(npz: np.lib.npyio.NpzFile, n_samples: int) -> np.ndarray:
    for key in ("sample_ids", "samples", "sample_id", "ids"):
        if key in npz.files:
            ids = np.asarray(npz[key]).astype(str)
            if ids.ndim != 1:
                ids = ids.reshape(-1)
            if len(ids) != n_samples:
                raise ValueError(
                    f"NPZ key {key!r} has {len(ids)} IDs but sequence array has "
                    f"{n_samples} samples."
                )
            return ids
    raise KeyError(
        "NPZ must contain sample identifiers under one of: "
        "sample_ids, samples, sample_id, ids."
    )


@dataclass
class SequenceUnits:
    onehot: np.ndarray               # (M, L, 4)
    sample_ids: np.ndarray           # (M,)
    haplotypes: np.ndarray           # (M,), '.' if unphased/single matrix
    unit_ids: np.ndarray             # (M,)
    source_shape: Tuple[int, ...]
    source_array_key: str


def _normalize_onehot_and_units(arr: np.ndarray, sample_ids: np.ndarray, key: str) -> SequenceUnits:
    """Normalize supported layouts to prediction units (M, L, 4).

    Supported:
      (N, L, 4)
      (N, 4, L)
      (L, 4) / (4, L) for a single sample
      (N, 2, L, 4) phased haplotypes
      (N, L, 2, 4) phased haplotypes

    For phased input, N samples become 2N prediction units while preserving the
    original sample ID and recording hap1/hap2.
    """
    arr = np.asarray(arr)
    source_shape = tuple(int(x) for x in arr.shape)

    # Single sequence -> N=1.
    if arr.ndim == 2:
        if len(sample_ids) != 1:
            raise ValueError(
                f"2D one-hot implies one sequence but found {len(sample_ids)} sample IDs."
            )
        if arr.shape[-1] == 4:
            out = arr[None, ...]
        elif arr.shape[0] == 4:
            out = arr.T[None, ...]
        else:
            raise ValueError(f"Cannot interpret 2D one-hot shape {arr.shape}; expected (L,4) or (4,L).")
        return SequenceUnits(
            onehot=out.astype(np.float32, copy=False),
            sample_ids=sample_ids.copy(),
            haplotypes=np.asarray(["."], dtype=str),
            unit_ids=sample_ids.copy(),
            source_shape=source_shape,
            source_array_key=key,
        )

    # Ordinary N x L x 4.
    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            out = arr
        elif arr.shape[1] == 4:
            out = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(
                f"Cannot interpret 3D one-hot shape {arr.shape}; expected (N,L,4) or (N,4,L)."
            )
        if out.shape[0] != len(sample_ids):
            raise ValueError(
                f"Sequence first dimension={out.shape[0]} but sample_ids={len(sample_ids)}."
            )
        return SequenceUnits(
            onehot=out.astype(np.float32, copy=False),
            sample_ids=sample_ids.copy(),
            haplotypes=np.full(len(sample_ids), ".", dtype=str),
            unit_ids=sample_ids.copy(),
            source_shape=source_shape,
            source_array_key=key,
        )

    # Phased N x 2 x L x 4 or N x L x 2 x 4.
    if arr.ndim == 4 and arr.shape[-1] == 4:
        if arr.shape[0] != len(sample_ids):
            raise ValueError(
                f"Phased sequence first dimension={arr.shape[0]} but sample_ids={len(sample_ids)}."
            )

        if arr.shape[1] == 2:
            phased = arr                         # (N,2,L,4)
        elif arr.shape[2] == 2:
            phased = np.transpose(arr, (0, 2, 1, 3))  # (N,2,L,4)
        else:
            raise ValueError(
                f"4D input {arr.shape} is only supported when a haplotype axis of length 2 "
                "is axis 1 or 2."
            )

        n, h, L, four = phased.shape
        out = phased.reshape(n * h, L, four).astype(np.float32, copy=False)
        sid = np.repeat(sample_ids, 2)
        hap = np.tile(np.asarray(["hap1", "hap2"], dtype=str), n)
        uid = np.asarray([f"{s}__{hh}" for s, hh in zip(sid, hap)], dtype=str)
        return SequenceUnits(
            onehot=out,
            sample_ids=sid,
            haplotypes=hap,
            unit_ids=uid,
            source_shape=source_shape,
            source_array_key=key,
        )

    raise ValueError(
        f"Unsupported one-hot shape {arr.shape}. Expected (N,L,4), (N,4,L), "
        "or phased (N,2,L,4)/(N,L,2,4)."
    )


def load_sequence_units(npz_path: str) -> SequenceUnits:
    print(f"Loading one-hot sequences: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as z:
        key, raw = _find_onehot_array(z)
        # First dimension is the biological sample dimension for all supported
        # multi-sample layouts; 2D input is a single sample.
        n_samples = 1 if raw.ndim == 2 else int(raw.shape[0])
        sample_ids = _find_sample_ids(z, n_samples)
        units = _normalize_onehot_and_units(raw, sample_ids, key)

    if units.onehot.shape[-1] != 4:
        raise AssertionError("Internal error: normalized one-hot does not end in 4 channels.")
    if not np.isfinite(units.onehot).all():
        raise ValueError("One-hot matrix contains NaN or infinite values.")

    print(f"  source key/shape : {units.source_array_key} {units.source_shape}")
    print(f"  prediction units : {units.onehot.shape}")
    print(f"  unique samples   : {len(np.unique(units.sample_ids))}")
    if np.any(units.haplotypes != "."):
        print("  phased input      : yes (hap1/hap2 predicted independently)")
    else:
        print("  phased input      : no explicit haplotype axis")
    return units


# -----------------------------------------------------------------------------
# Region reconstruction
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    chrom: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


def _chrom_key(chrom: str) -> str:
    chrom = str(chrom)
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def parse_region(region: str) -> Region:
    m = re.fullmatch(r"([^:]+):(\d+)-(\d+)", region.replace(",", ""))
    if not m:
        raise ValueError(
            f"Invalid --region {region!r}; expected chrN:START-END with 1-based inclusive coordinates."
        )
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start < 1 or end < start:
        raise ValueError(f"Invalid region coordinates: {region!r}")
    return Region(chrom, start, end)


def _gene_id_matches(candidate: str, requested: str) -> bool:
    if candidate == requested:
        return True
    # Permit ENSG... version mismatch only as a fallback.
    return candidate.split(".", 1)[0] == requested.split(".", 1)[0]


def reconstruct_region_from_tss_bed(tss_bed: str, gene_id: str, sequence_length: int) -> Region:
    exact: Optional[Tuple[str, int]] = None
    fallback: Optional[Tuple[str, int]] = None

    with open(tss_bed, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                fields = line.split()
            if len(fields) < 4:
                continue
            candidate = fields[3]
            if candidate == gene_id:
                exact = (fields[0], int(fields[1]))
                break
            if fallback is None and _gene_id_matches(candidate, gene_id):
                fallback = (fields[0], int(fields[1]))

    hit = exact or fallback
    if hit is None:
        raise KeyError(f"Gene {gene_id!r} not found in TSS BED: {tss_bed}")

    chrom, tss0 = hit
    tss1 = tss0 + 1
    left = sequence_length // 2
    start = tss1 - left
    end = start + sequence_length - 1

    if start < 1:
        raise ValueError(
            f"Reconstructed TSS-centred interval starts before chromosome base 1 ({start}). "
            "For a chromosome-edge gene, pass the exact interval used to build the NPZ via --region."
        )

    region = Region(chrom=chrom, start=start, end=end)
    print(f"Reconstructed NPZ interval from TSS BED: {region}")
    return region


def resolve_region(args: argparse.Namespace, sequence_length: int) -> Region:
    if args.region:
        region = parse_region(args.region)
    elif args.tss_bed:
        region = reconstruct_region_from_tss_bed(args.tss_bed, args.gene_id, sequence_length)
    else:
        raise ValueError("Provide either --region or --tss-bed so the one-hot bases can be mapped to genome coordinates.")

    if region.width != sequence_length:
        raise ValueError(
            f"Sequence length is {sequence_length:,} bp but resolved interval {region} is "
            f"{region.width:,} bp. The interval must exactly describe the supplied one-hot matrix."
        )
    return region


# -----------------------------------------------------------------------------
# GTF exon parsing and union mask
# -----------------------------------------------------------------------------

@dataclass
class GeneAnnotation:
    gene_id: str
    chrom: str
    strand: str
    exon_intervals: List[Tuple[int, int]]  # 1-based inclusive, merged union

    @property
    def total_exonic_bp(self) -> int:
        return int(sum(end - start + 1 for start, end in self.exon_intervals))


def _open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _extract_gtf_gene_id(attributes: str) -> Optional[str]:
    # Standard GTF: gene_id "ENSG...";
    m = re.search(r'(?:^|;\s*)gene_id\s+"([^"]+)"', attributes)
    if m:
        return m.group(1)
    # Small GFF/GTF compatibility fallback: gene_id=ENSG...
    m = re.search(r'(?:^|;)\s*gene_id=([^;]+)', attributes)
    return m.group(1).strip() if m else None


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted((int(s), int(e)) for s, e in intervals)
    if not ordered:
        return []
    merged: List[List[int]] = [[ordered[0][0], ordered[0][1]]]
    for s, e in ordered[1:]:
        last = merged[-1]
        if s <= last[1] + 1:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def load_gene_exons(gtf_path: str, gene_id: str) -> GeneAnnotation:
    exact_rows: List[Tuple[str, int, int, str]] = []
    fallback_rows: List[Tuple[str, int, int, str]] = []

    with _open_text(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            candidate = _extract_gtf_gene_id(fields[8])
            if candidate is None:
                continue
            row = (fields[0], int(fields[3]), int(fields[4]), fields[6])
            if candidate == gene_id:
                exact_rows.append(row)
            elif _gene_id_matches(candidate, gene_id):
                fallback_rows.append(row)

    rows = exact_rows if exact_rows else fallback_rows
    if not rows:
        raise KeyError(f"No exon records found for gene {gene_id!r} in {gtf_path}")

    chroms = {_chrom_key(r[0]) for r in rows}
    strands = {r[3] for r in rows}
    if len(chroms) != 1:
        raise ValueError(f"Gene {gene_id} has exon records on multiple chromosomes: {sorted(chroms)}")
    if strands - {"+", "-"} or len(strands) != 1:
        raise ValueError(f"Gene {gene_id} has unexpected/multiple exon strands: {sorted(strands)}")

    # Keep a representative chromosome label from the GTF.
    chrom = rows[0][0]
    strand = rows[0][3]
    merged = _merge_intervals((r[1], r[2]) for r in rows)
    return GeneAnnotation(gene_id=gene_id, chrom=chrom, strand=strand, exon_intervals=merged)


def build_exon_mask(annotation: GeneAnnotation, region: Region, sequence_length: int) -> Tuple[np.ndarray, int, float]:
    if _chrom_key(annotation.chrom) != _chrom_key(region.chrom):
        raise ValueError(
            f"GTF chromosome for {annotation.gene_id} is {annotation.chrom}, but NPZ interval is {region.chrom}."
        )

    mask = np.zeros(sequence_length, dtype=bool)
    for exon_start, exon_end in annotation.exon_intervals:
        ov_start = max(exon_start, region.start)
        ov_end = min(exon_end, region.end)
        if ov_start > ov_end:
            continue
        i0 = ov_start - region.start
        i1 = ov_end - region.start + 1  # half-open array slice
        mask[i0:i1] = True

    in_window = int(mask.sum())
    total = annotation.total_exonic_bp
    if in_window == 0:
        raise ValueError(
            f"No exonic bases for {annotation.gene_id} overlap input interval {region}. "
            "Check the NPZ interval / genome build / GTF."
        )
    fraction = in_window / total if total > 0 else float("nan")
    return mask, in_window, fraction


# -----------------------------------------------------------------------------
# AlphaGenome metadata and direct one-hot prediction
# -----------------------------------------------------------------------------


def _get_human_metadata(model, organism_enum):
    metadata_root = model._metadata
    # Current alphagenome_research stores metadata as {Organism: metadata}.
    try:
        return metadata_root[organism_enum]
    except (KeyError, TypeError):
        # Compatibility fallback for an older local wrapper exposing one organism
        # directly as the metadata object.
        if hasattr(metadata_root, "get") and metadata_root.get(RNA_OUTPUT) is not None:
            return metadata_root
        raise RuntimeError("Could not resolve AlphaGenome human output metadata from model._metadata")


def select_target_rna_tracks(metadata) -> Tuple[np.ndarray, pd.DataFrame, Dict[Tuple[str, str], int]]:
    all_meta = metadata.get(RNA_OUTPUT)
    if all_meta is None:
        raise RuntimeError("Loaded AlphaGenome checkpoint exposes no RNA_SEQ metadata.")

    wanted_names = {spec["track_name"] for spec in TARGET_TRACKS.values()}
    name_series = all_meta["name"].astype(str)
    mask = name_series.isin(wanted_names).to_numpy(dtype=bool)
    selected = all_meta.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "original_track_index"})

    if selected.empty:
        raise RuntimeError(
            "None of the requested astrocyte/glutamatergic RNA-seq tracks were found in this "
            "AlphaGenome checkpoint metadata."
        )

    # Validate exact expected +/- pair for each cell type and build index map in
    # the FILTERED prediction tensor.
    index_map: Dict[Tuple[str, str], int] = {}
    errors: List[str] = []
    for cell_type, spec in TARGET_TRACKS.items():
        sub = selected[selected["name"].astype(str) == spec["track_name"]]
        strands = list(sub["strand"].astype(str))
        for strand in ("+", "-"):
            loc = sub.index[sub["strand"].astype(str) == strand].tolist()
            if len(loc) != 1:
                errors.append(
                    f"{cell_type}: expected exactly one {strand} track named {spec['track_name']!r}; "
                    f"found {len(loc)} (all strands={strands})"
                )
            else:
                index_map[(cell_type, strand)] = int(loc[0])

    if errors:
        details = "\n  - ".join(errors)
        raise RuntimeError(f"Requested RNA-seq metadata did not match expectations:\n  - {details}")

    # Add user-facing cell type labels.
    track_to_cell = {spec["track_name"]: cell for cell, spec in TARGET_TRACKS.items()}
    selected.insert(0, "selected_track_index", np.arange(len(selected), dtype=int))
    selected.insert(1, "cell_type", selected["name"].map(track_to_cell))

    return mask, selected, index_map



def build_native_rna_predictor(model):
    """Build a JIT-compiled RNA-only AlphaGenome predictor.

    AlphaGenome's internal _predict function accepts ``requested_outputs`` as a
    Python/static argument.  In some JAX versions, calling the already-jitted
    private ``model._predict`` method directly with OutputType.RNA_SEQ can cause
    JAX to treat the enum as a dynamic array argument.

    To avoid that, recover the underlying (un-jitted) function and close over
    ``RNA_OUTPUT`` before applying our own jax.jit.  Runtime arguments are then
    JAX arrays / pytrees only.
    """
    base_predict = getattr(model._predict, "__wrapped__", None)
    if base_predict is None:
        raise RuntimeError(
            "Could not access the underlying AlphaGenome _predict function via "
            "model._predict.__wrapped__. This AlphaGenome version may expose a "
            "different private API."
        )

    def _predict_rna(
        params,
        state,
        sequences,
        organism_indices,
        negative_strand_mask,
        strand_reindexing,
    ):
        return base_predict(
            params,
            state,
            sequences,
            organism_indices,
            requested_outputs=(RNA_OUTPUT,),
            negative_strand_mask=negative_strand_mask,
            strand_reindexing=strand_reindexing,
        )

    return jax.jit(_predict_rna)


def predict_native_rna_batch(
    model,
    rna_predict_fn,
    metadata,
    track_mask_np: np.ndarray,
    batch_np: np.ndarray,
    organism_enum,
) -> np.ndarray:
    """Direct one-hot equivalent of AlphaGenome predict_sequence for RNA_SEQ.

    The public wrapper encodes a DNA string to [1,L,4] and then calls
    model._predict. Here the caller already owns [B,L,4], so only that encoding
    step is bypassed. Inputs are assumed to be forward genomic orientation.
    """
    B = int(batch_np.shape[0])
    org_idx = int(ag_dna_model.convert_to_organism_index(organism_enum))

    with model._device_context as device, jax.transfer_guard("disallow"):
        seq_dev = jax.device_put(batch_np.astype(np.float32, copy=False), device)
        org_dev = jax.device_put(np.full(B, org_idx, dtype=np.int32), device)
        neg_dev = jax.device_put(np.zeros(B, dtype=bool), device)
        strand_reindexing_dev = jax.device_put(metadata.strand_reindexing, device)

        predictions = rna_predict_fn(
            model._params,
            model._state,
            seq_dev,
            org_dev,
            neg_dev,
            strand_reindexing_dev,
        )

        # Filter channels on device before transferring the 1-bp tracks to CPU.
        selected_indices_dev = jax.device_put(np.flatnonzero(track_mask_np).astype(np.int32), device)
        rna_dev = jnp.take(predictions[RNA_OUTPUT], selected_indices_dev, axis=-1)
        rna_dev = rna_dev.astype(jnp.float32)

    rna = np.asarray(jax.device_get(rna_dev), dtype=np.float32)
    if rna.ndim != 3:
        raise RuntimeError(f"Unexpected RNA prediction shape: {rna.shape}; expected (B,L,K).")
    return rna


# -----------------------------------------------------------------------------
# Expression aggregation and outputs
# -----------------------------------------------------------------------------


def _safe_float(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def make_wide_expression(long_df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["unit_id", "sample_id", "haplotype", "gene_id"]
    pieces = []
    for metric in ("mean_exon_coverage", "log1p_mean_exon_coverage"):
        p = long_df.pivot(index=index_cols, columns="cell_type", values=metric)
        suffix = "AG_mean_exon_coverage" if metric == "mean_exon_coverage" else "AG_log1p_mean_exon_coverage"
        p.columns = [f"{c}_{suffix}" for c in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1).reset_index()
    return wide


def make_diploid_summary(long_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    phased = long_df[long_df["haplotype"].isin(["hap1", "hap2"])]
    if phased.empty:
        return None

    counts = phased.groupby(["sample_id", "gene_id", "cell_type"])["haplotype"].nunique()
    if (counts < 2).any():
        print("WARNING: Some sample/cell-type groups lack both haplotypes; diploid summary will omit them.", file=sys.stderr)

    # Combine alleles on the raw coverage scale first, then log1p. This avoids
    # averaging already log-transformed allelic predictions.
    dip = (
        phased.groupby(["sample_id", "gene_id", "cell_type"], as_index=False)
        .agg(
            n_haplotypes=("haplotype", "nunique"),
            diploid_mean_exon_coverage=("mean_exon_coverage", "mean"),
        )
    )
    dip = dip[dip["n_haplotypes"] == 2].copy()
    dip["diploid_log1p_mean_exon_coverage"] = np.log1p(dip["diploid_mean_exon_coverage"])
    return dip


def run(args: argparse.Namespace) -> None:
    units = load_sequence_units(args.npz_path)

    # Optional smoke-test mode: restrict inference to the first M sequence units.
    # This is useful for validating a 1-Mb AlphaGenome forward pass before
    # launching the full donor x haplotype cohort.
    if args.max_units is not None:
        max_units = int(args.max_units)
        if max_units < 1:
            raise ValueError("--max-units must be >= 1 when provided.")
        if max_units < units.onehot.shape[0]:
            print(
                f"Smoke-test mode: restricting prediction units "
                f"{units.onehot.shape[0]} -> {max_units}"
            )
            units = SequenceUnits(
                onehot=units.onehot[:max_units],
                sample_ids=units.sample_ids[:max_units],
                haplotypes=units.haplotypes[:max_units],
                unit_ids=units.unit_ids[:max_units],
                source_shape=units.source_shape,
                source_array_key=units.source_array_key,
            )

    n_units, sequence_length, _ = units.onehot.shape

    region = resolve_region(args, sequence_length)
    print(f"Input genomic interval: {region} ({region.width:,} bp)")

    annotation = load_gene_exons(args.gtf, args.gene_id)
    exon_mask, exonic_bp_in_window, exon_fraction = build_exon_mask(annotation, region, sequence_length)
    print(f"Gene strand: {annotation.strand}")
    print(f"Gene exonic union: {annotation.total_exonic_bp:,} bp total")
    print(
        f"Exonic bases inside input window: {exonic_bp_in_window:,} "
        f"({exon_fraction:.2%} of gene exon union)"
    )
    if exon_fraction < 0.999999:
        print(
            "WARNING: The input sequence does not contain all exonic bases for this gene. "
            "Expression scores will use only exonic bases present in the supplied interval.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir) / args.gene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    organism_enum = core_dna_model.Organism.HOMO_SAPIENS

    print("\nLoading AlphaGenome all_folds checkpoint (no training)...")
    model = ag_dna_model.create_from_huggingface(args.checkpoint)

    print("Building RNA-only AlphaGenome JIT forward...")
    rna_predict_fn = build_native_rna_predictor(model)

    metadata = _get_human_metadata(model, organism_enum)

    track_mask, selected_meta, track_index = select_target_rna_tracks(metadata)
    n_tracks = int(track_mask.sum())
    print("\nSelected native RNA-seq tracks:")
    cols_to_show = [c for c in ("selected_track_index", "cell_type", "name", "strand", "ontology_curie") if c in selected_meta.columns]
    print(selected_meta[cols_to_show].to_string(index=False))
    print(f"Selected {n_tracks} RNA-seq tracks total.")

    track_meta_path = out_dir / f"{args.gene_id}.rnaseq_track_metadata.tsv"
    selected_meta.to_csv(track_meta_path, sep="\t", index=False)

    units_df = pd.DataFrame({
        "prediction_row": np.arange(n_units, dtype=int),
        "unit_id": units.unit_ids,
        "sample_id": units.sample_ids,
        "haplotype": units.haplotypes,
    })
    units_path = out_dir / f"{args.gene_id}.sequence_units.tsv"
    units_df.to_csv(units_path, sep="\t", index=False)

    track_mmap = None
    tracks_path: Optional[Path] = None
    if args.save_tracks:
        tracks_path = out_dir / f"{args.gene_id}.native_rnaseq_tracks.npy"
        print(
            f"Saving selected native 1-bp RNA-seq tracks to {tracks_path}\n"
            f"  shape=({n_units}, {sequence_length}, {n_tracks}), dtype=float32"
        )
        track_mmap = np.lib.format.open_memmap(
            tracks_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_units, sequence_length, n_tracks),
        )

    records: List[Dict[str, object]] = []
    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    print("\nRunning native AlphaGenome RNA-seq prediction...")
    for i0 in range(0, n_units, batch_size):
        i1 = min(i0 + batch_size, n_units)
        batch = units.onehot[i0:i1]
        rna = predict_native_rna_batch(
            model=model,
            rna_predict_fn=rna_predict_fn,
            metadata=metadata,
            track_mask_np=track_mask,
            batch_np=batch,
            organism_enum=organism_enum,
        )

        if rna.shape[1] != sequence_length:
            raise RuntimeError(
                f"AlphaGenome returned RNA length {rna.shape[1]:,}, but input sequence length is "
                f"{sequence_length:,}. Cannot map predictions to exon coordinates safely."
            )
        if rna.shape[2] != n_tracks:
            raise RuntimeError(
                f"AlphaGenome returned {rna.shape[2]} selected RNA tracks; expected {n_tracks}."
            )

        if track_mmap is not None:
            track_mmap[i0:i1] = rna

        for local_i, global_i in enumerate(range(i0, i1)):
            for cell_type, spec in TARGET_TRACKS.items():
                idx = track_index[(cell_type, annotation.strand)]
                exon_values = rna[local_i, exon_mask, idx]
                mean_cov = float(np.mean(exon_values, dtype=np.float64))
                log_expr = float(np.log1p(mean_cov))

                row_meta = selected_meta.iloc[idx]
                records.append({
                    "unit_id": str(units.unit_ids[global_i]),
                    "sample_id": str(units.sample_ids[global_i]),
                    "haplotype": str(units.haplotypes[global_i]),
                    "gene_id": args.gene_id,
                    "cell_type": cell_type,
                    "ontology_curie": spec["ontology_curie"],
                    "track_name": spec["track_name"],
                    "track_strand": annotation.strand,
                    "gene_strand": annotation.strand,
                    "chrom": region.chrom,
                    "region_start_1based": region.start,
                    "region_end_1based": region.end,
                    "sequence_length_bp": sequence_length,
                    "gene_exonic_bp_total": annotation.total_exonic_bp,
                    "exonic_bp_in_window": exonic_bp_in_window,
                    "exon_coverage_fraction": exon_fraction,
                    "mean_exon_coverage": mean_cov,
                    "log1p_mean_exon_coverage": log_expr,
                    "selected_track_index": int(idx),
                    "original_track_index": int(row_meta["original_track_index"]),
                })

        print(f"  predicted {i1}/{n_units} sequence units")

    if track_mmap is not None:
        track_mmap.flush()
        del track_mmap

    long_df = pd.DataFrame.from_records(records)
    long_path = out_dir / f"{args.gene_id}.alphagenome_native_expression.tsv.gz"
    long_df.to_csv(long_path, sep="\t", index=False, compression="gzip")

    wide_df = make_wide_expression(long_df)
    wide_path = out_dir / f"{args.gene_id}.alphagenome_native_expression_wide.tsv.gz"
    wide_df.to_csv(wide_path, sep="\t", index=False, compression="gzip")

    diploid_df = make_diploid_summary(long_df)
    diploid_path: Optional[Path] = None
    if diploid_df is not None:
        diploid_path = out_dir / f"{args.gene_id}.alphagenome_native_expression_diploid.tsv.gz"
        diploid_df.to_csv(diploid_path, sep="\t", index=False, compression="gzip")

    run_meta = {
        "script": Path(__file__).name,
        "checkpoint": args.checkpoint,
        "model_mode": "native AlphaGenome zero-shot RNA-seq; no learned expression head",
        "npz_path": str(Path(args.npz_path).resolve()),
        "npz_array_key": units.source_array_key,
        "npz_source_shape": list(units.source_shape),
        "prediction_shape": [int(x) for x in units.onehot.shape],
        "gene_id": args.gene_id,
        "gtf": str(Path(args.gtf).resolve()),
        "gene_strand": annotation.strand,
        "region": str(region),
        "coordinate_convention": "region is 1-based inclusive; array index 0 corresponds to region.start",
        "input_orientation": "forward genomic",
        "gene_exonic_bp_total": annotation.total_exonic_bp,
        "exonic_bp_in_window": exonic_bp_in_window,
        "exon_coverage_fraction": exon_fraction,
        "expression_definition": "mean AlphaGenome RNA-seq prediction across union of target-gene exonic bases present in input interval",
        "expression_transform": "log1p(mean_exon_coverage) also reported",
        "expression_units": "AlphaGenome native RNA-seq prediction scale; not TPM",
        "target_tracks": TARGET_TRACKS,
        "n_selected_tracks": n_tracks,
        "save_tracks": bool(args.save_tracks),
        "tracks_file": str(tracks_path) if tracks_path is not None else None,
        "batch_size": batch_size,
        "max_units": args.max_units,
    }
    meta_path = out_dir / f"{args.gene_id}.run_metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(run_meta, handle, indent=2)

    print("\nDone.")
    print(f"  expression (long): {long_path}")
    print(f"  expression (wide): {wide_path}")
    if diploid_path is not None:
        print(f"  diploid summary   : {diploid_path}")
    print(f"  track metadata    : {track_meta_path}")
    print(f"  sequence units    : {units_path}")
    print(f"  run metadata      : {meta_path}")
    if tracks_path is not None:
        print(f"  native RNA tracks : {tracks_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Zero-shot native AlphaGenome RNA-seq -> gene expression for astrocyte "
            "and glutamatergic neuron from pre-existing one-hot matrices."
        )
    )
    ap.add_argument("--npz-path", required=True, help="NPZ containing one-hot sequence matrix and sample IDs")
    ap.add_argument("--gene-id", required=True, help="Target gene ID (e.g. ENSG00000182240)")
    ap.add_argument("--gtf", required=True, help="GENCODE GTF or GTF.gz used to define target-gene exon union and strand")

    coord = ap.add_mutually_exclusive_group(required=True)
    coord.add_argument(
        "--region",
        help="Exact genomic interval represented by NPZ, 1-based inclusive, e.g. chr21:32300000-32402399",
    )
    coord.add_argument(
        "--tss-bed",
        help=(
            "TSS BED with gene ID in column 4. Reconstructs a TSS-centred interval whose width equals "
            "the one-hot sequence length; compatible with the older 102,400-bp workflow."
        ),
    )

    ap.add_argument("--out-dir", required=True, help="Root output directory; a gene subdirectory is created")
    ap.add_argument(
        "--checkpoint",
        default="all_folds",
        help="alphagenome_research Hugging Face checkpoint name (default: all_folds)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of one-hot sequence units per AlphaGenome forward pass (default: 2)",
    )
    ap.add_argument(
        "--save-tracks",
        action="store_true",
        help=(
            "Save all selected 1-bp native RNA-seq tracks (+/- astrocyte and glutamatergic neuron) "
            "to a float32 NPY memmap."
        ),
    )
    ap.add_argument(
        "--max-units",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit on the number of sequence units to predict. "
            "For phased input, --max-units 2 predicts the first sample's hap1 and hap2 only."
        ),
    )

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()