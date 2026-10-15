"""Native astrocyte RNA adapter, derived from the supplied fixed native script.
Heavy AlphaGenome/JAX imports are deferred so preflight and recalculate need no GPU.
"""
from __future__ import annotations
import gzip
import re
import hashlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

TARGET_TRACKS = {"Astrocyte": {"ontology_curie": "CL:0000127", "track_name": "CL:0000127 total RNA-seq"}}

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
    selected = all_meta.loc[mask].copy().reset_index(drop=True)
    selected.insert(0, "original_track_index", np.flatnonzero(mask))

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
    """Use the direct path only when its inspected signature matches.

    Return None for installations with another private interface. Those use the
    installed public predict_sequence wrapper without guessing private arguments
    or internal tensor layouts. No package upgrade is required.
    """
    base_predict = getattr(getattr(model, "_predict", None), "__wrapped__", None)
    try:
        parameters = inspect.signature(base_predict).parameters if base_predict is not None else {}
    except (TypeError, ValueError):
        parameters = {}
    expected = {"params", "state", "sequences", "organism_indices",
                "requested_outputs", "negative_strand_mask", "strand_reindexing"}
    if not expected.issubset(parameters):
        print("Native inference path: public predict_sequence wrapper "
              "(installed private _predict signature differs).", flush=True)
        return None
    print("Native inference path: direct one-hot RNA forward (signature verified).", flush=True)

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


def decode_model_onehot(model, onehot):
    """Recover DNA text for the public wrapper and verify exact encoding parity.

    Canonical bases are decoded as A/C/G/T. Ambiguous encodings are represented
    by N; a round trip check rejects any encoding that N would change. Thus this
    fallback never silently changes the model input.
    """
    onehot = np.asarray(onehot, dtype=np.float32)
    if onehot.ndim != 2 or onehot.shape[1] != 4 or not np.isfinite(onehot).all():
        raise ValueError("Invalid one-hot sequence for native public inference")
    indices = onehot.argmax(axis=1)
    canonical = np.all(onehot == np.eye(4, dtype=np.float32)[indices], axis=1)
    text = np.full(len(onehot), ord("N"), dtype=np.uint8)
    text[canonical] = np.frombuffer(b"ACGT", dtype=np.uint8)[indices[canonical]]
    sequence = text.tobytes().decode("ascii")
    encoded = np.asarray(model._one_hot_encoder.encode(sequence), dtype=np.float32)
    if not np.array_equal(encoded, onehot):
        raise ValueError("Cannot losslessly decode this model's one-hot input for the public wrapper")
    return sequence


def predict_public_rna_batch(model, metadata, track_mask_np, batch_np, organism_enum):
    """Public wrapper fallback. Processes batch members serially on the GPU."""
    expected = metadata.get(RNA_OUTPUT).loc[track_mask_np].reset_index(drop=True)
    results = []
    for onehot in batch_np:
        sequence = decode_model_onehot(model, onehot)
        public = model.predict_sequence(
            sequence, organism=organism_enum,
            requested_outputs=[RNA_OUTPUT], ontology_terms=["CL:0000127"],
        ).rna_seq
        if public is None:
            raise RuntimeError("Public native prediction has no RNA_SEQ output")
        values = np.asarray(public.values, dtype=np.float32)
        public_meta = public.metadata
        if values.ndim != 2 or values.shape != (len(onehot), len(public_meta)):
            raise RuntimeError(f"Unexpected public native RNA shape {values.shape}")
        indices = []
        for row in expected.itertuples():
            hits = np.flatnonzero((public_meta["name"].astype(str) == str(row.name))
                                 & (public_meta["strand"].astype(str) == str(row.strand)))
            if len(hits) != 1:
                raise RuntimeError(f"Public RNA output must contain exactly one {row.name} / {row.strand} track")
            indices.append(int(hits[0]))
        results.append(values[:, indices])
    return np.stack(results, axis=0)


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
    if rna_predict_fn is None:
        return predict_public_rna_batch(model, metadata, track_mask_np, batch_np, organism_enum)
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


class NativeAstrocyte:
    """One frozen native head, no donor covariates or fitted HARI components."""

    def __init__(self, checkpoint="all_folds"):
        global jax, jnp, ag_dna_model, core_dna_model, dna_output, RNA_OUTPUT
        import jax
        import jax.numpy as jnp
        from alphagenome_research.model import dna_model as ag_dna_model
        from alphagenome.models import dna_model as core_dna_model
        from alphagenome.models import dna_output
        RNA_OUTPUT = dna_output.OutputType.RNA_SEQ
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("No JAX GPU device available. Run on an allocated GPU node.")
        self.organism = core_dna_model.Organism.HOMO_SAPIENS
        print(f"Loading native AlphaGenome {checkpoint}; JAX devices: {jax.devices()}", flush=True)
        self.model = ag_dna_model.create_from_huggingface(checkpoint)
        self.metadata = _get_human_metadata(self.model, self.organism)
        self.track_mask, self.track_metadata, self.track_index = select_target_rna_tracks(self.metadata)
        self.forward = build_native_rna_predictor(self.model)
        # Verify the exact loaded values, including state, rather than checkpoint name only.
        print("Fingerprinting loaded native weights and model state for safe resume...", flush=True)
        h = hashlib.sha256()
        for tree in (self.model._params, self.model._state):
            leaves, structure = jax.tree_util.tree_flatten(tree)
            h.update(str(structure).encode())
            for leaf in leaves:
                arr = np.asarray(jax.device_get(leaf))
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtype).encode())
                h.update(arr.tobytes(order="C"))
        versions = {}
        for package in ("alphagenome", "alphagenome-research", "jax", "jaxlib", "numpy", "pandas"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unregistered"
        model_file = Path(inspect.getfile(ag_dna_model))
        self.provenance = {
            "checkpoint": checkpoint,
            "inference_path": "public_predict_sequence" if self.forward is None else "direct_onehot_rna",
            "weights_and_state_sha256": h.hexdigest(),
            "versions": versions,
            "model_source_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            "devices": [str(d) for d in jax.devices()],
            "track_metadata": self.track_metadata.to_csv(sep="\t", index=False),
        }

    def encode(self, sequence):
        # Use the installed model's own encoding of A/C/G/T and unknown bases.
        x = np.asarray(self.model._one_hot_encoder.encode(sequence), dtype=np.float32)
        if x.shape != (len(sequence), 4):
            raise RuntimeError(f"Unexpected model one-hot shape {x.shape}")
        return x

    def tracks(self, batch):
        tracks = predict_native_rna_batch(
            self.model, self.forward, self.metadata, self.track_mask,
            batch, self.organism,
        )
        if tracks.shape != (len(batch), batch.shape[1], int(self.track_mask.sum())):
            raise RuntimeError("Native RNA output is not 1 bp over the full input interval.")
        if not np.isfinite(tracks).all() or (tracks < 0).any():
            raise RuntimeError("Invalid native RNA coverage: nonfinite or negative values.")
        return tracks

    def scores(self, batch, exon_mask, strand):
        tracks = self.tracks(batch)
        idx = self.track_index[("Astrocyte", strand)]
        return tracks[:, exon_mask, idx].mean(axis=1, dtype=np.float64)

    def baseline(self, sequence, onehot, exon_mask, strand, validate_public=True):
        tracks = self.tracks(onehot[None])
        if validate_public:
            # Direct/public parity, or public/public no-op check on the fallback.
            public = self.model.predict_sequence(
                sequence, organism=self.organism,
                requested_outputs=[RNA_OUTPUT], ontology_terms=["CL:0000127"],
            ).rna_seq
            public_meta = public.metadata
            for sign in ("+", "-"):
                positions = np.flatnonzero(
                    (public_meta["name"].astype(str) == TARGET_TRACKS["Astrocyte"]["track_name"])
                    & (public_meta["strand"].astype(str) == sign)
                )
                if len(positions) != 1:
                    raise RuntimeError("Public wrapper did not expose the exact astrocyte +/- pair.")
                np.testing.assert_allclose(
                    tracks[0, :, self.track_index[("Astrocyte", sign)]],
                    np.asarray(public.values)[:, positions[0]], rtol=1e-5, atol=1e-6,
                    err_msg="Native baseline predictions disagree with the intact public wrapper",
                )
            label = "public/public no-op" if self.forward is None else "direct/public"
            print(f"Intact reference: {label} native RNA tracks match.", flush=True)
        idx = self.track_index[("Astrocyte", strand)]
        return float(tracks[0, exon_mask, idx].mean(dtype=np.float64)), tracks[0]
