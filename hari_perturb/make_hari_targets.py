#!/usr/bin/env python3
"""Build the strict targets.tsv consumed by scan_hari.py.

Expected output columns (in this exact order):
    gene_id, chrom, tss0, strand, bundle_path

The TSS source must be a one-base BED6 file. BED start is already zero-based,
so tss0 is copied directly from column 2. By default, an otherwise exact primary
chromosome match may add/remove ``chr`` (and reconcile M/MT) to match the FASTA
index. Gene-symbol conversion, transcript selection, padding, and coordinate
conversion are never performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


OUTPUT_COLUMNS = ("gene_id", "chrom", "tss0", "strand", "bundle_path")
ENSEMBL_GENE_RE = re.compile(r"^ENSG[0-9]+(?:\.[0-9]+)?$")
DEFAULT_BUNDLE_TEMPLATE = (
    "{bundle_root}/{gene_id}/{gene_id}_celltype_exportable.inference_bundle.pkl"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate a validated HARI perturbation targets.tsv from Ensembl gene IDs, "
            "a one-base BED6 TSS annotation, full-context inference bundles, and a FASTA index."
        ),
    )
    p.add_argument(
        "--genes",
        required=True,
        help=(
            "Text/TSV file containing one Ensembl gene ID per line or a table with a "
            "gene_id column. Blank lines and lines beginning with # are ignored."
        ),
    )
    p.add_argument(
        "--tss-bed",
        required=True,
        help=(
            "One-base BED6 file: chrom, start0, end0=start0+1, gene_id, score, strand. "
            "Exactly one distinct TSS must remain for every requested gene."
        ),
    )
    p.add_argument(
        "--bundle-root",
        required=True,
        help="Root containing one subdirectory per gene and its exported inference bundle.",
    )
    p.add_argument(
        "--bundle-template",
        default=DEFAULT_BUNDLE_TEMPLATE,
        help="Python format string supporting {bundle_root} and {gene_id}.",
    )
    p.add_argument(
        "--fasta",
        required=True,
        help="Reference FASTA used by the scan. Its adjacent .fai index must exist.",
    )
    p.add_argument(
        "--chromosome-mode",
        choices=("auto", "exact"),
        default="auto",
        help=(
            "In auto mode, resolve an absent BED primary contig to one unambiguous "
            "FASTA equivalent (for example 4 -> chr4 or MT -> chrM). Exact mode "
            "requires literal equality."
        ),
    )
    p.add_argument("--output", required=True, help="Destination targets.tsv.")
    p.add_argument(
        "--missing-tss-policy",
        choices=("skip", "error"),
        default="skip",
        help=(
            "Skip requested genes absent from the TSS BED and continue with valid genes, "
            "or retain the earlier fail-fast behavior with error."
        ),
    )
    p.add_argument(
        "--skipped-output",
        default=None,
        help=(
            "Audit TSV for skipped genes. Default: <output stem>.skipped.tsv beside "
            "targets.tsv. The file is written even when no genes are skipped."
        ),
    )
    p.add_argument(
        "--required-sequence-length",
        type=int,
        default=1_048_576,
        help="Required complete TSS-centered input interval.",
    )
    p.add_argument(
        "--allow-missing-bundles",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --missing-bundle-policy include. "
            "Prefer the explicit policy option."
        ),
    )
    p.add_argument(
        "--missing-bundle-policy",
        choices=("skip", "error", "include"),
        default="skip",
        help=(
            "Skip genes without exported inference bundles, fail immediately, or include "
            "their unresolved paths. Skip is safest for a runnable targets.tsv."
        ),
    )
    return p


def read_genes(path: Path) -> list[str]:
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(re.split(r"\s+", stripped))
    if not rows:
        raise ValueError(f"Gene file is empty: {path}")

    first_lower = [x.lower() for x in rows[0]]
    if "gene_id" in first_lower:
        column = first_lower.index("gene_id")
        data = rows[1:]
    else:
        column = 0
        data = rows

    genes: list[str] = []
    seen: set[str] = set()
    for line_number, fields in enumerate(data, start=2 if data is not rows else 1):
        if column >= len(fields):
            raise ValueError(f"Missing gene_id at {path}:{line_number}")
        gene = fields[column]
        if not ENSEMBL_GENE_RE.fullmatch(gene):
            raise ValueError(
                f"Invalid Ensembl gene ID {gene!r} at {path}:{line_number}; "
                "gene-symbol conversion is intentionally not performed"
            )
        if gene not in seen:
            seen.add(gene)
            genes.append(gene)
    if not genes:
        raise ValueError(f"No genes remain after the header in {path}")
    return genes


def read_tss_bed(path: Path, requested: set[str]) -> dict[str, tuple[str, int, str]]:
    matches: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#") or raw.startswith("track "):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError(f"Expected BED6 at {path}:{line_number}")
            chrom, start_text, end_text, gene, _score, strand = fields[:6]
            if gene not in requested:
                continue
            try:
                start0, end0 = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"Non-integer BED coordinates at {path}:{line_number}") from exc
            if start0 < 0 or end0 != start0 + 1:
                raise ValueError(
                    f"{gene} at {path}:{line_number} is not one-base BED: "
                    f"[{start0}, {end0}); transcript/TSS selection will not be guessed"
                )
            if strand not in {"+", "-"}:
                raise ValueError(f"Invalid strand {strand!r} at {path}:{line_number}")
            if not chrom:
                raise ValueError(f"Empty chromosome at {path}:{line_number}")
            matches[gene].add((chrom, start0, strand))

    ambiguous = {gene: sorted(values) for gene, values in matches.items() if len(values) != 1}
    if ambiguous:
        details = "; ".join(f"{g}={v}" for g, v in list(sorted(ambiguous.items()))[:5])
        raise ValueError(
            f"{len(ambiguous)} genes have multiple distinct TSS records; provide a resolved "
            f"one-TSS BED instead of selecting implicitly: {details}"
        )
    return {gene: next(iter(values)) for gene, values in matches.items()}


def read_fai(fasta: Path) -> dict[str, int]:
    fai = Path(f"{fasta}.fai")
    if not fai.is_file():
        raise FileNotFoundError(f"FASTA index not found: {fai}; create it with samtools faidx")
    lengths: dict[str, int] = {}
    with fai.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed FASTA index at {fai}:{line_number}")
            chrom, length_text = fields[:2]
            if chrom in lengths:
                raise ValueError(f"Duplicate FASTA contig {chrom!r} in {fai}")
            lengths[chrom] = int(length_text)
    return lengths


def primary_contig_key(contig: str) -> str | None:
    """Return a conservative alias key for canonical human chromosomes only."""
    value = contig
    if value.lower().startswith("chr"):
        value = value[3:]
    value = value.upper()
    if value in {"M", "MT"}:
        return "MT"
    if value in {"X", "Y"}:
        return value
    if value.isdigit() and 1 <= int(value) <= 22:
        return str(int(value))
    return None


def resolve_fasta_contig(
    bed_contig: str,
    contig_lengths: dict[str, int],
    mode: str,
) -> tuple[str, bool]:
    """Resolve a BED contig to the literal FASTA name without guessing broadly."""
    if bed_contig in contig_lengths:
        return bed_contig, False
    if mode == "exact":
        raise ValueError(
            f"BED contig {bed_contig!r} is absent from the FASTA index in exact mode"
        )

    key = primary_contig_key(bed_contig)
    candidates = [
        contig
        for contig in contig_lengths
        if key is not None and primary_contig_key(contig) == key
    ]
    if len(candidates) == 1:
        return candidates[0], True
    if len(candidates) > 1:
        raise ValueError(
            f"BED contig {bed_contig!r} has multiple FASTA aliases {candidates}; "
            "use --chromosome-mode exact after resolving the annotation"
        )
    raise ValueError(
        f"BED contig {bed_contig!r} is absent from the FASTA index and has no "
        "unambiguous primary-chromosome alias"
    )


def bundle_for(gene: str, root: Path, template: str) -> Path:
    try:
        rendered = template.format(bundle_root=str(root), gene_id=gene)
    except KeyError as exc:
        raise ValueError(
            "--bundle-template may contain only {bundle_root} and {gene_id}"
        ) from exc
    return Path(rendered).expanduser().resolve()


def validate_bundle_sidecar(bundle: Path, gene: str, required_length: int) -> None:
    sidecar = Path(f"{bundle}.json")
    if not sidecar.is_file():
        return
    with sidecar.open() as handle:
        metadata = json.load(handle)
    recorded_gene = str(metadata.get("gene_id", ""))
    if recorded_gene and recorded_gene != gene:
        raise ValueError(f"Bundle sidecar gene mismatch: requested={gene}, recorded={recorded_gene}")
    generation = metadata.get("embedding_generation", {})
    schema = generation.get("training_embedding_schema", {}) if isinstance(generation, dict) else {}
    recorded_length = schema.get("required_sequence_length")
    if recorded_length is not None and int(recorded_length) != required_length:
        raise ValueError(
            f"{gene}: bundle was trained with {recorded_length} bp, expected {required_length} bp"
        )


def atomic_write(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parser().parse_args()
    if args.required_sequence_length < 1 or args.required_sequence_length % 2:
        raise ValueError("--required-sequence-length must be a positive even number")

    genes = read_genes(Path(args.genes))
    tss = read_tss_bed(Path(args.tss_bed), set(genes))
    missing_tss = [gene for gene in genes if gene not in tss]
    if missing_tss and args.missing_tss_policy == "error":
        preview = ", ".join(missing_tss[:10])
        suffix = " ..." if len(missing_tss) > 10 else ""
        noun = "gene is" if len(missing_tss) == 1 else "genes are"
        raise ValueError(
            f"{len(missing_tss)} requested {noun} absent from the TSS BED: "
            f"{preview}{suffix}"
        )
    available_genes = [gene for gene in genes if gene in tss]

    contig_lengths = read_fai(Path(args.fasta))
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    half = args.required_sequence_length // 2
    bundle_policy = "include" if args.allow_missing_bundles else args.missing_bundle_policy

    rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = [
        {"gene_id": gene, "reason": "missing_from_tss_bed"}
        for gene in missing_tss
    ]
    remapped_contigs: dict[tuple[str, str], int] = defaultdict(int)
    for gene in available_genes:
        bundle = bundle_for(gene, bundle_root, args.bundle_template)
        if not bundle.is_file():
            if bundle_policy == "error":
                raise FileNotFoundError(f"Inference bundle not found for {gene}: {bundle}")
            if bundle_policy == "skip":
                skipped_rows.append(
                    {"gene_id": gene, "reason": "missing_inference_bundle"}
                )
                continue
        else:
            validate_bundle_sidecar(bundle, gene, args.required_sequence_length)

        bed_chrom, tss0, strand = tss[gene]
        try:
            chrom, was_remapped = resolve_fasta_contig(
                bed_chrom, contig_lengths, args.chromosome_mode
            )
        except ValueError as exc:
            raise ValueError(f"{gene}: {exc}") from exc
        if was_remapped:
            remapped_contigs[(bed_chrom, chrom)] += 1
        interval_start = tss0 - half
        interval_end = interval_start + args.required_sequence_length
        if interval_start < 0 or interval_end > contig_lengths[chrom]:
            raise ValueError(
                f"{gene}: full TSS-centered interval [{interval_start}, {interval_end}) "
                f"falls outside {chrom} length {contig_lengths[chrom]}; no padding is performed"
            )

        rows.append(
            {
                "gene_id": gene,
                "chrom": chrom,
                "tss0": tss0,
                "strand": strand,
                "bundle_path": str(bundle),
            }
        )

    output = Path(args.output).expanduser().resolve()
    skipped_output = (
        Path(args.skipped_output).expanduser().resolve()
        if args.skipped_output
        else output.with_name(f"{output.stem}.skipped.tsv")
    )
    atomic_write(output, OUTPUT_COLUMNS, rows)
    atomic_write(skipped_output, ("gene_id", "reason"), skipped_rows)
    if remapped_contigs:
        summary = ", ".join(
            f"{source}->{target} ({count} gene{'s' if count != 1 else ''})"
            for (source, target), count in sorted(remapped_contigs.items())
        )
        print(f"Resolved BED contigs to FASTA names: {summary}", file=sys.stderr)
    if missing_tss:
        preview = ", ".join(missing_tss[:10])
        suffix = " ..." if len(missing_tss) > 10 else ""
        noun = "gene" if len(missing_tss) == 1 else "genes"
        print(
            f"Skipped {len(missing_tss)} {noun} absent from the TSS BED: "
            f"{preview}{suffix}",
            file=sys.stderr,
        )
    missing_bundles = [
        str(row["gene_id"])
        for row in skipped_rows
        if row["reason"] == "missing_inference_bundle"
    ]
    if missing_bundles:
        preview = ", ".join(missing_bundles[:10])
        suffix = " ..." if len(missing_bundles) > 10 else ""
        noun = "gene" if len(missing_bundles) == 1 else "genes"
        print(
            f"Skipped {len(missing_bundles)} {noun} without inference bundles: "
            f"{preview}{suffix}",
            file=sys.stderr,
        )
    if not rows:
        print(
            "WARNING: no runnable targets remain after applying the skip policies",
            file=sys.stderr,
        )
    print(
        f"Wrote {len(rows)} targets to {output} "
        f"and {len(skipped_rows)} skipped-gene records to {skipped_output} "
        f"(sequence_length={args.required_sequence_length:,} bp; exact FASTA contigs validated)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
