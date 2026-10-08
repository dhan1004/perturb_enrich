"""Dependency-light coordinates, shuffling, tracks, and effect-region calling."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

VERSION = "1.0.0"


def safe_name(value):
    value = str(value)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".")
    if not name:
        raise ValueError("Empty/unsafe identifier")
    # Avoid collisions between e.g. 'cell a' and 'cell_a'.
    return name if name == value else name + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_for(seed, gene, start, stop, repeat):
    digest = hashlib.sha256(f"{seed}|{gene}|{start}|{stop}|{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class IndexedFasta:
    """Read an existing uncompressed FASTA + .fai without an extra dependency.

    All intervals are genomic 0-based half-open. Out-of-contig sequence is N-padded
    without shifting the TSS. Windows crossing padding are excluded from testing.
    """
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix == ".gz":
            raise ValueError("Use an uncompressed FASTA with its .fai index")
        self.index = {}
        with open(str(self.path) + ".fai") as f:
            for line in f:
                col = line.rstrip().split("\t")
                if len(col) < 5 or col[0] in self.index:
                    raise ValueError("Malformed or duplicate FASTA index entry")
                self.index[col[0]] = tuple(map(int, col[1:5]))

    def fetch(self, chrom, start, stop):
        if chrom not in self.index:
            raise ValueError(f"Contig {chrom!r} absent from FASTA; no automatic chr renaming")
        length, offset, line_bases, line_bytes = self.index[chrom]
        if stop <= start or stop <= 0 or start >= length or line_bases < 1:
            raise ValueError(f"Invalid sequence interval {chrom}:{start}-{stop}")
        lo, hi = max(0, start), min(length, stop)
        byte_start = offset + (lo // line_bases) * line_bytes + lo % line_bases
        last = hi - 1
        byte_end = offset + (last // line_bases) * line_bytes + last % line_bases + 1
        with self.path.open("rb") as f:
            f.seek(byte_start)
            seq = f.read(byte_end - byte_start).replace(b"\n", b"").replace(b"\r", b"").upper()
        if len(seq) != hi - lo or b">" in seq:
            raise ValueError("FASTA/index mismatch: rebuild .fai for this exact FASTA")
        seq = b"N" * (lo - start) + seq + b"N" * (stop - hi)
        return seq, length


def onehot(sequence):
    lut = np.zeros((256, 4), dtype=np.float32)
    for i, b in enumerate(b"ACGT"):
        lut[b, i] = 1
    return lut[np.frombuffer(sequence, dtype=np.uint8)]


def shuffle_window(block: bytes, rng, method="mononucleotide"):
    """Mono = row permutation as in supplied script; di = random Euler trail.

    Di preserves exact internal dinucleotide counts AND first/last bases. It is
    not claimed to sample uniformly over all possible dinucleotide shuffles.
    Windows containing ambiguity characters are not silently rewritten.
    """
    if set(block) - set(b"ACGT"):
        raise ValueError("Only A/C/G/T windows can be shuffled")
    if method == "mononucleotide":
        arr = np.frombuffer(block, dtype=np.uint8)
        return arr[rng.permutation(len(arr))].tobytes()
    if method != "dinucleotide":
        raise ValueError(f"Unknown shuffle method {method}")
    if len(block) < 2:
        return block
    edges = defaultdict(list)
    for a, b in zip(block[:-1], block[1:]):
        edges[a].append(b)
    for destinations in edges.values():
        rng.shuffle(destinations)
    stack, trail = [block[0]], []
    while stack:
        if edges[stack[-1]]:
            stack.append(edges[stack[-1]].pop())
        else:
            trail.append(stack.pop())
    shuffled = bytes(trail[::-1])
    if (len(shuffled) != len(block) or shuffled[0] != block[0] or
            shuffled[-1] != block[-1] or
            Counter(zip(shuffled[:-1], shuffled[1:])) != Counter(zip(block[:-1], block[1:]))):
        raise AssertionError("Dinucleotide shuffle invariant failed")
    return shuffled


def make_windows(length, width, stride, start=0, stop=None):
    stop = length if stop is None else stop
    if not (0 <= start < stop <= length and 0 < stride <= width <= stop - start):
        raise ValueError("Require 0 < stride <= window <= scan span within model input")
    starts = list(range(start, stop - width + 1, stride))
    if starts[-1] + width < stop:
        starts.append(stop - width)  # keep full-width last window; overlap is explicit
    return np.asarray([(s, s + width) for s in starts], dtype=np.int64)


def smooth_windows(windows, values, size):
    """Centered, edge-truncated mean; never smooth through missing windows/gaps."""
    if size < 1 or size % 2 != 1:
        raise ValueError("smooth-windows must be a positive odd integer")
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    groups, current = [], []
    for i in range(len(windows)):
        valid = np.isfinite(values[i]).all()
        if current and (not valid or windows[i, 0] > windows[current[-1], 1]):
            groups.append(current)
            current = []
        if valid:
            current.append(i)
    if current:
        groups.append(current)
    half = size // 2
    for group in groups:
        for j, i in enumerate(group):
            out[i] = values[group[max(0, j-half):j+half+1]].mean(axis=0)
    return out


def project_windows(windows, values):
    """Non-overlapping bedGraph segments: mean of covering valid window scores.

    values[:,0] = absolute effect; values[:,1] = signed effect. A missing/unusable
    window masks its whole support so calls do not cross untested sequence.
    """
    events = defaultdict(list)
    values = np.asarray(values, dtype=float)
    for i, (a, b) in enumerate(windows):
        events[int(a)].append((i, 1))
        events[int(b)].append((i, -1))
    points = sorted(events)
    active, invalid = set(), set()
    result = []
    for j, a in enumerate(points[:-1]):
        for i, sign in events[a]:
            target = active if np.isfinite(values[i]).all() else invalid
            if sign > 0:
                target.add(i)
            else:
                target.remove(i)
        b = points[j+1]
        if active and not invalid and b > a:
            mean = values[sorted(active)].mean(axis=0)
            result.append({"start0": a, "end0": b, "abs_effect": float(mean[0]),
                           "signed_effect": float(mean[1]), "n_covering_windows": len(active)})
    return result


def direction(values, eps=1e-12):
    pos, neg = np.any(np.asarray(values) > eps), np.any(np.asarray(values) < -eps)
    return "mixed" if pos and neg else "increase" if pos else "decrease" if neg else "zero"


def call_regions(segments, threshold, min_bp, tss0, promoter_half_bp=5000):
    if not np.isfinite(threshold) or threshold <= 0 or min_bp < 1:
        raise ValueError("A positive finite threshold and minimum bp length are required")
    groups, current = [], []
    for s in segments:
        hit = s["abs_effect"] > threshold
        if current and (not hit or s["start0"] != current[-1]["end0"]):
            groups.append(current)
            current = []
        if hit:
            current.append(s)
    if current:
        groups.append(current)
    result = []
    for group in groups:
        a, b = group[0]["start0"], group[-1]["end0"]
        if b - a < min_bp:
            continue
        weights = np.array([s["end0"] - s["start0"] for s in group])
        signed = np.array([s["signed_effect"] for s in group])
        overlap = max(0, min(b, tss0 + promoter_half_bp) - max(a, tss0 - promoter_half_bp))
        result.append({"start0": a, "end0": b, "length_bp": b-a,
                       "mean_abs_effect": float(np.average([s["abs_effect"] for s in group], weights=weights)),
                       "max_abs_effect": max(s["abs_effect"] for s in group),
                       "mean_signed_effect": float(np.average(signed, weights=weights)),
                       "direction": direction(signed),
                       "positive_bp_fraction": float(np.sum(weights[signed > 1e-12]) / sum(weights)),
                       "negative_bp_fraction": float(np.sum(weights[signed < -1e-12]) / sum(weights)),
                       "tss_region_overlap_bp": overlap,
                       "annotation": "TSS_proximal" if overlap == b-a else "TSS_boundary" if overlap else "distal"})
    return result
