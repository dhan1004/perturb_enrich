#!/usr/bin/env python3

"""
motif x cell type enrichent matrix from HOMER knownRresults.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

LN10 = math.log(10.0)

# DEFAULT_MOTIFS = [
#     ("NURR1 (NR4A2)", ["NR4A2", "NURR1", "NR4A1", "NUR77"]),
#     ("FOXA2", ["FOXA2", "HNF3B", "HNF3BETA"]),
#     ("FOXA1", ["FOXA1", "HNF3A"]),
#     ("LMX1A", ["LMX1A"]),
#     ("LMX1B", ["LMX1B"]),
#     ("PITX3", ["PITX3"]),
#     ("EN1/EN2 (Engrailed)", ["EN1", "EN2", "ENGRAILED"]),
#     ("OTX2", ["OTX2"]),
#     ("NEUROD1", ["NEUROD1", "NEUROD"]),
#     ("ASCL1 (MASH1)", ["ASCL1", "MASH1"]),
#     ("NRF1", ["NRF1"]),
#     ("NRF2 (NFE2L2)", ["NFE2L2", "NRF2"]),
#     ("GATA2", ["GATA2"]),
#     ("YY1", ["YY1"]),
#     ("MEF2C", ["MEF2C", "MEF2"]),
# ]

MOTIFS = [
    ("NEUROD")
]

METRICS = {
    "neglog10p": ("neglog10p", "-log10 P (enrichment significance, not effect size)", "{:.1f}"),
    "fold": ("fold", "Fold enrichment (%target / %background)", "{:.1f}"),
    "log2fold": ("log2fold", "log2 fold enrichment", "{:.1f}"),
    "pct_target": ("pct_target", "% of target sequences with motif", "{:.0f}"),
}

def norm(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
 
 
def motif_symbol(name: str) -> str:
    head = name.split("/", 1)[0].split("(", 1)[0]
    return norm(head)
 
 
def alias_matches(symbol: str, alias: str, min_len: int = 3) -> bool:
    a = norm(alias)
    if len(a) < min_len or len(symbol) < min_len:
        return symbol == a
    return symbol == a or symbol.startswith(a) or a.startswith(symbol)

# ------------

def load_motif_list(path):
    if path is None:
        # Normalize the embedded aliases so downstream matching is consistent.
        return [(label, [norm(a) for a in aliases]) for label, aliases in DEFAULT_MOTIFS]
    entries = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        label = parts[0].strip()
        # If a row has no alias column, fall back to using the label as its own alias.
        aliases = [a.strip() for a in (parts[1].split(",") if len(parts) > 1 else [label]) if a.strip()]
        entries.append((label, [norm(a) for a in aliases]))
    if not entries:
        raise ValueError(f"No motif entries parsed from {path}")
    labels = [e[0] for e in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate motif labels in the motif list")
    return entries


def parse_known_results(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    col = {}
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("motif name"):
            col["name"] = c
        elif cl == "consensus":
            col["consensus"] = c
        elif cl == "p-value":
            col["pval"] = c
        elif cl.startswith("log p-value"):
            col["logp"] = c
        elif cl.startswith("q-value"):
            col["qval"] = c
        elif cl.startswith("# of target"):
            col["ntarget"] = c
        elif cl.startswith("% of target"):
            col["pcttarget"] = c
        elif cl.startswith("# of background"):
            col["nbg"] = c
        elif cl.startswith("% of background"):
            col["pctbg"] = c

    need = {"name", "logp", "qval", "pcttarget", "pctbg"}
    missing = need - set(col)
    if missing:
        raise ValueError(f"{path}: missing expected HOMER columns {sorted(missing)}; "
                         f"found {list(df.columns)}")
 
    # recover total target / background counts
    def total_from(header):
        m = re.search(r"\(of\s+([0-9]+)\)", header or "")
        return int(m.group(1)) if m else None
 
    total_targets = total_from(col.get("ntarget", ""))
    total_bg = total_from(col.get("nbg", ""))
 
    out = pd.DataFrame()
    out["motif"] = df[col["name"]].str.strip()
    out["consensus"] = df[col["consensus"]] if "consensus" in col else ""
    out["logp"] = pd.to_numeric(df[col["logp"]], errors="coerce")          # natural-log p
    out["qval"] = pd.to_numeric(df[col["qval"]], errors="coerce")          # Benjamini q
    out["pct_target"] = pd.to_numeric(df[col["pcttarget"]].str.rstrip("% "), errors="coerce")  # strip '%'
    out["pct_bg"] = pd.to_numeric(df[col["pctbg"]].str.rstrip("% "), errors="coerce")

    out["neglog10p"] = -out["logp"] / LN10

    with np.errstate(divide="ignore", invalid="ignore"):
        out["fold"] = out["pct_target"] / out["pct_bg"]
        out["log2fold"] = np.log2(out["fold"].replace(0, np.nan))
    
    out["symbol"] = out["motif"].map(motif_symbol)
    return out, total_targets, total_bg

def discover_cells(motifs_root):
    root = Path(motifs_root)
    cells = {}
    for kr in sorted(root.glob("*/knownResults.txt")):
        cells[kr.parent.name] = kr
    for kr in sorted(root.glob("*/knownResults/knownResults.txt")):
        cells.setdefault(kr.parent.parent.name, kr)
    if not cells:
        raise FileNotFoundError(f"No */knownResults.txt under {root}")
    return cells

def build(cell_paths, motif_list, metric):
    # build matrix and tables
    per_cell, totals = {}, {}
    for cell, path in cell_paths.items():
        df, n_tgt, n_bg = parse_known_results(path)
        per_cell[cell] = df
        totals[cell] = (n_tgt, n_bg)
 
    cells = list(per_cell)
    metric_col = METRICS[metric][0]
    matrix_rows, long_rows, audit_rows = [], [], []
 
    for label, aliases in motif_list:
        row = {"motif": label}
        matched_any = False
        matched_names_by_cell = {}
        for cell in cells:
            df = per_cell[cell]
            hits = df[df["symbol"].apply(lambda s: any(alias_matches(s, a) for a in aliases))]
            if len(hits):
                matched_any = True
                best = hits.loc[hits["neglog10p"].idxmax()]
                row[cell] = float(best[metric_col])
                matched_names_by_cell[cell] = sorted(hits["motif"].tolist())
                for _, h in hits.iterrows():
                    long_rows.append(dict(motif_label=label, cell_type=cell, homer_motif=h["motif"],
                                          consensus=h["consensus"], neglog10p=h["neglog10p"],
                                          logp=h["logp"], qvalue=h["qval"], pct_target=h["pct_target"],
                                          pct_background=h["pct_bg"], fold=h["fold"], log2fold=h["log2fold"],
                                          is_matrix_representative=(h["motif"] == best["motif"])))
            else:
                row[cell] = np.nan
        if matched_any:
            matrix_rows.append(row)
        union_names = sorted({n for names in matched_names_by_cell.values() for n in names})
        audit_rows.append(dict(motif_label=label, aliases=",".join(aliases),
                               matched=matched_any,
                               n_homer_motifs_matched=len(union_names),
                               homer_motifs_matched="; ".join(union_names) if union_names
                               else "NONE (not in HOMER known set you scanned)"))
 
    matrix = pd.DataFrame(matrix_rows).set_index("motif") if matrix_rows else pd.DataFrame()
    if not matrix.empty:
        matrix = matrix.reindex(columns=cells)          
        order = matrix.max(axis=1, skipna=True).sort_values(ascending=False).index
        matrix = matrix.loc[order]
    long_df = pd.DataFrame(long_rows)
    audit_df = pd.DataFrame(audit_rows)
    return matrix, long_df, audit_df, totals, cells

def qvalue_lookup(long_df):
    if long_df.empty:
        return {}
    rep = long_df[long_df["is_matrix_representative"]]
    return {(r.motif_label, r.cell_type): r.qvalue for r in rep.itertuples()}

def draw_heatmap(matrix, qlook, metric, totals, out_png, out_pdf):
    import matplotlib
    matplotlib.use("Agg")            
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
 
    _, cbar_label, fmt = METRICS[metric]
    cells = list(matrix.columns)
    labels = list(matrix.index)
    data = matrix.to_numpy(dtype=float)
 
    n_rows, n_cols = data.shape
    fig_w = max(5.2, 1.35 * n_cols + 3.2)
    fig_h = max(3.2, 0.46 * n_rows + 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
 
    finite = data[np.isfinite(data)]
    vmax = np.nanmax(finite) if finite.size else 1.0
    vmin = 0.0 if metric in ("neglog10p", "fold", "pct_target") else (np.nanmin(finite) if finite.size else 0.0)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#E6E3DC")             
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap,
                   norm=Normalize(vmin=vmin, vmax=vmax))
 
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cells, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    thresh = vmin + 0.55 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#8A8478")
                continue
            q = qlook.get((labels[i], cells[j]))
            star = "" if q is None or not np.isfinite(q) else ("**" if q < 0.01 else "*" if q < 0.05 else "")
            ax.text(j, i, fmt.format(v) + star, ha="center", va="center", fontsize=8.5,
                    color="#2A2420" if v >= thresh else "#EDE7DF")
 
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)     
    cbar.ax.tick_params(labelsize=8)
 
    ax.set_title("PD-relevant motif enrichment across cell types", fontsize=13, fontweight="bold", pad=10)
 
    small = {c: t for c, (t, _b) in totals.items() if t is not None and t < 10}
    caption = "* q<0.05  ** q<0.01 (Benjamini, HOMER)   grey = motif not matched in that cell"
    if small:
        caption += ("\nWARNING: tiny foreground — target sequences: "
                    + ", ".join(f"{c}={t}" for c, t in small.items())
                    + ". Enrichment p-values are unstable with so few foreground regions.")
    fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=7.5, color="#5D6B78")
 
    fig.tight_layout(rect=(0, 0.03 + 0.03 * caption.count("\n"), 1, 1))
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")    
    plt.close(fig)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--motifs-root", help="HOMER motifs dir; auto-discovers */knownResults.txt")
    src.add_argument("--known-results", nargs="+", metavar="CELL=PATH",
                     help="Explicit cell=path/to/knownResults.txt pairs")
    p.add_argument("--motif-list", help="TSV: label<TAB>alias1,alias2 (default: built-in PD set)")
    p.add_argument("--metric", choices=list(METRICS), default="neglog10p")
    p.add_argument("--out-prefix", required=True, help="Output path prefix, e.g. OUT/summary/pd_motif_matrix")
    p.add_argument("--cell-types", help="Optional comma-separated subset/order of cell types")
    args = p.parse_args(argv)
 
    if args.motifs_root:
        cell_paths = discover_cells(args.motifs_root)
    else:
        cell_paths = {}
        for item in args.known_results:
            if "=" not in item:
                p.error(f"--known-results entries must be CELL=PATH, got {item!r}")
            cell, path = item.split("=", 1)
            cell_paths[cell.strip()] = Path(path.strip())
    if args.cell_types:
        wanted = [c.strip() for c in args.cell_types.split(",")]
        absent = [c for c in wanted if c not in cell_paths]
        if absent:
            p.error(f"Requested cell types not found: {absent}. Available: {sorted(cell_paths)}")
        cell_paths = {c: cell_paths[c] for c in wanted}
 
    motif_list = load_motif_list(args.motif_list)
    matrix, long_df, audit_df, totals, cells = build(cell_paths, motif_list, args.metric)
 
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(f"{out_prefix}.audit.tsv", sep="\t", index=False)
    long_df.to_csv(f"{out_prefix}.long.tsv", sep="\t", index=False)
 
    print("Cell types:", ", ".join(f"{c} (targets={totals[c][0]}, bg={totals[c][1]})" for c in cells))
    small = [c for c in cells if totals[c][0] is not None and totals[c][0] < 10]
    if small:
        print(f"WARNING: very few foreground/target sequences in: {small}. "
              "HOMER enrichment p-values are unstable with tiny foregrounds.", file=sys.stderr)
 
    unmatched = audit_df.loc[~audit_df["matched"], "motif_label"].tolist()
    if unmatched:
        print("Not found in the HOMER known set you scanned (no library motif): "
              + ", ".join(unmatched))
 
    if matrix.empty:
        print("No PD motifs matched any cell type; nothing to plot. See the audit table.")
        return
 
    matrix.to_csv(f"{out_prefix}.matrix.tsv", sep="\t")
    draw_heatmap(matrix, qvalue_lookup(long_df), args.metric, totals,
                 f"{out_prefix}.heatmap.png", f"{out_prefix}.heatmap.pdf")
    print(f"\nMatrix ({args.metric}):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(matrix.round(2).to_string())
    print(f"\nWrote:\n  {out_prefix}.matrix.tsv\n  {out_prefix}.long.tsv\n"
          f"  {out_prefix}.audit.tsv\n  {out_prefix}.heatmap.png / .pdf")
 
 
if __name__ == "__main__":
    main()
 