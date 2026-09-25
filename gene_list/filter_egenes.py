import pandas as pd, numpy as np

df = pd.read_csv("Ast_eqtl_top_assoc.tsv", sep="\t")
df["gene_id"] = df["feature"].str.replace(r"\.\d+$", "", regex=True)

eg = df[df.qval < 0.05].copy()

# three roughly independent cohort blocks
z = eg[["z_tissue_0", "z_tissue_1", "z_tissue_2", "z_tissue_3"]]
eg["n_cohorts"] = z.notna().sum(axis=1)
sign = np.sign(eg["Random_Z"])

rosmap_ok = ((np.sign(eg.z_tissue_0) == sign) & (eg.z_tissue_0.abs() > 1.96)) | \
            ((np.sign(eg.z_tissue_1) == sign) & (eg.z_tissue_1.abs() > 1.96))
gabitto_ok = (np.sign(eg.z_tissue_2) == sign) & (eg.z_tissue_2.abs() > 1.96)
bryois_ok  = (np.sign(eg.z_tissue_3) == sign) & (eg.z_tissue_3.abs() > 1.96)

eg["n_blocks_supporting"] = rosmap_ok.astype(int) + gabitto_ok + bryois_ok
eg["n_blocks_concordant_sign"] = (
    ((np.sign(eg.z_tissue_0) == sign) | (np.sign(eg.z_tissue_1) == sign)).astype(int)
    + (np.sign(eg.z_tissue_2) == sign) + (np.sign(eg.z_tissue_3) == sign)
)

core = eg[eg.n_blocks_supporting >= 2]

tss = pd.read_csv("tss_one_base.bed", sep="\t", header=None,
                  names=["chrom","start0","end0","gene_id","score","strand"])
tss["gene_id"] = tss.gene_id.str.replace(r"\.\d+$", "", regex=True)

eg = core.merge(tss[["gene_id","chrom","start0","strand"]], on="gene_id", how="inner")
assert (eg["chr"].astype(str).str.replace("chr","") == eg["chrom"].str.replace("chr","")).all()

eg["dist_to_tss"] = (eg["pos"] - 1) - eg["start0"]     # pos is 1-based, start0 is 0-based
HALF = 524_288                                          # half of 1,048,576-bp model input
distal = eg[(eg.dist_to_tss.abs() > 5_000) & (eg.dist_to_tss.abs() < HALF - 1_000)]
print(distal.gene_id.nunique())

d = distal.dist_to_tss.abs()
print(d.describe())
print((eg.dist_to_tss.abs() <= 5_000).mean())   # fraction removed by the filter

other = {}
lead = {}
for ct in ["End", "Ext", "IN", "MG", "OD", "OPC"]:
    t = pd.read_csv(f"cell_eqtls/{ct}_eqtl_top_assoc.tsv.gz", sep="\t")
    t["gene_id"] = t.feature.str.replace(r"\.\d+$", "", regex=True)
    other[ct] = set(t.loc[t.qval < 0.05, "gene_id"])

    t = t[t.qval < 0.05]
    lead[ct] = dict(zip(t.gene_id, t.pos))

distal["n_other_celltypes"] = distal.gene_id.map(lambda g: sum(g in s for s in other.values()))
ast_specific = distal[distal.n_other_celltypes == 0]
print(ast_specific.gene_id.nunique())
print(distal.n_other_celltypes.value_counts().sort_index())

def n_shared_signal(row, window=10_000):
    return sum(abs(row.pos - lead[ct][row.gene_id]) <= window
               for ct in lead if row.gene_id in lead[ct])

distal["n_other_same_signal"] = distal.apply(n_shared_signal, axis=1)
print((distal.n_other_same_signal == 0).sum())