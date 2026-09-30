import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c = np.loadtxt("/home/dh2226/perturb_enrich/annotations/encode_astro/raw/barcode_counts.tsv", usecols=1)
c = np.sort(c)[::-1]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(np.arange(1, len(c) + 1), c)                 # knee plot
ax[0].set(xlabel="barcode rank", ylabel="fragments per barcode", title="Knee plot")
ax[1].hist(np.log10(c), bins=80)                          # histogram
ax[1].set(xlabel="log10(fragments per barcode)", ylabel="barcodes", title="Distribution")
for t in (1000, 3000, 10000):                             # candidate cutoffs
    ax[1].axvline(np.log10(t), ls="--", lw=0.8, color="gray")
fig.tight_layout()
fig.savefig("barcode_distribution.png", dpi=200)
print(f"barcodes: {len(c)}  max: {int(c.max())}  median: {int(np.median(c))}")
for t in (500, 1000, 2000, 5000, 10000):
    print(f">= {t}: {(c >= t).sum()}")