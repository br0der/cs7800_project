import matplotlib.pyplot as plt
import numpy as np

ops = ["insert", "erase", "set", "flip", "access", "rank1", "select1"]

series = {
    "Naive": [208000, 140500, 500, 0.0, 250, 2.91219e7, 1.0832e8],
    "Pbds": [np.nan, np.nan, 14804.7, 22798, 7281.92, 6370.74, 3610.84],
    "Jacobson/Munro": [np.nan, np.nan, np.nan, np.nan, 65.4224, 99.5068, 316.24],
    "BTree": [98865.3, 134055, 4208.57, 6202.82, 458.565, 1239.49, 667.206],
    "Navarro25": [412739, 18695.7, 16778.1, 17345.9, 5117.91, 4840.25, 3784.3],
}

colors = {
    "Naive": "black",
    "Pbds": "gray",
    "Jacobson/Munro": "yellow",
    "Navarro25": "green",
    "BTree": "blue",
}

labels = list(series.keys())
values = np.array([series[k] for k in labels], dtype=float)

x = np.arange(len(ops))
width = 0.15

fig, ax = plt.subplots(figsize=(13, 7))
offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width

for i, label in enumerate(labels):
    ax.bar(x + offsets[i], values[i], width, label=label, color=colors[label])

ax.set_title("Dynamic Bitvector Performance by Operation")
ax.set_xlabel("Operation")
ax.set_ylabel("Average time (ns)")
ax.set_xticks(x)
ax.set_xticklabels(ops, rotation=25)
ax.set_yscale("symlog", linthresh=1)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

out_path = "/mnt/data/dynamic_bitvector_bar_chart_2_colored.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved chart to {out_path}")