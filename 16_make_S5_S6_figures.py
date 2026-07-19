"""
16_make_S5_S6_figures.py
=========================
Companion figures for Supplementary Tables S5 (feature-set ablation) and S6
(building-block leakage check), matching the manuscript's existing figure
style (colorblind-safe palette, 300 dpi, Arial -- see regenerate_all_figures.py).

Output: scripts/figures_supp/FigS5_feature_ablation.png/.pdf
        scripts/figures_supp/FigS6_leakage_check.png/.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Rifat\Research\MOF_Screening")
SCRIPTS = ROOT / "scripts"
OUT_SUPP = SCRIPTS / "figures_supp"
OUT_SUPP.mkdir(exist_ok=True)

CB_COLORS = ["#0077BB", "#EE7733", "#009988", "#CC3311",
             "#33BBEE", "#EE3377", "#BBBBBB", "#999999"]

plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "mathtext.default":  "regular",
})

def save_fig(fig, name):
    fig.savefig(OUT_SUPP / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_SUPP / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")

# ---------------------------------------------------------------------------
# Figure S5: feature-set ablation
# ---------------------------------------------------------------------------
abl = pd.read_csv(SCRIPTS / "feature_ablation_results.csv")
targets = ["CO2_uptake", "WC", "Selectivity", "HoA"]
target_labels = {"CO2_uptake": r"CO$_2$ Uptake", "WC": "Working Capacity",
                  "Selectivity": r"CO$_2$/H$_2$ Selectivity" + "\n(log-space)", "HoA": "Heat of Adsorption"}
subset_labels = {
    "1_geometric_only": "Geometric\n(30 feat.)",
    "2_geom_plus_RAC": "+RAC\n(50 feat.)",
    "3_geom_plus_RAC_plus_RDF": "+RDF\n(70 feat.)",
    "4_full_77dim": "Full\n(77 feat.)",
}
subset_order = list(subset_labels.keys())

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=False)
for ax, target in zip(axes, targets):
    sub = abl[abl["target"] == target].set_index("feature_subset").reindex(subset_order)
    x = np.arange(len(subset_order))
    ax.bar(x, sub["R2"].values, color=CB_COLORS[0], width=0.6, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([subset_labels[s] for s in subset_order], fontsize=7)
    ax.set_title(target_labels[target], fontsize=9)
    ax.set_ylabel(r"Test $R^2$" if target == "CO2_uptake" else "")
    lo = sub["R2"].min()
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    for xi, v in zip(x, sub["R2"].values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

fig.suptitle("Figure S5. Feature-set ablation: test R² by descriptor subset", fontsize=10, y=1.05)
fig.tight_layout()
save_fig(fig, "FigS5_feature_ablation")

# ---------------------------------------------------------------------------
# Figure S6: building-block leakage check
# ---------------------------------------------------------------------------
lk = pd.read_csv(SCRIPTS / "leakage_check_results.csv")
split_labels = {"random_split": "Random split\n(primary metric)",
                 "group_split": "Group split\n(stress test)"}
split_order = ["random_split", "group_split"]

fig2, ax2 = plt.subplots(figsize=(6, 4))
x = np.arange(len(targets))
width = 0.35
for i, split in enumerate(split_order):
    sub = lk[lk["split"] == split].set_index("target").reindex(targets)
    offset = (i - 0.5) * width
    bars = ax2.bar(x + offset, sub["R2"].values, width, label=split_labels[split],
                    color=CB_COLORS[i], edgecolor="black", linewidth=0.5)
    for xi, v in zip(x + offset, sub["R2"].values):
        ax2.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels([target_labels[t].replace("\n", " ") for t in targets], fontsize=8)
ax2.set_ylabel(r"Test $R^2$")
ax2.set_ylim(0.6, 1.02)
ax2.legend(frameon=False, loc="lower left")
ax2.set_title("Figure S6. Random-split vs. building-block group-split test $R^2$", fontsize=10)
fig2.tight_layout()
save_fig(fig2, "FigS6_leakage_check")

print("\nDone.")
