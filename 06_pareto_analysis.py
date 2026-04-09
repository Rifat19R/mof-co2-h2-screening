"""
06_pareto_analysis.py
======================
Multi-objective screening analysis for the ARC-MOF CO2/H2 study.

1. Pareto front: working capacity vs. CO2/H2 selectivity
2. Unified score ranking (top 50 candidates)
3. Top-k retrieval: precision@k and recall@k vs. GCMC ground truth

IMPORTANT — selectivity model note:
  The selectivity model (xgb_selectivity_co2h2.json) was trained on
  log1p-transformed targets in 02b_improve_selectivity_hoa.py.
  Raw predictions must be back-transformed via np.expm1() before
  any analysis in physical units.

Outputs:
  data/top_candidates.csv
  data/topk_results.json
  figures/fig_pareto.png
  figures/fig_topk.png
"""

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT   = Path(r"D:\Rifat\MOF_Screening")
DATA   = ROOT / "data"
FIGS   = ROOT / "figures"
MODELS = DATA / "models"
FIGS.mkdir(exist_ok=True)

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
SKIP    = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])
COLORS  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
K_VALS  = [10, 20, 50, 100, 200, 500]
TOP_N   = 100

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.right": False, "axes.spines.top": False})

# ── Load data ─────────────────────────────────────────────────────────────────
df        = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df[c])]
X         = df[feat_cols].fillna(0).values.astype(np.float32)
split     = np.load(DATA / "train_test_idx.npz")
idx_te    = split["idx_te"]

print(f"Loaded {len(df):,} MOFs | {len(feat_cols)} features")

# ── Predict all 4 targets on full dataset ─────────────────────────────────────
print("Loading models and predicting...")
preds = {}
for tgt in TARGETS:
    m = xgb.XGBRegressor()
    m.load_model(str(MODELS / f"xgb_{tgt}.json"))
    raw = m.predict(X)

    # CRITICAL: selectivity model trained in log1p space — back-transform
    if tgt == "selectivity_co2h2":
        preds[tgt] = np.maximum(np.expm1(raw), 0)
        print(f"  {tgt}: log→raw back-transform applied "
              f"(range {preds[tgt].min():.1f}–{preds[tgt].max():.1f})")
    else:
        preds[tgt] = raw
        print(f"  {tgt}: range {raw.min():.2f}–{raw.max():.2f}")

# ── Unified score (normalised, higher = better) ───────────────────────────────
wc_n  = (preds["wc_mmol_g"]         - preds["wc_mmol_g"].min())         / \
        (preds["wc_mmol_g"].max()   - preds["wc_mmol_g"].min()         + 1e-9)
sel_n = (preds["selectivity_co2h2"] - preds["selectivity_co2h2"].min()) / \
        (preds["selectivity_co2h2"].max() - preds["selectivity_co2h2"].min() + 1e-9)
hoa_n = 1 - (preds["heat_of_ads"]   - preds["heat_of_ads"].min())       / \
            (preds["heat_of_ads"].max() - preds["heat_of_ads"].min()     + 1e-9)
co2_n = (preds["co2_uptake_mmol_g"] - preds["co2_uptake_mmol_g"].min()) / \
        (preds["co2_uptake_mmol_g"].max() - preds["co2_uptake_mmol_g"].min() + 1e-9)

unified_score = (wc_n + sel_n + hoa_n + co2_n) / 4

pred_df = df[["mof_id"]].copy()
for tgt in TARGETS:
    pred_df[f"pred_{tgt}"] = preds[tgt]
pred_df["unified_score"] = unified_score

top50 = pred_df.nlargest(50, "unified_score").reset_index(drop=True)
top50.insert(0, "Rank", range(1, 51))
top50.to_csv(DATA / "top_candidates.csv", index=False)
print(f"\nTop-50 saved.")
print(f"  #1: {top50['mof_id'].iloc[0]}")
print(f"  #1 WC       : {top50['pred_wc_mmol_g'].iloc[0]:.2f} mmol/g")
print(f"  #1 Selectivity: {top50['pred_selectivity_co2h2'].iloc[0]:.1f} (real units)")
print(f"  #1 HoA      : {top50['pred_heat_of_ads'].iloc[0]:.2f} kJ/mol")


# ── Pareto front (WC vs Selectivity) ─────────────────────────────────────────
def pareto_front(x, y):
    """Return indices of Pareto-optimal points (maximise both x and y)."""
    idx_sorted = np.argsort(-x)
    front      = []
    best_y     = -np.inf
    for i in idx_sorted:
        if y[i] > best_y:
            front.append(i)
            best_y = y[i]
    return np.array(front)

wc  = preds["wc_mmol_g"]
sel = preds["selectivity_co2h2"]   # real units after expm1
pf  = pareto_front(wc, sel)
print(f"\nPareto front: {len(pf)} MOFs (selectivity in real units)")

# Knee point — max sum of normalised objectives
wc_pf_n  = (wc[pf]  - wc[pf].min())  / (wc[pf].max()  - wc[pf].min()  + 1e-9)
sel_pf_n = (sel[pf] - sel[pf].min()) / (sel[pf].max() - sel[pf].min() + 1e-9)
knee_i   = np.argmax(wc_pf_n + sel_pf_n)
knee_idx = pf[knee_i]
print(f"  Knee point: {df['mof_id'].iloc[knee_idx]}")
print(f"    WC={wc[knee_idx]:.2f} mmol/g  Selectivity={sel[knee_idx]:.1f}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(wc, sel, s=1, alpha=0.08, color="lightsteelblue", rasterized=True)
ax.scatter(wc[pf], sel[pf], s=15, color=COLORS[0], zorder=4,
           label=f"Pareto front (n={len(pf)})")
ax.scatter(wc[knee_idx], sel[knee_idx], s=200, color="red",
           marker="*", zorder=5,
           label=f"Knee point\n{df['mof_id'].iloc[knee_idx][:30]}")
ax.set_xlabel(r"Predicted CO$_2$ Working Capacity [mmol/g]", fontsize=11)
ax.set_ylabel(r"Predicted CO$_2$/H$_2$ Selectivity", fontsize=11)
ax.set_title("Pareto Front: Working Capacity vs. CO₂/H₂ Selectivity\n"
             "278,885 ARC-MOF structures (selectivity in real units)",
             fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig_pareto.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: fig_pareto.png")


# ── Top-k retrieval (on test set only) ───────────────────────────────────────
print("\nComputing top-k retrieval metrics...")
topk_results = {}

for tgt in TARGETS:
    y_te  = df[tgt].iloc[idx_te].values
    p_te  = preds[tgt][idx_te]
    valid = np.isfinite(y_te)
    y_te  = y_te[valid]
    p_te  = p_te[valid]
    n_te  = len(y_te)

    pred_rank = np.argsort(-p_te)
    true_rank = np.argsort(-y_te)
    true_top  = set(true_rank[:TOP_N])

    rows = []
    for k in K_VALS:
        if k > n_te:
            continue
        pred_top = set(pred_rank[:k])
        prec = len(pred_top & true_top) / k
        rec  = len(pred_top & true_top) / TOP_N
        rows.append({"k": k, "precision": prec, "recall": rec})
        if k <= 20:
            print(f"  {tgt:<30} k={k:4d}  P@k={prec:.3f}  R@k={rec:.3f}")
    topk_results[tgt] = rows

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
TARGET_SHORT = {
    "co2_uptake_mmol_g": r"CO$_2$ Uptake",
    "wc_mmol_g"        : "Working Cap.",
    "selectivity_co2h2": "Selectivity",
    "heat_of_ads"      : "Heat of Ads.",
}
for i, tgt in enumerate(TARGETS):
    rows = topk_results[tgt]
    ks   = [r["k"]         for r in rows]
    prec = [r["precision"] for r in rows]
    rec  = [r["recall"]    for r in rows]
    ax1.plot(ks, prec, "o-", color=COLORS[i], lw=2, ms=6,
             label=TARGET_SHORT[tgt])
    ax2.plot(ks, rec,  "s-", color=COLORS[i], lw=2, ms=6)

for ax, ylabel, title in [
    (ax1, f"Precision@k (true top {TOP_N})", "Precision@k"),
    (ax2, f"Recall@k (true top {TOP_N})",    "Recall@k")]:
    ax.set_xlabel("k (structures selected)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
ax1.legend(fontsize=9)
fig.suptitle("Top-k Retrieval Performance — ML vs. GCMC Ground Truth",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig_topk.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig_pareto.png, fig_topk.png, top_candidates.csv, topk_results.json")

with open(DATA / "topk_results.json", "w") as f:
    json.dump(topk_results, f, indent=2)

print("Next: python 10_fix_all_bugs.py")