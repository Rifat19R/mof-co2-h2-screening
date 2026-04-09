"""
03_uncertainty.py
==================
Conformal prediction calibration using a held-out calibration set.
Produces calibrated 80% prediction intervals for all 4 targets.
Output: data/conformal_deltas.json, figures/uncertainty_calibration.png
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT    = Path(r"D:\Rifat\MOF_Screening")
DATA    = ROOT / "data"
FIGS    = ROOT / "figures"
MODELS  = DATA / "models"
FIGS.mkdir(exist_ok=True)

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
TARGET_LABELS = {"co2_uptake_mmol_g": r"CO$_2$ Uptake",
                 "wc_mmol_g": "Working Cap.",
                 "selectivity_co2h2": "Selectivity",
                 "heat_of_ads": "Heat of Ads."}
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
SKIP   = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])
LEVELS = np.arange(0.10, 0.96, 0.10)

df        = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df[c])]
X         = df[feat_cols].fillna(0).values.astype(np.float32)
split     = np.load(DATA / "train_test_idx.npz")
idx_tr    = split["idx_tr"]

# Carve out calibration set from training
idx_tr2, idx_cal = train_test_split(idx_tr, test_size=0.10, random_state=42)
idx_te = split["idx_te"]

all_deltas = {}
cov_before = {t: [] for t in TARGETS}
cov_after  = {t: [] for t in TARGETS}

for tgt in TARGETS:
    y = df[tgt].values.astype(np.float32)
    valid = np.isfinite(y)
    X_cal = X[idx_cal][valid[idx_cal]]; y_cal = y[idx_cal][valid[idx_cal]]
    X_te  = X[idx_te][valid[idx_te]];  y_te  = y[idx_te][valid[idx_te]]

    lo = xgb.XGBRegressor(); lo.load_model(str(MODELS / f"xgb_{tgt}_q10.json"))
    hi = xgb.XGBRegressor(); hi.load_model(str(MODELS / f"xgb_{tgt}_q90.json"))

    lo_cal = lo.predict(X_cal); hi_cal = hi.predict(X_cal)
    lo_te  = lo.predict(X_te);  hi_te  = hi.predict(X_te)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

    deltas = {}
    for nom in LEVELS:
        alpha   = 1 - nom
        q_level = min(np.ceil((len(scores)+1)*(1-alpha))/len(scores), 1.0)
        delta   = float(np.quantile(scores, q_level))
        cov_b   = float(np.mean((y_te >= lo_te) & (y_te <= hi_te)))
        cov_a   = float(np.mean((y_te >= lo_te-delta) & (y_te <= hi_te+delta)))
        deltas[float(nom)] = delta
        cov_before[tgt].append(cov_b)
        cov_after[tgt].append(cov_a)
    all_deltas[tgt] = deltas
    print(f"  {tgt}: 80% coverage after calibration = "
          f"{cov_after[tgt][7-1]:.3f}")  # index 7 ≈ 0.80

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
diag = np.linspace(0.05, 0.95, 100)
for ax, data, title in [
    (ax1, cov_before, "Before calibration"),
    (ax2, cov_after,  "After conformal calibration")]:
    ax.plot(diag, diag, "k--", lw=1.2, label="Perfect")
    ax.fill_between(diag, diag-0.05, diag+0.05, alpha=0.1, color="grey")
    for i, tgt in enumerate(TARGETS):
        ax.plot(LEVELS, data[tgt], "o-", color=COLORS[i],
                label=TARGET_LABELS[tgt], ms=5, lw=1.8)
    ax.set_xlabel("Nominal coverage", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
fig.suptitle("Conformal Prediction Interval Calibration", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig_uncertainty_calibration.png", dpi=300, bbox_inches="tight")
plt.close(fig)

with open(DATA / "conformal_deltas.json", "w") as f:
    json.dump(all_deltas, f, indent=2)
print(f"  Saved: conformal_deltas.json + fig_uncertainty_calibration.png")
print("Next: python 04_shap_analysis.py")
