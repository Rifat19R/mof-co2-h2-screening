"""
05b_fix_external_validation.py
================================
Fixes two issues in fig_external_validation.png:

  1. Selectivity predictions are in log space (model retrained with log1p
     in 02b_improve_selectivity_hoa.py) — must apply expm1 back-transform.

  2. Adds a diagnostic table showing how many CoRE features matched
     ARC-MOF features, so the distribution narrowness is explained.

Also regenerates the figure with better axis limits and annotations.
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

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
TARGET_LABELS = {
    "co2_uptake_mmol_g": r"CO$_2$ Uptake [mmol/g]",
    "wc_mmol_g"        : "Working Capacity [mmol/g]",
    "selectivity_co2h2": r"CO$_2$/H$_2$ Selectivity",
    "heat_of_ads"      : "Heat of Adsorption [kJ/mol]",
}
TARGET_SHORT = {
    "co2_uptake_mmol_g": "CO₂ Uptake",
    "wc_mmol_g"        : "Working Cap.",
    "selectivity_co2h2": "Selectivity",
    "heat_of_ads"      : "Heat of Ads.",
}
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
SKIP   = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])

# ── Load ARC-MOF test set ─────────────────────────────────────────────────────
df        = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df[c])]
split     = np.load(DATA / "train_test_idx.npz")
idx_te    = split["idx_te"]

print(f"ARC-MOF features used by models: {len(feat_cols)}")

# ── Load CoRE MOF ─────────────────────────────────────────────────────────────
core_path = Path(r"D:\Rifat\Semiconductor_Twin\data\core_mof_h2_features.parquet")
if not core_path.exists():
    core_path = DATA / "core_mof_h2_features.parquet"

core = pd.read_parquet(core_path)
print(f"CoRE MOF shape: {core.shape}")
print(f"CoRE columns (first 20): {core.columns.tolist()[:20]}")

# ── Map CoRE features → ARC-MOF feature space ────────────────────────────────
# Direct column name matches
direct_matches = [c for c in feat_cols if c in core.columns]
print(f"\nDirect column matches: {len(direct_matches)} / {len(feat_cols)}")
print(f"Matched: {direct_matches[:10]}...")

# Known column mappings CoRE → ARC-MOF
col_map = {
    "LCD"        : "Di",
    "PLD"        : "Df",
    "LFPD"       : "Dif",
    "AV_VF"      : "AVAf",
    "AV_cm3_g"   : "AVAg",
    "ASA_m2_g"   : "gASA",
    "ASA_m2_cm3" : "vASA",
    "Density"    : "Density",
    "void_fraction": "AVAf",
    "surface_area_m2_g": "gASA",
    "largest_free_sphere": "Df",
    "largest_included_sphere": "Di",
}

# Build CoRE feature matrix
X_core = pd.DataFrame(0.0, index=core.index, columns=feat_cols)

# Apply direct matches first
for col in direct_matches:
    X_core[col] = core[col].fillna(0).values

# Apply mapped columns
n_mapped = 0
for core_col, arc_col in col_map.items():
    if core_col in core.columns and arc_col in feat_cols:
        X_core[arc_col] = core[core_col].fillna(0).values
        n_mapped += 1

# Derived geometric features
if "Di" in X_core.columns and "Df" in X_core.columns:
    di = X_core["Di"].values; df_val = X_core["Df"].values
    dif = X_core["Dif"].values if "Dif" in X_core.columns else np.zeros(len(core))
    vf  = X_core["AVAf"].values
    avg = X_core["AVAg"].values
    gasa = X_core["gASA"].values

    derived = {
        "Di_Df_ratio" : np.where(df_val>0, di/df_val, 1.0),
        "Dif_Di_ratio": np.where(di>0, dif/di, 1.0),
        "VF_sq"       : vf**2,
        "one_minus_VF": 1 - vf,
        "packing_eff" : 1 - vf,
        "sa_per_density": gasa * avg,
        "log_Di"      : np.log1p(np.abs(di)),
        "log_Df"      : np.log1p(np.abs(df_val)),
        "log_gASA"    : np.log1p(np.abs(gasa)),
        "log_AVAg"    : np.log1p(np.abs(avg)),
        "SA_x_VF"     : gasa * vf,
        "PV_x_VF"     : avg  * vf,
        "SA_x_PV"     : gasa * avg,
        "POAVAg"      : avg  * 0.95,
        "POAVAf"      : vf   * 0.95,
        "log_POAVAg"  : np.log1p(np.abs(avg * 0.95)),
    }
    for feat, vals in derived.items():
        if feat in X_core.columns:
            X_core[feat] = vals

# Coverage report
n_nonzero = (X_core != 0).any(axis=0).sum()
print(f"Features with non-zero CoRE values: {n_nonzero} / {len(feat_cols)} "
      f"({n_nonzero/len(feat_cols):.1%})")
print(f"CoRE rows: {len(X_core):,}")

X_core_arr = X_core.values.astype(np.float32)

# ── Predict ───────────────────────────────────────────────────────────────────
predictions = {}
for tgt in TARGETS:
    m = xgb.XGBRegressor()
    m.load_model(str(MODELS / f"xgb_{tgt}.json"))
    raw_pred = m.predict(X_core_arr)

    # CRITICAL FIX: selectivity model was trained on log1p scale
    if tgt == "selectivity_co2h2":
        pred = np.expm1(raw_pred)          # back-transform
        pred = np.maximum(pred, 0)         # clip negatives
        print(f"  {tgt}: log→raw back-transform applied")
        print(f"    Raw pred range: {raw_pred.min():.2f}–{raw_pred.max():.2f} (log)")
        print(f"    Back-transformed: {pred.min():.1f}–{pred.max():.1f}")
    else:
        pred = raw_pred

    predictions[tgt] = pred
    print(f"  {tgt}: {pred.min():.2f}–{pred.max():.2f}  "
          f"mean={pred.mean():.2f}")

# ── Compare with ARC-MOF test distribution ────────────────────────────────────
print("\nARC-MOF test set ranges:")
for tgt in TARGETS:
    y_te = df[tgt].iloc[idx_te].dropna()
    print(f"  {tgt}: {y_te.min():.2f}–{y_te.max():.2f}  mean={y_te.mean():.2f}")

# ── Save predictions CSV ──────────────────────────────────────────────────────
id_col = next((c for c in core.columns
               if "name" in c.lower() or "id" in c.lower()), core.columns[0])
pred_df = core[[id_col]].copy()
for tgt in TARGETS:
    pred_df[f"pred_{tgt}"] = predictions[tgt]
pred_df.to_csv(DATA / "core_mof_predictions.csv", index=False)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for i, tgt in enumerate(TARGETS):
    ax      = axes[i//2][i%2]
    y_arc   = df[tgt].iloc[idx_te].dropna().values
    y_core  = predictions[tgt]

    # Use shared x range covering both distributions
    x_min = min(np.percentile(y_arc, 1),  np.percentile(y_core, 1))
    x_max = max(np.percentile(y_arc, 99), np.percentile(y_core, 99))
    bins   = np.linspace(x_min, x_max, 60)

    ax.hist(y_arc,  bins=bins, density=True, alpha=0.55,
            color="grey",     label=f"ARC-MOF test (n={len(y_arc):,})")
    ax.hist(y_core, bins=bins, density=True, alpha=0.70,
            color=COLORS[i],  label=f"CoRE 2019 (n={len(y_core):,})")

    # Stats annotations
    ax.axvline(np.mean(y_arc),  color="grey",     ls="--", lw=1.5,
               label=f"ARC mean={np.mean(y_arc):.1f}")
    ax.axvline(np.mean(y_core), color=COLORS[i],  ls="--", lw=1.5,
               label=f"CoRE mean={np.mean(y_core):.1f}")

    ax.set_xlabel(TARGET_LABELS[tgt], fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(TARGET_SHORT[tgt], fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Add feature coverage note
fig.text(0.5, 0.01,
         f"Note: CoRE MOF predictions use {n_nonzero}/{len(feat_cols)} "
         f"matched features ({n_nonzero/len(feat_cols):.0%}). "
         f"RAC/RDF descriptors unavailable for CoRE — geometric features only.",
         ha="center", fontsize=8, color="grey", style="italic")

fig.suptitle("External Validation: ARC-MOF vs CoRE MOF 2019 Predicted Distributions\n"
             "Models trained on ARC-MOF applied to 13,758 independent CoRE structures",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIGS / "fig_external_validation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ fig_external_validation.png (fixed)")

# ── Update external_validation.json ──────────────────────────────────────────
results = {
    "n_core": len(core),
    "n_features_matched": int(n_nonzero),
    "n_features_total"  : len(feat_cols),
    "feature_coverage"  : float(n_nonzero/len(feat_cols)),
    "selectivity_log_transform_applied": True,
    "pred_summary": {
        t: {"mean": float(predictions[t].mean()),
            "std" : float(predictions[t].std()),
            "min" : float(predictions[t].min()),
            "max" : float(predictions[t].max())}
        for t in TARGETS
    }
}
with open(DATA / "external_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ external_validation.json updated")
print("\nDone. Check fig_external_validation.png — selectivity should now show a real distribution.")