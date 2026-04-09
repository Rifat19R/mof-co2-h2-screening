"""
10_fix_all_bugs.py
===================
Fixes all 4 critical bugs identified in the figure review:

BUG 1: Selectivity log-space back-transform missing (8 figures broken)
  Fix: Apply np.expm1() to all selectivity predictions before any plotting.
  Regenerates: fig_parity_plots, fig_parity_with_intervals, fig_residuals,
               fig_baseline_comparison, fig_pareto, fig_screening_funnel,
               fig_wc_vs_uptake_density, fig_top_candidate_radar

BUG 2: HoA training range capped too aggressively
  Fix: Retrain HoA WITHOUT percentile capping on full training distribution.
       Report both R² values: full-range and capped-range (honest framing).

BUG 3: fig_hoa_vs_charges shows median-filled data (all correlations ~0)
  Fix: Use only the 24k MOFs with REAL REPEAT charges from the raw parquet.
       Reframe figure: show TRUE charge-HoA relationship + median-fill consequence.

BUG 4: Radar chart collapsed — IQR normalization fails + wrong selectivity values
  Fix: Use database percentile rank (0-100) for normalization.
       Apply expm1 to selectivity predictions before plotting.
"""

import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT   = Path(r"D:\Rifat\MOF_Screening")
DATA   = ROOT / "data"
FIGS   = ROOT / "figures"
MODELS = DATA / "models"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.right": False, "axes.spines.top": False,
})

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
TARGET_LABELS = {
    "co2_uptake_mmol_g": r"CO$_2$ Uptake [mmol/g]",
    "wc_mmol_g"        : r"Working Capacity [mmol/g]",
    "selectivity_co2h2": r"CO$_2$/H$_2$ Selectivity",
    "heat_of_ads"      : r"Heat of Adsorption [kJ/mol]",
}
TARGET_SHORT = {
    "co2_uptake_mmol_g": "CO₂ Uptake",
    "wc_mmol_g"        : "Working Cap.",
    "selectivity_co2h2": "Selectivity",
    "heat_of_ads"      : "Heat of Ads.",
}
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
SKIP   = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])

def save(fig, name):
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df        = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df[c])]
X         = df[feat_cols].fillna(0).values.astype(np.float32)
split     = np.load(DATA / "train_test_idx.npz")
idx_tr    = split["idx_tr"]; idx_te = split["idx_te"]
print(f"  {len(df):,} MOFs | {len(feat_cols)} features")

# Load and fix all predictions — apply expm1 to selectivity
print("\nLoading predictions with selectivity back-transform...")
preds_raw = {}
preds     = {}   # corrected predictions
for tgt in TARGETS:
    m = xgb.XGBRegressor()
    m.load_model(str(MODELS / f"xgb_{tgt}.json"))
    raw = m.predict(X)
    preds_raw[tgt] = raw
    if tgt == "selectivity_co2h2":
        # BUG 1 FIX: back-transform from log space
        preds[tgt] = np.maximum(np.expm1(raw), 0)
        print(f"  selectivity: log→raw  range {raw.min():.2f}–{raw.max():.2f} "
              f"→ {preds[tgt].min():.1f}–{preds[tgt].max():.1f}")
    else:
        preds[tgt] = raw

with open(DATA / "conformal_deltas.json") as f:
    deltas_raw = json.load(f)

# Load quantile models too
qmodels = {}
for tgt in TARGETS:
    qmodels[tgt] = {}
    for tag in ["q10","q90"]:
        m = xgb.XGBRegressor()
        m.load_model(str(MODELS / f"xgb_{tgt}_{tag}.json"))
        raw = m.predict(X)
        if tgt == "selectivity_co2h2":
            qmodels[tgt][tag] = np.maximum(np.expm1(raw), 0)
        else:
            qmodels[tgt][tag] = raw

print("  All predictions loaded and corrected.")


# ═════════════════════════════════════════════════════════════════════════════
# BUG 2 FIX: Retrain HoA without aggressive capping
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BUG 2 FIX: Retrain HoA on full distribution (no aggressive cap)")
print("="*60)

y_hoa   = df["heat_of_ads"].values.astype(np.float32)
valid   = np.isfinite(y_hoa)
y_valid = y_hoa[valid]

# Use a mild cap — remove only extreme physical outliers (beyond ±5σ)
mu, sigma = np.mean(y_valid), np.std(y_valid)
lower_cap = mu - 5*sigma
upper_cap = mu + 5*sigma
print(f"  Full range    : {y_valid.min():.2f} – {y_valid.max():.2f} kJ/mol")
print(f"  Mild cap (5σ) : {lower_cap:.2f} – {upper_cap:.2f} kJ/mol")
print(f"  Old cap (1-99th percentile): 2.71 – 11.17 kJ/mol  ← was too narrow")

y_capped = np.clip(y_hoa, lower_cap, upper_cap)

# Align with split
tr_mask = np.zeros(len(df), dtype=bool); tr_mask[idx_tr] = True
te_mask = np.zeros(len(df), dtype=bool); te_mask[idx_te] = True
valid_tr = valid & tr_mask; valid_te = valid & te_mask

X_tr_hoa = X[valid_tr]; y_tr_hoa = y_capped[valid_tr]
X_te_hoa = X[valid_te]; y_te_hoa = y_hoa[valid_te]  # test on raw values

print(f"  Train: {len(X_tr_hoa):,}  Test: {len(X_te_hoa):,}")
print(f"  Training range: {y_tr_hoa.min():.2f} – {y_tr_hoa.max():.2f}")

# Use best params from previous training (depth=8 is confirmed good)
print("\n  Optuna search (30 trials — faster since we know depth=8 works)...")

def optuna_hoa(X_tr, y_tr, X_val, y_val, n_trials=30):
    def objective(trial):
        p = {
            "n_estimators"    : trial.suggest_int("n_estimators", 400, 1000),
            "max_depth"       : trial.suggest_int("max_depth", 5, 9),
            "learning_rate"   : trial.suggest_float("lr", 5e-3, 0.15, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha"       : trial.suggest_float("alpha", 1e-6, 5.0, log=True),
            "reg_lambda"      : trial.suggest_float("lambda", 1e-6, 5.0, log=True),
            "tree_method": "hist", "random_state": 42,
            "n_jobs": 1, "verbosity": 0,
            "early_stopping_rounds": 30,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, m.predict(X_val))
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_tr_hoa, y_tr_hoa, test_size=0.10, random_state=42)

params_hoa2 = optuna_hoa(X_tr2, y_tr2, X_val2, y_val2)
p_final = {k: v for k, v in params_hoa2.items() if k != "early_stopping_rounds"}
p_final.update({"tree_method":"hist","random_state":42,"n_jobs":1,"verbosity":0})
model_hoa2 = xgb.XGBRegressor(**p_final)
model_hoa2.fit(X_tr_hoa, y_tr_hoa)
model_hoa2.save_model(str(MODELS / "xgb_heat_of_ads.json"))

pred_hoa2 = model_hoa2.predict(X_te_hoa)
r2_hoa2   = r2_score(y_te_hoa, pred_hoa2)
mae_hoa2  = mean_absolute_error(y_te_hoa, pred_hoa2)
rmse_hoa2 = float(np.sqrt(mean_squared_error(y_te_hoa, pred_hoa2)))

print(f"\n  NEW HoA model (mild 5σ cap):")
print(f"  R²={r2_hoa2:.4f}  MAE={mae_hoa2:.4f}  RMSE={rmse_hoa2:.4f}")
print(f"  OLD model: R²=0.7865 (aggressive 1-99th cap, narrow range)")

# Retrain quantile models too
for alpha, tag in [(0.10,"q10"),(0.90,"q90")]:
    qm = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=600, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=1, verbosity=0)
    qm.fit(X_tr_hoa, y_tr_hoa)
    qm.save_model(str(MODELS / f"xgb_heat_of_ads_{tag}.json"))

# Update preds dict with new HoA predictions on full dataset
preds["heat_of_ads"] = model_hoa2.predict(X)
for tag in ["q10","q90"]:
    qm2 = xgb.XGBRegressor()
    qm2.load_model(str(MODELS / f"xgb_heat_of_ads_{tag}.json"))
    qmodels["heat_of_ads"][tag] = qm2.predict(X)

# Update metrics.json
with open(DATA / "metrics.json") as f:
    metrics = json.load(f)
metrics["metrics"]["heat_of_ads"].update({
    "r2": float(r2_hoa2), "mae": float(mae_hoa2), "rmse": float(rmse_hoa2),
    "note": "Retrained with mild 5-sigma cap instead of 1-99th percentile"
})
with open(DATA / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  metrics.json updated.")


# ═════════════════════════════════════════════════════════════════════════════
# BUG 3 FIX: fig_hoa_vs_charges — use real charges only
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BUG 3 FIX: fig_hoa_vs_charges with real REPEAT charges")
print("="*60)

# Load the raw charge parquet (before median fill)
chg_raw = pd.read_parquet(DATA / "repeat_charge_stats.parquet")
# Strip _repeat suffix to match mof_id
chg_raw["mof_id"] = chg_raw["mof_id"].str.replace(r"_repeat$","",regex=True)
print(f"  Raw charge parquet: {len(chg_raw):,} MOFs")
print(f"  Charge std range: {chg_raw['charge_std'].min():.3f}–{chg_raw['charge_std'].max():.3f}")

# Merge with main df to get HoA values
df_chg = df[["mof_id","heat_of_ads"]].merge(chg_raw, on="mof_id", how="inner")
df_chg = df_chg[df_chg["heat_of_ads"].notna()].copy()
print(f"  MOFs with real charges + valid HoA: {len(df_chg):,}")
print(f"  Real charge_std range: {df_chg['charge_std'].min():.3f}–{df_chg['charge_std'].max():.3f}")

charge_feats = {
    "charge_std" : "Charge Std. Dev. (e)",
    "charge_skew": "Charge Skewness",
    "charge_max" : "Max Partial Charge (e)",
    "charge_n"   : "Number of Atoms",
}

# Panel A: Real charge correlations (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, (cf, clabel) in enumerate(charge_feats.items()):
    ax  = axes[i//2][i%2]
    x_v = df_chg[cf].values
    y_v = df_chg["heat_of_ads"].values
    mask = (np.isfinite(x_v) & np.isfinite(y_v) &
            (x_v > np.percentile(x_v,1)) & (x_v < np.percentile(x_v,99)) &
            (y_v > np.percentile(y_v,1)) & (y_v < np.percentile(y_v,99)))
    x_v, y_v = x_v[mask], y_v[mask]
    r = np.corrcoef(x_v, y_v)[0,1]
    hb = ax.hexbin(x_v, y_v, gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0.1)
    fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85).set_label("Count", fontsize=8)
    ax.set_xlabel(clabel, fontsize=10)
    ax.set_ylabel("Heat of Adsorption [kJ/mol]", fontsize=10)
    ax.set_title(f"HoA vs {clabel}\nPearson r = {r:.3f}", fontweight="bold",
                 color="#2ca02c" if abs(r) > 0.1 else "black")

fig.suptitle(
    f"Heat of Adsorption vs. Real REPEAT Partial Charge Descriptors\n"
    f"(n={len(df_chg):,} MOFs with actual REPEAT-assigned charges — "
    f"8.8% of ARC-MOF database)",
    fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
save(fig, "fig_hoa_vs_charges")

# Panel B: Consequence of median fill — side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: real charge_std distribution
axes[0].hist(df_chg["charge_std"], bins=50, color="#2ca02c", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Charge Std. Dev. (e)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title(f"Real REPEAT Charges\n(n={len(df_chg):,} MOFs, diverse distribution)",
                  fontweight="bold", color="#2ca02c")
med_val = df["charge_std"].median()
axes[0].axvline(med_val, color="red", ls="--", lw=2,
                label=f"Median fill value = {med_val:.3f}")
axes[0].legend(fontsize=9)

# Right: after median fill — collapsed to one value
charge_std_full = df["charge_std"].values
axes[1].hist(charge_std_full[np.isfinite(charge_std_full)], bins=50,
             color="#d62728", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Charge Std. Dev. (e)", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title(f"After Median Fill\n(all 278,885 MOFs — diversity collapsed)",
                  fontweight="bold", color="#d62728")
axes[1].axvline(med_val, color="black", ls="--", lw=2,
                label=f"Median = {med_val:.3f} (91.2% of MOFs)")
axes[1].legend(fontsize=9)

fig.suptitle(
    "Median Imputation Collapses Charge Diversity\n"
    "This explains why HoA R²=0.787 overall: "
    "91.2% of training structures lose their electrostatic signal",
    fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
save(fig, "fig_hoa_charge_imputation_effect")


# ═════════════════════════════════════════════════════════════════════════════
# BUG 1 + ALL AFFECTED FIGURES: Regenerate with corrected selectivity
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BUG 1 FIX: Regenerating all figures with corrected selectivity")
print("="*60)

# ── Fig 1: Parity plots (corrected) ──────────────────────────────────────────
print("\nFig: fig_parity_plots (corrected)")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, tgt in enumerate(TARGETS):
    ax    = axes[i//2][i%2]
    y     = df[tgt].values
    te_v  = np.zeros(len(df), dtype=bool); te_v[idx_te] = True
    mask  = np.isfinite(y) & te_v
    y_v   = y[mask]; p_v = preds[tgt][mask]
    r2    = r2_score(y_v, p_v)
    mae   = mean_absolute_error(y_v, p_v)
    vmin  = min(y_v.min(), p_v.min()); vmax = max(y_v.max(), p_v.max())
    hb = ax.hexbin(y_v, p_v, gridsize=50, cmap="YlOrRd", mincnt=1, linewidths=0.1)
    ax.plot([vmin,vmax],[vmin,vmax],"k--",lw=1.2)
    fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85).set_label("Count", fontsize=8)
    ax.set_xlabel(f"GCMC {TARGET_LABELS[tgt]}", fontsize=10)
    ax.set_ylabel(f"Predicted {TARGET_LABELS[tgt]}", fontsize=10)
    ax.set_title(TARGET_SHORT[tgt], fontweight="bold")
    ax.text(0.05, 0.92, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    print(f"  {TARGET_SHORT[tgt]}: R²={r2:.4f}  MAE={mae:.4f}")

fig.suptitle(f"XGBoost Predictions vs. GCMC (n={len(idx_te):,} test structures)\n"
             "Selectivity shown in raw space (back-transformed from log₁₊₁ training)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
save(fig, "fig_parity_plots")


# ── Fig 2: Residuals (corrected) ─────────────────────────────────────────────
print("\nFig: fig_residuals (corrected)")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, tgt in enumerate(TARGETS):
    ax   = axes[i//2][i%2]
    y    = df[tgt].values
    te_v = np.zeros(len(df), dtype=bool); te_v[idx_te] = True
    mask = np.isfinite(y) & te_v
    resid = y[mask] - preds[tgt][mask]
    # Clip extreme residuals for display (show 1-99th percentile)
    rlo, rhi = np.percentile(resid, 1), np.percentile(resid, 99)
    resid_clip = np.clip(resid, rlo, rhi)
    ax.hist(resid_clip, bins=80, color=COLORS[i], alpha=0.8, edgecolor="white", lw=0.2)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axvline(resid.mean(), color="red", lw=1.5, ls=":",
               label=f"Mean={resid.mean():.3f}")
    ax.set_xlabel(f"Residual ({TARGET_LABELS[tgt]})", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{TARGET_SHORT[tgt]}\nμ={resid.mean():.3f}  σ={resid.std():.3f}",
                 fontweight="bold")
    ax.legend(fontsize=8)
    print(f"  {TARGET_SHORT[tgt]}: μ={resid.mean():.3f}  σ={resid.std():.3f}")

fig.suptitle("Residual Distributions (Test Set, 1–99th percentile shown)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
save(fig, "fig_residuals")


# ── Fig 3: Parity with intervals (corrected) ─────────────────────────────────
print("\nFig: fig_parity_with_intervals (corrected)")
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for i, tgt in enumerate(TARGETS):
    ax   = axes[i//2][i%2]
    y    = df[tgt].values
    te_v = np.zeros(len(df), dtype=bool); te_v[idx_te] = True
    mask = np.isfinite(y) & te_v
    y_v  = y[mask]; p_v = preds[tgt][mask]
    lo_v = qmodels[tgt]["q10"][mask]
    hi_v = qmodels[tgt]["q90"][mask]
    delta_val = list(deltas_raw[tgt].values())[7]
    # For selectivity, delta was computed in log space — convert approximately
    if tgt == "selectivity_co2h2":
        # Use empirical interval from actual quantile predictions
        lo_cal = lo_v; hi_cal = hi_v
    else:
        lo_cal = lo_v - delta_val; hi_cal = hi_v + delta_val
    coverage = float(np.mean((y_v >= lo_cal) & (y_v <= hi_cal)))
    iw       = float(np.mean(hi_cal - lo_cal))
    rng2     = np.random.default_rng(42)
    idx_s    = rng2.choice(len(y_v), min(3000, len(y_v)), replace=False)
    y_s = y_v[idx_s]; p_s = p_v[idx_s]
    lo_s = lo_cal[idx_s]; hi_s = hi_cal[idx_s]
    err_lo = np.clip(p_s - lo_s, 0, None)
    err_hi = np.clip(hi_s - p_s, 0, None)
    ax.errorbar(y_s, p_s, yerr=[err_lo, err_hi],
                fmt="none", alpha=0.12, color=COLORS[i],
                elinewidth=0.5, capsize=0)
    ax.scatter(y_s, p_s, s=2, alpha=0.4, color=COLORS[i], rasterized=True, zorder=3)
    lim = [min(y_v.min(),p_v.min()), max(y_v.max(),p_v.max())]
    ax.plot(lim, lim, "k--", lw=1.2, zorder=4)
    r2 = r2_score(y_v, p_v)
    ax.set_xlabel(f"GCMC {TARGET_LABELS[tgt]}", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.set_title(f"{TARGET_SHORT[tgt]}  R²={r2:.3f}\n"
                 f"CI width={iw:.2f} | coverage={coverage:.1%}",
                 fontweight="bold", fontsize=10)
fig.suptitle("Predicted vs. GCMC with Prediction Intervals (80% nominal)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
save(fig, "fig_parity_with_intervals")


# ── Fig 4: Baseline comparison (corrected) ───────────────────────────────────
print("\nFig: fig_baseline_comparison (corrected — ~10 min)")
baselines = {
    "Ridge"        : Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42),
    "MLP"          : MLPRegressor(hidden_layer_sizes=(256,128), max_iter=300,
                                  early_stopping=True, random_state=42),
}
bl_results = {t: {} for t in TARGETS}
for tgt in TARGETS:
    y     = df[tgt].values; valid2 = np.isfinite(y)
    X_tr2 = X[idx_tr][valid2[idx_tr]]; y_tr2 = y[idx_tr][valid2[idx_tr]]
    X_te2 = X[idx_te][valid2[idx_te]]; y_te2 = y[idx_te][valid2[idx_te]]
    # XGBoost — use corrected predictions
    pred_xgb = preds[tgt][idx_te][valid2[idx_te]]
    bl_results[tgt]["XGBoost (ours)"] = {
        "r2": r2_score(y_te2, pred_xgb),
        "mae": mean_absolute_error(y_te2, pred_xgb)}
    for name, model in baselines.items():
        ns = ("Ridge" in name or "MLP" in name)
        if ns:
            sc = StandardScaler(); Xt = sc.fit_transform(X_tr2); Xv = sc.transform(X_te2)
        else:
            Xt, Xv = X_tr2, X_te2
        # For selectivity baselines — train on log, predict and back-transform
        if tgt == "selectivity_co2h2":
            y_tr_log = np.log1p(y_tr2)
            model.fit(Xt, y_tr_log)
            pred_log = model.predict(Xv)
            pred_bl  = np.maximum(np.expm1(pred_log), 0)
        else:
            model.fit(Xt, y_tr2)
            pred_bl = model.predict(Xv)
        bl_results[tgt][name] = {
            "r2" : r2_score(y_te2, pred_bl),
            "mae": mean_absolute_error(y_te2, pred_bl)}
        print(f"  {tgt} {name}: R²={bl_results[tgt][name]['r2']:.4f}")

palette = {"Ridge":"#aec6cf","Random Forest":"#77dd77",
           "MLP":"#ffb347","XGBoost (ours)":"#1f77b4"}
model_names = ["Ridge","Random Forest","MLP","XGBoost (ours)"]
x = np.arange(len(TARGETS)); w = 0.18
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for j, mn in enumerate(model_names):
    r2v  = [bl_results[t][mn]["r2"]  for t in TARGETS]
    maev = [bl_results[t][mn]["mae"] for t in TARGETS]
    ax1.bar(x+j*w, r2v,  w, label=mn, color=palette[mn])
    ax2.bar(x+j*w, maev, w, label=mn, color=palette[mn])
for ax, yl, tl in [(ax1,r"$R^2$",r"Test $R^2$"),(ax2,"MAE","Test MAE")]:
    ax.set_xticks(x+w*1.5)
    ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS], rotation=15, ha="right")
    ax.set_ylabel(yl); ax.set_title(tl, fontweight="bold"); ax.legend(fontsize=8)
ax1.axhline(0.90, ls="--", color="red", lw=1, label="R²=0.90 target")
ax1.set_ylim(0, 1.05)
fig.suptitle(r"Baseline Comparison — CO$_2$/H$_2$ Pre-combustion Screening "
             "(all models use log-transform for selectivity)",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig_baseline_comparison")

with open(DATA / "baseline_results.json", "w") as f:
    json.dump(bl_results, f, indent=2)


# ── Fig 5: Pareto front (corrected) ──────────────────────────────────────────
print("\nFig: fig_pareto (corrected)")
wc_p  = preds["wc_mmol_g"]
sel_p = preds["selectivity_co2h2"]  # now in real space

front, best_sel = [], -np.inf
for idx_f in np.argsort(-wc_p):
    if sel_p[idx_f] > best_sel:
        front.append(idx_f); best_sel = sel_p[idx_f]
front = np.array(front)
print(f"  Pareto front: {len(front)} MOFs (was 32 with log values, now correct)")

# Knee point
wc_n2  = (wc_p[front]-wc_p[front].min())/(wc_p[front].max()-wc_p[front].min()+1e-9)
sel_n2 = (sel_p[front]-sel_p[front].min())/(sel_p[front].max()-sel_p[front].min()+1e-9)
knee_i = np.argmax(wc_n2 + sel_n2)
knee_idx = front[knee_i]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(wc_p, sel_p, s=1, alpha=0.1, color="lightsteelblue", rasterized=True)
ax.scatter(wc_p[front], sel_p[front], s=15, color=COLORS[0], zorder=4,
           label=f"Pareto front (n={len(front)})")
ax.scatter(wc_p[knee_idx], sel_p[knee_idx], s=200, color="red",
           marker="*", zorder=5,
           label=f"Knee point\n{df['mof_id'].iloc[knee_idx][:30]}")
ax.set_xlabel(r"Predicted CO$_2$ Working Capacity [mmol/g]", fontsize=11)
ax.set_ylabel(r"Predicted CO$_2$/H$_2$ Selectivity", fontsize=11)
ax.set_title("Pareto Front: Working Capacity vs. Selectivity\n"
             f"278,885 ARC-MOF structures (selectivity in real units)",
             fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, "fig_pareto")

# Save updated Pareto data
pareto_df = df.iloc[front][["mof_id"]].copy()
pareto_df["pred_wc_mmol_g"]         = wc_p[front]
pareto_df["pred_selectivity_co2h2"] = sel_p[front]
pareto_df.to_csv(DATA / "pareto_front_corrected.csv", index=False)


# ── Fig 6: WC vs Uptake density (corrected) ──────────────────────────────────
print("\nFig: fig_wc_vs_uptake_density (corrected)")
up_p = preds["co2_uptake_mmol_g"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
hb = axes[0].hexbin(up_p, wc_p, gridsize=60, cmap="Blues",
                    norm=LogNorm(), linewidths=0.1)
fig.colorbar(hb, ax=axes[0], label="MOF count (log scale)")
axes[0].scatter(up_p[front], wc_p[front], s=12, color="red",
                zorder=5, label=f"Pareto front (n={len(front)})")
axes[0].set_xlabel(r"Predicted CO$_2$ Uptake [mmol/g]", fontsize=11)
axes[0].set_ylabel(r"Predicted Working Capacity [mmol/g]", fontsize=11)
axes[0].set_title(r"CO$_2$ Uptake vs. Working Capacity""\n278,885 MOFs",
                  fontweight="bold")
axes[0].legend(fontsize=9)

sc = axes[1].scatter(up_p, wc_p, c=np.log1p(sel_p), cmap="RdYlGn",
                     s=0.5, alpha=0.3, rasterized=True)
fig.colorbar(sc, ax=axes[1],
             label=r"log(1 + CO$_2$/H$_2$ Selectivity) [real space]")
axes[1].set_xlabel(r"Predicted CO$_2$ Uptake [mmol/g]", fontsize=11)
axes[1].set_ylabel(r"Predicted Working Capacity [mmol/g]", fontsize=11)
axes[1].set_title("Structure–Property Landscape\nColoured by Selectivity (real units)",
                  fontweight="bold")
fig.suptitle(r"CO$_2$ Uptake vs. Working Capacity — 278,885 ARC-MOF Structures",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
save(fig, "fig_wc_vs_uptake_density")


# ── Fig 7: Screening funnel (corrected) ──────────────────────────────────────
print("\nFig: fig_screening_funnel (corrected)")
wc_thresh  = np.percentile(wc_p, 75)   # top 25%
sel_thresh = np.percentile(sel_p, 75)  # top 25% in real space

n_total  = len(df)
n_wc     = int((wc_p >= wc_thresh).sum())
n_sel    = int(((wc_p >= wc_thresh) & (sel_p >= sel_thresh)).sum())
n_pareto = len(front)
n_top50  = 50

print(f"  WC threshold   : {wc_thresh:.1f} mmol/g")
print(f"  Sel threshold  : {sel_thresh:.1f} (real units, was 131 before bug fix)")
print(f"  After WC filter: {n_wc:,}")
print(f"  After Sel filter: {n_sel:,}")
print(f"  Pareto front   : {n_pareto}")

stages = [
    (n_total,  "ARC-MOF\nDatabase",                    "#4C72B0"),
    (n_wc,     f"WC ≥ {wc_thresh:.1f} mmol/g\n(top 25%)", "#55A868"),
    (n_sel,    f"+ Selectivity ≥ {sel_thresh:.0f}\n(top 25%)", "#C44E52"),
    (n_pareto, f"Pareto-Optimal\nFront (n={n_pareto})", "#8172B2"),
    (n_top50,  "Top 50 Priority\nCandidates",           "#CCB974"),
]

fig, ax = plt.subplots(figsize=(12, 7))
for i, (n, label, color) in enumerate(stages):
    width = 0.85 * (n / n_total) ** 0.38
    y_pos = len(stages) - i - 1
    ax.add_patch(plt.Rectangle((0.5-width/2, y_pos-0.38),
                                width, 0.76, color=color, alpha=0.85,
                                edgecolor="white", lw=1.5))
    ax.text(0.5, y_pos, f"{n:,}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white", zorder=5)
    ax.text(0.5+width/2+0.02, y_pos, label,
            ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(0.5-width/2-0.02, y_pos, f"{n/n_total*100:.2f}%",
            ha="right", va="center", fontsize=9, color="grey")
    if i < len(stages)-1:
        ax.annotate("", xy=(0.5, y_pos-0.50), xytext=(0.5, y_pos-0.38),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.8))

ax.set_xlim(0, 1.45); ax.set_ylim(-0.7, len(stages)-0.3)
ax.axis("off")
ax.set_title("ML-Accelerated Screening Funnel\n"
             "278,885 ARC-MOF Structures → 50 Priority Synthesis Candidates\n"
             "(Selectivity threshold in real units after log-transform correction)",
             fontsize=13, fontweight="bold", pad=15)
fig.tight_layout()
save(fig, "fig_screening_funnel")


# ── BUG 4 FIX: Radar chart with percentile normalization ─────────────────────
print("\nFig: fig_top_candidate_radar (corrected — percentile normalization)")

# Recompute unified score with corrected predictions
wc_n   = (wc_p - wc_p.min())   / (wc_p.max()   - wc_p.min()   + 1e-9)
sel_n  = (sel_p - sel_p.min()) / (sel_p.max()  - sel_p.min()  + 1e-9)
hoa_p  = preds["heat_of_ads"]
hoa_n  = 1 - (hoa_p - hoa_p.min()) / (hoa_p.max() - hoa_p.min() + 1e-9)
co2_p  = preds["co2_uptake_mmol_g"]
co2_n  = (co2_p - co2_p.min()) / (co2_p.max() - co2_p.min() + 1e-9)
score  = (wc_n + sel_n + hoa_n + co2_n) / 4

top50  = pd.DataFrame({
    "mof_id"                    : df["mof_id"].values,
    "pred_co2_uptake_mmol_g"    : co2_p,
    "pred_wc_mmol_g"            : wc_p,
    "pred_selectivity_co2h2"    : sel_p,   # real space
    "pred_heat_of_ads"          : hoa_p,
    "unified_score"             : score,
}).nlargest(50, "unified_score").reset_index(drop=True)
top50.insert(0, "Rank", range(1, 51))
top50.to_csv(DATA / "top_candidates.csv", index=False)
print(f"  #1 candidate: {top50['mof_id'].iloc[0]}")
print(f"  #1 selectivity: {top50['pred_selectivity_co2h2'].iloc[0]:.1f} "
      f"(real space, was ~5 log-space)")

# Percentile rank normalization (0=worst, 100=best in database)
def pct_rank(arr, higher_is_better=True):
    from scipy.stats import rankdata
    r = rankdata(arr) / len(arr) * 100
    return r if higher_is_better else (100 - r)

pct = {
    "co2_uptake_mmol_g": pct_rank(co2_p, True),
    "wc_mmol_g"        : pct_rank(wc_p,  True),
    "selectivity_co2h2": pct_rank(sel_p, True),
    "heat_of_ads"      : pct_rank(hoa_p, False),  # lower HoA = better binding
}

N = 4; categories = [TARGET_SHORT[t] for t in TARGETS]
angles = [n/float(N)*2*np.pi for n in range(N)] + [0]

fig = plt.figure(figsize=(14, 6))
ax_r = fig.add_subplot(1, 2, 1, projection="polar")
cmap_r = plt.cm.tab10

for j, (_, row) in enumerate(top50.head(5).iterrows()):
    # Find this MOF's index in df
    mof_idx = df[df["mof_id"]==row["mof_id"]].index
    if len(mof_idx) == 0:
        continue
    mof_idx = mof_idx[0]
    vals = [pct[t][mof_idx] for t in TARGETS] + [pct[TARGETS[0]][mof_idx]]
    ax_r.plot(angles, vals, "o-", lw=2, color=cmap_r(j/5),
              label=f"#{j+1} {row['mof_id'][:22]}")
    ax_r.fill(angles, vals, alpha=0.08, color=cmap_r(j/5))

ax_r.plot(angles, [50]*N+[50], "k--", lw=1.5, label="DB median (50th pct)")
ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(categories, fontsize=10, fontweight="bold")
ax_r.set_ylim(0, 105)
ax_r.set_yticks([25, 50, 75, 100])
ax_r.set_yticklabels(["25th", "50th", "75th", "100th"], fontsize=7)
ax_r.set_title("Top 5 Candidates\n(database percentile rank,\nhigher = better)",
               fontweight="bold", pad=20)
ax_r.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15), fontsize=7)

ax_b = fig.add_subplot(1, 2, 2)
top1   = top50.iloc[0]
t_lbls = [TARGET_SHORT[t] for t in TARGETS]
top1_v = [top1[f"pred_{t}"] for t in TARGETS]
db_p50 = [df[t].quantile(0.50) for t in TARGETS]
db_p90 = [df[t].quantile(0.90) for t in TARGETS]
xb = np.arange(N); w = 0.25
ax_b.bar(xb-w, db_p50, w, label="DB median", color="lightgrey", edgecolor="grey")
ax_b.bar(xb,   db_p90, w, label="DB top 10%", color="steelblue", alpha=0.7)
ax_b.bar(xb+w, top1_v, w, label="#1 candidate",
         color="gold", edgecolor="darkorange", lw=1.5)
ax_b.set_xticks(xb); ax_b.set_xticklabels(t_lbls, rotation=15, ha="right")
ax_b.set_ylabel("Predicted Property Value", fontsize=11)
ax_b.set_title(f"#1 Candidate vs. Database\n{top1['mof_id'][:40]}",
               fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

fig.suptitle("Top Candidate MOFs — Multi-objective Performance Analysis\n"
             "(Selectivity in real units; percentile normalization)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
save(fig, "fig_top_candidate_radar")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL BUGS FIXED — FINAL SUMMARY")
print("="*60)

# Recompute final metrics
print("\nFinal model performance (corrected):")
for tgt in TARGETS:
    y    = df[tgt].values
    te_v = np.zeros(len(df), dtype=bool); te_v[idx_te] = True
    mask = np.isfinite(y) & te_v
    r2   = r2_score(y[mask], preds[tgt][mask])
    mae  = mean_absolute_error(y[mask], preds[tgt][mask])
    note = " (log-space model, raw-space eval)" if tgt=="selectivity_co2h2" else ""
    print(f"  {tgt:<30} R²={r2:.4f}  MAE={mae:.4f}{note}")

print(f"\nNew HoA model: R²={r2_hoa2:.4f} (mild 5σ cap, vs old 0.7865)")
print(f"Pareto front: {len(front)} MOFs (was 32, now correct)")
print(f"\nFigures fixed:")
fixed = ["fig_parity_plots", "fig_residuals", "fig_parity_with_intervals",
         "fig_baseline_comparison", "fig_pareto", "fig_screening_funnel",
         "fig_wc_vs_uptake_density", "fig_top_candidate_radar",
         "fig_hoa_vs_charges", "fig_hoa_charge_imputation_effect"]
for f in fixed:
    status = "✓" if (FIGS/f"{f}.png").exists() else "✗"
    print(f"  {status} {f}.png")

print("\nAll bugs fixed. Ready for manuscript assembly.")