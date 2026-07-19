"""
generate_outputs.py
===================
Generates all CSV/parquet outputs needed for figure generation.

Run from your scripts folder:
    python generate_outputs.py

All four targets use XGBoost regressors with Optuna-optimised
hyperparameters. Trained models are cached to data/models/ so
subsequent runs skip retraining.

Steps:
  1  Load full_features.parquet
  2  Train/test split (seed=42, 90/10, stratified by CO2 uptake decile)
  3  Train/load XGBoost models (all four targets)
  4  test_predictions.csv
  5  shap_values.parquet  [skipped if already exists]
  6  conformal_results.csv
  7  learning_curves.csv
  8  pareto_front.csv + top_candidates.csv
  9  back_calculated_results.csv
  10 synthesizability_results.csv
  11 weight_sensitivity_results.csv
  12 topology_selectivity.csv
  13 baseline_comparison.csv
  14 topk_metrics.csv
  15 screening_funnel_counts.csv
  16 charge_data.csv  [skipped if repeat_charge_stats.parquet missing]
  17 robustness_metrics.csv (3-fold CV, conservative lower bound)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

ROOT   = Path(r"D:\Rifat\Research\MOF_Screening")
DATA   = ROOT / "data"
SCRPT  = ROOT / "scripts"
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

SEP = "=" * 65

def hdr(text):
    print(f"\n{SEP}\n{text}\n{SEP}")

# =============================================================================
# CONFIG
# =============================================================================

TARGET_COLS = {
    "co2_uptake_mmol_g": {"log": False, "clip": False, "label": "CO2_uptake"},
    "wc_mmol_g":         {"log": False, "clip": False, "label": "WC"},
    "selectivity_co2h2": {"log": True,  "clip": False, "label": "Selectivity"},
    "heat_of_ads":       {"log": False, "clip": True,  "label": "HoA"},
}

NON_FEAT  = {"mof_id", "co2_uptake_wt_pct", "co2_uptake_vol", "wc_wt_pct"}
STRAT_COL = "co2_uptake_mmol_g"
HOA_COL   = "heat_of_ads"
HOA_SIGMA = 5
SEED      = 42
N_BINS    = 10

# Optuna-optimised hyperparameters (same for all four targets)
HPARAMS = dict(
    n_estimators=900, max_depth=8, learning_rate=0.07,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=4,
    reg_alpha=0.02, reg_lambda=0.05,
    tree_method="hist", n_jobs=-1, random_state=SEED,
)

TIER1_METALS   = {"cu","zn","al","zr","fe","mg","na","k","li"}
TIER2_METALS   = {"co","ni","mn","cd","cr","v","in"}
KNOWN_FAMILIES = ["irmof","mof-177","pcn","zif","uio","hkust","mil",
                  "nott","mof-5","mof-74","mof-199"]

# =============================================================================
# HELPERS
# =============================================================================

def clip_hoa(s):
    mu, std = s.mean(), s.std()
    return s.clip(mu - HOA_SIGMA * std, mu + HOA_SIGMA * std)

def get_feat_cols(df):
    excl = set(TARGET_COLS.keys()) | NON_FEAT
    return [c for c in df.columns
            if c not in excl and pd.api.types.is_numeric_dtype(df[c])]

def make_strat(series):
    return pd.qcut(series, q=N_BINS, labels=False, duplicates="drop").to_numpy()

def prepare_y(df, col):
    s = df[col].copy()
    if col == HOA_COL:
        s = clip_hoa(s)
    if TARGET_COLS[col]["log"]:
        s = np.log1p(s)
    return s.values

def r2_mae(true, pred):
    return r2_score(true, pred), mean_absolute_error(true, pred)

def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

# =============================================================================
# STEP 1 — Load data
# =============================================================================

hdr("STEP 1 — Load full_features.parquet")

df = pd.read_parquet(DATA / "full_features.parquet")
print(f"  Shape        : {df.shape}")

before = len(df)
df = df.dropna(subset=list(TARGET_COLS.keys())).reset_index(drop=True)
print(f"  Rows kept    : {len(df):,} / {before:,}  ({before - len(df)} NaN rows removed)")

feat_cols = get_feat_cols(df)
print(f"  Feature cols : {len(feat_cols)}")

X_all = df[feat_cols].values.astype(np.float32)
strat = make_strat(df[STRAT_COL])

# =============================================================================
# STEP 2 — Train/test split
# =============================================================================

hdr("STEP 2 — Train/test split (seed=42, 90/10, stratified)")

tr_idx, te_idx = train_test_split(
    np.arange(len(df)), test_size=0.10,
    stratify=strat, random_state=SEED
)
X_tr = X_all[tr_idx]
X_te = X_all[te_idx]
print(f"  Train : {len(tr_idx):,}   Test : {len(te_idx):,}")

# =============================================================================
# STEP 3 — Train/load XGBoost models (all four targets)
# =============================================================================

hdr("STEP 3 — XGBoost models (all four targets)")

models    = {}
true_dict = {}
pred_dict = {}

for col in TARGET_COLS:
    y_all = prepare_y(df, col)
    y_tr  = y_all[tr_idx]
    y_te  = y_all[te_idx]
    true_dict[col] = y_te

    model_path = MODELS / f"xgb_{col}.json"

    if model_path.exists():
        print(f"  Loading  {col} from {model_path.name}")
        model = xgb.XGBRegressor(**HPARAMS)
        model.load_model(str(model_path))
    else:
        t0 = time.time()
        print(f"  Training {col} ...", end=" ", flush=True)
        model = xgb.XGBRegressor(**HPARAMS)
        model.fit(X_tr, y_tr, verbose=False)
        model.save_model(str(model_path))
        print(f"done ({time.time()-t0:.0f}s)  saved -> {model_path.name}")

    pred = model.predict(X_te)
    pred_dict[col] = pred
    models[col]    = model
    r2, mae = r2_mae(y_te, pred)
    print(f"    R2={r2:.4f}  MAE={mae:.4f}")

# =============================================================================
# STEP 4 — test_predictions.csv
# =============================================================================

hdr("STEP 4 — test_predictions.csv")

test_preds = pd.DataFrame({
    "true_CO2":     true_dict["co2_uptake_mmol_g"],
    "pred_CO2":     pred_dict["co2_uptake_mmol_g"],
    "true_WC":      true_dict["wc_mmol_g"],
    "pred_WC":      pred_dict["wc_mmol_g"],
    "true_sel_log": true_dict["selectivity_co2h2"],
    "pred_sel_log": pred_dict["selectivity_co2h2"],
    "true_HoA":     true_dict["heat_of_ads"],
    "pred_HoA":     pred_dict["heat_of_ads"],
})
test_preds.to_csv(DATA / "test_predictions.csv", index=False)
print(f"  Saved: data/test_predictions.csv  ({len(test_preds):,} rows)")

# =============================================================================
# STEP 5 — SHAP values (5,000 test samples)
# =============================================================================

hdr("STEP 5 — SHAP values -> shap_values.parquet")

shap_path = DATA / "shap_values.parquet"
if shap_path.exists():
    print(f"  Already exists — skipping ({shap_path.stat().st_size/1e6:.1f} MB)")
else:
    try:
        import shap

        SHAP_N   = min(5000, len(te_idx))
        rng      = np.random.default_rng(SEED)
        shap_idx = rng.choice(len(te_idx), SHAP_N, replace=False)
        X_shap   = X_te[shap_idx]

        shap_df = pd.DataFrame(X_shap, columns=feat_cols)

        for col in ["co2_uptake_mmol_g","wc_mmol_g","selectivity_co2h2",
                    "heat_of_ads","POAVAg","Density"]:
            if col in df.columns:
                shap_df[col] = df.iloc[te_idx[shap_idx]][col].values

        prefixes = {
            "co2_uptake_mmol_g": "shap_CO2",
            "wc_mmol_g":         "shap_WC",
            "selectivity_co2h2": "shap_sel",
            "heat_of_ads":       "shap_HoA",
        }

        for col, prefix in prefixes.items():
            t0        = time.time()
            explainer = shap.TreeExplainer(models[col])
            sv        = explainer.shap_values(X_shap)
            for fi, fname in enumerate(feat_cols):
                shap_df[f"{prefix}_{fname}"] = sv[:, fi]
            print(f"  {prefix:<12s} done  ({time.time()-t0:.0f}s)")

        shap_df.to_parquet(shap_path, index=False)
        print(f"  Saved: data/shap_values.parquet  ({SHAP_N} structures)")

    except ImportError:
        print("  shap not installed — run: pip install shap")
        print("  Figures 6/7/8 will use synthetic placeholders until then.")

# =============================================================================
# STEP 6 — Conformal calibration
# =============================================================================

hdr("STEP 6 — Conformal calibration -> conformal_results.csv")

rng_cal       = np.random.default_rng(SEED)
cal_size      = int(0.10 * len(tr_idx))
cal_local_idx = rng_cal.choice(len(tr_idx), cal_size, replace=False)
fit_local_idx = np.setdiff1d(np.arange(len(tr_idx)), cal_local_idx)

X_fit = X_tr[fit_local_idx]
X_cal = X_tr[cal_local_idx]

nominals  = np.round(np.linspace(0.10, 0.90, 9), 2)
conf_rows = []

for col, cfg in TARGET_COLS.items():
    label  = cfg["label"]
    y_all  = prepare_y(df, col)
    y_fit  = y_all[tr_idx[fit_local_idx]]
    y_cal  = y_all[tr_idx[cal_local_idx]]
    y_test = y_all[te_idx]

    emp_before = []
    emp_after  = []

    for alpha in nominals:
        lo_q = (1 - alpha) / 2
        hi_q = 1 - lo_q

        m_lo = xgb.XGBRegressor(**{**HPARAMS,
               "objective": "reg:quantileerror",
               "quantile_alpha": lo_q, "n_estimators": 400})
        m_hi = xgb.XGBRegressor(**{**HPARAMS,
               "objective": "reg:quantileerror",
               "quantile_alpha": hi_q, "n_estimators": 400})

        m_lo.fit(X_fit, y_fit, verbose=False)
        m_hi.fit(X_fit, y_fit, verbose=False)

        lo_te = m_lo.predict(X_te)
        hi_te = m_hi.predict(X_te)
        cov_before = float(np.mean((y_test >= lo_te) & (y_test <= hi_te)))
        emp_before.append(cov_before)

        lo_cal_pred = m_lo.predict(X_cal)
        hi_cal_pred = m_hi.predict(X_cal)
        scores      = np.maximum(lo_cal_pred - y_cal, y_cal - hi_cal_pred)
        q_conf      = float(np.quantile(scores, alpha * (1 + 1 / len(y_cal))))

        cov_after = float(np.mean(
            (y_test >= lo_te - q_conf) & (y_test <= hi_te + q_conf)))
        emp_after.append(cov_after)

    for nom, eb, ea in zip(nominals, emp_before, emp_after):
        conf_rows.append({
            "target":           label,
            "nominal":          float(nom),
            "empirical_before": round(eb, 4),
            "empirical_after":  round(ea, 4),
        })
    print(f"  {label:<15s} calibrated  "
          f"(80%: before={emp_before[7]:.3f}  after={emp_after[7]:.3f})")

conf_df = pd.DataFrame(conf_rows)
conf_df.to_csv(DATA / "conformal_results.csv", index=False)
print(f"\n  Saved: data/conformal_results.csv  ({len(conf_df)} rows)")

# =============================================================================
# STEP 7 — Learning curves
# =============================================================================

hdr("STEP 7 — Learning curves -> learning_curves.csv")

lc_sizes = [1000, 3000, 7000, 15000, 30000,
            60000, 100000, 150000, 200000, len(tr_idx)]
lc_rows  = []

for col, cfg in TARGET_COLS.items():
    y_all  = prepare_y(df, col)
    y_test = y_all[te_idx]
    label  = cfg["label"]
    print(f"  {label}")

    for sz in lc_sizes:
        sz  = min(sz, len(tr_idx))
        sub = np.random.default_rng(SEED).choice(len(tr_idx), sz, replace=False)
        m   = xgb.XGBRegressor(**{**HPARAMS, "n_estimators": 300})
        m.fit(X_tr[sub], y_all[tr_idx[sub]], verbose=False)
        r2, _ = r2_mae(y_test, m.predict(X_te))
        lc_rows.append({"target": label, "train_size": sz, "r2": round(r2, 5)})
        print(f"    n={sz:>7,}  R2={r2:.4f}")

lc_df = pd.DataFrame(lc_rows)
lc_df.to_csv(DATA / "learning_curves.csv", index=False)
print(f"\n  Saved: data/learning_curves.csv  ({len(lc_df)} rows)")

# =============================================================================
# STEP 8 — Full-database predictions -> Pareto front + Top-50
# =============================================================================

hdr("STEP 8 — Pareto front + Top-50 candidates")

print("  Predicting all four targets on full database...")
co2_all = models["co2_uptake_mmol_g"].predict(X_all)
wc_all  = models["wc_mmol_g"].predict(X_all)
sel_all = np.expm1(models["selectivity_co2h2"].predict(X_all))
hoa_all = models["heat_of_ads"].predict(X_all)

wc_thresh  = float(np.percentile(df["wc_mmol_g"].values, 75))
sel_thresh = 130.0

print(f"  WC  threshold : {wc_thresh:.2f} mmol/g  (75th percentile)")
print(f"  Sel threshold : {sel_thresh}")

# Sequential filters
mask_wc   = wc_all >= wc_thresh
mask_both = mask_wc & (sel_all >= sel_thresh)

n_wc_only = int(mask_wc.sum())
n_both    = int(mask_both.sum())
filt_idx  = np.where(mask_both)[0]

print(f"  After WC filter only  : {n_wc_only:,}")
print(f"  After both filters    : {n_both:,}")

# Pareto non-dominance within the filtered pool
def pareto_efficient(wc_vals, sel_vals):
    costs  = np.column_stack([wc_vals, sel_vals])
    n      = len(costs)
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if is_eff[i]:
            dominated = (np.all(costs >= costs[i], axis=1) &
                         np.any(costs  > costs[i], axis=1))
            is_eff[dominated] = False
    return is_eff

pf_mask_local = pareto_efficient(wc_all[filt_idx], sel_all[filt_idx])
pf_idx        = filt_idx[pf_mask_local]
print(f"  Pareto front (within filtered pool): {len(pf_idx)}")

mof_ids = (df["mof_id"].values if "mof_id" in df.columns
           else np.array([f"MOF_{i}" for i in range(len(df))]))

pf_df = pd.DataFrame({
    "mof_id":           mof_ids[pf_idx],
    "WC_pred":          wc_all[pf_idx].round(4),
    "selectivity_pred": sel_all[pf_idx].round(2),
    "WC_gcmc":          df["wc_mmol_g"].values[pf_idx].round(4),
    "selectivity_gcmc": df["selectivity_co2h2"].values[pf_idx].round(2),
    "co2_uptake_gcmc":  df["co2_uptake_mmol_g"].values[pf_idx].round(4),
    "hoa_pred":         hoa_all[pf_idx].round(3),
}).sort_values("WC_pred", ascending=False).reset_index(drop=True)

pf_df.to_csv(SCRPT / "pareto_front.csv", index=False)
print(f"  Saved: scripts/pareto_front.csv  ({len(pf_df)} non-dominated structures)")

# Top-50 by four-target scalarisation across the filtered pool
score = (minmax(wc_all[filt_idx])
       + minmax(sel_all[filt_idx])
       + minmax(co2_all[filt_idx])
       + (1 - minmax(hoa_all[filt_idx]))) / 4

top50_local = np.argsort(score)[::-1][:50]
top50_idx   = filt_idx[top50_local]

top50_df = pd.DataFrame({
    "rank":             np.arange(1, 51),
    "mof_id":           mof_ids[top50_idx],
    "WC_gcmc":          df["wc_mmol_g"].values[top50_idx].round(3),
    "WC_ml":            wc_all[top50_idx].round(3),
    "selectivity_gcmc": df["selectivity_co2h2"].values[top50_idx].round(1),
    "selectivity_ml":   sel_all[top50_idx].round(1),
    "co2_uptake_gcmc":  df["co2_uptake_mmol_g"].values[top50_idx].round(3),
    "co2_uptake_ml":    co2_all[top50_idx].round(3),
    "hoa_ml":           hoa_all[top50_idx].round(3),
    "score":            score[top50_local].round(5),
    "Density":          (df["Density"].values[top50_idx].round(4)
                         if "Density" in df.columns else np.nan),
    "POAVAg":           (df["POAVAg"].values[top50_idx].round(4)
                         if "POAVAg" in df.columns else np.nan),
})
top50_df.to_csv(SCRPT / "top_candidates.csv", index=False)
print(f"  Saved: scripts/top_candidates.csv  (50 priority candidates)")

# =============================================================================
# STEP 9 — Candidate validation
# =============================================================================

hdr("STEP 9 — Candidate validation -> back_calculated_results.csv")

back_df = pd.DataFrame({
    "mof_id":   top50_df["mof_id"],
    "gcmc_CO2": top50_df["co2_uptake_gcmc"],
    "ml_CO2":   top50_df["co2_uptake_ml"],
    "gcmc_WC":  top50_df["WC_gcmc"],
    "ml_WC":    top50_df["WC_ml"],
    "gcmc_sel": top50_df["selectivity_gcmc"],
    "ml_sel":   top50_df["selectivity_ml"],
    "gcmc_HoA": df["heat_of_ads"].values[top50_idx].round(3),
    "ml_HoA":   top50_df["hoa_ml"],
})
back_df.to_csv(SCRPT / "back_calculated_results.csv", index=False)

print(f"  {'Target':<15s}  {'R2':>6s}  {'MAE':>7s}  {'MBE':>8s}")
print(f"  {'-'*45}")
for gcol, mcol, label, unit in [
    ("gcmc_CO2", "ml_CO2",  "CO2 Uptake",  "mmol/g"),
    ("gcmc_WC",  "ml_WC",   "Work. Cap.",  "mmol/g"),
    ("gcmc_sel", "ml_sel",  "Selectivity", "raw"),
    ("gcmc_HoA", "ml_HoA",  "HoA",         "kJ/mol"),
]:
    g   = back_df[gcol].values.astype(float)
    m   = back_df[mcol].values.astype(float)
    r2, mae = r2_mae(g, m)
    mbe = float((m - g).mean())
    print(f"  {label:<15s}  {r2:>6.3f}  {mae:>7.3f}  {mbe:>+8.3f}  {unit}")

print(f"\n  Saved: scripts/back_calculated_results.csv")

# =============================================================================
# STEP 10 — Synthesizability
# =============================================================================

hdr("STEP 10 — Synthesizability -> synthesizability_results.csv")

def synth_score(mof_id):
    mid = str(mof_id).lower()
    sc  = 0
    for fam in KNOWN_FAMILIES:
        if fam in mid:
            sc += 2
            break
    metal_sc = 0
    for m in TIER1_METALS:
        if m in mid:
            metal_sc = 1
            break
    if metal_sc == 0:
        for m in TIER2_METALS:
            if m in mid:
                metal_sc = 0.5
                break
    sc += metal_sc
    if mid.startswith("db1"):
        sc += 1
    sc  = min(sc, 4)
    cat = "High" if sc >= 3 else ("Moderate" if sc >= 2 else "Low")
    return {"score": sc, "category": cat}

synth_rows = [{"rank": r["rank"], "mof_id": r["mof_id"],
               **synth_score(r["mof_id"])}
              for _, r in top50_df.iterrows()]
synth_df = pd.DataFrame(synth_rows)
synth_df.to_csv(SCRPT / "synthesizability_results.csv", index=False)
counts = synth_df["category"].value_counts()
for cat in ["High", "Moderate", "Low"]:
    print(f"  {cat:<10s}: {counts.get(cat, 0):>3d}")
print(f"  Saved: scripts/synthesizability_results.csv")

# =============================================================================
# STEP 11 — Weight sensitivity
# =============================================================================

hdr("STEP 11 — Weight sensitivity -> weight_sensitivity_results.csv")

wc_f  = wc_all[filt_idx]
sel_f = sel_all[filt_idx]
co2_f = co2_all[filt_idx]
hoa_f = hoa_all[filt_idx]

def ranked_top50(w_wc, w_sel, w_co2, w_hoa):
    s = (w_wc  * minmax(wc_f)
       + w_sel * minmax(sel_f)
       + w_co2 * minmax(co2_f)
       + w_hoa * (1 - minmax(hoa_f)))
    return set(np.argsort(s)[::-1][:50])

baseline_set = ranked_top50(0.25, 0.25, 0.25, 0.25)

named = [
    ("WC-dominant",   0.50, 0.20, 0.20, 0.10),
    ("Sel-focused",   0.15, 0.55, 0.20, 0.10),
    ("CO2-focused",   0.20, 0.20, 0.50, 0.10),
    ("HoA-focused",   0.10, 0.10, 0.10, 0.70),
    ("Balanced",      0.25, 0.25, 0.25, 0.25),
    ("WC+Sel",        0.40, 0.40, 0.10, 0.10),
    ("CO2+WC",        0.40, 0.10, 0.40, 0.10),
    ("No-HoA",        0.33, 0.33, 0.33, 0.00),
    ("HoA-penalised", 0.30, 0.30, 0.30, 0.10),
]

ws_rows = []
for name, ww, ws, wc2, wh in named:
    top_ = ranked_top50(ww, ws, wc2, wh)
    j    = len(baseline_set & top_) / len(baseline_set | top_)
    ws_rows.append({"scenario": name, "w_WC": ww, "w_sel": ws,
                    "w_CO2": wc2, "w_HoA": wh, "jaccard": round(j, 4)})

rng_ws = np.random.default_rng(SEED)
for i in range(41):
    w    = rng_ws.dirichlet([1, 1, 1, 1])
    top_ = ranked_top50(*w)
    j    = len(baseline_set & top_) / len(baseline_set | top_)
    ws_rows.append({"scenario": f"random_{i:02d}",
                    "w_WC":  round(w[0], 4), "w_sel": round(w[1], 4),
                    "w_CO2": round(w[2], 4), "w_HoA": round(w[3], 4),
                    "jaccard": round(j, 4)})

ws_df = pd.DataFrame(ws_rows)
ws_df.to_csv(SCRPT / "weight_sensitivity_results.csv", index=False)
jbar, jstd = ws_df["jaccard"].mean(), ws_df["jaccard"].std()
print(f"  Mean Jaccard : {jbar:.3f} +/- {jstd:.3f}")
print(f"  Saved: scripts/weight_sensitivity_results.csv  ({len(ws_df)} rows)")

# =============================================================================
# STEP 12 — Topology selectivity
# =============================================================================

hdr("STEP 12 — Topology selectivity -> topology_selectivity.csv")

if "mof_id" in df.columns:
    def extract_topo(mof_id):
        parts = str(mof_id).replace(".sym", "").split("_")
        for p in reversed(parts):
            t = p.split(".")[0]
            if 2 <= len(t) <= 6 and t.isalpha() and t.islower():
                return t
        return "unknown"

    df["topology"] = df["mof_id"].apply(extract_topo)
    counts = df["topology"].value_counts()
    valid  = counts[counts >= 200].index.tolist()[:20]

    topo_rows = []
    for topo in valid:
        sel_vals = df[df["topology"] == topo]["selectivity_co2h2"].dropna().values
        rng_t    = np.random.default_rng(SEED)
        boots    = [float(np.median(
                        rng_t.choice(sel_vals, len(sel_vals), replace=True)))
                    for _ in range(1000)]
        topo_rows.append({
            "topology":   topo,
            "count":      int(len(sel_vals)),
            "median_sel": round(float(np.median(sel_vals)), 2),
            "ci_low":     round(float(np.percentile(boots, 2.5)), 2),
            "ci_high":    round(float(np.percentile(boots, 97.5)), 2),
        })

    topo_df = (pd.DataFrame(topo_rows)
               .sort_values("median_sel", ascending=False)
               .reset_index(drop=True))
    print(f"\n  {'Topology':<12s} {'n':>7s}  {'Median':>8s}  {'95% CI'}")
    print(f"  {'-'*50}")
    for _, row in topo_df.head(8).iterrows():
        print(f"  {row['topology']:<12s} {int(row['count']):>7,}  "
              f"{row['median_sel']:>8.1f}  "
              f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]")
else:
    # Fallback: use confirmed manuscript values directly
    topo_df = pd.DataFrame({
        "topology":   ["fof","fsc","clean","sra","pts","bcu","opt","pcu"],
        "count":      [689, 45297, 4848, 10772, 37068, 670, 1655, 62359],
        "median_sel": [172.0, 145.5, 139.8, 110.6, 110.1, 99.7, 83.2, 73.6],
        "ci_low":     [168.9, 145.1, 137.0, 109.7, 109.7, 95.3, 77.2, 73.0],
        "ci_high":    [175.7, 145.9, 142.2, 111.3, 110.5, 103.8, 89.8, 74.1],
    })
    print("  mof_id column not found — using manuscript values directly")

topo_df.to_csv(DATA / "topology_selectivity.csv", index=False)
print(f"\n  Saved: data/topology_selectivity.csv")

# =============================================================================
# STEP 13 — Baseline comparison
# =============================================================================

hdr("STEP 13 — Baseline comparison -> baseline_comparison.csv")

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

baseline_rows = []

print("  Ridge regression...")
for col, cfg in TARGET_COLS.items():
    y_tr_  = prepare_y(df, col)[tr_idx]
    y_te_  = prepare_y(df, col)[te_idx]
    m_     = Ridge(alpha=1.0)
    m_.fit(X_tr_sc, y_tr_)
    r2, _  = r2_mae(y_te_, m_.predict(X_te_sc))
    baseline_rows.append({"model": "Ridge", "target": cfg["label"], "R2": round(r2, 4)})
    print(f"    {cfg['label']:<15s}  R2={r2:.4f}")

print("  Random Forest (200 trees)...")
for col, cfg in TARGET_COLS.items():
    y_tr_  = prepare_y(df, col)[tr_idx]
    y_te_  = prepare_y(df, col)[te_idx]
    m_     = RandomForestRegressor(n_estimators=200, max_depth=12,
                                   n_jobs=-1, random_state=SEED)
    m_.fit(X_tr, y_tr_)
    r2, _  = r2_mae(y_te_, m_.predict(X_te))
    baseline_rows.append({"model": "Random Forest", "target": cfg["label"], "R2": round(r2, 4)})
    print(f"    {cfg['label']:<15s}  R2={r2:.4f}")

print("  MLP (256-128)...")
for col, cfg in TARGET_COLS.items():
    y_tr_  = prepare_y(df, col)[tr_idx]
    y_te_  = prepare_y(df, col)[te_idx]
    m_     = MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                          max_iter=300, early_stopping=True,
                          random_state=SEED, verbose=False)
    m_.fit(X_tr_sc, y_tr_)
    r2, _  = r2_mae(y_te_, m_.predict(X_te_sc))
    baseline_rows.append({"model": "MLP", "target": cfg["label"], "R2": round(r2, 4)})
    print(f"    {cfg['label']:<15s}  R2={r2:.4f}")

# XGBoost: reuse already-computed test predictions
for col, cfg in TARGET_COLS.items():
    r2, _ = r2_mae(true_dict[col], pred_dict[col])
    baseline_rows.append({"model": "XGBoost", "target": cfg["label"], "R2": round(r2, 4)})

# CGCNN: fixed manuscript values.
# CGCNN requires graph-format inputs not available in this pipeline.
# Values were computed separately using the original Xie & Grossman architecture.
# Lower R2 reflects an information gap (no long-range geometric descriptors),
# not architectural inferiority — see manuscript Section 3.2.
for tgt, r2_val in {"CO2_uptake": 0.742, "WC": 0.751,
                    "Selectivity": 0.621, "HoA": 0.488}.items():
    baseline_rows.append({"model": "CGCNN", "target": tgt, "R2": r2_val})

baseline_df   = pd.DataFrame(baseline_rows)
baseline_wide = baseline_df.pivot(
    index="model", columns="target", values="R2").reset_index()
baseline_wide.to_csv(DATA / "baseline_comparison.csv", index=False)
print(f"\n  Saved: data/baseline_comparison.csv")

# =============================================================================
# STEP 14 — Top-k retrieval
# =============================================================================

hdr("STEP 14 — Top-k retrieval -> topk_metrics.csv")

co2_gcmc_all = df["co2_uptake_mmol_g"].values
wc_gcmc_all  = df["wc_mmol_g"].values

k_vals = (list(range(1, 51))
        + list(range(55, 201, 5))
        + list(range(210, 501, 10)))

topk_rows = []
for k in k_vals:
    gt_co2 = set(np.argsort(co2_gcmc_all)[::-1][:k])
    gt_wc  = set(np.argsort(wc_gcmc_all)[::-1][:k])
    ml_co2 = set(np.argsort(co2_all)[::-1][:k])
    ml_wc  = set(np.argsort(wc_all)[::-1][:k])
    topk_rows.append({
        "k":             k,
        "precision_CO2": round(len(gt_co2 & ml_co2) / k, 4),
        "precision_WC":  round(len(gt_wc  & ml_wc)  / k, 4),
        "recall_CO2":    round(len(gt_co2 & ml_co2) / len(gt_co2), 4),
        "recall_WC":     round(len(gt_wc  & ml_wc)  / len(gt_wc),  4),
    })

topk_df = pd.DataFrame(topk_rows)
topk_df.to_csv(DATA / "topk_metrics.csv", index=False)
print(f"  Saved: data/topk_metrics.csv  ({len(topk_df)} rows)")

# =============================================================================
# STEP 15 — Screening funnel counts
# =============================================================================

hdr("STEP 15 — Screening funnel -> screening_funnel_counts.csv")

n_full    = len(df)
n_wc_only = int(mask_wc.sum())
n_both    = int(mask_both.sum())
n_pareto  = int(len(pf_idx))
n_top50   = 50

funnel_df = pd.DataFrame([
    {"stage": "Full ARC-MOF database",           "count": n_full},
    {"stage": f"WC >= {wc_thresh:.2f} mmol/g",  "count": n_wc_only},
    {"stage": "Selectivity >= 130",              "count": n_both},
    {"stage": "Pareto front (non-dominated)",     "count": n_pareto},
    {"stage": "Priority candidates (top-50)",     "count": n_top50},
])
funnel_df.to_csv(DATA / "screening_funnel_counts.csv", index=False)

print(f"\n  {'Stage':<40s} {'Count':>8s}  {'%':>7s}")
print(f"  {'-'*60}")
for _, row in funnel_df.iterrows():
    pct = row["count"] / n_full * 100
    print(f"  {row['stage']:<40s} {int(row['count']):>8,}  {pct:>6.2f}%")
print(f"\n  Saved: data/screening_funnel_counts.csv")

# =============================================================================
# STEP 16 — Charge data
# =============================================================================

hdr("STEP 16 — Charge data -> charge_data.csv")

# repeat_charge_stats.parquet is an external pre-computed file containing
# REPEAT partial charge statistics for the 24,483 structures that have
# REPEAT charges. Generated by the REPEAT code (external tool).
# Available at the Zenodo archive [DOI].
# Expected columns: charge_mean, charge_std, charge_skew,
#                   charge_kurt, charge_min, charge_max, atom_count

charge_parquet_path = DATA / "repeat_charge_stats.parquet"

if not charge_parquet_path.exists():
    print(f"  WARNING: {charge_parquet_path.name} not found.")
    print(f"  Skipping — Figure 10 will use synthetic placeholder.")
    placeholder = pd.DataFrame({
        "charge_std": [0.4112],
        "is_real":    [0],
    })
    placeholder.to_csv(DATA / "charge_data.csv", index=False)
    print(f"  Saved placeholder: data/charge_data.csv")
else:
    chg         = pd.read_parquet(charge_parquet_path)
    imputed_val = float(chg["charge_std"].median())
    real_chg = pd.DataFrame({
        "charge_std": chg["charge_std"].values,
        "is_real":    np.ones(len(chg), dtype=int),
    })
    n_imputed   = min(10000, len(df) - len(chg))
    imputed_chg = pd.DataFrame({
        "charge_std": np.full(n_imputed, imputed_val),
        "is_real":    np.zeros(n_imputed, dtype=int),
    })
    charge_data = pd.concat([real_chg, imputed_chg], ignore_index=True)
    charge_data.to_csv(DATA / "charge_data.csv", index=False)
    print(f"  Real structures    : {len(real_chg):,}")
    print(f"  Imputed median std : {imputed_val:.4f} e")
    print(f"  Real std range     : {chg['charge_std'].min():.4f} - "
          f"{chg['charge_std'].max():.4f} e")
    print(f"  Saved: data/charge_data.csv")

# =============================================================================
# STEP 17 — 3-fold CV robustness (conservative lower bound)
# =============================================================================

hdr("STEP 17 — 3-fold CV robustness -> robustness_metrics.csv")

print("  3-fold stratified CV (n_estimators=300, conservative lower bound)")
print("  For seed-stability results (Table S2) run: robustness_metrics.py --data data/full_features.parquet")

skf_rb  = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
rb_rows = []

for col, cfg in TARGET_COLS.items():
    y_all_ = prepare_y(df, col)
    label  = cfg["label"]
    for fi, (tri_, vi_) in enumerate(skf_rb.split(X_all, strat), 1):
        m = xgb.XGBRegressor(**{**HPARAMS, "n_estimators": 300})
        m.fit(X_all[tri_], y_all_[tri_], verbose=False)
        r2, mae = r2_mae(y_all_[vi_], m.predict(X_all[vi_]))
        rb_rows.append({
            "split_type":   "3-fold CV (n_est=300)",
            "fold_or_seed": f"fold_{fi}",
            "target":       label,
            "R2":           round(r2, 6),
            "MAE":          round(mae, 6),
        })
        print(f"    {label:<15s}  fold {fi}  R2={r2:.4f}  MAE={mae:.4f}")

rb_df = pd.DataFrame(rb_rows)
rb_df.to_csv(SCRPT / "robustness_metrics.csv", index=False)
print(f"\n  Saved: scripts/robustness_metrics.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

hdr("ALL STEPS COMPLETE")

outputs = [
    (DATA  / "test_predictions.csv",          "Fig 3, 4"),
    (DATA  / "shap_values.parquet",            "Fig 6, 7, 8"),
    (DATA  / "conformal_results.csv",          "Fig 5"),
    (DATA  / "learning_curves.csv",            "Fig 9"),
    (DATA  / "topology_selectivity.csv",       "Fig 13"),
    (DATA  / "baseline_comparison.csv",        "Fig S1"),
    (DATA  / "topk_metrics.csv",               "Fig S2"),
    (DATA  / "screening_funnel_counts.csv",    "Fig 12"),
    (DATA  / "charge_data.csv",                "Fig 10"),
    (SCRPT / "pareto_front.csv",               "Fig 11"),
    (SCRPT / "top_candidates.csv",             "Table S3"),
    (SCRPT / "back_calculated_results.csv",    "Fig S3, Table 5"),
    (SCRPT / "synthesizability_results.csv",   "Section 3.9"),
    (SCRPT / "weight_sensitivity_results.csv", "Section 3.8"),
    (SCRPT / "robustness_metrics.csv",         "Table S1"),
    (MODELS / "xgb_co2_uptake_mmol_g.json",   "model"),
    (MODELS / "xgb_wc_mmol_g.json",           "model"),
    (MODELS / "xgb_selectivity_co2h2.json",   "model"),
    (MODELS / "xgb_heat_of_ads.json",         "model"),
]

print(f"\n  {'File':<42s} {'Status':>8s}  Used in")
print(f"  {'-'*72}")
for path, usage in outputs:
    status = "FOUND   " if path.exists() else "MISSING "
    size   = f"{path.stat().st_size/1e3:.0f}KB" if path.exists() else "---"
    print(f"  {status}  {str(path.name):<38s} {size:>7s}  {usage}")

print(f"\nNext step:")
print(f"  python regenerate_all_figures.py")