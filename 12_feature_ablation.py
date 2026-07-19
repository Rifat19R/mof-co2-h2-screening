"""
12_feature_ablation.py
=======================
Priority 1.1 (Digital Discovery resubmission, Referee 3 checklist item 3b):
feature-set ablation comparison. Same XGBoost config and same 90/10 stratified
split (seed 42) as scripts/generate_outputs.py, evaluated on four nested
feature subsets:
  1. Geometric only (30 features)
  2. Geometric + RAC PCs (50 features)
  3. Geometric + RAC + RDF PCs (70 features)
  4. Full 77-dimensional set (+ REPEAT charge stats)

Column families (see FEATURE_ENGINEERING.md / full_features.parquet columns):
  Geometric (30): UC_volume ... Dens_x_VF   (columns 10-39 of the parquet)
  RAC PCs   (20): RAC_PC1 ... RAC_PC20
  RDF PCs   (20): RDF_PC1 ... RDF_PC20
  Charge     (7): charge_mean ... charge_n

Output: scripts/feature_ablation_results.csv (4 subsets x 4 targets, R2 + MAE)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Rifat\Research\MOF_Screening")
DATA = ROOT / "data"
OUT  = ROOT / "scripts" / "feature_ablation_results.csv"

SEED = 42
N_BINS = 10

TARGET_COLS = {
    "co2_uptake_mmol_g": {"log": False, "clip": False, "label": "CO2_uptake"},
    "wc_mmol_g":         {"log": False, "clip": False, "label": "WC"},
    "selectivity_co2h2": {"log": True,  "clip": False, "label": "Selectivity"},
    "heat_of_ads":       {"log": False, "clip": True,  "label": "HoA"},
}
HOA_SIGMA = 5

HPARAMS = dict(
    n_estimators=900, max_depth=8, learning_rate=0.07,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=4,
    reg_alpha=0.02, reg_lambda=0.05,
    tree_method="hist", n_jobs=-1, random_state=SEED,
)

GEOM_COLS = [
    "UC_volume","Density","ASA","vASA","gASA","AVA","AVAf","AVAg","POAVA","POAVAf",
    "POAVAg","Di","Df","Dif","Di_Df_ratio","Dif_Di_ratio","sa_per_density","VF_sq",
    "one_minus_VF","packing_eff","log_gASA","log_UC_volume","log_AVAg","log_POAVAg",
    "log_Di","log_Df","SA_x_VF","PV_x_VF","SA_x_PV","Dens_x_VF",
]
RAC_COLS    = [f"RAC_PC{i}" for i in range(1, 21)]
RDF_COLS    = [f"RDF_PC{i}" for i in range(1, 21)]
CHARGE_COLS = ["charge_mean","charge_std","charge_skew","charge_kurt","charge_min","charge_max","charge_n"]

SUBSETS = {
    "1_geometric_only":            GEOM_COLS,
    "2_geom_plus_RAC":             GEOM_COLS + RAC_COLS,
    "3_geom_plus_RAC_plus_RDF":    GEOM_COLS + RAC_COLS + RDF_COLS,
    "4_full_77dim":                GEOM_COLS + RAC_COLS + RDF_COLS + CHARGE_COLS,
}

def hdr(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")

def clip_hoa(s):
    mu, std = s.mean(), s.std()
    return s.clip(mu - HOA_SIGMA * std, mu + HOA_SIGMA * std)

def prepare_y(df, col):
    s = df[col].copy()
    if col == "heat_of_ads":
        s = clip_hoa(s)
    if TARGET_COLS[col]["log"]:
        s = np.log1p(s)
    return s.values

def make_strat(series):
    return pd.qcut(series, q=N_BINS, labels=False, duplicates="drop").to_numpy()

hdr("Load full_features.parquet")
df = pd.read_parquet(DATA / "full_features.parquet")
before = len(df)
df = df.dropna(subset=list(TARGET_COLS.keys())).reset_index(drop=True)
print(f"Rows kept: {len(df):,} / {before:,}")

for name, cols in SUBSETS.items():
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{name}: missing columns {missing}"
assert set(SUBSETS["4_full_77dim"]) - set(GEOM_COLS+RAC_COLS+RDF_COLS+CHARGE_COLS) == set()
assert len(SUBSETS["4_full_77dim"]) == 77, len(SUBSETS["4_full_77dim"])

strat = make_strat(df["co2_uptake_mmol_g"])
tr_idx, te_idx = train_test_split(
    np.arange(len(df)), test_size=0.10, stratify=strat, random_state=SEED
)
print(f"Train: {len(tr_idx):,}   Test: {len(te_idx):,}")

rows = []
t_start = time.time()
for subset_name, cols in SUBSETS.items():
    hdr(f"Subset: {subset_name}  ({len(cols)} features)")
    X_all = df[cols].values.astype(np.float32)
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]

    for col in TARGET_COLS:
        y_all = prepare_y(df, col)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        t0 = time.time()
        model = xgb.XGBRegressor(**HPARAMS)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_te)
        r2, mae = r2_score(y_te, pred), mean_absolute_error(y_te, pred)
        dt = time.time() - t0
        print(f"  {TARGET_COLS[col]['label']:12s} R2={r2:.4f}  MAE={mae:.4f}  ({dt:.0f}s)")
        rows.append({
            "feature_subset": subset_name, "n_features": len(cols),
            "target": TARGET_COLS[col]["label"], "R2": r2, "MAE": mae,
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print(f"Saved: {OUT}")
print(out_df.pivot(index="feature_subset", columns="target", values="R2"))
