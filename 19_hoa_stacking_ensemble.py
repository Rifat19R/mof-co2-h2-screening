"""
19_hoa_stacking_ensemble.py
=============================
Rebuilds the heat-of-adsorption (HoA) stacking ensemble described in
manuscript Section 2.6 (XGBoost + LightGBM + Random Forest + Extra Trees
base learners, 5-fold out-of-fold predictions, Ridge meta-learner), on the
corrected (100% real-charge) feature matrix.

No script implementing this ensemble existed anywhere in the repository --
only two trained artifacts survived from the original (pre-charge-fix) run:
  data/models_PRE_CHARGE_FIX_BACKUP/hoa_stacked_ensemble.pkl  (2026-05-13,
      exploratory: custom 4-feature meta-learner, Ridge alpha=0.001, R2=0.762)
  data/models_PRE_CHARGE_FIX_BACKUP/stacking_hoa.pkl          (2026-05-17,
      final: sklearn StackingRegressor, cv=5, Ridge alpha=1.0 [default])

This script reproduces the second (final, more rigorous) architecture
exactly -- hyperparameters extracted directly from the saved
StackingRegressor object via joblib.load() -- and reruns it on the
corrected full_features.parquet so the number in Section 2.6 is backed by
a real, reproducible script rather than an orphaned artifact.

Output: data/models/stacking_hoa_corrected.pkl
        19_hoa_stacking_ensemble_log.txt (via redirect)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
N_BINS = 10
HOA_SIGMA = 5

NON_FEAT = {"mof_id", "co2_uptake_wt_pct", "co2_uptake_vol", "wc_wt_pct",
            "co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"}


def hdr(t):
    print(f"\n{'=' * 65}\n{t}\n{'=' * 65}")


def clip_hoa(s):
    mu, std = s.mean(), s.std()
    return s.clip(mu - HOA_SIGMA * std, mu + HOA_SIGMA * std)


def get_feat_cols(df):
    return [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]


def make_strat(series):
    return pd.qcut(series, q=N_BINS, labels=False, duplicates="drop").to_numpy()


hdr("STEP 1 -- Load corrected full_features.parquet")
df = pd.read_parquet(DATA / "full_features.parquet")
df = df.dropna(subset=["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]).reset_index(drop=True)
feat_cols = get_feat_cols(df)
print(f"Rows: {len(df):,}  Features: {len(feat_cols)}")

X_all = df[feat_cols].values.astype(np.float32)
y_all = clip_hoa(df["heat_of_ads"]).values
strat = make_strat(df["co2_uptake_mmol_g"])

hdr("STEP 2 -- Train/test split (seed=42, 90/10, stratified)  -- identical to generate_outputs.py")
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.10, stratify=strat, random_state=SEED)
X_tr, X_te = X_all[tr_idx], X_all[te_idx]
y_tr, y_te = y_all[tr_idx], y_all[te_idx]
print(f"Train: {len(tr_idx):,}   Test: {len(te_idx):,}")

hdr("STEP 3 -- Build stacking ensemble (hyperparameters extracted from stacking_hoa.pkl)")

xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror", colsample_bytree=0.7, learning_rate=0.07,
    max_depth=8, min_child_weight=4, n_estimators=500, random_state=SEED,
    reg_alpha=0.02, reg_lambda=0.05, subsample=0.7, tree_method="hist", n_jobs=-1,
)
lgbm_base = LGBMRegressor(
    colsample_bytree=0.7, learning_rate=0.07, max_depth=8, min_child_weight=4,
    n_estimators=500, random_state=SEED, reg_alpha=0.02, reg_lambda=0.05,
    subsample=0.7, verbose=-1,
)
rf_base = RandomForestRegressor(max_depth=10, n_estimators=300, n_jobs=-1, random_state=SEED)
et_base = ExtraTreesRegressor(max_depth=10, n_estimators=300, n_jobs=-1, random_state=SEED)

stack = StackingRegressor(
    estimators=[("xgb", xgb_base), ("lgbm", lgbm_base), ("rf", rf_base), ("et", et_base)],
    final_estimator=Ridge(),  # alpha=1.0 default, matching the saved artifact exactly
    cv=5, n_jobs=1, passthrough=False,
)

print("Fitting StackingRegressor (5-fold OOF for 4 base learners + final refit)...")
t0 = time.time()
stack.fit(X_tr, y_tr)
print(f"Done in {time.time()-t0:.0f}s")

hdr("STEP 4 -- Evaluate on held-out test set")
pred = stack.predict(X_te)
r2 = r2_score(y_te, pred)
mae = mean_absolute_error(y_te, pred)
print(f"Stacking ensemble: R2 = {r2:.4f}   MAE = {mae:.4f}")

# Compare against the plain XGBoost primary model already reported in Table 1
plain_model_path = MODELS / "xgb_heat_of_ads.json"
if plain_model_path.exists():
    plain = xgb.XGBRegressor()
    plain.load_model(str(plain_model_path))
    plain_pred = plain.predict(X_te)
    plain_r2 = r2_score(y_te, plain_pred)
    plain_mae = mean_absolute_error(y_te, plain_pred)
    print(f"Plain XGBoost (primary model): R2 = {plain_r2:.4f}   MAE = {plain_mae:.4f}")
    print(f"\nStacking ensemble {'IMPROVES on' if r2 > plain_r2 else 'does NOT improve on'} plain XGBoost "
          f"(delta R2 = {r2 - plain_r2:+.4f})")

joblib.dump(stack, MODELS / "stacking_hoa_corrected.pkl")
print(f"\nSaved: {MODELS / 'stacking_hoa_corrected.pkl'}")
print("\nDONE.")
