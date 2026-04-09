"""
02_train_models.py
===================
Trains XGBoost regressors (Optuna-tuned) for all 4 targets.
Also trains quantile models for uncertainty estimation.

Output:
  data/models/xgb_{target}.json        — point estimate models
  data/models/xgb_{target}_q10.json    — 10th percentile quantile model
  data/models/xgb_{target}_q90.json    — 90th percentile quantile model
  data/metrics.json                     — R², MAE, RMSE for all targets
  data/train_test_idx.npz               — fixed split indices
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT      = Path(r"D:\Rifat\MOF_Screening")
DATA      = ROOT / "data"
MODELS    = DATA / "models"
MODELS.mkdir(parents=True, exist_ok=True)

ID        = "mof_id"
TARGETS   = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
SKIP_COLS = set(TARGETS + [ID, "co2_uptake_wt_pct", "co2_uptake_vol", "wc_wt_pct"])

TEST_SIZE    = 0.10
RANDOM_STATE = 42
N_TRIALS     = 40   # Optuna trials — increase to 60 for final run

SEP = "=" * 60


def get_features(df):
    return [c for c in df.columns
            if c not in SKIP_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def optuna_xgb(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        p = {
            "n_estimators"    : trial.suggest_int("n_estimators", 400, 1000),
            "max_depth"       : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"   : trial.suggest_float("lr", 5e-3, 0.15, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha"       : trial.suggest_float("alpha", 1e-6, 5.0, log=True),
            "reg_lambda"      : trial.suggest_float("lambda", 1e-6, 5.0, log=True),
            "tree_method"     : "hist",
            "random_state"    : RANDOM_STATE,
            "n_jobs"          : 1,
            "verbosity"       : 0,
            "early_stopping_rounds": 30,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params


def train_final(X_tr, y_tr, params: dict) -> xgb.XGBRegressor:
    """Refit on full training set without early stopping."""
    p = {k: v for k, v in params.items()
         if k not in ("early_stopping_rounds",)}
    p.update({"tree_method": "hist", "random_state": RANDOM_STATE,
              "n_jobs": 1, "verbosity": 0})
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr)
    return m


def train_quantile(X_tr, y_tr, alpha: float) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=RANDOM_STATE,
        n_jobs=1, verbosity=0)
    m.fit(X_tr, y_tr)
    return m


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nLoading full_features.parquet\n{SEP}")

df       = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = get_features(df)
X        = df[feat_cols].fillna(0).values.astype(np.float32)
print(f"  MOFs     : {len(df):,}")
print(f"  Features : {len(feat_cols)}")

# Fixed train/test split — saved so all scripts use the same split
idx      = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
np.savez(DATA / "train_test_idx.npz", idx_tr=idx_tr, idx_te=idx_te)
print(f"  Train: {len(idx_tr):,}  Test: {len(idx_te):,}")


# ── Train per target ──────────────────────────────────────────────────────────
all_metrics  = {}
best_params  = {}

for tgt in TARGETS:
    print(f"\n{SEP}\nTarget: {tgt}\n{SEP}")

    y     = df[tgt].values.astype(np.float32)
    valid = np.isfinite(y)
    if not valid.all():
        print(f"  Dropping {(~valid).sum():,} NaN/Inf rows for this target")
    y_clean   = y[valid]
    X_clean   = X[valid]
    idx_clean = idx[valid]

    # Align train/test with valid rows
    tr_mask = np.isin(idx_clean, idx_tr)
    te_mask = ~tr_mask
    X_tr_full = X_clean[tr_mask];  y_tr_full = y_clean[tr_mask]
    X_te      = X_clean[te_mask];  y_te      = y_clean[te_mask]

    # Val split from training for Optuna
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.10, random_state=RANDOM_STATE)

    # ── Optuna search ──────────────────────────────────────────────────────────
    print(f"  Optuna search ({N_TRIALS} trials)...")
    params = optuna_xgb(X_tr, y_tr, X_val, y_val)
    best_params[tgt] = params
    print(f"  Best params: n_est={params.get('n_estimators')} "
          f"depth={params.get('max_depth')} "
          f"lr={params.get('lr', params.get('learning_rate', '?')):.4f}")

    # ── Final model (full train set) ───────────────────────────────────────────
    model = train_final(X_tr_full, y_tr_full, params)
    model.save_model(str(MODELS / f"xgb_{tgt}.json"))
    pred  = model.predict(X_te)

    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    print(f"  Test R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    all_metrics[tgt] = {"r2": r2, "mae": mae, "rmse": rmse,
                         "n_train": int(len(y_tr_full)),
                         "n_test":  int(len(y_te))}

    # ── Quantile models for conformal prediction ───────────────────────────────
    print(f"  Training quantile models (q10, q90)...")
    for q, tag in [(0.10, "q10"), (0.90, "q90")]:
        qm = train_quantile(X_tr_full, y_tr_full, q)
        qm.save_model(str(MODELS / f"xgb_{tgt}_{tag}.json"))

# ── Save metrics ──────────────────────────────────────────────────────────────
print(f"\n{SEP}\nSummary\n{SEP}")
for tgt, m in all_metrics.items():
    print(f"  {tgt:<30} R²={m['r2']:.4f}  MAE={m['mae']:.4f}")

out = {"metrics": all_metrics, "best_params": best_params,
       "feat_cols": feat_cols, "n_features": len(feat_cols)}
with open(DATA / "metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: {DATA / 'metrics.json'}")
print("Next: python 03_uncertainty.py")
