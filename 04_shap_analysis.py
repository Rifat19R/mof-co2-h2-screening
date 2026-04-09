"""
04_shap_analysis.py
====================
SHAP TreeExplainer for all 4 XGBoost models.
Produces beeswarm plots (top 20 features) and a summary bar chart.
Output: figures/fig_shap_{target}.png, data/shap_importance.json

Requires: pip install shap --break-system-packages
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT   = Path(r"D:\Rifat\MOF_Screening")
DATA   = ROOT / "data"
FIGS   = ROOT / "figures"
MODELS = DATA / "models"

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
TARGET_LABELS = {
    "co2_uptake_mmol_g": r"CO$_2$ Uptake [mmol/g]",
    "wc_mmol_g"        : r"Working Capacity [mmol/g]",
    "selectivity_co2h2": r"CO$_2$/H$_2$ Selectivity",
    "heat_of_ads"      : r"Heat of Adsorption [kJ/mol]",
}
SKIP = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])

try:
    import shap
    SHAP_OK = True
except ImportError:
    print("shap not installed. Run: pip install shap --break-system-packages")
    SHAP_OK = False

df        = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df[c])]
split     = np.load(DATA / "train_test_idx.npz")
idx_te    = split["idx_te"]
X_te      = df[feat_cols].fillna(0).iloc[idx_te].values.astype(np.float32)

shap_importance = {}

for tgt in TARGETS:
    print(f"\nSHAP for {tgt}...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / f"xgb_{tgt}.json"))

    # Use at most 5000 test samples for speed
    n_shap = min(5000, len(X_te))
    rng    = np.random.default_rng(42)
    idx_s  = rng.choice(len(X_te), n_shap, replace=False)
    X_s    = X_te[idx_s]

    if SHAP_OK:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        importance  = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)
        shap_importance[tgt] = importance.head(20).to_dict()

        # Beeswarm-style: mean |SHAP| bar
        top20 = importance.head(20)
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, 20))
        bars = ax.barh(range(20)[::-1], top20.values, color=colors,
                       edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(20))
        ax.set_yticklabels(
            [c.replace("_PC", " PC").replace("_", " ") for c in top20.index[::-1]],
            fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"Feature Importance (SHAP)\n{TARGET_LABELS[tgt]}",
                     fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / f"fig_shap_{tgt}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Top feature: {top20.index[0]}  |SHAP|={top20.iloc[0]:.4f}")
    else:
        # Fallback: XGBoost gain importance
        fi = pd.Series(model.get_booster().get_score(importance_type="gain"))
        fi = fi.reindex(feat_cols, fill_value=0).sort_values(ascending=False)
        shap_importance[tgt] = fi.head(20).to_dict()
        print(f"  (SHAP not available — used XGBoost gain importance)")

with open(DATA / "shap_importance.json", "w") as f:
    json.dump(shap_importance, f, indent=2)
print(f"\nSaved: shap_importance.json + fig_shap_*.png")
print("Next: python 05_external_validation.py")
