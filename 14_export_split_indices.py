"""
14_export_split_indices.py
===========================
Materialises the fixed train/test/calibration split as files, reproducing
generate_outputs.py's Step 2 (train/test) and Step 6 (calibration) index
logic exactly (same seed=42), so these can be archived on Zenodo as promised
in the README / manuscript Data Availability section (they were previously
computed at runtime and never saved to disk).

Output: data/split_indices.npz (integer positional indices into the
dropna-cleaned, reset_index(drop=True) full_features.parquet) and
data/split_indices_mof_ids.csv (mof_id + split label, for human/portable use).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\Rifat\Research\MOF_Screening")
DATA = ROOT / "data"
SEED = 42
N_BINS = 10
TARGET_COLS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]

df = pd.read_parquet(DATA / "full_features.parquet")
df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
print(f"Rows: {len(df):,}")

strat = pd.qcut(df["co2_uptake_mmol_g"], q=N_BINS, labels=False, duplicates="drop").to_numpy()
tr_idx, te_idx = train_test_split(
    np.arange(len(df)), test_size=0.10, stratify=strat, random_state=SEED
)
print(f"Train: {len(tr_idx):,}  Test: {len(te_idx):,}")

rng_cal = np.random.default_rng(SEED)
cal_size = int(0.10 * len(tr_idx))
cal_local_idx = rng_cal.choice(len(tr_idx), cal_size, replace=False)
fit_local_idx = np.setdiff1d(np.arange(len(tr_idx)), cal_local_idx)
cal_idx = tr_idx[cal_local_idx]
fit_idx = tr_idx[fit_local_idx]
print(f"Fit (train minus calibration): {len(fit_idx):,}  Calibration: {len(cal_idx):,}")

assert len(set(tr_idx) & set(te_idx)) == 0
assert len(set(fit_idx) & set(cal_idx)) == 0
assert len(set(fit_idx) | set(cal_idx)) == len(tr_idx)

np.savez(DATA / "split_indices.npz", train_idx=tr_idx, test_idx=te_idx,
         calibration_idx=cal_idx, fit_idx=fit_idx, seed=SEED)
print(f"Saved: {DATA / 'split_indices.npz'}")

split_label = np.full(len(df), "", dtype=object)
split_label[te_idx] = "test"
split_label[fit_idx] = "train_fit"
split_label[cal_idx] = "train_calibration"
out = pd.DataFrame({"mof_id": df["mof_id"], "split": split_label})
out.to_csv(DATA / "split_indices_mof_ids.csv", index=False)
print(f"Saved: {DATA / 'split_indices_mof_ids.csv'}")
print(out["split"].value_counts())
