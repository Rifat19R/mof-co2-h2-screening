"""
01_build_features.py
=====================
Merges geometric features (from arcmof_co2_features.parquet) with:
  - RAC descriptors  (RACs.csv)       → PCA-reduced to 20 components
  - RDF descriptors  (RDFs.csv)       → PCA-reduced to 20 components
  - REPEAT charges   (repeat_charge_stats.parquet) → 7 stats columns

ID normalisation (all → base mof_id format, e.g. DB0-m2_o1_o10_f0_pcu.sym.66):
  RACs     : filename        e.g. DB1-Cu2O8.cif        → strip .cif
  RDFs     : Structure_Name  e.g. DB0-m29_repeat.cif   → strip .cif + _repeat
  Charges  : mof_id          e.g. DB0-m12_o10_repeat   → strip _repeat
  Base     : mof_id          e.g. DB0-m2_o1_pcu.sym.66 → already clean

Output: data/full_features.parquet
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT   = Path(r"D:\Rifat\MOF_Screening")
DB     = Path(r"D:\Rifat\Semiconductor_Twin\database")
DATA   = ROOT / "data"
ID     = "mof_id"
PCA_N  = 20

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
RAC_PREFIXES = ("f-", "mc-", "D_mc-", "lc-", "D_lc-", "func-", "D_func-")

SEP = "=" * 60


def clean_id(s: pd.Series, strip_cif=True, strip_repeat=True) -> pd.Series:
    if strip_cif:
        s = s.str.replace(r"\.cif$", "", regex=True)
    if strip_repeat:
        s = s.str.replace(r"_repeat$", "", regex=True)
    return s


def pca_block(wide: pd.DataFrame, prefix: str, n: int) -> pd.DataFrame:
    n_comp = min(n, wide.shape[1], wide.shape[0] - 1)
    X      = StandardScaler().fit_transform(wide.fillna(0).values)
    pcs    = PCA(n_components=n_comp, random_state=42).fit_transform(X)
    cols   = [f"{prefix}{i+1}" for i in range(n_comp)]
    out    = pd.DataFrame(pcs, index=wide.index, columns=cols)
    var    = PCA(n_components=n_comp, random_state=42).fit(X).explained_variance_ratio_.cumsum()[-1]
    print(f"    PCA {n_comp} components → {var:.1%} variance explained")
    return out


def overlap(base: pd.Series, other: pd.Series, label: str) -> float:
    n   = len(set(base) & set(other))
    pct = n / len(base) * 100
    print(f"  {label} overlap: {n:,} / {len(base):,} ({pct:.1f}%)")
    if pct < 50:
        print(f"    ⚠ LOW OVERLAP — sample base IDs : {base.iloc[:2].tolist()}")
        print(f"    ⚠ LOW OVERLAP — sample other IDs: {other.iloc[:2].tolist()}")
    return pct


# ── Step 1: Base ──────────────────────────────────────────────────────────────
print(f"\n{SEP}\nStep 1 — Load base (geometry + targets)\n{SEP}")

df = pd.read_parquet(DATA / "arcmof_co2_features.parquet")
print(f"  Shape   : {df.shape}")
print(f"  ID      : '{ID}' = {df[ID].iloc[0]}")
print(f"  Targets : {[t for t in TARGETS if t in df.columns]}")

# Drop rows with all targets missing
before = len(df)
df = df[df[TARGETS].notna().any(axis=1)].reset_index(drop=True)
print(f"  Kept {len(df):,} / {before:,} rows with at least one target")

# Geometric feature columns (numeric, not targets, not ID)
GEOM_COLS = [c for c in df.columns
             if c not in TARGETS + [ID, "co2_uptake_wt_pct",
             "co2_uptake_vol", "wc_wt_pct"]
             and pd.api.types.is_numeric_dtype(df[c])]
print(f"  Geometric features: {len(GEOM_COLS)}")


# ── Step 2: RACs ──────────────────────────────────────────────────────────────
print(f"\n{SEP}\nStep 2 — RAC descriptors (472k rows)\n{SEP}")

racs              = pd.read_csv(DB / "RACs.csv")
racs["_id"]       = clean_id(racs["filename"], strip_cif=True, strip_repeat=False)
rac_feat          = [c for c in racs.columns if c.startswith(RAC_PREFIXES)]
print(f"  RAC feature columns: {len(rac_feat)}")
overlap(df[ID], racs["_id"], "RAC")

racs_wide = racs.set_index("_id")[rac_feat]
if racs_wide.index.duplicated().any():
    racs_wide = racs_wide.groupby(level=0).mean()
    print(f"  Deduplicated RAC index")

racs_pca = pca_block(racs_wide, "RAC_PC", PCA_N)
racs_df  = racs_pca.reset_index().rename(columns={"_id": ID})
del racs, racs_wide
print(f"  RAC output: {racs_df.shape}")


# ── Step 3: RDFs ──────────────────────────────────────────────────────────────
print(f"\n{SEP}\nStep 3 — RDF descriptors (280k rows × 680 cols)\n{SEP}")

rdf         = pd.read_csv(DB / "RDFs.csv")
rdf["_id"]  = clean_id(rdf["Structure_Name"], strip_cif=True, strip_repeat=True)
rdf_feat    = [c for c in rdf.columns if c.startswith("RDF_")]
print(f"  RDF feature columns: {len(rdf_feat)}")
overlap(df[ID], rdf["_id"], "RDF")

rdf_wide = rdf.set_index("_id")[rdf_feat]
if rdf_wide.index.duplicated().any():
    rdf_wide = rdf_wide.groupby(level=0).mean()
    print(f"  Deduplicated RDF index")

rdf_pca = pca_block(rdf_wide, "RDF_PC", PCA_N)
rdf_df  = rdf_pca.reset_index().rename(columns={"_id": ID})
del rdf, rdf_wide
print(f"  RDF output: {rdf_df.shape}")


# ── Step 4: Charges ───────────────────────────────────────────────────────────
print(f"\n{SEP}\nStep 4 — REPEAT charge statistics\n{SEP}")

chg        = pd.read_parquet(DATA / "repeat_charge_stats.parquet")
chg[ID]    = clean_id(chg[ID], strip_cif=False, strip_repeat=True)
chg_cols   = [c for c in chg.columns if c.startswith("charge_")]
print(f"  Charge feature columns: {len(chg_cols)}")
overlap(df[ID], chg[ID], "Charges")


# ── Step 5: Merge ─────────────────────────────────────────────────────────────
print(f"\n{SEP}\nStep 5 — Merge all blocks\n{SEP}")

df = df.merge(racs_df,          on=ID, how="left")
df = df.merge(rdf_df,           on=ID, how="left")
df = df.merge(chg[[ID]+chg_cols], on=ID, how="left")

chem_cols = [c for c in df.columns
             if c.startswith(("RAC_", "RDF_", "charge_"))]

# Coverage report
for label, prefix in [("RAC", "RAC_"), ("RDF", "RDF_"), ("Charge", "charge_")]:
    cols = [c for c in chem_cols if c.startswith(prefix)]
    if cols:
        n = df[cols[0]].notna().sum()
        print(f"  {label} coverage: {n:,} / {len(df):,} ({n/len(df):.1%})")

# Fill missing chemical features with column median
for col in chem_cols:
    df[col] = df[col].fillna(df[col].median())

nan_left = df.isnull().sum().sum()
print(f"  Remaining NaN: {nan_left}")


# ── Step 6: Save ──────────────────────────────────────────────────────────────
print(f"\n{SEP}\nFinal feature matrix\n{SEP}")

all_feat = GEOM_COLS + [c for c in chem_cols]
print(f"  Total MOFs         : {len(df):,}")
print(f"  Geometric features : {len(GEOM_COLS)}")
print(f"  RAC PCs            : {sum(c.startswith('RAC_') for c in df.columns)}")
print(f"  RDF PCs            : {sum(c.startswith('RDF_') for c in df.columns)}")
print(f"  Charge features    : {sum(c.startswith('charge_') for c in df.columns)}")
print(f"  Total feature cols : {len(all_feat)}")

out = DATA / "full_features.parquet"
df.to_parquet(out, index=False)
print(f"\n  Saved : {out}")
print(f"  Size  : {out.stat().st_size / 1e6:.1f} MB")
print("\nNext: python 02_train_models.py")
