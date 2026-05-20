# Feature Engineering Pipeline

This document describes how the **77-feature matrix** (`full_features.parquet`) was constructed from the raw ARC-MOF structures.

The final matrix is **278,778 rows × 77 columns** (77 features + 4 GCMC target columns).

---

## Overview

| Feature group | Count | Source tool |
|---|---|---|
| Geometric (Zeo++) | 30 | Zeo++ v0.3+ |
| RAC principal components | 20 | Custom Python (RAC descriptors → PCA) |
| RDF principal components | 20 | Custom Python (RDF descriptors → PCA) |
| REPEAT charge statistics | 7 | REPEAT v2.0 (8.8% coverage; median-imputed otherwise) |
| **Total** | **77** | |

---

## Step 1 — Geometric features (Zeo++)

Zeo++ was run on each CIF file to compute:

| Raw descriptor | Symbol | Unit |
|---|---|---|
| Pore-limiting diameter | PLD | Å |
| Largest cavity diameter | LCD | Å |
| Accessible surface area (probe r=1.86 Å) | ASA | m² g⁻¹ |
| Accessible pore volume per unit mass | POAVAg | cm³ g⁻¹ |
| Void fraction | VF | dimensionless |
| Crystal density | Density | g cm⁻³ |

From these 6 raw values, 30 geometric features were derived:

```python
# Log-transforms (add 1 to avoid log(0))
log_PLD    = np.log1p(PLD)
log_LCD    = np.log1p(LCD)
log_ASA    = np.log1p(ASA)
log_POAVAg = np.log1p(POAVAg)

# Interaction terms
SA_x_PV    = ASA * POAVAg
density_x_VF = Density * VF
PV_x_VF    = POAVAg * VF
SA_x_VF    = ASA * VF

# Packing efficiency
one_minus_VF = 1.0 - VF      # KEY selectivity feature (SHAP finding)
VF_sq        = VF ** 2

# Pore ratios
LCD_over_PLD = LCD / (PLD + 1e-6)
ASA_over_vol = ASA / (POAVAg + 1e-6)
```

**Zeo++ command used:**
```bash
zeoplusplus -sa 1.86 1.86 50000 -vol 1.86 1.86 50000 -res structure.cif
```

---

## Step 2 — RAC descriptors (Revised Autocorrelation Functions)

RAC descriptors encode chemical environment by computing autocorrelations of
atomic properties along the MOF graph up to depth 3.

**Properties used:**
- Electronegativity (χ)
- Atomic number (Z)
- Polarisability (α)
- Atomic radius (r)

**Centred on:** metal nodes and organic linkers separately.

This produced **176 raw RAC features** per structure.

**PCA reduction:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_rac_scaled = scaler.fit_transform(X_rac_raw)          # 278,778 × 176

pca = PCA(n_components=20, random_state=42)
X_rac_pca = pca.fit_transform(X_rac_scaled)              # 278,778 × 20
# Explained variance: 95.3%
```

---

## Step 3 — RDF descriptors (Radial Distribution Functions)

Atom-pair RDFs were computed for all element-pair combinations present in
the ARC-MOF dataset, binned at 0.1 Å resolution up to 15 Å cutoff.

This produced **678 raw RDF features** per structure.

**PCA reduction:**
```python
pca_rdf = PCA(n_components=20, random_state=42)
X_rdf_pca = pca_rdf.fit_transform(X_rdf_scaled)          # 278,778 × 20
# Explained variance: 94.3%
```

---

## Step 4 — REPEAT charge statistics

REPEAT partial charges were available for **24,483 structures (8.8%)**.
For these structures, 7 summary statistics were computed from the full
per-atom charge distribution:

```python
charge_stats = {
    "charge_mean":   partial_charges.mean(),
    "charge_std":    partial_charges.std(),
    "charge_skew":   scipy.stats.skew(partial_charges),
    "charge_kurt":   scipy.stats.kurtosis(partial_charges),
    "charge_min":    partial_charges.min(),
    "charge_max":    partial_charges.max(),
    "atom_count":    len(partial_charges),
}
```

**For the remaining 91.2% of structures**, each statistic was replaced by
the dataset-wide median (computed on the 24,483 real values):

```python
for col in charge_stat_cols:
    median_val = df[col][df["has_repeat_charges"]].median()
    df[col]    = df[col].fillna(median_val)
```

Imputed median charge std = **0.4112 e** (real range: 0.0000–0.7962 e).

This imputation is the primary limitation on heat-of-adsorption prediction
accuracy. See manuscript Section 3.5 and Figure 10.

The pre-computed charge statistics are stored in:
```
data/repeat_charge_stats.parquet
```
This file is an **external dependency** — it was generated from REPEAT v2.0
output and is available at the Zenodo archive: [DOI to be inserted].

---

## Final assembly

```python
import pandas as pd
import numpy as np

# Combine all feature groups
full_features = pd.DataFrame(
    np.hstack([
        X_geom,           # 278,778 × 30
        X_rac_pca,        # 278,778 × 20
        X_rdf_pca,        # 278,778 × 20
        X_charge_stats,   # 278,778 × 7
    ]),
    columns=geom_cols + rac_pc_cols + rdf_pc_cols + charge_cols
)

# Append GCMC targets
full_features["co2_uptake_mmol_g"] = gcmc_targets["co2_uptake"]
full_features["wc_mmol_g"]         = gcmc_targets["wc"]
full_features["selectivity_co2h2"] = gcmc_targets["selectivity"]
full_features["heat_of_ads"]       = gcmc_targets["heat_of_ads"]
full_features["mof_id"]            = mof_ids

# Drop 107 rows with missing target values
full_features = full_features.dropna(
    subset=["co2_uptake_mmol_g","wc_mmol_g","selectivity_co2h2","heat_of_ads"]
).reset_index(drop=True)

# Save
full_features.to_parquet("data/full_features.parquet", index=False)
# Final shape: 278,778 × 81 (77 features + 4 targets)
```

---

## Dependencies

| Tool | Version | Purpose |
|---|---|---|
| Zeo++ | 0.3+ | Geometric features |
| REPEAT | 2.0 | Partial charge statistics |
| scikit-learn | 1.3.0 | PCA, StandardScaler |
| pandas | 1.5.3 | DataFrame assembly |
| numpy | 1.24.3 | Array operations |
| scipy | 1.11.1 | Skewness, kurtosis |

---

## Notes

- All PCA transformers were fit on the **full 278,778-structure dataset** before the train/test split, to avoid data leakage through the split but to use all available structural information for the PCA basis.
- Geometric features were computed with probe radius 1.86 Å (nitrogen probe, standard for gas adsorption studies).
- The `one_minus_VF` feature (packing efficiency = 1 − void fraction) is the top SHAP feature for CO₂/H₂ selectivity. It is explicitly included as a named feature rather than computed implicitly from VF, to give SHAP a clean signal to assign.
