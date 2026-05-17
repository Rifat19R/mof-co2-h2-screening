# Packing Efficiency Governs CO₂/H₂ Selectivity in Metal–Organic Frameworks

**Uncertainty-Guided Machine Learning Screening of 278,778 Structures Reveals Topology-Level Reticular Design Rules**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Journal](https://img.shields.io/badge/Journal-Communications%20Chemistry-green.svg)](https://www.nature.com/commschem/)

> **Md. Rifat Khandaker\*, Mohammad Asaduzzaman Chowdhury, Sujan Hossain**  
> Department of Chemical Engineering, DUET, Gazipur-1700, Bangladesh  
> \*Corresponding: rifatkh.duet@gmail.com

---

## Overview

This repository contains all code, processed data, and trained models for the large-scale ML screening of 278,778 hypothetical MOF structures from the [ARC-MOF database](https://doi.org/10.1021/acs.chemmater.1c03517) for pre-combustion CO₂/H₂ separation at **40 bar, 298 K**.

### Key results at a glance

| Target | R² (test set) | MAE |
|---|---|---|
| CO₂ uptake | 0.981 | 0.564 mmol g⁻¹ |
| Working capacity | 0.985 | 0.567 mmol g⁻¹ |
| CO₂/H₂ selectivity | 0.975 (log-space) | 0.101 |
| Heat of adsorption | 0.817 | 0.552 kJ mol⁻¹ |

- **Principal finding:** Packing efficiency (1 − void fraction) governs selectivity, not pore diameter
- **Top topologies:** *fof* (median selectivity 172.0) and *fsc* (145.5) vs *pcu* (73.6)
- **Pareto front:** 4 non-dominated structures from 803 high-performance candidates
- **Priority candidates:** 50 structures; 72% rated High synthesizability, all Tier 1 metals
- **Uncertainty quantification:** Split conformal prediction → ≥79.8% empirical coverage at 80% nominal on all four targets

---

## Repository structure

```
mof-co2-h2-screening/
│
├── data/
│   ├── full_features.parquet          # 278,778 × 77: all features + GCMC targets
│   ├── test_predictions.csv           # 27,878 test-set predictions vs GCMC
│   ├── shap_values.parquet            # SHAP values for 5,000 test structures
│   ├── conformal_results.csv          # Conformal calibration coverage results
│   ├── learning_curves.csv            # Learning curve data (4 targets)
│   ├── topology_selectivity.csv       # Topology-wise median selectivity + CI
│   ├── baseline_comparison.csv        # Baseline model R² comparison
│   ├── topk_metrics.csv               # Precision@k and Recall@k
│   ├── screening_funnel_counts.csv    # Funnel step counts
│   └── charge_data.csv                # REPEAT charge coverage and imputation stats
│
├── models/
│   ├── xgb_co2_uptake_mmol_g.json     # Trained XGBoost — CO₂ uptake
│   ├── xgb_wc_mmol_g.json             # Trained XGBoost — working capacity
│   ├── xgb_selectivity_co2h2.json     # Trained XGBoost — selectivity (log-space)
│   └── xgb_heat_of_ads.json           # Stacking ensemble base model — HoA
│
├── results/
│   ├── pareto_front.csv               # 4 Pareto-optimal MOF structures
│   ├── top_candidates.csv             # Top-50 priority candidates (all metrics)
│   ├── back_calculated_results.csv    # ML vs GCMC validation (top-50)
│   ├── synthesizability_results.csv   # SA scores + metal tier classification
│   ├── weight_sensitivity_results.csv # Jaccard similarity across 50 weight sets
│   └── robustness_metrics.csv         # Seed stability (seeds 42, 0, 123)
│
├── scripts/
│   ├── 01_feature_engineering.py      # Geometric + RAC + RDF + REPEAT features
│   ├── 02_data_split.py               # Stratified 90/10 split, seed 42
│   ├── 03_train_xgboost.py            # Optuna HPO + XGBoost training (4 targets)
│   ├── 04_stacking_ensemble_hoa.py    # Stacking ensemble for heat of adsorption
│   ├── 05_conformal_prediction.py     # Split conformal calibration + coverage
│   ├── 06_shap_analysis.py            # SHAP TreeExplainer + dependence plots
│   ├── 07_baseline_models.py          # Ridge, RF, MLP, CGCNN baselines
│   ├── 08_topology_analysis.py        # Topology-wise selectivity + bootstrap CI
│   ├── 09_screening_funnel.py         # WC + selectivity filters + Pareto front
│   ├── 10_candidate_validation.py     # GCMC rerun + ML vs GCMC comparison
│   ├── 11_synthesizability.py         # SA score + metal tier classification
│   ├── 12_weight_sensitivity.py       # Jaccard stability across weight sets
│   ├── 13_cross_validation.py         # 3-fold CV + seed stability analysis
│   └── 14_figures.py                  # All manuscript figures (main + SI)
│
├── figures/
│   ├── main/                          # Figures 01–13 (PNG + PDF)
│   └── supplementary/                 # Figures S1–S4 (PNG + PDF)
│
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip-installable dependencies
├── LICENSE                            # MIT licence (code)
├── LICENSE-DATA                       # CC BY 4.0 (data)
└── README.md                          # This file
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Rifat19R/mof-co2-h2-screening.git
cd mof-co2-h2-screening
```

### 2. Set up the environment

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate mof-screening
```

Using pip:
```bash
pip install -r requirements.txt
```

### 3. Reproduce the results

Run scripts in numbered order:

```bash
# Feature engineering
python scripts/01_feature_engineering.py

# Train models
python scripts/03_train_xgboost.py
python scripts/04_stacking_ensemble_hoa.py

# Conformal calibration
python scripts/05_conformal_prediction.py

# Screening funnel → top-50 candidates
python scripts/09_screening_funnel.py

# Generate all figures
python scripts/14_figures.py
```

Or reproduce the full pipeline end-to-end:
```bash
bash run_pipeline.sh
```

### 4. Load pre-trained models and make predictions

```python
import xgboost as xgb
import pandas as pd

# Load a trained model
model = xgb.XGBRegressor()
model.load_model('models/xgb_co2_uptake_mmol_g.json')

# Load features for new structures
features = pd.read_parquet('data/full_features.parquet').drop(
    columns=['co2_uptake_mmol_g', 'wc_mmol_g', 'selectivity_co2h2', 'heat_of_ads']
)

# Predict CO₂ uptake for all 278,778 structures
predictions = model.predict(features)
```

---

## Data

| File | Description | Size |
|---|---|---|
| `data/full_features.parquet` | 278,778 × 77 feature matrix + 4 GCMC targets | ~180 MB |
| `data/test_predictions.csv` | 27,878 test-set ML predictions vs GCMC | ~8 MB |
| `data/shap_values.parquet` | SHAP values for 5,000 test structures | ~12 MB |
| `results/top_candidates.csv` | Top-50 priority candidates, all metrics | <1 MB |

**ARC-MOF source database:** Raza et al., *Chem. Mater.* 34, 2864–2884 (2022).  
The raw CIF structures and original GCMC data are available from the ARC-MOF authors.  
This repository contains derived features and model outputs only.

---

## Feature description

The 77-feature input matrix comprises:

| Feature group | Count | Source |
|---|---|---|
| Geometric (Zeo++) | 30 | PLD, LCD, ASA, POAVAg, VF + log-transforms + interaction terms |
| RAC principal components | 20 | 95.3% variance from 176 revised autocorrelation descriptors |
| RDF principal components | 20 | 94.3% variance from 678 radial distribution function features |
| REPEAT charge statistics | 7 | mean, std, skew, kurt, min, max, atom count |

**REPEAT charge coverage:** 24,483 / 278,778 structures (8.8%). Median imputation applied to the remaining 91.2% (imputed median std = 0.4112 e; real range 0.0000–0.7962 e).

---

## Models

| Model | Target | Architecture | R² (test) |
|---|---|---|---|
| `xgb_co2_uptake_mmol_g.json` | CO₂ uptake | XGBoost (n_est=900, depth=8) | 0.981 |
| `xgb_wc_mmol_g.json` | Working capacity | XGBoost (n_est=900, depth=8) | 0.985 |
| `xgb_selectivity_co2h2.json` | Selectivity (log-space) | XGBoost (n_est=900, depth=8) | 0.975 |
| `xgb_heat_of_ads.json` | Heat of adsorption | Stacking ensemble (XGB+LGBM+RF+ET+Ridge) | 0.817 |

Hyperparameters optimised by Optuna Bayesian search (200 trials, 5-fold CV).

---

## Citation

If you use this code, data, or models, please cite:

```bibtex
@article{khandaker2025mof,
  title   = {Packing Efficiency Governs CO$_2$/H$_2$ Selectivity in Metal--Organic
             Frameworks: Uncertainty-Guided Machine Learning Screening of
             278,778 Structures Reveals Topology-Level Reticular Design Rules},
  author  = {Khandaker, Md. Rifat and Chowdhury, Mohammad Asaduzzaman and Hossain, Sujan},
  journal = {Communications Chemistry},
  year    = {2025},
  note    = {Under review},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## Dependencies

Core:
- Python ≥ 3.9
- xgboost ≥ 1.7
- lightgbm ≥ 3.3
- scikit-learn ≥ 1.2
- pandas ≥ 1.5
- numpy ≥ 1.23
- shap ≥ 0.41
- optuna ≥ 3.0
- pymatgen ≥ 2023.1
- rdkit ≥ 2022.09

Visualisation:
- matplotlib ≥ 3.6
- seaborn ≥ 0.12

See `environment.yml` for the full pinned environment used to produce manuscript figures.

---

## Licence

**Code:** MIT — see [LICENSE](LICENSE)  
**Data and figures:** CC BY 4.0 — see [LICENSE-DATA](LICENSE-DATA)

---

## Contact

Md. Rifat Khandaker — rifatkh.duet@gmail.com  
Google Scholar: [scholar.google.com/citations?user=dp3Vs-QAAAAJ](https://scholar.google.com/citations?user=dp3Vs-QAAAAJ)  
LinkedIn: [linkedin.com/in/quantum-boy](https://linkedin.com/in/quantum-boy/)
