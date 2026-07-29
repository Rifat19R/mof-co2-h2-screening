# Packing Efficiency Governs CO₂/H₂ Selectivity in Machine-Learning-Screened MOFs

This repository contains the analysis code and reproducibility archive links for the manuscript:

Packing efficiency governs CO₂/H₂ selectivity in machine-learning-screened metal–organic frameworks

(Previously titled "Packing Efficiency Governs CO₂/H₂ Selectivity in Metal–Organic Frameworks: Uncertainty-Guided Machine Learning Screening of 278,778 Structures Reveals Topology-Level Reticular Design Rules" in an earlier resubmission round.)

Authors: Md. Rifat Khandaker (Department of Chemical Engineering), Mohammad Asaduzzaman Chowdhury (Department of Mechanical Engineering), Sujan Hossain (Department of Mechanical Engineering)

Affiliation: Dhaka University of Engineering & Technology (DUET), Gazipur-1700, Bangladesh

Corresponding author: rifatkh.duet@gmail.com

Target journal: *Digital Discovery*, Royal Society of Chemistry

Data archive: https://doi.org/10.5281/zenodo.20305724

---

What this project does

This project uses an uncertainty-guided machine-learning workflow to screen 278,778 ARC-MOF structures for pre-combustion CO₂/H₂ separation at 298 K and 40 bar. The goal is not only to rank materials. The goal is to extract design rules that are useful for reticular MOF synthesis and experimental follow-up.

The workflow predicts four adsorption targets simultaneously:

- CO₂ uptake
- CO₂ working capacity
- CO₂/H₂ selectivity
- Heat of adsorption

The final model stack combines descriptor-based learning, split conformal uncertainty quantification, SHAP interpretation, topology-level analysis, Pareto screening, synthesizability scoring, and weight-sensitivity testing. Every screened candidate carries a calibrated uncertainty interval alongside its point prediction, so candidate selection can account for both performance and prediction risk.

---

Main result

The central finding is straightforward:

> CO₂/H₂ selectivity is governed more strongly by packing efficiency than by pore diameter.

Working capacity follows a different rule. It peaks near POAVAg ≈ 1.5 cm³ g⁻¹ and weakens beyond roughly 2 cm³ g⁻¹. This gives a practical synthesis design window:

> POAVAg = 1–2 cm³ g⁻¹ with crystal density below 0.5 g cm⁻³.

At topology level, fof and fsc nets show consistently high median CO₂/H₂ selectivities of 172.0 and 145.5, respectively, compared with a database-wide median of 89. This is, to our knowledge, the first topology–selectivity map reported at full ARC-MOF database resolution.

---

Dataset and descriptors

The workflow starts from the ARC-MOF database (Burner et al., *Chemistry of Materials*, 2023), which reports 279,384 structures. Of these, 499 failed geometric/RAC/RDF feature computation and were dropped before assembly, and a further 107 had missing or non-finite target values and were dropped after assembly, for 606 discarded in total, leaving 278,778 retained structures.

The final descriptor matrix contains 77 features:

| Descriptor family | Dimensions | Description |
|---|---|---|
| Geometric descriptors | 30 | Pore volume, surface area, void fraction, density, sphere diameters, interaction terms, log-transforms |
| RAC principal components | 20 | PCA on 176 raw revised autocorrelation features (95.3% variance retained) |
| RDF principal components | 20 | PCA on 678 raw radial distribution function features (94.3% variance retained) |
| REPEAT charge statistics | 7 | Mean, standard deviation, skewness, kurtosis, min, max, atom count |

**Correction (2026-07-29):** an earlier version of this pipeline recovered real REPEAT charges for only 24,481 structures (8.8%) and silently imputed a database-wide constant for the remaining 91.2%, due to a bug in the charge-extraction step, not a genuine data limitation. Real, per-atom charges are now extracted directly from the official ARC-MOF CIF archive ([Zenodo record 6908728](https://zenodo.org/records/6908728)) for all 278,885 structures (100% coverage) by `18_extract_real_repeat_charges.py`. Restoring full coverage left heat-of-adsorption accuracy essentially unchanged (see below), which rules out charge-data availability as the limiting factor -- see manuscript Section 2.2, Section 3.5, and Supplementary Table S7/Figure S7 for the full disclosure.

---

Model performance

Held-out test set: 27,878 structures (10% of the database, stratified by CO₂ uptake decile, never used during training, hyperparameter search, or conformal calibration).

| Target | Model | R² | MAE |
|---|---|---|---|
| CO₂ uptake | XGBoost | 0.982 | 0.555 mmol g⁻¹ |
| Working capacity | XGBoost | 0.986 | 0.558 mmol g⁻¹ |
| CO₂/H₂ selectivity | XGBoost, log(1+x) target | 0.975 (log-space) | 0.099 log units |
| Heat of adsorption | XGBoost | 0.817 | 0.552 kJ mol⁻¹ |

XGBoost is the reported model for all four targets. A stacking ensemble (XGBoost + LightGBM + Random Forest + Extra Trees, Ridge meta-learner; `19_hoa_stacking_ensemble.py`) was also tested for heat of adsorption and reaches a modestly higher R² = 0.822, but XGBoost is retained for exact TreeExplainer SHAP compatibility, native quantile regression for conformal calibration, and consistency across all four targets (manuscript Section 2.6).

Full baseline comparison against Ridge, Random Forest, MLP, and CGCNN (`baseline_comparison.csv`), matching Table 1 of the manuscript:

| Target | XGBoost R² | Ridge R² | RF R² | MLP R² | CGCNN R² |
|---|---|---|---|---|---|
| CO₂ uptake | 0.982 | 0.904 | 0.944 | 0.984 | 0.742 |
| Working capacity | 0.986 | 0.918 | 0.959 | 0.987 | 0.751 |
| CO₂/H₂ selectivity | 0.975 (log) / 0.866 (raw) | 0.923 | 0.954 | 0.972 | 0.621 |
| Heat of adsorption | 0.817 | 0.793 | 0.809 | 0.815 | 0.488 |

MLP is marginally ahead of XGBoost on the capacity targets (< 0.002 R²) but XGBoost is retained as the primary model for exact TreeExplainer SHAP values and native quantile regression for conformal calibration (Section 2.9 / 3.2 of the manuscript). CGCNN trails substantially because local crystal graphs do not directly encode the pore-volume and surface-area descriptors that dominate these targets, not because graph architectures are broadly weaker; CGCNN also used its published default hyperparameters and was not tuned for this dataset, unlike the other baselines, so the comparison is an information-content check rather than a tuned-for-tuned benchmark (Section 2.9).

Three-fold cross-validation confirms stability: R² = 0.978 ± 0.000 (CO₂ uptake), 0.982 ± 0.000 (working capacity), 0.971 ± 0.000 (selectivity), 0.817 ± 0.001 (HoA). Seed-stability analysis across seeds 42, 0, and 123 gives R² variation below 0.002 for the three primary targets (HoA range/mean = 0.0045, reflecting its lower R² denominator, not partition sensitivity).

The heat-of-adsorption model reaches a descriptor-level ceiling that is **not** explained by charge coverage: restoring 100% real charge coverage left HoA R² essentially unchanged (0.817 before and after), and SHAP analysis on the corrected model ranks all seven charge-derived features outside the top 15 for HoA. The ceiling instead reflects a representational limit of aggregate charge summary statistics -- see manuscript Section 3.5.

Two additional robustness checks (`12_feature_ablation.py`, `13_buildingblock_leakage_check.py`) are included in this repository. The feature-set ablation (`feature_ablation_results.csv`) shows geometric descriptors alone already reach R² = 0.935-0.953 for the capacity targets, with RDF descriptors contributing the largest incremental gain. The building-block leakage check (`leakage_check_results.csv`, `leakage_check_group_summary.txt`) uses a group-aware split so no shared metal-node or linker token crosses train/test; because the two combinatorially-assembled data sources (81% of the database) each collapse into a single connected component under that constraint, the resulting group-split test set is drawn almost entirely (88%) from a structurally distinct source, making it an out-of-family generalization stress test rather than a narrow leakage probe. Test R² under that stricter split is 0.944/0.954/0.838/0.737 (CO₂ uptake/WC/selectivity/HoA), versus 0.982/0.986/0.975/0.817 under the primary random split reported above. See Supplementary Tables S5-S6 and manuscript Sections 2.4/3.2/3.6 for full discussion.

---

Uncertainty quantification

Split conformal prediction provides finite-sample calibrated intervals for all four targets. Before correction, uncalibrated quantile models covered only ~0.75 of test values at 80% nominal level. After conformal calibration on ~25,090 held-out structures, empirical coverages on the test set are:

| Target | Empirical coverage at 80% nominal | Interval width |
|---|---|---|
| CO₂ uptake | 0.802 | 2.30 mmol g⁻¹ |
| Working capacity | 0.804 | 2.33 mmol g⁻¹ |
| CO₂/H₂ selectivity | 0.797 | 33.6 raw units |
| Heat of adsorption | 0.801 | 1.68 kJ mol⁻¹ |

---

Screening output

The candidate-selection workflow is filter-first, then Pareto-ranked.

| Stage | Criterion | Structures retained |
|---|---|---|
| Full retained ARC-MOF set | Valid target values | 278,778 |
| Working-capacity filter | WC ≥ 19.57 mmol g⁻¹ (75th percentile) | 69,744 |
| Selectivity filter | CO₂/H₂ selectivity ≥ 130 | 790 |
| Restricted Pareto front | WC vs. selectivity inside the 790-structure pool | 10 |
| Final priority set | Four-target scalarisation | 50 |

The final priority set contains 50 MOFs. Of these, 76% are assigned High synthesizability based on structural family precedent (IRMOF-family name-string match: 38/50) and Tier 1 or Tier 2 metal-node criteria.

Top-ranked candidate: `DB1-Zn2O8N2-ADC_A-irmof14_A_No822`

- GCMC working capacity: 32.674 mmol g⁻¹ | ML predicted: 32.996 mmol g⁻¹
- GCMC selectivity: 151.5 | ML predicted: 157.0
- GCMC CO₂ uptake: 37.789 mmol g⁻¹ | ML predicted: 38.051 mmol g⁻¹
- ML heat of adsorption: 7.258 kJ mol⁻¹
- Synthesizability: High (IRMOF family, Zn node)

---

Topology–selectivity results

| Topology | Structures (n) | Median CO₂/H₂ selectivity | 95% bootstrap CI |
|---|---|---|---|
| fof | 689 | 172.0 | 168.9–175.7 |
| fsc | 45,297 | 145.5 | 145.1–145.9 |
| clean | — | 139.8 | 137.0–142.2 |
| pcu | 62,359 | 73.6 | — |
| Database-wide median | 278,778 | 89.0 | — |

Bootstrap CIs from 1,000 resamples at seed 42. fof and fsc are the primary reticular design targets identified by this study.

---

Repository structure

```text
.
├── generate_outputs.py         # Full pipeline: features -> models -> metrics -> screening (17 steps, see docstring)
├── regenerate_all_figures.py   # Rebuilds all manuscript figures from generate_outputs.py's CSV/parquet outputs
├── robustness_metrics.py       # Standalone 3-fold CV robustness check
├── FEATURE_ENGINEERING.md      # How the 77-feature matrix was built from raw ARC-MOF structures
├── top_candidates.csv, pareto_front.csv, back_calculated_results.csv,
│   synthesizability_results.csv, weight_sensitivity_results.csv,
│   robustness_metrics.csv, robustness_tables.txt   # Pipeline outputs, committed for inspection
├── requirements.txt            # Python package requirements
├── environment.yml             # Optional conda environment
└── README.md
```

`generate_outputs.py` expects `data/full_features.parquet` (archived on Zenodo, not in this repo) and writes cached model files to `data/models/`. It is idempotent: cached models and existing output files are skipped and reused on re-run.

The processed feature matrix, trained models, fixed split indices, computed outputs, and candidate lists are all archived on Zenodo:

DOI: `10.5281/zenodo.20305724`
URL: https://doi.org/10.5281/zenodo.20305724

---

Installation

Clone the repository:

```bash
git clone https://github.com/Rifat19R/mof-co2-h2-screening.git
cd mof-co2-h2-screening
```

Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core packages used in this study: Python 3.13, XGBoost 2.x, LightGBM 4.x, scikit-learn 1.x, Optuna 3.x, SHAP, NumPy, Pandas, Matplotlib, SciPy.

---

Reproducing the workflow

`generate_outputs.py` runs the full workflow as 17 sequential steps (see the script's own docstring for the exact list): load features, split train/test, train/load the four XGBoost models, write test predictions, compute SHAP values, build conformal intervals, compute the Pareto front and top candidates, run synthesizability scoring, weight-sensitivity analysis, topology analysis, baseline comparison, and robustness metrics.

Command pattern:

```bash
python generate_outputs.py            # runs all 17 steps; trained models are cached under data/models/
python regenerate_all_figures.py      # rebuilds all manuscript figures from the CSV/parquet outputs above
```

Both scripts are idempotent: cached models and existing output files are skipped and reused on re-run, so re-running `generate_outputs.py` after fetching pre-trained models from Zenodo only regenerates missing outputs, not a full retrain.

---

Data availability

The reproducibility archive on Zenodo contains:

- Processed 77-dimensional feature matrix (full_features.parquet, 278,885 structures; 278,778 after the 107 rows with missing/non-finite targets are dropped at pipeline runtime)
- Fixed train/test split indices (seed 42, stratified by CO₂ uptake decile)
- Calibration indices for conformal prediction
- Trained XGBoost model files (primary + quantile models for all four targets)
- Stacking ensemble weights for heat of adsorption
- Computed model metrics (test-set, cross-validation, seed-stability)
- Conformal prediction calibration outputs and interval widths
- SHAP outputs and feature importance rankings
- Pareto-screening results and scalarisation scores
- Weight-sensitivity outputs (50 weight sets)
- Final prioritised top-50 candidate list with GCMC and ML values

The raw ARC-MOF adsorption data should be obtained directly from the original source:

> Burner, J. et al. ARC–MOF: a diverse database of metal-organic frameworks with DFT-derived partial atomic charges and descriptors for machine learning. *Chemistry of Materials* 35, 900–916 (2023). https://doi.org/10.1021/acs.chemmater.2c02485

---

Important scope limits

This work is a GCMC-benchmarked comparative screen at 298 K, not an experimental validation study.

The screen is fixed at 298 K and 40 bar. Industrial pre-combustion streams operate at 150–400°C. Absolute capacities reported here are upper-bound comparative estimates and must be temperature-corrected before any process-level interpretation. Relative rankings are expected to remain qualitatively stable across moderate temperature changes.

Heat-of-adsorption prediction reaches a descriptor-level ceiling at R² ≈ 0.82 regardless of training set size. This is not a charge-coverage problem — see the correction note above — but a limit of the seven aggregate charge statistics used here. Spatially-resolved charge descriptors, not higher charge coverage, are the next step.

Synthesizability scores indicate structural family precedent and established secondary building units. They do not prove that any specific hypothetical structure has already been synthesised or will be straightforward to realise experimentally. Experimental validation remains essential for the shortlisted candidates.

The group-split accuracy above is a lower bound for out-of-family generalization, not the expected accuracy for the 50 priority candidates. Checking candidate structure IDs against the realized group-split assignment shows 49 of 50 fall on the training (in-family) side; one candidate falls on the held-out side, for which the group-split numbers, not the primary random-split numbers, are the relevant benchmark.

---

Citation

If this code or data contributes to your work, please cite the manuscript, the Zenodo archive, and the ARC-MOF database.

Manuscript (update after publication):

```bibtex
@article{khandaker2025mof,
  title   = {Packing efficiency governs {CO$_2$/H$_2$} selectivity in machine-learning-screened
             metal--organic frameworks},
  author  = {Khandaker, Md. Rifat and Chowdhury, Mohammad Asaduzzaman and Hossain, Sujan},
  journal = {Digital Discovery},
  year    = {2025},
  publisher = {Royal Society of Chemistry}
}
```

Zenodo data archive:

```bibtex
@misc{khandaker2026zenodo,
  title  = {Data and trained models for ``Packing Efficiency Governs {CO$_2$/H$_2$} Selectivity in {MOFs}"},
  author = {Khandaker, Md. Rifat and Chowdhury, Mohammad Asaduzzaman and Hossain, Sujan},
  year   = {2026},
  doi    = {10.5281/zenodo.20305724},
  url    = {https://doi.org/10.5281/zenodo.20305724}
}
```

ARC-MOF database:

```bibtex
@article{burner2023arcmof,
  title   = {{ARC-MOF}: a diverse database of metal-organic frameworks with {DFT}-derived
             partial atomic charges and descriptors for machine learning},
  author  = {Burner, Jake and Luo, Jun and White, Andrew and Mirmiran, Adam and Kwon, Ohmin
             and Boyd, Peter G. and Maley, Stephen and Gibaldi, Marco and Simrod, Scott
             and Ogden, Victoria and Woo, Tom K.},
  journal = {Chemistry of Materials},
  volume  = {35},
  pages   = {900--916},
  year    = {2023},
  doi     = {10.1021/acs.chemmater.2c02485}
}
```

---

License

- Code: MIT License — see [LICENSE](LICENSE)
- Trained model weights and processed data outputs: CC BY 4.0

Do not redistribute raw ARC-MOF files unless the original database licence permits it. Check the ARC-MOF source before redistribution.

---

Contact

Md. Rifat Khandaker
Department of Chemical Engineering
Dhaka University of Engineering & Technology (DUET)
Gazipur-1700, Bangladesh
Email: rifatkh.duet@gmail.com
GitHub: https://github.com/Rifat19R
