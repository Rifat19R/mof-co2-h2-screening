Packing efficiency governs CO₂/H₂ selectivity in machine-learning-screened MOFs
This repository contains the analysis code for the manuscript:
Packing Efficiency Governs CO₂/H₂ Selectivity in Metal–Organic Frameworks: Uncertainty-Guided Machine Learning Screening of 278,778 Structures Reveals Topology-Level Reticular Design Rules
Authors: Md. Rifat Khandaker, Mohammad Asaduzzaman Chowdhury, Sujan Hossain  
Affiliation: Department of Chemical Engineering, Dhaka University of Engineering & Technology (DUET), Gazipur-1700, Bangladesh  
Contact: rifatkh.duet@gmail.com
---
What this project does
This project uses an uncertainty-guided machine-learning workflow to screen 278,778 ARC-MOF structures for pre-combustion CO₂/H₂ separation at 298 K and 40 bar. The goal is not only to rank materials. The goal is to extract design rules that are useful for reticular MOF design.
The workflow predicts four adsorption targets:
CO₂ uptake
CO₂ working capacity
CO₂/H₂ selectivity
Heat of adsorption
The final model stack combines descriptor-based learning, split conformal uncertainty quantification, SHAP interpretation, topology-level analysis, Pareto screening, synthesizability scoring, and weight-sensitivity testing.
---
Main result
The central finding is simple:
> **CO₂/H₂ selectivity is governed more strongly by packing efficiency than by pore diameter.**
Working capacity follows a different rule. It peaks near POAVAg ≈ 1.5 cm³ g⁻¹ and weakens beyond roughly 2 cm³ g⁻¹. This gives a practical design window:
> **POAVAg = 1–2 cm³ g⁻¹ with crystal density below 0.5 g cm⁻³.**
At topology level, fof and fsc nets show strong median CO₂/H₂ selectivity, with median selectivities of 172.0 and 145.5, compared with a database median of 89.
---
Dataset and descriptors
The workflow starts from the ARC-MOF database and retains 278,778 structures after removing entries with missing or non-finite target values.
The final descriptor matrix contains 77 features:
Descriptor family	Count	Description
Geometric descriptors	30	Pore volume, surface area, void fraction, density, sphere diameters, interaction terms, log-transforms
RAC principal components	20	PCA-reduced revised autocorrelation descriptors
RDF principal components	20	PCA-reduced radial distribution function descriptors
REPEAT charge statistics	7	Mean, standard deviation, skewness, kurtosis, min, max, atom count
REPEAT partial charges are available for only 8.8% of the database. The remaining structures use median-imputed charge statistics. This is the main limitation for heat-of-adsorption prediction.
---
Model performance
Held-out test set: 27,878 structures.
Target	Model	R²	MAE
CO₂ uptake	XGBoost	0.981	0.564 mmol g⁻¹
Working capacity	XGBoost	0.985	0.567 mmol g⁻¹
CO₂/H₂ selectivity	XGBoost, log(1+x) target	0.975	0.101 log units
Heat of adsorption	Stacking ensemble	0.817	0.552 kJ mol⁻¹
The heat-of-adsorption model reaches a descriptor-level ceiling because charge coverage is sparse. This is treated as a limitation, not hidden as model success.
---
Screening output
The candidate-selection workflow is filter-first, then Pareto-ranked.
Stage	Criterion	Structures retained
Full retained ARC-MOF set	Valid target values	278,778
Working-capacity filter	WC ≥ 19.57 mmol g⁻¹	69,685
Selectivity filter	CO₂/H₂ selectivity ≥ 130	803
Restricted Pareto front	WC vs selectivity inside the 803-structure pool	4
Final priority set	Four-target scalarisation	50
The final priority set contains 50 MOFs. Of these, 72% are assigned High synthesizability based on family-level structural precedent and metal-node criteria.
---
Repository structure
Recommended layout:
```text
.
├── data/
│   ├── raw/                    # Raw ARC-MOF inputs, if locally available
│   ├── processed/              # Processed descriptor matrices and target files
│   ├── splits/                 # Fixed train/test/calibration split indices
│   └── candidates/             # Final screened and prioritized MOF lists
├── models/                     # Trained model files
├── outputs/
│   ├── metrics/                # Test, CV, seed-stability, and candidate-set metrics
│   ├── conformal/              # Split conformal interval outputs
│   ├── shap/                   # SHAP values and feature rankings
│   ├── screening/              # Pareto, scalarisation, and weight-sensitivity outputs
│   └── figures/                # Final manuscript figures
├── scripts/                    # Reproducible analysis scripts
├── notebooks/                  # Optional exploratory notebooks
├── requirements.txt            # Python package requirements
├── environment.yml             # Optional conda environment file
└── README.md
```
The exact processed data, trained models, fixed split indices, computed outputs, and candidate lists are archived on Zenodo:
DOI: `10.5281/zenodo.20305725`  
URL: `https://doi.org/10.5281/zenodo.20305725`
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
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
```
Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Core packages used in the study:
Python 3.13
XGBoost 2.x
LightGBM 4.x
scikit-learn 1.x
Optuna 3.x
SHAP
NumPy
Pandas
Matplotlib
SciPy
---
Reproducing the workflow
The full workflow follows this order:
```text
1. Prepare target table and descriptor matrix
2. Build fixed train/test/calibration splits
3. Train XGBoost models for CO₂ uptake, working capacity, and selectivity
4. Train stacking ensemble for heat of adsorption
5. Run cross-validation and seed-stability checks
6. Build split conformal prediction intervals
7. Compute SHAP values and dependence plots
8. Run high-performance filtering
9. Compute restricted Pareto front
10. Run four-target scalarisation
11. Run synthesizability scoring and weight-sensitivity analysis
12. Generate final figures and tables
```
Suggested command pattern:
```bash
python scripts/01_prepare_features.py
python scripts/02_train_models.py
python scripts/03_cross_validation.py
python scripts/04_conformal_prediction.py
python scripts/05_shap_analysis.py
python scripts/06_screen_candidates.py
python scripts/07_weight_sensitivity.py
python scripts/08_make_figures.py
```
If your local script names differ, keep the same order. The important part is the fixed split indices and the archived processed inputs.
---
Data availability
The reproducibility archive contains:
Processed feature matrix
Fixed train/test split indices
Calibration indices for conformal prediction
Trained model files
Computed model metrics
Conformal prediction outputs
SHAP outputs
Pareto-screening results
Weight-sensitivity outputs
Final prioritized candidate list
The raw ARC-MOF adsorption data should be obtained from the original ARC-MOF database source. This repository is intended to reproduce the analysis from the processed and archived inputs.
---
Important scope limits
This work is a GCMC-benchmarked comparative screen, not an experimental validation study.
Key limits:
The screen is fixed at 298 K.
Industrial pre-combustion streams often operate at higher temperature.
Absolute capacities should not be treated as process-level values without temperature-dependent isotherms.
Heat-of-adsorption prediction is limited by sparse REPEAT charge coverage.
Synthesizability scores indicate family-level precedent. They do not prove that a specific hypothetical structure has already been synthesized.
These limits are part of the workflow. They define the next step: targeted experimental or higher-fidelity computational validation of the shortlisted MOFs.
---
Citation
Manuscript citation will be updated after publication.
For now, cite the repository and Zenodo archive:
```bibtex
@misc{khandaker_mof_co2_h2_screening,
  title  = {Packing efficiency governs CO2/H2 selectivity in machine-learning-screened metal-organic frameworks},
  author = {Khandaker, Md. Rifat and Chowdhury, Mohammad Asaduzzaman and Hossain, Sujan},
  year   = {2026},
  doi    = {10.5281/zenodo.20305725},
  url    = {https://doi.org/10.5281/zenodo.20305725}
}
```
---
License
Add the repository license before publication. Recommended:
MIT License for code
CC BY 4.0 for processed data, figures, and documentation, if compatible with the original data-source terms
Do not redistribute raw ARC-MOF files unless the original license permits it.
---
Contact
Md. Rifat Khandaker  
Department of Chemical Engineering  
Dhaka University of Engineering & Technology (DUET)  
Gazipur-1700, Bangladesh  
Email: rifatkh.duet@gmail.com
