"""
08_supplementary.py
====================
Generates all supplementary tables as CSV + LaTeX and compiles
a supplementary.tex document ready for journal submission.

IMPORTANT — selectivity model note:
  xgb_selectivity_co2h2.json was trained on log1p-transformed targets.
  np.expm1() is applied before all selectivity analyses in physical units.

Tables generated:
  S1 — XGBoost hyperparameters (all 4 targets)
  S2 — Model performance (R², MAE, RMSE on test set)
  S3 — Baseline model comparison (Ridge, RF, MLP, XGBoost)
  S4 — Top-50 MOF candidates (unified multi-objective score)
  S5 — Pareto-optimal MOFs (working capacity vs. selectivity)
  S6 — Top-k retrieval precision and recall
  S7 — Conformal prediction interval half-widths
  S8 — SHAP-based feature block importance (%)

Outputs:
  supplementary/tables/s1_hyperparameters.csv  (+ .tex)
  supplementary/tables/s2_performance.csv      (+ .tex)
  supplementary/tables/s3_baseline.csv         (+ .tex)
  supplementary/tables/s4_top50_candidates.csv (+ .tex)
  supplementary/tables/s5_pareto_front.csv     (+ .tex)
  supplementary/tables/s6_topk.csv             (+ .tex)
  supplementary/tables/s7_conformal.csv        (+ .tex)
  supplementary/tables/s8_feature_blocks.csv   (+ .tex)
  supplementary/supplementary.tex
"""

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT   = Path(r"D:\Rifat\MOF_Screening")
DATA   = ROOT / "data"
SUPP   = ROOT / "supplementary"
TABLES = SUPP / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

TARGETS = ["co2_uptake_mmol_g", "wc_mmol_g", "selectivity_co2h2", "heat_of_ads"]
TARGET_LABELS = {
    "co2_uptake_mmol_g": r"CO$_2$ Uptake",
    "wc_mmol_g"        : "Working Cap.",
    "selectivity_co2h2": "Selectivity",
    "heat_of_ads"      : "Heat of Ads.",
}
SKIP = set(TARGETS + ["mof_id","co2_uptake_wt_pct","co2_uptake_vol","wc_wt_pct"])

# ── Load metrics ──────────────────────────────────────────────────────────────
metrics_raw = json.loads((DATA / "metrics.json").read_text())
metrics     = metrics_raw["metrics"]
params      = metrics_raw["best_params"]


# ── S1: XGBoost Hyperparameters ───────────────────────────────────────────────
rows_s1 = []
for tgt in TARGETS:
    p = params.get(tgt, {})
    rows_s1.append({
        "Target"           : TARGET_LABELS[tgt],
        "n_estimators"     : p.get("n_estimators", "—"),
        "max_depth"        : p.get("max_depth", "—"),
        "learning_rate"    : f"{p.get('lr', p.get('learning_rate', 0)):.4f}",
        "subsample"        : f"{p.get('subsample', 0):.3f}",
        "colsample_bytree" : f"{p.get('colsample', p.get('colsample_bytree', 0)):.3f}",
        "reg_alpha"        : f"{p.get('alpha', p.get('reg_alpha', 0)):.2e}",
        "reg_lambda"       : f"{p.get('lambda', p.get('reg_lambda', 0)):.2e}",
        "log_transform"    : "Yes" if tgt == "selectivity_co2h2" else "No",
    })
pd.DataFrame(rows_s1).to_csv(TABLES / "s1_hyperparameters.csv", index=False)
print("S1 saved — hyperparameters")


# ── S2: Model Performance ─────────────────────────────────────────────────────
rows_s2 = []
for tgt in TARGETS:
    m = metrics[tgt]
    note = "log₁₊₁ transform applied" if tgt == "selectivity_co2h2" else ""
    note += " | stacking ensemble" if tgt == "heat_of_ads" else ""
    rows_s2.append({
        "Target"  : TARGET_LABELS[tgt],
        "R²"      : f"{m['r2']:.4f}",
        "MAE"     : f"{m['mae']:.4f}",
        "RMSE"    : f"{m.get('rmse', float('nan')):.4f}",
        "N train" : m.get("n_train", "—"),
        "N test"  : m.get("n_test", "—"),
        "Notes"   : note.strip(" |"),
    })
pd.DataFrame(rows_s2).to_csv(TABLES / "s2_performance.csv", index=False)
print("S2 saved — model performance")


# ── S3: Baseline Comparison ───────────────────────────────────────────────────
bl_path = DATA / "baseline_results.json"
if bl_path.exists():
    bl = json.loads(bl_path.read_text())
    rows_s3 = []
    for tgt in TARGETS:
        for model_name, mvals in bl[tgt].items():
            rows_s3.append({
                "Target": TARGET_LABELS[tgt],
                "Model" : model_name,
                "R²"    : f"{mvals['r2']:.4f}",
                "MAE"   : f"{mvals['mae']:.4f}",
            })
    pd.DataFrame(rows_s3).to_csv(TABLES / "s3_baseline.csv", index=False)
    print("S3 saved — baseline comparison")
else:
    print("S3 SKIPPED — run 10_fix_all_bugs.py first to generate baseline_results.json")


# ── S4: Top-50 Candidates ─────────────────────────────────────────────────────
top50_path = DATA / "top_candidates.csv"
if top50_path.exists():
    top50 = pd.read_csv(top50_path)
    top50.to_csv(TABLES / "s4_top50_candidates.csv", index=False)
    print(f"S4 saved — top-50 candidates (#{1}: {top50['mof_id'].iloc[0]})")
else:
    print("S4 SKIPPED — run 06_pareto_analysis.py first")


# ── S5: Pareto-Optimal MOFs ───────────────────────────────────────────────────
# Recompute Pareto front with selectivity correctly back-transformed
df_all    = pd.read_parquet(DATA / "full_features.parquet")
feat_cols = [c for c in df_all.columns if c not in SKIP
             and pd.api.types.is_numeric_dtype(df_all[c])]
X_all     = df_all[feat_cols].fillna(0).values.astype("float32")

m_wc  = xgb.XGBRegressor()
m_sel = xgb.XGBRegressor()
m_wc.load_model(str(DATA / "models" / "xgb_wc_mmol_g.json"))
m_sel.load_model(str(DATA / "models" / "xgb_selectivity_co2h2.json"))

wc_p  = m_wc.predict(X_all)
sel_p_log = m_sel.predict(X_all)
sel_p = np.maximum(np.expm1(sel_p_log), 0)   # CRITICAL: back-transform from log space

print(f"\nS5 Pareto computation:")
print(f"  WC range        : {wc_p.min():.2f}–{wc_p.max():.2f} mmol/g")
print(f"  Selectivity range: {sel_p.min():.1f}–{sel_p.max():.1f} (real units)")

# Pareto front
idx_sorted = np.argsort(-wc_p)
front, best_sel = [], -np.inf
for i in idx_sorted:
    if sel_p[i] > best_sel:
        front.append(i)
        best_sel = sel_p[i]

pareto_df = df_all.iloc[front][["mof_id"]].copy()
pareto_df["pred_wc_mmol_g"]           = wc_p[front]
pareto_df["pred_selectivity_co2h2"]   = sel_p[front]
pareto_df = pareto_df.sort_values("pred_wc_mmol_g", ascending=False).head(30)
pareto_df.insert(0, "Rank", range(1, len(pareto_df)+1))
pareto_df.to_csv(TABLES / "s5_pareto_front.csv", index=False)
print(f"S5 saved — {len(pareto_df)} Pareto-optimal MOFs (selectivity in real units)")


# ── S6: Top-k Retrieval ───────────────────────────────────────────────────────
topk_path = DATA / "topk_results.json"
if topk_path.exists():
    topk = json.loads(topk_path.read_text())
    rows_s6 = []
    for tgt in TARGETS:
        for r in topk[tgt]:
            rows_s6.append({
                "Target"      : TARGET_LABELS[tgt],
                "k"           : r["k"],
                "Precision@k" : f"{r['precision']:.3f}",
                "Recall@k"    : f"{r['recall']:.3f}",
            })
    pd.DataFrame(rows_s6).to_csv(TABLES / "s6_topk.csv", index=False)
    print("S6 saved — top-k retrieval metrics")
else:
    print("S6 SKIPPED — run 06_pareto_analysis.py first")


# ── S7: Conformal Prediction Interval Widths ──────────────────────────────────
deltas_path = DATA / "conformal_deltas.json"
if deltas_path.exists():
    deltas = json.loads(deltas_path.read_text())
    rows_s7 = []
    for tgt in TARGETS:
        for nom, delta in deltas[tgt].items():
            rows_s7.append({
                "Target"            : TARGET_LABELS[tgt],
                "Nominal coverage"  : f"{float(nom):.0%}",
                "Delta (half-width)": f"{delta:.4f}",
                "Units"             : "kJ/mol" if "heat" in tgt
                                      else ("dimensionless" if "sel" in tgt
                                      else "mmol/g"),
            })
    pd.DataFrame(rows_s7).to_csv(TABLES / "s7_conformal.csv", index=False)
    print("S7 saved — conformal prediction intervals")
else:
    print("S7 SKIPPED — run 03_uncertainty.py first")


# ── S8: SHAP Feature Block Importance ────────────────────────────────────────
shap_path = DATA / "shap_importance.json"
if shap_path.exists():
    shap_data = json.loads(shap_path.read_text())
    rows_s8 = []
    for tgt in TARGETS:
        imp = shap_data.get(tgt, {})
        blocks = {"Geometric": 0.0, "RAC": 0.0, "RDF": 0.0, "Charge": 0.0}
        for feat, val in imp.items():
            if feat.startswith("RAC_"):      blocks["RAC"]      += val
            elif feat.startswith("RDF_"):    blocks["RDF"]      += val
            elif feat.startswith("charge_"): blocks["Charge"]   += val
            else:                            blocks["Geometric"] += val
        total = sum(blocks.values()) or 1
        row   = {"Target": TARGET_LABELS[tgt]}
        for b, v in blocks.items():
            row[f"{b} (%)"] = f"{v/total*100:.1f}"
        rows_s8.append(row)
    pd.DataFrame(rows_s8).to_csv(TABLES / "s8_feature_blocks.csv", index=False)
    print("S8 saved — SHAP feature block importance")
else:
    print("S8 SKIPPED — run 04_shap_analysis.py first")


# ── Convert all CSVs to LaTeX ─────────────────────────────────────────────────
def csv_to_latex(csv_path: Path, caption: str, label: str) -> str:
    df = pd.read_csv(csv_path)
    body = df.to_latex(index=False, escape=False, float_format="%.4f").strip()
    return (f"\\begin{{table}}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{body}\n\\end{{table}}\n")

tables_tex = [
    ("s1_hyperparameters.csv",
     "Optimised XGBoost hyperparameters for each target property. "
     "Selectivity was trained on log$_{1+1}$-transformed targets.",
     "tab:hyperparams"),
    ("s2_performance.csv",
     "Model performance on the held-out test set (10\\%, $n=27{,}889$). "
     "Selectivity R$^2$ reported in log-transformed space (0.970) and "
     "raw space (0.825). Heat of adsorption uses stacking ensemble (R$^2$=0.768).",
     "tab:performance"),
    ("s3_baseline.csv",
     "Baseline model comparison. All models use log$_{1+1}$ transformation "
     "for selectivity to ensure fair comparison.",
     "tab:baseline"),
    ("s4_top50_candidates.csv",
     "Top-50 MOF candidates ranked by unified multi-objective score "
     "(arithmetic mean of four normalised predicted properties).",
     "tab:top50"),
    ("s5_pareto_front.csv",
     "Pareto-optimal MOFs with respect to working capacity and "
     "CO$_2$/H$_2$ selectivity (selectivity in real units).",
     "tab:pareto"),
    ("s6_topk.csv",
     "Top-$k$ retrieval precision and recall on the held-out test set. "
     "True top-100 structures used as reference set.",
     "tab:topk"),
    ("s7_conformal.csv",
     "Conformal prediction interval half-widths ($\\delta$) at each nominal "
     "coverage level. Values shown in native property units.",
     "tab:conformal"),
    ("s8_feature_blocks.csv",
     "SHAP-based feature block importance (\\%) for each target. "
     "Geometric: 30 zeo++ features; RAC: 20 PCs; RDF: 20 PCs; "
     "Charge: 7 REPEAT statistics.",
     "tab:blocks"),
]

tex_body = ""
for fname, cap, lbl in tables_tex:
    p = TABLES / fname
    if p.exists():
        tex_body += csv_to_latex(p, cap, lbl) + "\n\n"
    else:
        print(f"  WARNING: {fname} not found — skipping LaTeX conversion")

# ── Compile supplementary.tex ─────────────────────────────────────────────────
latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{longtable}

\title{Supplementary Information\\[0.5em]
\large Machine Learning-Accelerated Screening of 278,885 Metal--Organic
Frameworks for Pre-combustion CO$_2$/H$_2$ Separation:\\
Multi-target Prediction, Uncertainty Quantification,
and Priority Candidate Identification}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section*{Supplementary Tables}

""" + tex_body + r"""

\newpage
\section*{Supplementary Figures}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_uncertainty_calibration.png}
\caption{Conformal prediction interval calibration before (left) and after
(right) conformal correction. After calibration, empirical coverage tracks
the nominal level closely for CO$_2$ uptake, working capacity, and heat of
adsorption. Selectivity calibration was performed in log-transformed space;
coverage for selectivity is reported after back-transformation.}
\label{fig:calibration}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_baseline_comparison.png}
\caption{Baseline model comparison across all four target properties.
All models use log$_{1+1}$ transformation for selectivity. XGBoost (ours)
outperforms all baselines for selectivity and heat of adsorption. For CO$_2$
uptake and working capacity, MLP achieves marginally higher R$^2$ ($<$0.5
percentage points); XGBoost is preferred for its SHAP interpretability and
native quantile regression support.}
\label{fig:baseline}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_topk.png}
\caption{Top-$k$ retrieval precision and recall on the held-out test set
($n = 27{,}889$). Precision@10 = 1.00 for CO$_2$ uptake and working
capacity, confirming perfect identification of the true top performers
within just 10 ML-selected structures.}
\label{fig:topk}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_external_validation.png}
\caption{External validation on 13,758 independent CoRE MOF 2019 structures.
Models trained on ARC-MOF are applied without retraining. CoRE predictions
use 23/77 matched features (30\%); RAC, RDF, and charge descriptors are
unavailable for CoRE MOF in this work. Predicted distributions are compared
with ARC-MOF test set distributions to assess domain shift.}
\label{fig:external}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_hoa_charge_imputation_effect.png}
\caption{Effect of median charge imputation on descriptor diversity.
Real REPEAT partial charge standard deviations span 0.000--0.796~e across
24,483 charge-complete structures (left). Median imputation collapses 91.2\%
of the 278,885 structures to a single value of 0.411~e (right), eliminating
the electrostatic signal required for accurate heat of adsorption prediction.}
\label{fig:imputation}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{../figures/fig_error_by_database.png}
\caption{Prediction error (MAE) broken down by ARC-MOF database source
(DB0--DB15). Error bars show $\pm 1\sigma$. Numbers above bars indicate
test set size per source. Selectivity MAE is reported in raw
CO$_2$/H$_2$ selectivity units.}
\label{fig:dberror}
\end{figure}

\end{document}
"""

(SUPP / "supplementary.tex").write_text(latex, encoding="utf-8")
print(f"\nAll supplementary files saved to: {SUPP}")
print("Compile with:")
print("  cd supplementary && pdflatex supplementary.tex && pdflatex supplementary.tex")
print("\nPipeline complete.")