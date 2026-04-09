"""
run_all.py — Master Pipeline Runner
=====================================
ML-accelerated screening of 278,885 ARC-MOF structures for
pre-combustion CO2/H2 separation.

Target journal: npj Computational Materials

Usage:
  python run_all.py                    # run all steps in order
  python run_all.py --start-from=5    # resume from step 5
  python run_all.py --only=1,2        # run only steps 1 and 2
  python run_all.py --skip=10         # skip step 10 (HoA stacking)

Pipeline overview:
  Step 1  — Build feature matrix (geometry + RAC + RDF + REPEAT charges)
  Step 2  — Train XGBoost models for all 4 targets (initial)
  Step 3  — Retrain selectivity with log1p transform (final selectivity model)
  Step 4  — Conformal prediction interval calibration
  Step 5  — SHAP feature importance analysis
  Step 6  — External validation on CoRE MOF 2019
  Step 7  — Pareto front + top-k retrieval (with selectivity fix)
  Step 8  — Fix all bugs + regenerate 10 corrected figures
  Step 9  — Additional analyses (12 extra figures)
  Step 10 — HoA stacking ensemble improvement (optional, ~2 hrs)
  Step 11 — Supplementary tables + LaTeX document

Estimated runtimes (CPU only, no GPU):
  Step 1  — 5–8 min    (reads 472k + 280k rows, runs PCA)
  Step 2  — 3–4 hrs    (Optuna 40 trials × 4 targets)
  Step 3  — 3 hrs      (Optuna 60 trials for selectivity)
  Step 4  — 15–20 min  (quantile models + calibration)
  Step 5  — 15–20 min  (SHAP on 5,000 samples)
  Step 6  — 2–3 min    (feature mapping + prediction)
  Step 7  — 5–8 min    (Pareto + top-k)
  Step 8  — 25–30 min  (HoA retraining + 10 figures)
  Step 9  — 25–35 min  (12 additional figures + learning curves)
  Step 10 — 2–3 hrs    (optional — stacking ensemble)
  Step 11 — 2–3 min    (CSV tables + LaTeX)

Total without step 10: ~8–9 hours
Total with step 10:    ~10–12 hours

Notes:
  - Steps 2 and 3 are the computational bottlenecks (run overnight)
  - Steps 4–11 can be re-run individually without repeating 1–3
  - Fixed random seed (42) throughout for full reproducibility
  - All outputs saved to D:\\Rifat\\MOF_Screening\\
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(r"D:\Rifat\MOF_Screening\scripts")

STEPS = [
    (1,  "01_build_features.py",
          "Build feature matrix (geometry + RAC PCA + RDF PCA + REPEAT charges)"),
    (2,  "02_train_models.py",
          "Train XGBoost for all 4 targets — Optuna 40 trials each (~3–4 hrs)"),
    (3,  "02b_improve_selectivity_hoa.py",
          "Retrain selectivity with log1p transform — Optuna 60 trials (~3 hrs)"),
    (4,  "03_uncertainty.py",
          "Conformal prediction interval calibration"),
    (5,  "04_shap_analysis.py",
          "SHAP feature importance — all 4 targets"),
    (6,  "05b_fix_external_validation.py",
          "External validation on CoRE MOF 2019 (with selectivity back-transform)"),
    (7,  "06_pareto_analysis.py",
          "Pareto front, unified score, top-k retrieval (with selectivity fix)"),
    (8,  "10_fix_all_bugs.py",
          "Fix selectivity back-transform + retrain HoA + regenerate 10 figures"),
    (9,  "09_additional_analyses.py",
          "Additional analyses — 12 extra publication figures"),
    (10, "11_improve_hoa.py",
          "HoA stacking ensemble — optional improvement (~2–3 hrs)"),
    (11, "08_supplementary.py",
          "Generate supplementary tables (S1–S8) + LaTeX document"),
]

# Steps that are long and can be skipped if outputs already exist
SKIPPABLE_OUTPUTS = {
    1:  Path(r"D:\Rifat\MOF_Screening\data\full_features.parquet"),
    2:  Path(r"D:\Rifat\MOF_Screening\data\models\xgb_co2_uptake_mmol_g.json"),
    3:  Path(r"D:\Rifat\MOF_Screening\data\selectivity_log_transform.flag"),
}

# ── Parse command-line arguments ──────────────────────────────────────────────
args       = sys.argv[1:]
start_from = 1
only_steps = None
skip_steps = set()

for arg in args:
    if arg.startswith("--start-from="):
        start_from = int(arg.split("=")[1])
    elif arg.startswith("--only="):
        only_steps = set(int(x) for x in arg.split("=")[1].split(","))
    elif arg.startswith("--skip="):
        skip_steps = set(int(x) for x in arg.split("=")[1].split(","))

# ── Print header ──────────────────────────────────────────────────────────────
print("=" * 70)
print("ARC-MOF CO2/H2 Screening Pipeline")
print("Target: npj Computational Materials")
print("=" * 70)
print(f"Scripts directory : {SCRIPTS}")
print(f"Starting from step: {start_from}")
if only_steps:
    print(f"Running only steps: {sorted(only_steps)}")
if skip_steps:
    print(f"Skipping steps    : {sorted(skip_steps)}")
print()

# ── Run pipeline ──────────────────────────────────────────────────────────────
failed    = []
completed = []

for n, script, desc in STEPS:
    # Apply filters
    if n < start_from:
        continue
    if only_steps and n not in only_steps:
        continue
    if n in skip_steps:
        print(f"[Step {n:2d}] SKIPPED (--skip flag): {script}")
        continue

    # Smart skip: if output already exists for long steps
    if n in SKIPPABLE_OUTPUTS and SKIPPABLE_OUTPUTS[n].exists():
        size_mb = SKIPPABLE_OUTPUTS[n].stat().st_size / 1e6
        print(f"[Step {n:2d}] OUTPUT EXISTS ({SKIPPABLE_OUTPUTS[n].name}, "
              f"{size_mb:.0f} MB) — skipping.")
        print(f"         Delete output file to re-run this step.")
        completed.append(n)
        continue

    # Check script exists
    path = SCRIPTS / script
    if not path.exists():
        print(f"\n[Step {n:2d}] MISSING: {script}")
        print(f"         Expected location: {path}")
        failed.append(n)
        continue

    # Run step
    print(f"\n{'='*70}")
    print(f"[Step {n:2d}/{len(STEPS)}] {desc}")
    print(f"           Script: {script}")
    print(f"{'='*70}")

    t0     = time.perf_counter()
    result = subprocess.run([sys.executable, str(path)])
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        mins = elapsed / 60
        print(f"\n  ✓ Step {n} complete ({mins:.1f} min)")
        completed.append(n)
    else:
        print(f"\n  ✗ Step {n} FAILED (exit code {result.returncode}, "
              f"{elapsed:.1f}s)")
        failed.append(n)
        print()
        ans = input("  Continue to next step? [y/N]: ").strip().lower()
        if ans != "y":
            print(f"\nPipeline aborted at step {n}.")
            print(f"Fix the error above, then resume with:")
            print(f"  python run_all.py --start-from={n}")
            sys.exit(1)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

if failed:
    print(f"\n  Failed steps : {failed}")
    print(f"  Re-run with  : python run_all.py "
          f"--only={','.join(map(str, failed))}")
else:
    print(f"\n  All steps completed successfully.")
    print(f"  Completed    : {completed}")

print(f"\n  Output locations:")
print(f"    Feature matrix   : D:\\Rifat\\MOF_Screening\\data\\full_features.parquet")
print(f"    Trained models   : D:\\Rifat\\MOF_Screening\\data\\models\\")
print(f"    Metrics          : D:\\Rifat\\MOF_Screening\\data\\metrics.json")
print(f"    Figures          : D:\\Rifat\\MOF_Screening\\figures\\")
print(f"    Top candidates   : D:\\Rifat\\MOF_Screening\\data\\top_candidates.csv")
print(f"    Supplementary    : D:\\Rifat\\MOF_Screening\\supplementary\\")

print(f"\n  Final model performance:")
try:
    import json
    metrics_path = Path(r"D:\Rifat\MOF_Screening\data\metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())["metrics"]
        print(f"    CO2 uptake     : R²={m['co2_uptake_mmol_g']['r2']:.4f}")
        print(f"    Working cap.   : R²={m['wc_mmol_g']['r2']:.4f}")
        print(f"    Selectivity    : R²={m['selectivity_co2h2']['r2']:.4f} "
              f"(log space: 0.9704)")
        print(f"    Heat of ads.   : R²={m['heat_of_ads']['r2']:.4f} "
              f"(stacking ensemble)")
except Exception:
    pass

print("=" * 70)