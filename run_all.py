"""
run_all.py
===========
Executes the full MOF CO₂/H₂ screening pipeline in order.

Usage
-----
    python run_all.py              # run everything
    python run_all.py --from 04   # resume from step 04
    python run_all.py --only 07   # run a single step

Pipeline order
--------------
  01  Build feature matrix (geometric, RAC, RDF, REPEAT charges)
  02  Train XGBoost models (CO₂ uptake, WC, selectivity, HoA)
  02b Improve selectivity and HoA models
  03  Conformal prediction (split conformal, quantile models)
  04  SHAP analysis (feature importance, dependence plots)
  05  External validation
  06  Pareto analysis and candidate selection
  07  Main manuscript figures
  08  Supplementary figures
  09  Additional analyses and figure patches
  10  Bug fixes (HoA retrain, selectivity back-transform, bootstrap CIs)
  11  HoA improvement
  12  Synthesizability assessment
  13  Weight sensitivity analysis
  14  Back-calculated values
  F23 Figure 23 — candidate-level validation

Excluded from pipeline (one-time or deprecated):
  15_setup_raspa_validation.py     — one-time RASPA setup, already run
  15_setup_raspa_validation_values.py
  16_parse_raspa_results.py        — one-time RASPA parsing, already run
  gnn.py / gnn_baseline.py        — baseline comparison only, not core pipeline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Script definitions ────────────────────────────────────────────────────────
# Each entry: (step_id, filename, description)
PIPELINE = [
    ("01",  "01_build_features.py",              "Build 77-dimensional feature matrix"),
    ("02",  "02_train_models.py",                "Train XGBoost regressors (all 4 targets)"),
    ("02b", "02b_improve_selectivity_hoa.py",    "Improve selectivity and HoA models"),
    ("03",  "03_uncertainty.py",                 "Conformal prediction — calibrated intervals"),
    ("04",  "04_shap_analysis.py",               "SHAP feature importance and dependence plots"),
    ("05",  "05_external_validation.py",         "External validation"),
    ("06",  "06_pareto_analysis.py",             "Pareto front and candidate selection"),
    ("07",  "07_figures.py",                     "Main manuscript figures"),
    ("08",  "08_supplementary.py",               "Supplementary figures"),
    ("09",  "09_additional_analyses.py",         "Additional analyses and figure patches (bootstrap CIs)"),
    ("10",  "10_fix_all_bugs.py",                "Bug fixes: HoA retrain, selectivity back-transform, topology CIs"),
    ("11",  "11_improve_hoa.py",                 "HoA model improvement"),
    ("12",  "12_synthesizability_analysis.py",   "Synthesizability scoring — top 50 candidates"),
    ("13",  "13_weight_sensitivity.py",          "Weight sensitivity analysis (50 weight sets)"),
    ("14",  "14_back_calculated_values.py",      "Back-calculated validation values"),
    ("F23", "fig23_candidate_validation.py",     "Figure 23 — candidate-level validation"),
]

SCRIPTS_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full MOF CO₂/H₂ screening pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", metavar="STEP",
        help="Resume pipeline from this step ID (e.g. --from 04)")
    group.add_argument(
        "--only", dest="only_step", metavar="STEP",
        help="Run only this step ID (e.g. --only 07)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print steps that would run without executing them")
    return parser.parse_args()


def select_steps(args):
    """Return the subset of PIPELINE steps to run given CLI args."""
    if args.only_step:
        steps = [s for s in PIPELINE if s[0] == args.only_step]
        if not steps:
            print(f"[ERROR] Step '{args.only_step}' not found. "
                  f"Valid IDs: {[s[0] for s in PIPELINE]}")
            sys.exit(1)
        return steps

    if args.from_step:
        ids = [s[0] for s in PIPELINE]
        if args.from_step not in ids:
            print(f"[ERROR] Step '{args.from_step}' not found. "
                  f"Valid IDs: {ids}")
            sys.exit(1)
        start = ids.index(args.from_step)
        return PIPELINE[start:]

    return PIPELINE


def run_step(step_id, filename, description, dry_run=False):
    script = SCRIPTS_DIR / filename
    if not script.exists():
        print(f"\n  [SKIP] {filename} not found — skipping.")
        return True   # non-fatal: skip missing scripts

    print(f"\n{'='*65}")
    print(f"  STEP {step_id}: {description}")
    print(f"  Script : {filename}")
    print(f"{'='*65}")

    if dry_run:
        print("  [DRY RUN] would execute:", sys.executable, filename)
        return True

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPTS_DIR)
    )
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    if result.returncode != 0:
        print(f"\n  [FAILED] Step {step_id} exited with code {result.returncode}.")
        print(f"  Elapsed: {mins}m {secs}s")
        return False

    print(f"\n  [DONE] Step {step_id} completed in {mins}m {secs}s.")
    return True


def main():
    args = parse_args()
    steps = select_steps(args)

    print("\n" + "="*65)
    print("  MOF CO₂/H₂ Screening Pipeline")
    print(f"  Steps to run: {len(steps)}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print("="*65)

    total_start = time.time()
    failed = []

    for step_id, filename, description in steps:
        ok = run_step(step_id, filename, description, dry_run=args.dry_run)
        if not ok:
            failed.append((step_id, filename))
            print("\n  Pipeline halted at failed step. "
                  "Fix the error above and resume with:")
            print(f"      python run_all.py --from {step_id}\n")
            break

    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)

    print("\n" + "="*65)
    if not failed:
        print(f"  ALL STEPS COMPLETE  ({total_mins}m {total_secs}s total)")
    else:
        print(f"  PIPELINE FAILED at step {failed[0][0]}: {failed[0][1]}")
        print(f"  Total time before failure: {total_mins}m {total_secs}s")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
