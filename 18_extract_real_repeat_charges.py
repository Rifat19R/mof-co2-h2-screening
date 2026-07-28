"""
18_extract_real_repeat_charges.py
===================================
CRITICAL DATA-PIPELINE FIX (discovered during resubmission audit, 2026-07-28).

The manuscript claims REPEAT partial charges are available for only 8.8% of
the database (24,483 structures), with the remaining 91.2% median-imputed.
This claim traces to data/repeat_charge_stats.parquet, an external file that
no longer exists in the repo and whose original extraction evidently failed
for ~91% of structures.

Direct inspection of the official ARC-MOF v2022-06-10 CIF archive (Zenodo
record 6908728, ARCMOF_20220610.tar.gz, the exact version cited in the
manuscript) shows every CIF already carries per-atom `_atom_type_partial_charge`
values from the REPEAT Assigner. A random sample of 500 structures flagged
"imputed" in full_features.parquet showed 493/500 (98.6%) actually have real,
structure-specific charges in the source CIF.

This script re-extracts real per-atom REPEAT charges for every structure in
full_features.parquet directly from the CIF archive in a single sequential
pass (avoids 278k+ repeated archive scans), computes the same 7 summary
statistics used throughout the pipeline (mean, std, skew, kurtosis, min, max,
atom count), and writes them to data/repeat_charge_stats_full.parquet.

Output: data/repeat_charge_stats_full.parquet
        (mof_id, charge_mean, charge_std, charge_skew, charge_kurt,
         charge_min, charge_max, charge_n, charge_source)
        charge_source in {"real_cif", "missing_from_archive"}
"""

import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TAR_PATH = ROOT / "scratch_dft_charges" / "ARCMOF_20220610.tar.gz"
OUT_PATH = DATA / "repeat_charge_stats_full.parquet"
LOG_PATH = ROOT / "scripts" / "18_extract_real_repeat_charges_log.txt"

_log = []


def log(msg=""):
    print(msg)
    _log.append(str(msg))


def hdr(t):
    log(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def parse_cif_charges(content: str):
    """Extract per-atom partial charges from a REPEAT-assigned CIF's atom loop."""
    charges = []
    in_atom_loop = False
    header_done = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "_atom_type_partial_charge":
            header_done = True
            continue
        if header_done:
            parts = stripped.split()
            if len(parts) == 6:
                try:
                    charges.append(float(parts[5]))
                except ValueError:
                    pass
            elif stripped == "" or stripped.startswith("_") or stripped.startswith("loop_"):
                if charges:
                    break
    return np.array(charges, dtype=np.float64)


hdr("STEP 1 -- Load target mof_id list from full_features.parquet")
feat = pd.read_parquet(DATA / "full_features.parquet", columns=["mof_id"])
target_ids = set(feat["mof_id"].tolist())
log(f"Target structures: {len(target_ids):,}")

name_to_mid = {f"repeat_cifs/{mid}_repeat.cif": mid for mid in target_ids}
target_names = set(name_to_mid.keys())

hdr("STEP 2 -- Single-pass scan of ARC-MOF CIF archive, extracting charges")
assert TAR_PATH.exists(), f"Missing {TAR_PATH} -- download from Zenodo record 6908728 first"

rows = []
n_scanned = 0
n_matched = 0
t0 = time.time()
with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar:
        n_scanned += 1
        if member.name in target_names:
            mid = name_to_mid[member.name]
            f = tar.extractfile(member)
            content = f.read().decode("utf-8", errors="ignore")
            charges = parse_cif_charges(content)
            if len(charges) > 0:
                rows.append({
                    "mof_id": mid,
                    "charge_mean": charges.mean(),
                    "charge_std": charges.std(),
                    "charge_skew": skew(charges),
                    "charge_kurt": kurtosis(charges),
                    "charge_min": charges.min(),
                    "charge_max": charges.max(),
                    "charge_n": len(charges),
                    "charge_source": "real_cif",
                })
            else:
                rows.append({"mof_id": mid, "charge_source": "empty_charge_data"})
            n_matched += 1
        if n_scanned % 50000 == 0:
            dt = time.time() - t0
            log(f"  scanned {n_scanned:,}/279,611 archive entries "
                f"({n_matched:,}/{len(target_names):,} targets matched, {dt:.0f}s elapsed)")
        if n_matched == len(target_names):
            log(f"  all {n_matched:,} targets found after scanning {n_scanned:,} entries "
                f"({time.time()-t0:.0f}s)")
            break

dt_total = time.time() - t0
log(f"\nScan complete in {dt_total:.0f}s. Matched {n_matched:,}/{len(target_ids):,} target structures.")

hdr("STEP 3 -- Assemble and save")
out_df = pd.DataFrame(rows)
n_real = (out_df["charge_source"] == "real_cif").sum()
n_missing_from_archive = len(target_ids) - n_matched
n_empty = (out_df["charge_source"] == "empty_charge_data").sum()

log(f"Real per-atom charges recovered : {n_real:,} ({100*n_real/len(target_ids):.1f}%)")
log(f"Present in archive but no charge data parsed : {n_empty:,}")
log(f"Not found in archive at all      : {n_missing_from_archive:,}")

out_df.to_parquet(OUT_PATH, index=False)
log(f"\nSaved: {OUT_PATH}")

LOG_PATH.write_text("\n".join(_log), encoding="utf-8")
log(f"Log written to: {LOG_PATH}")
log("\nDONE.")
