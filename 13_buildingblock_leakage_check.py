"""
13_buildingblock_leakage_check.py
==================================
Priority 1.2 (Digital Discovery resubmission, Referee 3 checklist item 5c):
building-block leakage check. The random 90/10 structure-level split does not
guarantee structures sharing the same linker/node/topology building block are
kept on the same side. This script parses mof_id, groups structures that share
a building block via union-find (so no shared block crosses the split), and
retrains/evaluates under a group-respecting 90/10 split for direct comparison
against the existing random-split R2/MAE.

mof_id naming is NOT one convention -- it is 14 distinct DBx source prefixes.
Field cardinality was inspected empirically first (see chat record / field_
cardinality analysis) before deciding which fields are real shared building-
block tokens vs unique per-row serial numbers. Decodable, high-confidence
combinatorial sources:

  DB0 (203,022 rows, 73% of data): "DB0-m<M>_o<L1>_o<L2>_f<F>_<topo>.sym.<K>"
      -> building blocks = metal index M, linker indices L1, L2.
  DB1 (23,259 rows, 8%):           "DB1-<node>-<linker1>-<linker2>_No<idx>"
      -> building blocks = node formula, linker1 (name+isomer), linker2.
  DB5 (22,360 rows, 8%):           "DB5-hypotheticalMOF_<uid>_<f2>_<f3>_<f4>_<f5>_<f6>_<f7>"
      -> fields f2..f7 have low cardinality (4-36 distinct values across 22,360
         rows), consistent with reused topology/node/linker index codes in the
         hMOF combinatorial scheme, but the exact field->meaning mapping is not
         independently confirmed from documentation available in this repo.
         Treated conservatively (all six fields as block tokens) since over-
         grouping only makes the leakage check MORE conservative, not less.

Sources NOT treated as combinatorially decomposable (no shared-block grouping
applied beyond ordinary random assignment -- see printed rationale per prefix):
  DB6, DB7 (index-only IDs, no encoded components)
  DB12 (CSD refcodes / literature codes -- real, individually distinct structures)
  DB13 (topology + "Syn" literature serial -- not a shared parts-bin scheme)
  DB15 (topology + generated graph node/edge counts -- unique generated graphs)
  DB2, DB3, DB4, DB8, DB10, DB14 (n<120 each, together <0.1% of data)

Output: scripts/leakage_check_results.csv (random-split vs group-split R2/MAE,
4 targets), plus scripts/leakage_check_group_summary.txt (group size stats).
"""

import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Rifat\Research\MOF_Screening")
DATA = ROOT / "data"
OUT_CSV = ROOT / "scripts" / "leakage_check_results.csv"
OUT_TXT = ROOT / "scripts" / "leakage_check_group_summary.txt"

SEED = 42
N_BINS = 10

TARGET_COLS = {
    "co2_uptake_mmol_g": {"log": False, "clip": False, "label": "CO2_uptake"},
    "wc_mmol_g":         {"log": False, "clip": False, "label": "WC"},
    "selectivity_co2h2": {"log": True,  "clip": False, "label": "Selectivity"},
    "heat_of_ads":       {"log": False, "clip": True,  "label": "HoA"},
}
NON_FEAT = {"mof_id", "co2_uptake_wt_pct", "co2_uptake_vol", "wc_wt_pct"}
HOA_SIGMA = 5

HPARAMS = dict(
    n_estimators=900, max_depth=8, learning_rate=0.07,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=4,
    reg_alpha=0.02, reg_lambda=0.05,
    tree_method="hist", n_jobs=-1, random_state=SEED,
)

# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

DB0_RE = re.compile(r"^DB0-m(\d+)")
DB5_RE = re.compile(r"^DB5-hypotheticalMOF_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")

def extract_blocks(mof_id):
    """Return a list of building-block token strings for one mof_id, or [] if
    the source is not treated as combinatorially decomposable (singleton).
    DB0/DB1 use robust token extraction (not full-string regex) because a
    minority of IDs are single-linker variants (~1.4% of DB0, ~7.6% of DB1)
    that don't follow the two-linker pattern; verified 100% coverage on the
    full dataset before use."""
    if mof_id.startswith("DB0-"):
        m = DB0_RE.match(mof_id)
        if m:
            os_ = re.findall(r"_o(\d+)", mof_id)
            return [f"DB0:m{m.group(1)}"] + [f"DB0:o{o}" for o in os_]
    elif mof_id.startswith("DB1-"):
        body = mof_id[len("DB1-"):]
        parts = body.split("-")
        if len(parts) >= 2:
            parts = list(parts)
            parts[-1] = re.sub(r"_No\d+$", "", parts[-1])
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return [f"DB1:node{parts[0]}"] + [f"DB1:lk{p}" for p in parts[1:]]
    elif mof_id.startswith("DB5-"):
        m = DB5_RE.match(mof_id)
        if m:
            uid, f2, f3, f4, f5, f6, f7 = m.groups()
            # uid (field1) is unique per row (cardinality ratio 1.0) -> excluded.
            return [f"DB5:f2={f2}", f"DB5:f3={f3}", f"DB5:f4={f4}",
                    f"DB5:f5={f5}", f"DB5:f6={f6}", f"DB5:f7={f7}"]
    return []

def hdr(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")

def clip_hoa(s):
    mu, std = s.mean(), s.std()
    return s.clip(mu - HOA_SIGMA * std, mu + HOA_SIGMA * std)

def prepare_y(df, col):
    s = df[col].copy()
    if col == "heat_of_ads":
        s = clip_hoa(s)
    if TARGET_COLS[col]["log"]:
        s = np.log1p(s)
    return s.values

def make_strat(series):
    return pd.qcut(series, q=N_BINS, labels=False, duplicates="drop").to_numpy()

def get_feat_cols(df):
    excl = set(TARGET_COLS.keys()) | NON_FEAT
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]

# ---------------------------------------------------------------------------
hdr("Load full_features.parquet")
df = pd.read_parquet(DATA / "full_features.parquet")
before = len(df)
df = df.dropna(subset=list(TARGET_COLS.keys())).reset_index(drop=True)
print(f"Rows kept: {len(df):,} / {before:,}")

feat_cols = get_feat_cols(df)
print(f"Feature cols: {len(feat_cols)}")

hdr("Build building-block groups (union-find)")
uf = UnionFind()
decodable_count = 0
prefix_counts = defaultdict(int)
for mid in df["mof_id"]:
    prefix_counts[mid.split("-")[0]] += 1
    blocks = extract_blocks(mid)
    if blocks:
        decodable_count += 1
        for i in range(1, len(blocks)):
            uf.union(blocks[0], blocks[i])
        # tie this row to its first block via a row-specific marker
        uf.union(mid, blocks[0])
    else:
        uf.parent.setdefault(mid, mid)  # singleton group

group_ids = np.array([uf.find(mid) for mid in df["mof_id"]])
n_groups = len(set(group_ids))
print(f"Rows with decodable building blocks: {decodable_count:,} / {len(df):,} "
      f"({100*decodable_count/len(df):.1f}%)")
print(f"Total groups after union-find: {n_groups:,} (vs {len(df):,} rows)")

group_sizes = pd.Series(group_ids).value_counts()
summary_lines = [
    f"Rows: {len(df):,}",
    f"Decodable (grouped) rows: {decodable_count:,} ({100*decodable_count/len(df):.1f}%)",
    f"Total groups: {n_groups:,}",
    f"Largest group size: {group_sizes.max():,}",
    f"Groups with size > 1: {(group_sizes > 1).sum():,}",
    "",
    "Prefix counts (all sources):",
] + [f"  {k}: {v:,}" for k, v in sorted(prefix_counts.items(), key=lambda x: -x[1])]
Path(OUT_TXT).write_text("\n".join(summary_lines), encoding="utf-8")
print("\n".join(summary_lines))

# ---------------------------------------------------------------------------
hdr("Random 90/10 split (existing baseline, for direct comparison)")
strat = make_strat(df["co2_uptake_mmol_g"])
rand_tr, rand_te = train_test_split(
    np.arange(len(df)), test_size=0.10, stratify=strat, random_state=SEED
)
print(f"Random split  -> train {len(rand_tr):,}  test {len(rand_te):,}")

hdr("Group-aware 90/10 split (GroupShuffleSplit, seed 42)")
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
grp_tr, grp_te = next(gss.split(np.arange(len(df)), groups=group_ids))
print(f"Group split   -> train {len(grp_tr):,}  test {len(grp_te):,}")
overlap = set(group_ids[grp_tr]) & set(group_ids[grp_te])
print(f"Group overlap between grouped train/test: {len(overlap)} (must be 0)")
assert len(overlap) == 0, "GroupShuffleSplit leaked a group across the split!"

# ---------------------------------------------------------------------------
X_all = df[feat_cols].values.astype(np.float32)
rows = []
t_start = time.time()
for split_name, (tr_idx, te_idx) in [("random_split", (rand_tr, rand_te)),
                                       ("group_split", (grp_tr, grp_te))]:
    hdr(f"Split: {split_name}")
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    for col in TARGET_COLS:
        y_all = prepare_y(df, col)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        t0 = time.time()
        model = xgb.XGBRegressor(**HPARAMS)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_te)
        r2, mae = r2_score(y_te, pred), mean_absolute_error(y_te, pred)
        dt = time.time() - t0
        print(f"  {TARGET_COLS[col]['label']:12s} R2={r2:.4f}  MAE={mae:.4f}  ({dt:.0f}s)")
        rows.append({"split": split_name, "target": TARGET_COLS[col]["label"],
                      "R2": r2, "MAE": mae, "n_train": len(tr_idx), "n_test": len(te_idx)})

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print(f"Saved: {OUT_CSV}")
print(out_df.pivot(index="split", columns="target", values="R2"))
