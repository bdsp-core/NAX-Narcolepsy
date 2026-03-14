#!/usr/bin/env python3
"""
Verify key numbers stated in the NAX-Narcolepsy manuscript against actual data
and model outputs.

This script loads parquet files, pickle files, and CSV results, then compares
computed values against what is reported in the manuscript. Each check prints
PASS or FAIL with the expected and actual values.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
DISC_DIR = os.path.join(DATA_DIR, "discriminative-modeling")
PRED_DIR = os.path.join(DATA_DIR, "predictive-modeling")
RISK_DIR = os.path.join(REPO, "predictive-modeling", "risk_score_v2")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
passed = 0
failed = 0
skipped = 0


def check(label, expected, actual, tol=0):
    """Compare expected vs actual, print PASS/FAIL."""
    global passed, failed
    if isinstance(expected, float) and isinstance(actual, float):
        ok = abs(expected - actual) <= tol
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         Expected: {expected}")
        print(f"         Actual:   {actual}")
    return ok


def check_float(label, expected, actual, tol=0.002):
    """Compare floats with tolerance."""
    return check(label, expected, actual, tol=tol)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _derive_site(patient_id):
    """Derive hospital site from patient ID prefix."""
    s = str(patient_id)
    if s.startswith(('111', '112', '113', '114', '115',
                     '116', '117', '118', '119', '120', '121', '122')):
        return 'MGH'
    elif s.startswith(('150', '151')):
        return 'BIDMC'
    elif s.startswith('175'):
        return 'BCH'
    elif s.startswith('177'):
        return 'Stanford'
    elif s.startswith('179'):
        return 'Emory'
    return 'unknown'


# ===================================================================
# 1. Cross-sectional cohort counts
# ===================================================================
section("1. Cross-Sectional Cohort Counts")

notes = pd.read_parquet(os.path.join(DISC_DIR, "notes.parquet"))
# Rename site label: MGB → MGH
if "cohort" in notes.columns:
    notes["cohort"] = notes["cohort"].replace("MGB", "MGH")
# Use 'id' column if present, fall back to 'bdsp_patient_id'
id_col = "id" if "id" in notes.columns else "bdsp_patient_id"
n_patients = notes[id_col].nunique()
n_notes = len(notes)
n_usable = len(notes[notes["annot"] != 3])  # annot=3 is Unclear

check("Total patients = 6,492", 6492, n_patients)
check("Total notes = 9,356", 9356, n_notes)
check("Usable notes (excl Unclear) = 8,937", 8937, n_usable)

# ===================================================================
# 2. Annotation counts
# ===================================================================
section("2. Annotation Counts")

annot_counts = notes["annot"].value_counts().to_dict()
# annot mapping: 1=NT1, 2=NT2/IH, 3=Unclear, 4=Absent
check("NT1 annotations = 838", 838, annot_counts.get(1, 0))
check("NT2/IH annotations = 550", 550, annot_counts.get(2, 0))
check("Absent annotations = 7,549", 7549, annot_counts.get(4, 0))
check("Unclear annotations = 419", 419, annot_counts.get(3, 0))

# Per-site annotation counts (Table 2B)
section("2B. Annotation Counts by Site")
expected_site = {
    "BCH":      {"NT1": 237, "NT2/IH": 55,  "Unclear": 83,  "Absent": 1506, "Total": 1881},
    "BIDMC":    {"NT1": 349, "NT2/IH": 198, "Unclear": 130, "Absent": 1433, "Total": 2110},
    "Emory":    {"NT1": 65,  "NT2/IH": 101, "Unclear": 47,  "Absent": 1628, "Total": 1841},
    "MGH":      {"NT1": 119, "NT2/IH": 92,  "Unclear": 117, "Absent": 1633, "Total": 1961},
    "Stanford": {"NT1": 68,  "NT2/IH": 104, "Unclear": 42,  "Absent": 1349, "Total": 1563},
}
annot_map = {1: "NT1", 2: "NT2/IH", 3: "Unclear", 4: "Absent"}
for site, expected_vals in expected_site.items():
    site_df = notes[notes["cohort"] == site]
    site_counts = site_df["annot"].value_counts().to_dict()
    for annot_code, annot_name in annot_map.items():
        actual = site_counts.get(annot_code, 0)
        check(f"{site} {annot_name} = {expected_vals[annot_name]}", expected_vals[annot_name], actual)
    check(f"{site} Total = {expected_vals['Total']}", expected_vals["Total"], len(site_df))

# ===================================================================
# 3. Longitudinal cohort initial counts
# ===================================================================
section("3. Longitudinal Cohort - Initial Counts")

# Load new 3-file format parquets (metadata only for memory efficiency)
META_COLS_LOAD = ['bdsp_patient_id', 'id', 'site', 'cohort', 'n+_state',
                  'days_since_first_visit', 'num_visits_since_first_visit']


def _load_task_parquets(task_dir, columns=None):
    """Load n+ and controls parquets for a task (n- excluded); rename id->bdsp_patient_id, derive site."""
    parts = []
    for fname in ['n+_features_3.parquet',
                   'controls_features_3.parquet']:
        fpath = os.path.join(PRED_DIR, task_dir, fname)
        if os.path.exists(fpath):
            # Read only requested columns if specified (for memory efficiency)
            if columns is not None:
                import pyarrow.parquet as pq
                schema = pq.read_schema(fpath)
                avail = [c for c in columns if c in schema.names]
                dfp = pd.read_parquet(fpath, columns=avail)
            else:
                dfp = pd.read_parquet(fpath)
            if 'id' in dfp.columns and 'bdsp_patient_id' not in dfp.columns:
                dfp.rename(columns={'id': 'bdsp_patient_id'}, inplace=True)
            parts.append(dfp)
    df = pd.concat(parts, ignore_index=True)
    if 'site' not in df.columns:
        df['site'] = df['bdsp_patient_id'].apply(_derive_site)
    return df


# Load metadata-only versions for cohort counting (memory efficient)
df_nt1_meta = _load_task_parquets("nt1", columns=META_COLS_LOAD)
df_nt2_meta = _load_task_parquets("nt2ih", columns=META_COLS_LOAD)

nt1_label = df_nt1_meta.groupby("bdsp_patient_id")["n+_state"].max()
nt1_case_ids = set(nt1_label[nt1_label == 1].index)
nt1_noncase_ids = set(nt1_label[nt1_label == 0].index)

nt2_label = df_nt2_meta.groupby("bdsp_patient_id")["n+_state"].max()
nt2_case_ids = set(nt2_label[nt2_label == 1].index)
nt2_noncase_ids = set(nt2_label[nt2_label == 0].index)

# Build combined dataframe (metadata only - mirrors load_all_data)
keep_cols_meta = ["bdsp_patient_id", "site", "n+_state",
                  "days_since_first_visit", "case_type"]

df_nt1_cases = df_nt1_meta[df_nt1_meta["bdsp_patient_id"].isin(nt1_case_ids)].copy()
df_nt1_cases["case_type"] = "nt1"
df_nt1_noncases = df_nt1_meta[df_nt1_meta["bdsp_patient_id"].isin(nt1_noncase_ids)].copy()
df_nt1_noncases["case_type"] = "control"
df_nt2_cases = df_nt2_meta[df_nt2_meta["bdsp_patient_id"].isin(nt2_case_ids)].copy()
df_nt2_cases["case_type"] = "nt2ih"
df_nt2_noncases = df_nt2_meta[df_nt2_meta["bdsp_patient_id"].isin(nt2_noncase_ids)].copy()
df_nt2_noncases["case_type"] = "control"

dfs = []
for df_part in [df_nt1_cases, df_nt1_noncases, df_nt2_cases, df_nt2_noncases]:
    cols = [c for c in keep_cols_meta if c in df_part.columns]
    dfs.append(df_part[cols])
df_combined = pd.concat(dfs, axis=0, ignore_index=True)
df_combined = df_combined.sort_values(
    ["bdsp_patient_id", "days_since_first_visit"]).reset_index(drop=True)

all_patient_ids = set(df_combined["bdsp_patient_id"].unique())
all_case_ids = nt1_case_ids | nt2_case_ids
all_ctrl_ids = all_patient_ids - all_case_ids

check("Initial patients = 10,401", 10401, len(all_patient_ids))
check("Initial visits = 1,308,247", 1308247, len(df_combined))
check("Total cases = 543", 543, len(all_case_ids))
check("NT1 cases = 266", 266, len(nt1_case_ids))
check("NT2/IH cases = 277", 277, len(nt2_case_ids))
check("Controls = 9,858", 9858, len(all_ctrl_ids))

# ===================================================================
# 4. Longitudinal filtering (gap exclusion)
# ===================================================================
section("4. Longitudinal Cohort - After Gap Exclusion (cases only)")

# Replicate gap detection from risk_score_v2.py
# Gap exclusion applies only to cases (controls kept regardless)
pat_info = {}
for sid, grp in df_combined.groupby("bdsp_patient_id"):
    t_vals = grp["days_since_first_visit"].values
    y_vals = grp["n+_state"].values
    case_type = grp["case_type"].iloc[0]
    has_event = y_vals.max() == 1
    has_gap = len(t_vals) >= 2 and np.any(np.diff(t_vals) / 365.25 >= 5)
    pat_info[sid] = {
        "has_event": has_event,
        "has_gap": has_gap,
        "case_type": case_type,
    }

exclude_gap = {sid for sid, info in pat_info.items()
               if info["has_gap"] and info["has_event"]}
remaining_ids = all_patient_ids - exclude_gap
df_after_gap = df_combined[~df_combined["bdsp_patient_id"].isin(exclude_gap)]

cases_after_gap = remaining_ids & all_case_ids
nt1_after_gap = remaining_ids & nt1_case_ids
nt2ih_after_gap = remaining_ids & nt2_case_ids
ctrls_after_gap = remaining_ids - all_case_ids

check("After gap excl: patients = 10,346", 10346, len(remaining_ids))
check("After gap excl: visits = 1,299,801", 1299801, len(df_after_gap))
check("After gap excl: cases (any narcolepsy) = 488", 488, len(cases_after_gap))
check("After gap excl: NT1 cases = 241", 241, len(nt1_after_gap))
check("After gap excl: NT2/IH cases = 247", 247, len(nt2ih_after_gap))
check("After gap excl: controls = 9,858", 9858, len(ctrls_after_gap))

# ===================================================================
# 5. Predictive model performance (from pickle files)
# ===================================================================
section("5. Predictive Model Performance")

for outcome, pickle_name in [("any_narcolepsy", "v2_results_any_narcolepsy.pickle"),
                               ("nt1", "v2_results_nt1.pickle"),
                               ("nt2ih", "v2_results_nt2ih.pickle")]:
    pickle_path = os.path.join(RISK_DIR, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"  [SKIP] {pickle_path} not found")
        skipped += 1
        continue

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    r = data["results"][0.5]

    # 5-fold CV (pooled) performance
    pooled_perf = r["pooled"]["perf"]
    mean_auc_cv = pooled_perf["AUC"].mean()
    mean_auprc_cv = pooled_perf["AUPRC"].mean()

    # LOSO performance
    loso_perf = r["loso"]["perf"]
    mean_auc_loso = loso_perf["AUC"].mean()
    mean_auprc_loso = loso_perf["AUPRC"].mean()

    print(f"\n  --- {outcome} ---")
    expected_perf = {
        "any_narcolepsy": {"cv_auc": 0.842, "cv_auprc": 0.496,
                           "loso_auc": 0.822, "loso_auprc": 0.473},
        "nt1":            {"cv_auc": 0.780, "cv_auprc": 0.310,
                           "loso_auc": 0.763, "loso_auprc": 0.213},
        "nt2ih":          {"cv_auc": 0.818, "cv_auprc": 0.388,
                           "loso_auc": 0.726, "loso_auprc": 0.142},
    }
    exp = expected_perf[outcome]
    check_float(f"{outcome} 5-fold CV AUC = {exp['cv_auc']}", exp["cv_auc"], mean_auc_cv)
    check_float(f"{outcome} 5-fold CV AUPRC = {exp['cv_auprc']}", exp["cv_auprc"], mean_auprc_cv)
    check_float(f"{outcome} LOSO AUC = {exp['loso_auc']}", exp["loso_auc"], mean_auc_loso)
    check_float(f"{outcome} LOSO AUPRC = {exp['loso_auprc']}", exp["loso_auprc"], mean_auprc_loso)

    # Print actual values for reference
    print(f"         Actual 5-fold CV AUC:   {mean_auc_cv:.3f}")
    print(f"         Actual 5-fold CV AUPRC: {mean_auprc_cv:.3f}")
    print(f"         Actual LOSO AUC:        {mean_auc_loso:.3f}")
    print(f"         Actual LOSO AUPRC:      {mean_auprc_loso:.3f}")

# ===================================================================
# 6. Feature counts (non-zero L1 coefficients in final model)
# ===================================================================
section("6. Non-Zero L1 Feature Counts")

for outcome, pickle_name, expected_count in [
    ("any_narcolepsy", "v2_results_any_narcolepsy.pickle", 82),
    ("nt1", "v2_results_nt1.pickle", 70),
    ("nt2ih", "v2_results_nt2ih.pickle", 69),
]:
    pickle_path = os.path.join(RISK_DIR, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"  [SKIP] {pickle_path} not found")
        skipped += 1
        continue

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    clf = data["final_model"]["clf"]
    n_nonzero = int((clf.coef_ != 0).sum())
    check(f"{outcome} non-zero L1 features = {expected_count}", expected_count, n_nonzero)

# ===================================================================
# 7. Feature heatmap patient counts
# ===================================================================
section("7. Feature Heatmap Patient Counts")

# Replicate the heatmap script's patient selection logic
MAX_YEARS = 2.5
MIN_VISITS = 5

# Build the merged dataframe using metadata-only DFs (memory efficient)
keep_cols_hm = ["bdsp_patient_id", "n+_state", "days_since_first_visit"]

df_hm = pd.concat([
    df_nt1_meta[[c for c in keep_cols_hm if c in df_nt1_meta.columns]],
    df_nt2_meta[[c for c in keep_cols_hm if c in df_nt2_meta.columns]],
], ignore_index=True)
df_hm = df_hm.drop_duplicates(subset=["bdsp_patient_id", "days_since_first_visit"])
df_hm = df_hm.sort_values(["bdsp_patient_id", "days_since_first_visit"]).reset_index(drop=True)

ctrl_ids_hm = (nt1_noncase_ids | nt2_noncase_ids) - (nt1_case_ids | nt2_case_ids)


def filter_by_min_visits_hm(df_grouped, patient_ids, ref_times, max_years, min_visits):
    """Keep only patients with >= min_visits in the [-max_years, 0] window.
    Uses pre-grouped data for efficiency."""
    kept = {}
    for sid in patient_ids:
        if sid not in ref_times:
            continue
        ref_t = ref_times[sid]
        if sid not in df_grouped.groups:
            continue
        t_vals = df_grouped.get_group(sid)["days_since_first_visit"].values
        t_rel = (t_vals - ref_t) / 365.25
        n_in_window = int(((t_rel >= -max_years) & (t_rel <= 0)).sum())
        if n_in_window >= min_visits:
            kept[sid] = ref_t
    return kept


# Pre-group for efficient lookups
df_hm_grouped = df_hm.groupby("bdsp_patient_id")

for outcome, expected_cases, expected_ctrls in [
    ("any_narcolepsy", 232, 232),
    ("nt1", 88, 88),
]:
    if outcome == "any_narcolepsy":
        case_ids_out = nt1_case_ids | nt2_case_ids
    else:
        case_ids_out = nt1_case_ids

    # Compute reference times for cases (vectorized)
    case_diag_t = {}
    pos_df = df_hm[(df_hm["bdsp_patient_id"].isin(case_ids_out)) & (df_hm["n+_state"] == 1)]
    first_pos = pos_df.groupby("bdsp_patient_id")["days_since_first_visit"].min()
    case_diag_t = first_pos.to_dict()

    # Compute pseudo-reference for controls (vectorized)
    ctrl_df = df_hm[df_hm["bdsp_patient_id"].isin(ctrl_ids_hm)]
    ctrl_pseudo_t = ctrl_df.groupby("bdsp_patient_id")["days_since_first_visit"].max().to_dict()

    # Filter by min visits
    case_diag_t = filter_by_min_visits_hm(df_hm_grouped, case_ids_out, case_diag_t,
                                           MAX_YEARS, MIN_VISITS)
    ctrl_pseudo_t_filtered = filter_by_min_visits_hm(df_hm_grouped, ctrl_ids_hm, ctrl_pseudo_t,
                                                      MAX_YEARS, MIN_VISITS)

    n_cases_hm = len(case_diag_t)

    # Match controls to case count (same RNG as feature_heatmap.py)
    rng = np.random.RandomState(42)
    ctrl_list = list(ctrl_pseudo_t_filtered.keys())
    if len(ctrl_list) > n_cases_hm:
        ctrl_sample = list(rng.choice(ctrl_list, size=n_cases_hm, replace=False))
    else:
        ctrl_sample = ctrl_list
    n_ctrls_hm = len(ctrl_sample)

    check(f"{outcome} heatmap cases = {expected_cases}", expected_cases, n_cases_hm)
    check(f"{outcome} heatmap controls = {expected_ctrls}", expected_ctrls, n_ctrls_hm)

# ===================================================================
# 8. NNT analysis operating points
# ===================================================================
section("8. NNT Analysis Operating Points")

PREVALENCE = 0.0008

for outcome, pickle_name in [("any_narcolepsy", "v2_results_any_narcolepsy.pickle"),
                               ("nt1", "v2_results_nt1.pickle")]:
    pickle_path = os.path.join(RISK_DIR, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"  [SKIP] {pickle_path} not found")
        skipped += 1
        continue

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    r = data["results"][0.5]
    # Use trajectory data (traj_ = patient-level mean scores)
    traj_scores = r["pooled"]["traj_scores"]
    traj_sids = r["pooled"]["traj_sids"]
    traj_y = r["pooled"]["traj_y"]
    traj_T = r["pooled"]["traj_T"]

    # Aggregate to patient level
    pat_df = pd.DataFrame({
        "sid": traj_sids, "score": traj_scores, "y": traj_y, "T": traj_T
    })
    pat = pat_df.groupby("sid").agg(score=("score", "mean"), y=("y", "max")).reset_index()

    case_scores = pat[pat["y"] == 1]["score"].values
    ctrl_scores = pat[pat["y"] == 0]["score"].values

    thresholds = np.linspace(0.005, 0.995, 500)
    sens = np.array([np.mean(case_scores >= t) for t in thresholds])
    spec = np.array([np.mean(ctrl_scores < t) for t in thresholds])

    ppv = (sens * PREVALENCE) / (
        sens * PREVALENCE + (1 - spec) * (1 - PREVALENCE))
    nnt = np.where(ppv > 0, 1.0 / ppv, np.nan)

    print(f"\n  --- {outcome} ---")

    # Find operating points for NNT = 10 and NNT = 20
    for target_nnt in [10, 20]:
        valid = ~np.isnan(nnt) & np.isfinite(nnt) & (nnt > 0)
        valid_idx = np.where(valid)[0]
        nnt_valid = nnt[valid_idx]
        closest = valid_idx[np.argmin(np.abs(nnt_valid - target_nnt))]
        t_val = thresholds[closest]
        s_val = sens[closest]
        n_val = nnt[closest]
        print(f"    NNT~{target_nnt}: threshold={t_val:.2f}, NNT={n_val:.0f}, "
              f"sensitivity={s_val:.0%}")

    # Verify specific manuscript claims
    if outcome == "any_narcolepsy":
        # NNT~20 threshold~0.98 sens~70%, NNT~10 threshold~0.99 sens~69%
        for target_nnt, exp_thr, exp_sens in [(20, 0.98, 0.70), (10, 0.99, 0.69)]:
            valid = ~np.isnan(nnt) & np.isfinite(nnt) & (nnt > 0)
            valid_idx = np.where(valid)[0]
            nnt_valid = nnt[valid_idx]
            closest = valid_idx[np.argmin(np.abs(nnt_valid - target_nnt))]
            t_val = thresholds[closest]
            s_val = sens[closest]
            n_val = nnt[closest]
            check_float(f"{outcome} NNT={target_nnt} threshold ~ {exp_thr}",
                        exp_thr, t_val, tol=0.03)
            check_float(f"{outcome} NNT={target_nnt} sensitivity ~ {exp_sens:.0%}",
                        exp_sens, s_val, tol=0.03)

    elif outcome == "nt1":
        # NNT~20 threshold~0.93 sens~81%, NNT~10 threshold~0.98 sens~80%
        for target_nnt, exp_thr, exp_sens in [(20, 0.93, 0.81), (10, 0.98, 0.80)]:
            valid = ~np.isnan(nnt) & np.isfinite(nnt) & (nnt > 0)
            valid_idx = np.where(valid)[0]
            nnt_valid = nnt[valid_idx]
            closest = valid_idx[np.argmin(np.abs(nnt_valid - target_nnt))]
            t_val = thresholds[closest]
            s_val = sens[closest]
            n_val = nnt[closest]
            check_float(f"{outcome} NNT={target_nnt} threshold ~ {exp_thr}",
                        exp_thr, t_val, tol=0.05)
            check_float(f"{outcome} NNT={target_nnt} sensitivity ~ {exp_sens:.0%}",
                        exp_sens, s_val, tol=0.05)

# ===================================================================
# 9. LOSO per-site case counts and performance (Supplementary Table 1)
# ===================================================================
section("9. LOSO Per-Site Case Counts and Performance (Supp Table 1)")

for outcome, pickle_name in [("any_narcolepsy", "v2_results_any_narcolepsy.pickle"),
                               ("nt1", "v2_results_nt1.pickle"),
                               ("nt2ih", "v2_results_nt2ih.pickle")]:
    pickle_path = os.path.join(RISK_DIR, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"  [SKIP] {pickle_path} not found")
        skipped += 1
        continue

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    loso_perf = data["results"][0.5]["loso"]["perf"]
    print(f"\n  --- {outcome} ---")

    # Expected values - LOSO restricted to BIDMC and MGH only (sites with >=50 controls)
    all_expected_sites = {
        "any_narcolepsy": {
            "BIDMC":    {"N_diag": 44, "N_ctrl": 4976, "AUC": 0.918, "AUPRC": 0.734},
            "MGH":      {"N_diag": 54, "N_ctrl": 4882, "AUC": 0.725, "AUPRC": 0.211},
        },
        "nt1": {
            "BIDMC":    {"N_diag": 11, "N_ctrl": 4976, "AUC": 0.775, "AUPRC": 0.152},
            "MGH":      {"N_diag": 26, "N_ctrl": 4882, "AUC": 0.751, "AUPRC": 0.275},
        },
        "nt2ih": {
            "BIDMC":    {"N_diag": 33, "N_ctrl": 4976, "AUC": 0.829, "AUPRC": 0.235},
            "MGH":      {"N_diag": 28, "N_ctrl": 4882, "AUC": 0.623, "AUPRC": 0.050},
        },
    }
    expected_sites = all_expected_sites[outcome]

    for _, row in loso_perf.iterrows():
        site = row["site"]
        if site in expected_sites:
            exp = expected_sites[site]
            check(f"{outcome} {site} N_diag = {exp['N_diag']}",
                  exp["N_diag"], int(row["N_diag"]))
            check(f"{outcome} {site} N_ctrl = {exp['N_ctrl']}",
                  exp["N_ctrl"], int(row["N_ctrl"]))
            check_float(f"{outcome} {site} AUC = {exp['AUC']:.3f}",
                        exp["AUC"], row["AUC"], tol=0.002)
            check_float(f"{outcome} {site} AUPRC = {exp['AUPRC']:.3f}",
                        exp["AUPRC"], row["AUPRC"], tol=0.002)

    # Mean LOSO
    expected_means = {
        "any_narcolepsy": {"auc": 0.822, "auprc": 0.473},
        "nt1": {"auc": 0.763, "auprc": 0.213},
        "nt2ih": {"auc": 0.726, "auprc": 0.142},
    }
    mean_auc = loso_perf["AUC"].mean()
    mean_auprc = loso_perf["AUPRC"].mean()
    em = expected_means[outcome]
    check_float(f"{outcome} mean LOSO AUC = {em['auc']}", em["auc"], mean_auc)
    check_float(f"{outcome} mean LOSO AUPRC = {em['auprc']}", em["auprc"], mean_auprc)

# ===================================================================
# 10. Demographics (Table 1A) - from cross-sectional notes
# ===================================================================
section("10. Demographics (Table 1A)")

# NOTE: Demographics (sex, race, age) are NOT stored in the notes.parquet.
# The manuscript demographics were computed from an external patient
# demographics table not included in the data directory.
# We verify what we can: patient and note counts per site.

print("  [INFO] Sex, race, and age demographics are not available in the")
print("         parquet files shipped with the repo. Those numbers were")
print("         derived from a BDSP demographics table not present here.")
print("         Verifying site-level patient and note counts instead.\n")

expected_site_patients = {
    "BIDMC": 1549, "Stanford": 1454, "Emory": 1292, "BCH": 1138, "MGH": 1059,
}
expected_site_notes = {
    "BIDMC": 2110, "Stanford": 1563, "Emory": 1841, "BCH": 1881, "MGH": 1961,
}

for site in expected_site_patients:
    site_df = notes[notes["cohort"] == site]
    n_pts = site_df[id_col].nunique()
    n_nts = len(site_df)
    check(f"{site} patients = {expected_site_patients[site]}",
          expected_site_patients[site], n_pts)
    check(f"{site} notes = {expected_site_notes[site]}",
          expected_site_notes[site], n_nts)

# ===================================================================
# Cross-sectional model performance (Tables 4 & 5)
# ===================================================================
section("BONUS: Cross-Sectional Best Model AUROC (from per_fold_results)")

for task, best_model, exp_auroc, exp_auprc in [
    ("nt1_vs_others", "RandomForest", 0.997, 0.956),
    ("nt2ih_vs_others", "XGBoost", 0.988, 0.854),
    ("any_narcolepsy_vs_others", "GradientBoosting", 0.990, 0.948),
]:
    fold_path = os.path.join(RESULTS_DIR, task, "per_fold_results.csv")
    if not os.path.exists(fold_path):
        print(f"  [SKIP] {fold_path} not found")
        skipped += 1
        continue

    fold_df = pd.read_csv(fold_path)
    rf = fold_df[fold_df["model"] == best_model]
    mean_auroc = rf["roc_auc"].mean()
    mean_auprc = rf["auprc"].mean()
    print(f"\n  --- {task} ({best_model}) ---")
    check_float(f"{task} {best_model} AUROC = {exp_auroc:.3f}", exp_auroc, mean_auroc)
    check_float(f"{task} {best_model} AUPRC = {exp_auprc:.3f}", exp_auprc, mean_auprc)


# ===================================================================
# Summary
# ===================================================================
section("SUMMARY")
total = passed + failed
print(f"  Passed: {passed}/{total}")
print(f"  Failed: {failed}/{total}")
if skipped:
    print(f"  Skipped: {skipped}")
if failed == 0:
    print("\n  All checks PASSED.")
else:
    print(f"\n  {failed} check(s) FAILED -- review above.")
    sys.exit(1)
