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
FEAT_UPDATE_DIR = os.path.join(REPO, "predictive-modeling", "features_update")
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


# ===================================================================
# 1. Cross-sectional cohort counts
# ===================================================================
section("1. Cross-Sectional Cohort Counts")

notes = pd.read_parquet(os.path.join(DISC_DIR, "notes.parquet"))
n_patients = notes["bdsp_patient_id"].nunique()
n_notes = len(notes)
n_usable = len(notes[notes["annot"] != 3])  # annot=3 is Unclear

check("Total patients = 6,498", 6498, n_patients)
check("Total notes = 8,990", 8990, n_notes)
check("Usable notes (excl Unclear) = 8,694", 8694, n_usable)

# ===================================================================
# 2. Annotation counts
# ===================================================================
section("2. Annotation Counts")

annot_counts = notes["annot"].value_counts().to_dict()
# annot mapping: 1=NT1, 2=NT2/IH, 3=Unclear, 4=Absent
check("NT1 annotations = 620", 620, annot_counts.get(1, 0))
check("NT2/IH annotations = 360", 360, annot_counts.get(2, 0))
check("Absent annotations = 7,714", 7714, annot_counts.get(4, 0))
check("Unclear annotations = 296", 296, annot_counts.get(3, 0))

# Per-site annotation counts (Table 2B)
section("2B. Annotation Counts by Site")
expected_site = {
    "bch":   {"NT1": 194, "NT2/IH": 46,  "Unclear": 74, "Absent": 1563, "Total": 1877},
    "bidmc": {"NT1": 265, "NT2/IH": 126, "Unclear": 77, "Absent": 1453, "Total": 1921},
    "emory": {"NT1": 56,  "NT2/IH": 71,  "Unclear": 33, "Absent": 1698, "Total": 1858},
    "mgb":   {"NT1": 77,  "NT2/IH": 61,  "Unclear": 73, "Absent": 1646, "Total": 1857},
    "stan":  {"NT1": 28,  "NT2/IH": 56,  "Unclear": 39, "Absent": 1354, "Total": 1477},
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

# Load features_update parquets (same as risk_score_v2.py load_all_data)
df_nt1 = pd.read_parquet(os.path.join(FEAT_UPDATE_DIR, "nt1", "features_3.parquet"))
df_nt2 = pd.read_parquet(os.path.join(FEAT_UPDATE_DIR, "nt2ih", "features_3.parquet"))
df_nt1 = df_nt1.rename(columns={"cohort": "site"})
df_nt2 = df_nt2.rename(columns={"cohort": "site"})

nt1_label = df_nt1.groupby("bdsp_patient_id")["n+_state"].max()
nt1_case_ids = set(nt1_label[nt1_label == 1].index)
nt1_noncase_ids = set(nt1_label[nt1_label == 0].index)

nt2_label = df_nt2.groupby("bdsp_patient_id")["n+_state"].max()
nt2_case_ids = set(nt2_label[nt2_label == 1].index)
nt2_noncase_ids = set(nt2_label[nt2_label == 0].index)

# Build combined dataframe (mirrors load_all_data)
meta_cols = {"bdsp_patient_id", "site", "cohort", "date", "n+_state",
             "days_since_first_visit", "num_visits_since_first_visit",
             "case_type", "filename"}
feat_cols = sorted(set(df_nt1.columns) - meta_cols)
keep_cols = ["bdsp_patient_id", "site", "n+_state",
             "days_since_first_visit", "case_type"] + feat_cols

df_nt1_cases = df_nt1[df_nt1["bdsp_patient_id"].isin(nt1_case_ids)].copy()
df_nt1_cases["case_type"] = "nt1"
df_nt1_noncases = df_nt1[df_nt1["bdsp_patient_id"].isin(nt1_noncase_ids)].copy()
df_nt1_noncases["case_type"] = "control"
df_nt2_cases = df_nt2[df_nt2["bdsp_patient_id"].isin(nt2_case_ids)].copy()
df_nt2_cases["case_type"] = "nt2ih"
df_nt2_noncases = df_nt2[df_nt2["bdsp_patient_id"].isin(nt2_noncase_ids)].copy()
df_nt2_noncases["case_type"] = "control"

dfs = []
for df_part in [df_nt1_cases, df_nt1_noncases, df_nt2_cases, df_nt2_noncases]:
    cols = [c for c in keep_cols if c in df_part.columns]
    dfs.append(df_part[cols])
df_combined = pd.concat(dfs, axis=0, ignore_index=True)
df_combined = df_combined.sort_values(
    ["bdsp_patient_id", "days_since_first_visit"]).reset_index(drop=True)

all_patient_ids = set(df_combined["bdsp_patient_id"].unique())
all_case_ids = nt1_case_ids | nt2_case_ids
all_ctrl_ids = all_patient_ids - all_case_ids

check("Initial patients = 13,342", 13342, len(all_patient_ids))
check("Initial visits = 1,022,458", 1022458, len(df_combined))
check("Total cases = 596", 596, len(all_case_ids))
check("NT1 cases = 282", 282, len(nt1_case_ids))
check("NT2/IH cases = 314", 314, len(nt2_case_ids))
check("Controls = 12,746", 12746, len(all_ctrl_ids))

# ===================================================================
# 4. Longitudinal filtering (gap exclusion)
# ===================================================================
section("4. Longitudinal Cohort - After Gap Exclusion")

# Replicate gap detection from risk_score_v2.py
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

exclude_gap = {sid for sid, info in pat_info.items() if info["has_gap"]}
remaining_ids = all_patient_ids - exclude_gap
df_after_gap = df_combined[~df_combined["bdsp_patient_id"].isin(exclude_gap)]

cases_after_gap = remaining_ids & all_case_ids
nt1_after_gap = remaining_ids & nt1_case_ids
ctrls_after_gap = remaining_ids - all_case_ids

check("After gap excl: patients = 11,588", 11588, len(remaining_ids))
check("After gap excl: visits = 876,318", 876318, len(df_after_gap))
check("After gap excl: cases (any narcolepsy) = 539", 539, len(cases_after_gap))
check("After gap excl: NT1 cases = 258", 258, len(nt1_after_gap))
check("After gap excl: controls = 11,049", 11049, len(ctrls_after_gap))

# ===================================================================
# 5. Predictive model performance (from pickle files)
# ===================================================================
section("5. Predictive Model Performance")

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

    # 5-fold CV (pooled) performance
    pooled_perf = r["pooled"]["perf"]
    mean_auc_cv = pooled_perf["AUC"].mean()
    mean_auprc_cv = pooled_perf["AUPRC"].mean()

    # LOSO performance
    loso_perf = r["loso"]["perf"]
    mean_auc_loso = loso_perf["AUC"].mean()
    mean_auprc_loso = loso_perf["AUPRC"].mean()

    print(f"\n  --- {outcome} ---")
    if outcome == "any_narcolepsy":
        check_float(f"{outcome} 5-fold CV AUC = 0.835", 0.835, mean_auc_cv)
        check_float(f"{outcome} 5-fold CV AUPRC = 0.377", 0.377, mean_auprc_cv)
        check_float(f"{outcome} LOSO AUC = 0.797", 0.797, mean_auc_loso)
        check_float(f"{outcome} LOSO AUPRC = 0.428", 0.428, mean_auprc_loso)
    elif outcome == "nt1":
        check_float(f"{outcome} 5-fold CV AUC = 0.838", 0.838, mean_auc_cv)
        check_float(f"{outcome} 5-fold CV AUPRC = 0.298", 0.298, mean_auprc_cv)
        check_float(f"{outcome} LOSO AUC = 0.788", 0.788, mean_auc_loso)
        check_float(f"{outcome} LOSO AUPRC = 0.285", 0.285, mean_auprc_loso)

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
    ("nt1", "v2_results_nt1.pickle", 84),
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

# Build the merged dataframe the same way feature_heatmap.py does
meta_cols_hm = {"bdsp_patient_id", "site", "cohort", "date", "n+_state",
                "days_since_first_visit", "num_visits_since_first_visit",
                "filename"}
feat_cols_hm = sorted(set(df_nt1.columns) - meta_cols_hm)
keep_cols_hm = ["bdsp_patient_id", "n+_state", "days_since_first_visit"] + feat_cols_hm

df_hm = pd.concat([
    df_nt1[[c for c in keep_cols_hm if c in df_nt1.columns]],
    df_nt2[[c for c in keep_cols_hm if c in df_nt2.columns]],
], ignore_index=True)
df_hm = df_hm.drop_duplicates(subset=["bdsp_patient_id", "days_since_first_visit"])
df_hm = df_hm.sort_values(["bdsp_patient_id", "days_since_first_visit"]).reset_index(drop=True)

ctrl_ids_hm = (nt1_noncase_ids | nt2_noncase_ids) - (nt1_case_ids | nt2_case_ids)


def filter_by_min_visits_hm(df, patient_ids, ref_times, max_years, min_visits):
    """Keep only patients with >= min_visits in the [-max_years, 0] window."""
    kept = {}
    for sid in patient_ids:
        if sid not in ref_times:
            continue
        ref_t = ref_times[sid]
        sub = df[df["bdsp_patient_id"] == sid].sort_values("days_since_first_visit")
        t_rel = (sub["days_since_first_visit"].values - ref_t) / 365.25
        n_in_window = ((t_rel >= -max_years) & (t_rel <= 0)).sum()
        if n_in_window >= min_visits:
            kept[sid] = ref_t
    return kept


for outcome, expected_cases, expected_ctrls in [
    ("any_narcolepsy", 234, 234),
    ("nt1", 81, 81),
]:
    if outcome == "any_narcolepsy":
        case_ids_out = nt1_case_ids | nt2_case_ids
    else:
        case_ids_out = nt1_case_ids

    # Compute reference times for cases
    case_diag_t = {}
    for sid in case_ids_out:
        sub = df_hm[df_hm["bdsp_patient_id"] == sid].sort_values("days_since_first_visit")
        pos_visits = sub[sub["n+_state"] == 1]
        if len(pos_visits) > 0:
            case_diag_t[sid] = pos_visits["days_since_first_visit"].iloc[0]

    # Compute pseudo-reference for controls
    ctrl_pseudo_t = {}
    for sid in ctrl_ids_hm:
        sub = df_hm[df_hm["bdsp_patient_id"] == sid]
        if len(sub) > 0:
            ctrl_pseudo_t[sid] = sub["days_since_first_visit"].max()

    # Filter by min visits
    case_diag_t = filter_by_min_visits_hm(df_hm, case_ids_out, case_diag_t,
                                           MAX_YEARS, MIN_VISITS)
    ctrl_pseudo_t_filtered = filter_by_min_visits_hm(df_hm, ctrl_ids_hm, ctrl_pseudo_t,
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
        # threshold 0.95 -> NNT=20, sens 68%
        # threshold 0.99 -> NNT=10, sens 67%
        for target_nnt, exp_thr, exp_sens in [(20, 0.95, 0.68), (10, 0.99, 0.67)]:
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
        # threshold 0.85 -> NNT=20, sens 84%
        # threshold 0.95 -> NNT=10, sens 79%
        for target_nnt, exp_thr, exp_sens in [(20, 0.85, 0.84), (10, 0.95, 0.79)]:
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

# ===================================================================
# 9. LOSO per-site case counts and performance (Supplementary Table 1)
# ===================================================================
section("9. LOSO Per-Site Case Counts and Performance (Supp Table 1)")

for outcome, pickle_name in [("any_narcolepsy", "v2_results_any_narcolepsy.pickle"),
                               ("nt1", "v2_results_nt1.pickle")]:
    pickle_path = os.path.join(RISK_DIR, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"  [SKIP] {pickle_path} not found")
        skipped += 1
        continue

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    loso_perf = data["results"][0.5]["loso"]["perf"]
    print(f"\n  --- {outcome} ---")

    # Expected values from Supplementary Table 1
    if outcome == "any_narcolepsy":
        expected_sites = {
            "bch":   {"N_diag": 29, "N_ctrl": 605,  "AUC": 0.797, "AUPRC": 0.454},
            "bidmc": {"N_diag": 48, "N_ctrl": 5133, "AUC": 0.779, "AUPRC": 0.376},
            "emory": {"N_diag": 33, "N_ctrl": 483,  "AUC": 0.894, "AUPRC": 0.691},
            "mgb":   {"N_diag": 54, "N_ctrl": 4182, "AUC": 0.773, "AUPRC": 0.157},
            "stan":  {"N_diag": 32, "N_ctrl": 646,  "AUC": 0.740, "AUPRC": 0.462},
        }
    else:  # nt1
        expected_sites = {
            "bch":   {"N_diag": 16, "N_ctrl": 605,  "AUC": 0.941, "AUPRC": 0.311},
            "bidmc": {"N_diag": 13, "N_ctrl": 5133, "AUC": 0.828, "AUPRC": 0.321},
            "emory": {"N_diag": 9,  "N_ctrl": 483,  "AUC": 0.764, "AUPRC": 0.618},
            "mgb":   {"N_diag": 21, "N_ctrl": 4182, "AUC": 0.628, "AUPRC": 0.133},
            "stan":  {"N_diag": 7,  "N_ctrl": 646,  "AUC": 0.779, "AUPRC": 0.040},
        }

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
    mean_auc = loso_perf["AUC"].mean()
    mean_auprc = loso_perf["AUPRC"].mean()
    if outcome == "any_narcolepsy":
        check_float(f"{outcome} mean LOSO AUC = 0.797", 0.797, mean_auc)
        check_float(f"{outcome} mean LOSO AUPRC = 0.428", 0.428, mean_auprc)
    else:
        check_float(f"{outcome} mean LOSO AUC = 0.788", 0.788, mean_auc)
        check_float(f"{outcome} mean LOSO AUPRC = 0.285", 0.285, mean_auprc)

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
    "bidmc": 1549, "stan": 1454, "emory": 1294, "bch": 1141, "mgb": 1060,
}
expected_site_notes = {
    "bidmc": 1921, "stan": 1477, "emory": 1858, "bch": 1877, "mgb": 1857,
}

for site in expected_site_patients:
    site_df = notes[notes["cohort"] == site]
    n_pts = site_df["bdsp_patient_id"].nunique()
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
    ("nt1_vs_others", "RandomForest", 0.996, 0.935),
    ("nt2ih_vs_others", "XGBoost", 0.977, 0.676),
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
