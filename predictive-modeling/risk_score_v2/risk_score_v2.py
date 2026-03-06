"""
Narcolepsy risk score model — version 2

Key changes from v1:
1. Three outcomes: 'any_narcolepsy' (NT1 + NT2/IH combined), 'nt1', and 'nt2ih'
2. Case visits truncated to ≤2 years before diagnosis
3. Primary: pooled stratified 5-fold CV
4. Secondary: leave-one-site-out (LOSO) CV as sensitivity analysis
5. Tests h=0 vs h=0.5 exclusion horizon

Usage:
    python risk_score_v2.py any_narcolepsy
    python risk_score_v2.py nt1
    python risk_score_v2.py nt2ih
    python risk_score_v2.py all
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Apply shared publication style
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'paper_figures'))
from pub_style import (apply_style, CASE_COLOR, CTRL_COLOR, AUROC_COLOR,
                        NNT_COLOR, SENS_COLOR, FEAT_POS_COLOR, FEAT_NEG_COLOR,
                        BAR_COLORS, add_panel_label, savefig as pub_savefig,
                        LINE_WIDTH, LINE_WIDTH_THICK, REFERENCE_LINE_WIDTH,
                        FONT_SIZE_ANNOTATION, FONT_SIZE_LEGEND, FONT_SIZE_TITLE,
                        FONT_SIZE_SUPTITLE, DOUBLE_COL_IN, MAX_HEIGHT_IN)
apply_style()

MAX_VISITS_PER_PATIENT = 20
TOP_K_FEATURES = 100
MAX_YEARS_BEFORE_DIAG_TRAIN = 2.5  # training window: [-2.5yr, -0.5yr]
MAX_YEARS_BEFORE_DIAG_TEST  = 5.0  # testing/scoring window: [-5yr, 0yr]

MANUSCRIPT_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', '..', 'manuscript', 'figures')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """
    Load NT1 and NT2/IH parquets (cases AND non-cases from each).
    Non-case patients from both disease parquets serve as controls,
    giving every site adequate controls for LOSO analysis.
    Returns: df, feat_names, pat_info dict
    """
    base = os.path.dirname(os.path.abspath(__file__))
    # resolve features_update relative to repo root if running from risk_score_v2/
    feat_base = os.path.join(base, 'features_update')
    if not os.path.isdir(feat_base):
        feat_base = os.path.join(base, '..', 'features_update')
    print("Loading all data...")

    # --- NT1 parquet (cases + non-cases) ---
    df_nt1 = pd.read_parquet(os.path.join(feat_base, 'nt1/features_3.parquet'))
    df_nt1 = df_nt1.rename(columns={'cohort': 'site'}).drop(columns=['filename'], errors='ignore')
    nt1_label = df_nt1.groupby('bdsp_patient_id')['n+_state'].max()
    nt1_case_ids = set(nt1_label[nt1_label == 1].index)
    nt1_noncase_ids = set(nt1_label[nt1_label == 0].index)

    df_nt1_cases = df_nt1[df_nt1['bdsp_patient_id'].isin(nt1_case_ids)].copy()
    df_nt1_cases['case_type'] = 'nt1'
    df_nt1_noncases = df_nt1[df_nt1['bdsp_patient_id'].isin(nt1_noncase_ids)].copy()
    df_nt1_noncases['case_type'] = 'control'
    print(f"  NT1 parquet: {len(nt1_case_ids)} cases, "
          f"{len(nt1_noncase_ids)} non-cases")

    # --- NT2/IH parquet (cases + non-cases) ---
    df_nt2 = pd.read_parquet(os.path.join(feat_base, 'nt2ih/features_3.parquet'))
    df_nt2 = df_nt2.rename(columns={'cohort': 'site'}).drop(columns=['filename'], errors='ignore')
    nt2_label = df_nt2.groupby('bdsp_patient_id')['n+_state'].max()
    nt2_case_ids = set(nt2_label[nt2_label == 1].index)
    nt2_noncase_ids = set(nt2_label[nt2_label == 0].index)

    df_nt2_cases = df_nt2[df_nt2['bdsp_patient_id'].isin(nt2_case_ids)].copy()
    df_nt2_cases['case_type'] = 'nt2ih'
    df_nt2_noncases = df_nt2[df_nt2['bdsp_patient_id'].isin(nt2_noncase_ids)].copy()
    df_nt2_noncases['case_type'] = 'control'
    print(f"  NT2/IH parquet: {len(nt2_case_ids)} cases, "
          f"{len(nt2_noncase_ids)} non-cases")

    # --- Align feature columns ---
    meta_cols = {'bdsp_patient_id', 'site', 'cohort', 'date', 'n+_state',
                 'days_since_first_visit', 'num_visits_since_first_visit',
                 'case_type', 'filename'}
    feat_cols = sorted(set(df_nt1.columns) - meta_cols)
    for name, df_check in [('NT2/IH', df_nt2)]:
        missing = set(feat_cols) - set(df_check.columns)
        if missing:
            print(f"  Warning: {len(missing)} features missing from {name}")
            feat_cols = [f for f in feat_cols if f not in missing]

    keep_cols = ['bdsp_patient_id', 'site', 'n+_state',
                 'days_since_first_visit', 'case_type'] + feat_cols

    dfs = []
    for df_part in [df_nt1_cases, df_nt1_noncases, df_nt2_cases, df_nt2_noncases]:
        cols = [c for c in keep_cols if c in df_part.columns]
        dfs.append(df_part[cols])
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values(['bdsp_patient_id', 'days_since_first_visit']).reset_index(drop=True)

    n_ctrl = df[df['case_type'] == 'control']['bdsp_patient_id'].nunique()
    print(f"  Combined: {len(df)} rows, {df['bdsp_patient_id'].nunique()} patients "
          f"({len(nt1_case_ids)} NT1, {len(nt2_case_ids)} NT2/IH, {n_ctrl} controls)")

    # --- Patient-level metadata ---
    print("  Computing per-patient metadata...")
    pat_info = {}
    for sid, grp in df.groupby('bdsp_patient_id'):
        t_vals = grp['days_since_first_visit'].values
        y_vals = grp['n+_state'].values
        site = grp['site'].iloc[0]
        case_type = grp['case_type'].iloc[0]
        has_event = y_vals.max() == 1
        diag_t = None
        if has_event:
            diag_t = t_vals[np.where(y_vals == 1)[0][0]]
        has_gap = len(t_vals) >= 2 and np.any(np.diff(t_vals) / 365.25 >= 5)
        pat_info[sid] = {
            'site': site, 'has_event': has_event, 'diag_t': diag_t,
            'n_visits': len(t_vals), 'has_gap': has_gap, 'case_type': case_type,
        }

    # exclude patients with >5yr gaps
    exclude = {sid for sid, info in pat_info.items()
               if info['has_gap']}
    df = df[~df['bdsp_patient_id'].isin(exclude)].reset_index(drop=True)
    for sid in exclude:
        del pat_info[sid]
    print(f"  After exclusions: {len(df)} rows, {df['bdsp_patient_id'].nunique()} patients")

    # --- Subsample visits per patient ---
    print(f"  Subsampling to max {MAX_VISITS_PER_PATIENT} visits/patient...")
    rng = np.random.RandomState(42)
    keep_idx = []
    for sid, grp in df.groupby('bdsp_patient_id'):
        idx = grp.index.values
        if len(idx) <= MAX_VISITS_PER_PATIENT:
            keep_idx.extend(idx)
        else:
            middle = idx[1:-1]
            n_sample = MAX_VISITS_PER_PATIENT - 2
            sampled = rng.choice(middle, size=min(n_sample, len(middle)), replace=False)
            keep_idx.extend([idx[0]] + sorted(sampled) + [idx[-1]])
    df = df.loc[sorted(keep_idx)].reset_index(drop=True)
    print(f"  After subsampling: {len(df)} rows, {df['bdsp_patient_id'].nunique()} patients")

    # --- Remove sparse features ---
    X_all = df[feat_cols].values
    nonzero_counts = (np.abs(X_all) > 0).sum(axis=0)
    good_mask = nonzero_counts >= 50
    feat_names = [f for f, m in zip(feat_cols, good_mask) if m]
    print(f"  Features: {len(feat_cols)} -> {len(feat_names)} with >=50 non-zeros")

    return df, feat_names, pat_info


def prepare_dataset(df, pat_info, outcome, horizon_years=0.5, max_years=None):
    """
    Filter df for the given outcome and apply truncation + horizon exclusion.

    For cases:
      - Keep only visits in [diag_t - max_years*365.25, diag_t - horizon*365.25]
    For controls:
      - Keep all visits

    outcome: 'any_narcolepsy' (NT1 + NT2/IH), 'nt1' (NT1 only), or 'nt2ih' (NT2/IH only)
    max_years: how far back from diagnosis to include (default: MAX_YEARS_BEFORE_DIAG_TRAIN)
    """
    if max_years is None:
        max_years = MAX_YEARS_BEFORE_DIAG_TRAIN
    # Determine which patients are cases for this outcome
    if outcome == 'any_narcolepsy':
        case_types = {'nt1', 'nt2ih'}
    elif outcome == 'nt1':
        case_types = {'nt1'}
    elif outcome == 'nt2ih':
        case_types = {'nt2ih'}
    else:
        raise ValueError(f"Unknown outcome: {outcome}")

    # Build case/control labels for this outcome
    case_pids = {sid for sid, info in pat_info.items()
                 if info['case_type'] in case_types and info['has_event']}
    ctrl_pids = {sid for sid, info in pat_info.items()
                 if info['case_type'] == 'control'}

    # For single-subtype outcomes, exclude the other subtype entirely
    if outcome == 'nt1':
        exclude_pids = {sid for sid, info in pat_info.items()
                        if info['case_type'] == 'nt2ih'}
    elif outcome == 'nt2ih':
        exclude_pids = {sid for sid, info in pat_info.items()
                        if info['case_type'] == 'nt1'}
    else:
        exclude_pids = set()

    # Filter to relevant patients
    keep_pids = (case_pids | ctrl_pids) - exclude_pids
    df_out = df[df['bdsp_patient_id'].isin(keep_pids)].copy().reset_index(drop=True)

    # Add outcome-specific label
    df_out['ever_diagnosed'] = df_out['bdsp_patient_id'].isin(case_pids).astype(int)

    # --- Truncate case visits: [diag_t - max_years, diag_t - h] ---
    horizon_days = horizon_years * 365.25
    max_before_days = max_years * 365.25
    keep = np.ones(len(df_out), dtype=bool)

    for sid in case_pids:
        if sid not in pat_info:
            continue
        info = pat_info[sid]
        diag_t = info['diag_t']
        if diag_t is None:
            continue
        mask = df_out['bdsp_patient_id'] == sid
        grp_idx = df_out.index[mask]
        t_vals = df_out.loc[grp_idx, 'days_since_first_visit'].values

        # Drop visits outside the window [diag_t - max_before, diag_t - horizon]
        too_early = t_vals < (diag_t - max_before_days)
        too_late = t_vals >= (diag_t - horizon_days) if horizon_years > 0 else t_vals > diag_t
        drop_mask = too_early | too_late
        keep[grp_idx[drop_mask]] = False

    df_out = df_out[keep].reset_index(drop=True)

    # Re-exclude patients with 0 visits after truncation
    vpp = df_out.groupby('bdsp_patient_id').size()
    keep_pids2 = vpp[vpp >= 1].index
    df_out = df_out[df_out['bdsp_patient_id'].isin(keep_pids2)].reset_index(drop=True)

    n_cases = df_out[df_out['ever_diagnosed'] == 1]['bdsp_patient_id'].nunique()
    n_ctrls = df_out[df_out['ever_diagnosed'] == 0]['bdsp_patient_id'].nunique()
    print(f"  {outcome} h={horizon_years}: {len(df_out)} rows, "
          f"{n_cases} cases, {n_ctrls} controls")
    return df_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_patient_index(sids, y):
    sid_to_rows = {}
    for i, (sid, label) in enumerate(zip(sids, y)):
        if sid not in sid_to_rows:
            sid_to_rows[sid] = {'label': label, 'rows': []}
        sid_to_rows[sid]['rows'].append(i)
    pos = {sid: info['rows'] for sid, info in sid_to_rows.items() if info['label'] == 1}
    neg = {sid: info['rows'] for sid, info in sid_to_rows.items() if info['label'] == 0}
    return pos, neg


def sample_balanced_batch(pos_patients, neg_patients, rng):
    pos_sids = list(pos_patients.keys())
    neg_sids = list(neg_patients.keys())
    n_pos = len(pos_sids)
    n_neg_sample = min(n_pos, len(neg_sids))
    pos_indices = [rng.choice(pos_patients[sid]) for sid in pos_sids]
    neg_sids_sample = rng.choice(neg_sids, size=n_neg_sample, replace=False)
    neg_indices = [rng.choice(neg_patients[sid]) for sid in neg_sids_sample]
    return np.array(pos_indices + neg_indices)


def chi2_feature_select(X_tr, y_tr, top_k):
    X_nn = np.maximum(X_tr, 0)
    scores, _ = chi2(X_nn, y_tr)
    scores = np.nan_to_num(scores, nan=0)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return np.sort(top_idx)


def _train_one_fold(X, y, sids, tr_mask, te_mask, feat_names,
                    n_epochs=200, top_k=TOP_K_FEATURES, fold_seed=0,
                    alpha_values=None, inner_fold_key=None, inner_fold_vals=None):
    """
    Train SGD on tr_mask, evaluate on te_mask.
    Returns: prob_te, coef_full, metrics_dict, model_artifacts
    """
    if alpha_values is None:
        alpha_values = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    y_tr, y_te = y[tr_mask], y[te_mask]
    sids_tr, sids_te = sids[tr_mask], sids[te_mask]

    # chi2 feature selection
    sel_idx = chi2_feature_select(X[tr_mask], y_tr, top_k)
    X_tr_sel = X[tr_mask][:, sel_idx]
    X_te_sel = X[te_mask][:, sel_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_sel)
    X_te_s = scaler.transform(X_te_sel)

    pos_tr, neg_tr = build_patient_index(sids_tr, y_tr)
    if len(pos_tr) == 0:
        return None, None, None, None

    # --- Inner CV for alpha selection ---
    if inner_fold_key is not None and inner_fold_vals is not None:
        # LOSO inner CV
        inner_keys = inner_fold_key[tr_mask]
        inner_folds = sorted(set(inner_keys))
    else:
        # Stratified 3-fold inner CV
        pat_df = pd.DataFrame({'sid': sids_tr, 'y': y_tr})
        pat_level = pat_df.groupby('sid')['y'].max().reset_index()
        if len(pat_level) < 6 or pat_level['y'].sum() < 3:
            # too few patients for inner CV, use default alpha
            best_alpha = alpha_values[len(alpha_values) // 2]
        else:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            inner_fold_map = {}
            for fi, (_, val_idx) in enumerate(skf.split(pat_level, pat_level['y'])):
                for pid in pat_level.iloc[val_idx]['sid']:
                    inner_fold_map[pid] = fi
            inner_keys = np.array([inner_fold_map.get(s, 0) for s in sids_tr])
            inner_folds = [0, 1, 2]

    best_alpha, best_auc = alpha_values[len(alpha_values) // 2], 0.5
    try:
        for alpha_val in alpha_values:
            aucs = []
            for ifold in inner_folds:
                i_tr = inner_keys != ifold
                i_te = inner_keys == ifold
                pos_i, neg_i = build_patient_index(sids_tr[i_tr], y_tr[i_tr])
                if len(pos_i) == 0:
                    continue
                clf = SGDClassifier(loss='log_loss', penalty='l1', alpha=alpha_val,
                                    max_iter=1, warm_start=True, random_state=42,
                                    learning_rate='optimal')
                rng = np.random.RandomState(42)
                for _ in range(30):
                    idx = sample_balanced_batch(pos_i, neg_i, rng)
                    clf.partial_fit(X_tr_s[i_tr][idx], y_tr[i_tr][idx], classes=[0, 1])
                prob = clf.predict_proba(X_tr_s[i_te])[:, 1]
                idf = pd.DataFrame({'sid': sids_tr[i_te], 'p': prob, 'y': y_tr[i_te]})
                pp = idf.groupby('sid').agg(p=('p', 'mean'), y=('y', 'max')).reset_index()
                if pp['y'].nunique() < 2:
                    continue
                aucs.append(roc_auc_score(pp['y'], pp['p']))
            if aucs and np.mean(aucs) > best_auc:
                best_auc = np.mean(aucs)
                best_alpha = alpha_val
    except (NameError, UnboundLocalError):
        pass  # inner_folds not defined due to too few patients

    # --- Train final model ---
    clf = SGDClassifier(loss='log_loss', penalty='l1', alpha=best_alpha,
                        max_iter=1, warm_start=True, random_state=42,
                        learning_rate='optimal')
    rng = np.random.RandomState(42 + fold_seed)
    for _ in range(n_epochs):
        idx = sample_balanced_batch(pos_tr, neg_tr, rng)
        clf.partial_fit(X_tr_s[idx], y_tr[idx], classes=[0, 1])

    # Full-space coefficients (from uncalibrated model)
    full_coefs = np.zeros(X.shape[1])
    full_coefs[sel_idx] = clf.coef_[0]

    prob_te = clf.predict_proba(X_te_s)[:, 1]

    # Patient-level metrics
    tdf = pd.DataFrame({'sid': sids_te, 'p': prob_te, 'y': y_te})
    pte = tdf.groupby('sid').agg(p=('p', 'mean'), y=('y', 'max')).reset_index()
    if pte['y'].nunique() >= 2:
        auc = roc_auc_score(pte['y'], pte['p'])
        auprc = average_precision_score(pte['y'], pte['p'])
    else:
        auc = auprc = np.nan

    metrics = {
        'AUC': auc, 'AUPRC': auprc,
        'N_diag': int(pte['y'].sum()),
        'N_ctrl': int((pte['y'] == 0).sum()),
        'alpha': best_alpha,
    }
    model_artifacts = {'clf': clf, 'scaler': scaler, 'sel_idx': sel_idx}
    return prob_te, full_coefs, metrics, model_artifacts


def train_pooled_cv(df, feat_names, n_folds=5):
    """Primary analysis: stratified k-fold CV across all patients."""
    print("\n  --- Pooled 5-fold CV ---")
    sids = df['bdsp_patient_id'].values
    X = df[feat_names].values.astype(np.float64)
    y = df['ever_diagnosed'].values.astype(int)
    T = df['days_since_first_visit'].values / 365.25

    # Patient-level stratified split
    pat_df = df.groupby('bdsp_patient_id').agg(
        y=('ever_diagnosed', 'max')).reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_map = {}
    for fi, (_, test_idx) in enumerate(skf.split(pat_df, pat_df['y'])):
        for pid in pat_df.iloc[test_idx]['bdsp_patient_id']:
            fold_map[pid] = fi

    fold_ids = np.array([fold_map[s] for s in sids])
    scores_cv = np.full(len(X), np.nan)
    coefs_all = []
    metrics_all = []
    artifacts_all = {}  # fold_i -> model_artifacts

    for fold_i in range(n_folds):
        tr = fold_ids != fold_i
        te = fold_ids == fold_i

        prob_te, coefs, met, artifacts = _train_one_fold(
            X, y, sids, tr, te, feat_names, fold_seed=fold_i)
        if prob_te is None:
            print(f"    Fold {fold_i}: skipped (no cases)")
            continue
        scores_cv[te] = prob_te
        coefs_all.append(coefs)
        met['fold'] = fold_i
        metrics_all.append(met)
        artifacts_all[fold_i] = artifacts
        print(f"    Fold {fold_i}: AUC={met['AUC']:.3f}  AUPRC={met['AUPRC']:.3f}  "
              f"({met['N_diag']} cases, {met['N_ctrl']} ctrls, alpha={met['alpha']})")

    perf = pd.DataFrame(metrics_all)
    coefs_arr = np.array(coefs_all) if coefs_all else np.zeros((1, X.shape[1]))
    print(f"    ** Pooled mean: AUC={perf['AUC'].mean():.3f}  "
          f"AUPRC={perf['AUPRC'].mean():.3f} **")
    return scores_cv, coefs_arr, feat_names, perf, sids, y, T, fold_map, artifacts_all


def train_loso_cv(df, feat_names):
    """Secondary analysis: leave-one-site-out CV."""
    print("\n  --- LOSO CV (sensitivity) ---")
    sids = df['bdsp_patient_id'].values
    sites = df['site'].values
    X = df[feat_names].values.astype(np.float64)
    y = df['ever_diagnosed'].values.astype(int)
    T = df['days_since_first_visit'].values / 365.25

    site_names = sorted(df['site'].unique())
    scores_cv = np.full(len(X), np.nan)
    coefs_all = []
    metrics_all = []

    for fold_i, held_out in enumerate(site_names):
        tr = sites != held_out
        te = sites == held_out

        prob_te, coefs, met, _ = _train_one_fold(
            X, y, sids, tr, te, feat_names, fold_seed=fold_i,
            inner_fold_key=sites, inner_fold_vals=site_names)
        if prob_te is None:
            print(f"    {held_out}: skipped (no cases)")
            continue
        scores_cv[te] = prob_te
        coefs_all.append(coefs)
        met['site'] = held_out
        metrics_all.append(met)
        print(f"    {held_out}: AUC={met['AUC']:.3f}  AUPRC={met['AUPRC']:.3f}  "
              f"({met['N_diag']} cases, {met['N_ctrl']} ctrls)")

    perf = pd.DataFrame(metrics_all)
    coefs_arr = np.array(coefs_all) if coefs_all else np.zeros((1, X.shape[1]))
    print(f"    ** LOSO mean: AUC={perf['AUC'].mean():.3f}  "
          f"AUPRC={perf['AUPRC'].mean():.3f} **")
    return scores_cv, coefs_arr, feat_names, perf, sids, y, T


def train_final_model(df_train, feat_names, best_alpha, n_epochs=200):
    """
    Train a single final model on ALL training data using the selected alpha.
    This model is used for scoring (trajectory plots) and deployment.
    CV is used only for performance estimation.

    Returns: model_artifacts dict {'clf', 'scaler', 'sel_idx'}
    """
    print(f"\n  --- Training final model on ALL data (alpha={best_alpha}) ---")
    sids = df_train['bdsp_patient_id'].values
    X = df_train[feat_names].values.astype(np.float64)
    y = df_train['ever_diagnosed'].values.astype(int)

    # Chi-squared feature selection on full training set
    sel_idx = chi2_feature_select(X, y, TOP_K_FEATURES)
    X_sel = X[:, sel_idx]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_sel)

    pos_idx, neg_idx = build_patient_index(sids, y)
    clf = SGDClassifier(loss='log_loss', penalty='l1', alpha=best_alpha,
                        max_iter=1, warm_start=True, random_state=42,
                        learning_rate='optimal')
    rng = np.random.RandomState(42)
    for _ in range(n_epochs):
        idx = sample_balanced_batch(pos_idx, neg_idx, rng)
        clf.partial_fit(X_s[idx], y[idx], classes=[0, 1])

    n_nonzero = np.sum(clf.coef_[0] != 0)
    print(f"    Trained on {len(X)} visits ({int(y.sum())} case visits, "
          f"{int((y == 0).sum())} control visits)")
    print(f"    {n_nonzero}/{len(sel_idx)} features with non-zero coefficients")

    return {'clf': clf, 'scaler': scaler, 'sel_idx': sel_idx}


def score_with_final_model(df_full, feat_names, final_artifacts):
    """
    Score ALL visits in df_full using the single final model.
    Returns: scores, sids, y, T arrays.
    """
    print("\n  --- Scoring all visits with final model ---")
    sids = df_full['bdsp_patient_id'].values
    X = df_full[feat_names].values.astype(np.float64)
    y = df_full['ever_diagnosed'].values.astype(int)
    T = df_full['days_since_first_visit'].values / 365.25

    clf = final_artifacts['clf']
    scaler = final_artifacts['scaler']
    sel_idx = final_artifacts['sel_idx']

    X_sel = X[:, sel_idx]
    X_s = scaler.transform(X_sel)
    scores = clf.predict_proba(X_s)[:, 1]

    print(f"    Scored {len(scores)} visits")
    return scores, sids, y, T


def score_full_timeline(df_full, feat_names, fold_map, artifacts_all):
    """
    Score ALL visits in df_full (test dataset, [-5yr, 0yr] window)
    using models trained on the h=0.5 dataset ([-2.5yr, -0.5yr]).

    Patients in fold_map: scored by their assigned fold's model (out-of-sample).
    Patients NOT in fold_map: scored by averaging all fold models (they were
    not in any fold's training data, so all folds give valid OOS predictions).
    Returns: scores, sids, y, T arrays matching df_full rows.
    """
    print("\n  --- Scoring test dataset with trained h=0.5 models ---")
    sids = df_full['bdsp_patient_id'].values
    X = df_full[feat_names].values.astype(np.float64)
    y = df_full['ever_diagnosed'].values.astype(int)
    T = df_full['days_since_first_visit'].values / 365.25

    scores = np.full(len(X), np.nan)

    # 1) Score patients WITH fold assignments (out-of-sample by their fold)
    for fold_i, artifacts in artifacts_all.items():
        clf = artifacts['clf']
        scaler = artifacts['scaler']
        sel_idx = artifacts['sel_idx']

        te_mask = np.array([fold_map.get(s, -1) == fold_i for s in sids])
        if te_mask.sum() == 0:
            continue
        X_te_sel = X[te_mask][:, sel_idx]
        X_te_s = scaler.transform(X_te_sel)
        scores[te_mask] = clf.predict_proba(X_te_s)[:, 1]

    # 2) Score patients WITHOUT fold assignments (average all fold models)
    no_fold_mask = np.isnan(scores)
    if no_fold_mask.sum() > 0:
        X_nf = X[no_fold_mask]
        accum = np.zeros(no_fold_mask.sum())
        for fold_i, artifacts in artifacts_all.items():
            clf = artifacts['clf']
            scaler = artifacts['scaler']
            sel_idx = artifacts['sel_idx']
            X_sel = X_nf[:, sel_idx]
            X_s = scaler.transform(X_sel)
            accum += clf.predict_proba(X_s)[:, 1]
        scores[no_fold_mask] = accum / len(artifacts_all)

    n_scored = np.sum(~np.isnan(scores))
    print(f"    Scored {n_scored}/{len(scores)} visits "
          f"({no_fold_mask.sum()} via ensemble average)")
    return scores, sids, y, T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_traj_data(results, h, pat_info, cv_type='pooled', rng=None):
    """Build case and control trajectory DataFrames for one outcome/horizon."""
    r = results[h][cv_type]
    sids_arr = r.get('traj_sids', r['sids'])
    scores = r.get('traj_scores', r['scores'])
    y_arr = r.get('traj_y', r['y'])
    T_arr = r.get('traj_T', r['T'])

    dtmp = pd.DataFrame({'sid': sids_arr, 'score': scores, 'y': y_arr, 'T': T_arr})
    dtmp = dtmp.dropna(subset=['score'])

    pat_label = dtmp.groupby('sid')['y'].max()
    case_sids = set(pat_label[pat_label == 1].index)
    ctrl_sids = set(pat_label[pat_label == 0].index)

    # Cases: align to diagnosis time
    case_rows = []
    for sid in case_sids:
        sub = dtmp[dtmp['sid'] == sid].sort_values('T')
        info = pat_info.get(sid)
        if info is None or info['diag_t'] is None:
            continue
        diag_t_yr = info['diag_t'] / 365.25
        for _, row in sub.iterrows():
            t_rel = row['T'] - diag_t_yr
            if -MAX_YEARS_BEFORE_DIAG_TEST - 0.1 <= t_rel <= 0.1:
                case_rows.append({'t_rel': t_rel, 'score': row['score'], 'sid': sid})
    case_df = pd.DataFrame(case_rows) if case_rows else pd.DataFrame(
        columns=['t_rel', 'score', 'sid'])

    # Controls: random pseudo-diagnosis, keep visits in [-5yr, 0]
    ctrl_rows = []
    ctrl_sample = list(ctrl_sids)
    if len(ctrl_sample) > 500:
        ctrl_sample = list(rng.choice(ctrl_sample, size=500, replace=False))
    for sid in ctrl_sample:
        sub = dtmp[dtmp['sid'] == sid].sort_values('T')
        t_vals = sub['T'].values
        t_span = t_vals[-1] - t_vals[0]
        if t_span < 0.5:
            continue
        earliest_pseudo = t_vals[0] + 0.5
        latest_pseudo = t_vals[-1]
        if earliest_pseudo >= latest_pseudo:
            pseudo_diag = latest_pseudo
        else:
            pseudo_diag = rng.uniform(earliest_pseudo, latest_pseudo)
        for _, row in sub.iterrows():
            t_rel = row['T'] - pseudo_diag
            if -MAX_YEARS_BEFORE_DIAG_TEST - 0.1 <= t_rel <= 0.1:
                ctrl_rows.append({'t_rel': t_rel, 'score': row['score'], 'sid': sid})
    ctrl_df = pd.DataFrame(ctrl_rows) if ctrl_rows else pd.DataFrame(
        columns=['t_rel', 'score', 'sid'])

    return case_df, ctrl_df


def _logit(p, eps=1e-3):
    """Logit transform with clipping to avoid infinities."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sliding_window_percentiles(t_rel, values, percentiles, window=1.0,
                                 step=0.1, min_count=10):
    """
    Compute percentiles using a sliding window of given width.
    Returns: t_centers array, dict {pct: values_array} for each percentile.
    """
    t_centers = np.arange(t_rel.min() + window / 2,
                          t_rel.max() + step, step)
    result = {p: np.full(len(t_centers), np.nan) for p in percentiles}
    for i, tc in enumerate(t_centers):
        mask = (t_rel >= tc - window / 2) & (t_rel < tc + window / 2)
        vals = values[mask]
        if len(vals) >= min_count:
            for p in percentiles:
                result[p][i] = np.percentile(vals, p * 100)
    return t_centers, result


def _sliding_window_auc(case_df, ctrl_df, window=1.0, step=0.1,
                        min_cases=5, min_ctrls=10):
    """
    Compute patient-level AUC in a sliding time window.
    For each window, average each patient's visit-level scores, then compute AUC.
    Returns: t_centers, auc_values arrays.
    """
    all_df = pd.concat([case_df.assign(label=1), ctrl_df.assign(label=0)])
    t_min = max(all_df['t_rel'].min(), -5.0)
    t_max = min(all_df['t_rel'].max(), 0.1)
    t_centers = np.arange(t_min + window / 2, t_max + step, step)
    auc_vals = np.full(len(t_centers), np.nan)

    for i, tc in enumerate(t_centers):
        mask = (all_df['t_rel'] >= tc - window / 2) & (all_df['t_rel'] < tc + window / 2)
        sub = all_df[mask]
        if len(sub) == 0:
            continue
        # Patient-level: average score per patient, max label
        pat = sub.groupby('sid').agg(score=('score', 'mean'),
                                     label=('label', 'max')).reset_index()
        n_pos = pat['label'].sum()
        n_neg = (pat['label'] == 0).sum()
        if n_pos >= min_cases and n_neg >= min_ctrls and pat['label'].nunique() == 2:
            auc_vals[i] = roc_auc_score(pat['label'], pat['score'])
    return t_centers, auc_vals


def plot_trajectories_combined(all_results, pat_info, cv_type='pooled'):
    """
    Combined trajectory plot: 2×N grid (N = number of outcomes).
    Top row: risk score trajectories (logit scale) with P25/P50/P75 curves.
    Bottom row: time-dependent AUROC.
    """
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    n_cols = len(outcomes)
    fig, axes = plt.subplots(2, n_cols, figsize=(DOUBLE_COL_IN * n_cols / 2, 6.0),
                             sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    rng = np.random.RandomState(123)
    h = 0.5

    outcome_labels = {'any_narcolepsy': 'Any narcolepsy', 'nt1': 'NT1 only',
                      'nt2ih': 'NT2/IH only'}
    all_panel = [chr(ord('A') + i) for i in range(2 * n_cols)]
    panel_labels = [all_panel[:n_cols], all_panel[n_cols:]]

    for col_i, outcome in enumerate(outcomes):
        ax_traj = axes[0, col_i]
        ax_auc = axes[1, col_i]
        add_panel_label(ax_traj, panel_labels[0][col_i])
        add_panel_label(ax_auc, panel_labels[1][col_i])
        results = all_results[outcome]
        case_df, ctrl_df = _build_traj_data(results, h, pat_info, cv_type, rng)

        n_cases = case_df['sid'].nunique() if len(case_df) > 0 else 0
        n_ctrls = ctrl_df['sid'].nunique() if len(ctrl_df) > 0 else 0

        # Apply logit transform
        if len(case_df) > 0:
            case_df = case_df.copy()
            case_df['logit_score'] = _logit(case_df['score'].values)
        if len(ctrl_df) > 0:
            ctrl_df = ctrl_df.copy()
            ctrl_df['logit_score'] = _logit(ctrl_df['score'].values)

        # === Top row: trajectories ===
        if len(ctrl_df) > 0:
            for sid in ctrl_df['sid'].unique():
                sub = ctrl_df[ctrl_df['sid'] == sid].sort_values('t_rel')
                if len(sub) >= 2:
                    ax_traj.plot(sub['t_rel'], sub['logit_score'], color=CTRL_COLOR,
                                 alpha=0.12, linewidth=0.5)
                elif len(sub) == 1:
                    ax_traj.scatter(sub['t_rel'], sub['logit_score'], color=CTRL_COLOR,
                                    alpha=0.10, s=8, zorder=1)
        if len(case_df) > 0:
            for sid in case_df['sid'].unique():
                sub = case_df[case_df['sid'] == sid].sort_values('t_rel')
                if len(sub) >= 2:
                    ax_traj.plot(sub['t_rel'], sub['logit_score'], color=CASE_COLOR,
                                 alpha=0.20, linewidth=0.6)
                elif len(sub) == 1:
                    ax_traj.scatter(sub['t_rel'], sub['logit_score'], color=CASE_COLOR,
                                    alpha=0.15, s=10, zorder=2)

        # Percentile curves via sliding 1yr window
        pcts = [0.25, 0.50, 0.75]
        pct_styles = {0.25: ('--', 1.2), 0.50: ('-', LINE_WIDTH_THICK), 0.75: ('--', 1.2)}
        p50_endpoints = {}  # store rightmost P50 point for inline labels
        for df_group, color, group_label in [
            (ctrl_df, CTRL_COLOR, 'Controls'),
            (case_df, CASE_COLOR, 'Cases'),
        ]:
            if len(df_group) == 0:
                continue
            t_arr = df_group['t_rel'].values
            v_arr = df_group['logit_score'].values
            t_ctr, pct_dict = _sliding_window_percentiles(
                t_arr, v_arr, pcts, window=1.0, step=0.1, min_count=10)
            for p in pcts:
                ls, lw = pct_styles[p]
                valid = ~np.isnan(pct_dict[p])
                ax_traj.plot(t_ctr[valid], pct_dict[p][valid], color=color,
                             linestyle=ls, linewidth=lw, zorder=10)
                if p == 0.50 and valid.any():
                    # Store endpoint for inline label
                    idx = np.where(valid)[0][-1]
                    p50_endpoints[group_label] = (t_ctr[idx], pct_dict[p][idx], color)

        # Inline labels on P50 curves (Tufte style)
        for grp, (t_end, y_end, clr) in p50_endpoints.items():
            n_grp = n_cases if grp == 'Cases' else n_ctrls
            ax_traj.annotate(f'{grp} (n={n_grp})',
                             xy=(t_end, y_end), xytext=(6, 0),
                             textcoords='offset points',
                             fontsize=FONT_SIZE_ANNOTATION, color=clr,
                             va='center', ha='left', fontweight='bold', zorder=15)

        if col_i == 0:
            ax_traj.set_ylabel('Risk score (logit scale)')

        ax_traj.set_title(f'{outcome_labels[outcome]}')

        # === Bottom row: time-dependent AUROC ===
        if len(case_df) > 0 and len(ctrl_df) > 0:
            t_auc, auc_curve = _sliding_window_auc(
                case_df, ctrl_df, window=1.0, step=0.1,
                min_cases=5, min_ctrls=10)
            valid = ~np.isnan(auc_curve)
            ax_auc.plot(t_auc[valid], auc_curve[valid], color='0.3',
                        linewidth=LINE_WIDTH_THICK)
            ax_auc.fill_between(t_auc[valid], 0.5, auc_curve[valid],
                                color='0.6', alpha=0.10)

        ax_auc.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
        ax_auc.axhline(0.8, color='gray', linewidth=0.6, linestyle='--', alpha=0.4)
        ax_auc.axhline(0.9, color='gray', linewidth=0.6, linestyle='--', alpha=0.4)
        ax_auc.set_ylim(0.50, 1.02)
        ax_auc.set_xlabel('Years relative to diagnosis')
        if col_i == 0:
            ax_auc.set_ylabel('AUROC')

    # Shared x-axis settings
    for ax in axes.flat:
        ax.set_xlim(-2.65, 0.15)
        ax.set_xticks(np.arange(-2.5, 0.1, 0.5))

    plt.tight_layout()
    os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)
    pub_savefig(fig, os.path.join(MANUSCRIPT_FIG_DIR, 'figure2_risk_score_trajectories.png'))
    plt.close()
    print("Saved: figure2_risk_score_trajectories.png")


def plot_score_distributions_combined(all_results, cv_type='pooled'):
    """Combined score distribution plot, h=0.5 only.
    Uses final model scores (traj_ data) so cohort matches trajectory plot."""
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    n_rows = len(outcomes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(DOUBLE_COL_IN, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    h = 0.5
    outcome_labels = {'any_narcolepsy': 'Any narcolepsy (NT1 + NT2/IH)',
                      'nt1': 'NT1 only', 'nt2ih': 'NT2/IH only'}

    for row_i, outcome in enumerate(outcomes):
        ax = axes[row_i]
        add_panel_label(ax, chr(ord('A') + row_i))
        r = all_results[outcome][h][cv_type]
        sids = r.get('traj_sids', r['sids'])
        scores = r.get('traj_scores', r['scores'])
        y_arr = r.get('traj_y', r['y'])
        dtmp = pd.DataFrame({'sid': sids, 'score': scores, 'y': y_arr})
        dtmp = dtmp.dropna(subset=['score'])
        pat = dtmp.groupby('sid').agg(score=('score', 'mean'), y=('y', 'max')).reset_index()

        ctrl = pat[pat['y'] == 0]['score']
        case = pat[pat['y'] == 1]['score']
        ax.hist(ctrl, bins=50, alpha=0.5, color=CTRL_COLOR,
                label=f'Controls (n = {len(ctrl)})', density=True)
        ax.hist(case, bins=30, alpha=0.6, color=CASE_COLOR,
                label=f'Cases (n = {len(case)})', density=True)
        ax.set_xlabel('Risk score (patient-level mean)')
        ax.set_ylabel('Density')
        auc_val = r['perf']['AUC'].mean()
        ax.set_title(f'{outcome_labels[outcome]} (AUC = {auc_val:.3f})')
        ax.legend(fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout()
    pub_savefig(fig, os.path.join(MANUSCRIPT_FIG_DIR, 'efigure10_risk_score_distributions.png'))
    plt.close()
    print("Saved: efigure10_risk_score_distributions.png")


def plot_feature_importance_combined(all_results, cv_type='pooled'):
    """Combined feature importance, h=0.5 only."""
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    n_cols = len(outcomes)
    fig, axes = plt.subplots(1, n_cols, figsize=(DOUBLE_COL_IN * n_cols / 2 * 2, 6))
    if n_cols == 1:
        axes = [axes]
    h = 0.5
    top_n = 20
    outcome_labels = {'any_narcolepsy': 'Any narcolepsy (NT1 + NT2/IH)',
                      'nt1': 'NT1 only', 'nt2ih': 'NT2/IH only'}

    for col_i, outcome in enumerate(outcomes):
        ax = axes[col_i]
        add_panel_label(ax, chr(ord('A') + col_i))
        r = all_results[outcome][h][cv_type]
        coefs = r['coefs']
        fnames = r['feat_names']
        mean_c = coefs.mean(axis=0)
        abs_c = np.abs(mean_c)
        nz_frac = (coefs != 0).mean(axis=0)
        sel = np.where(nz_frac >= 0.4)[0]
        if len(sel) == 0:
            sel = np.argsort(abs_c)[::-1][:top_n]
        else:
            sel = sel[np.argsort(abs_c[sel])[::-1]]
        n_show = min(top_n, len(sel))
        idx = sel[:n_show][::-1]
        names = [fnames[i][:40] for i in idx]
        vals = mean_c[idx]
        colors = [FEAT_POS_COLOR if v > 0 else FEAT_NEG_COLOR for v in vals]
        ax.barh(np.arange(n_show), vals, color=colors, alpha=0.8,
                edgecolor='k', linewidth=0.5)
        ax.set_yticks(np.arange(n_show))
        ax.set_yticklabels(names)
        ax.axvline(0, color='k', linewidth=0.8)
        auc_val = r['perf']['AUC'].mean()
        ax.set_title(f'{outcome_labels[outcome]} (AUC = {auc_val:.3f})')
        ax.set_xlabel('Mean coefficient (across CV folds)')

    plt.tight_layout()
    pub_savefig(fig, os.path.join(MANUSCRIPT_FIG_DIR, 'efigure11_top_predictive_features.png'))
    plt.close()
    print("Saved: efigure11_top_predictive_features.png")


def plot_performance_combined(all_results):
    """Combined performance figure: Nx2 grid.
    Rows: outcomes.  Cols: AUC, AUPRC.
    Each panel: 3 bars (Pooled CV, LOSO, Final model) with per-fold/site dots.
    """
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    n_rows = len(outcomes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(DOUBLE_COL_IN, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)
    h = 0.5
    outcome_labels = {'any_narcolepsy': 'Any narcolepsy (NT1 + NT2/IH)',
                      'nt1': 'NT1 only', 'nt2ih': 'NT2/IH only'}
    bar_labels = ['5-fold CV', 'LOSO', 'Final model\n(all data)']
    all_panel = [chr(ord('A') + i) for i in range(2 * n_rows)]
    panel_labels = [[all_panel[r * 2], all_panel[r * 2 + 1]] for r in range(n_rows)]

    for row_i, outcome in enumerate(outcomes):
        results = all_results[outcome]
        for col_i, metric in enumerate(['AUC', 'AUPRC']):
            ax = axes[row_i, col_i]
            add_panel_label(ax, panel_labels[row_i][col_i])

            pooled_vals = results[h]['pooled']['perf'][metric].dropna().values
            loso_vals = results[h]['loso']['perf'][metric].dropna().values
            resub_val = results[h]['resubstitution']['perf'][metric].iloc[0]

            means = [np.mean(pooled_vals), np.mean(loso_vals), resub_val]
            all_dots = [pooled_vals, loso_vals, [resub_val]]

            for i in range(3):
                ax.bar(i, means[i], width=0.6, color=BAR_COLORS[i], alpha=0.35,
                       edgecolor='k', linewidth=0.5)
                rng = np.random.RandomState(42)
                jit = rng.uniform(-0.12, 0.12, size=len(all_dots[i]))
                ax.scatter(i + jit, all_dots[i], color=BAR_COLORS[i], s=45,
                           edgecolor='k', linewidth=0.5, zorder=5, alpha=0.9)
                ax.text(i, means[i] + 0.012, f'{means[i]:.3f}', ha='center',
                        fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')

            ax.set_xticks(range(3))
            ax.set_xticklabels(bar_labels)
            ax.grid(True, alpha=0.3, axis='y')

            if metric == 'AUC':
                ax.set_ylim(0.45, 1.05)
            else:
                ax.set_ylim(0, 1.05)

            ax.set_title(f'{outcome_labels[outcome]} — {metric}')

    plt.tight_layout()
    pub_savefig(fig, os.path.join(MANUSCRIPT_FIG_DIR, 'efigure9_predictive_performance.png'))
    plt.close()
    print("Saved: efigure9_predictive_performance.png")


def save_loso_table(all_results):
    """Save a LOSO performance table broken down by site for all outcomes."""
    h = 0.5
    outcome_labels = {'any_narcolepsy': 'Any Narcolepsy (NT1 + NT2/IH)',
                      'nt1': 'NT1 Only', 'nt2ih': 'NT2/IH Only'}
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    rows = []
    for outcome in outcomes:
        results = all_results[outcome]
        perf = results[h]['loso']['perf']
        for _, row in perf.iterrows():
            rows.append({
                'Outcome': outcome_labels[outcome],
                'Site': row['site'].upper(),
                'AUC': f"{row['AUC']:.3f}",
                'AUPRC': f"{row['AUPRC']:.3f}",
                'N_cases': int(row['N_diag']),
                'N_controls': int(row['N_ctrl']),
            })
        # Add mean row
        rows.append({
            'Outcome': outcome_labels[outcome],
            'Site': 'Mean',
            'AUC': f"{perf['AUC'].mean():.3f}",
            'AUPRC': f"{perf['AUPRC'].mean():.3f}",
            'N_cases': int(perf['N_diag'].sum()),
            'N_controls': int(perf['N_ctrl'].sum()),
        })

    table = pd.DataFrame(rows)
    table.to_csv('v2_loso_by_site.csv', index=False)
    print("\nLOSO results by site:")
    print(table.to_string(index=False))
    print("Saved: v2_loso_by_site.csv")
    return table


def plot_nnt_analysis(all_results, prevalence=0.0008, cv_type='pooled'):
    """
    Number Needed to Test (NNT) analysis using final model scores.

    At threshold tau:
      Sensitivity = P(score >= tau | case)
      Specificity = P(score < tau | control)
      PPV = (Sens * pi) / (Sens * pi + (1-Spec) * (1-pi))
      NNT = 1 / PPV

    Uses final-model patient-level mean scores (traj_ data).
    """
    outcomes = sorted(all_results.keys(),
                      key=lambda x: ['any_narcolepsy', 'nt1', 'nt2ih'].index(x))
    n_cols = len(outcomes)
    fig, axes = plt.subplots(1, n_cols, figsize=(DOUBLE_COL_IN * n_cols / 2, 3.5))
    if n_cols == 1:
        axes = [axes]
    h = 0.5
    outcome_labels = {'any_narcolepsy': 'Any narcolepsy (NT1 + NT2/IH)',
                      'nt1': 'NT1 only', 'nt2ih': 'NT2/IH only'}

    for col_i, outcome in enumerate(outcomes):
        results = all_results[outcome]
        r = results[h][cv_type]

        sids = r.get('traj_sids', r['sids'])
        scores = r.get('traj_scores', r['scores'])
        y_arr = r.get('traj_y', r['y'])

        dtmp = pd.DataFrame({'sid': sids, 'score': scores, 'y': y_arr})
        dtmp = dtmp.dropna(subset=['score'])
        pat = dtmp.groupby('sid').agg(
            score=('score', 'mean'), y=('y', 'max')).reset_index()

        case_scores = pat[pat['y'] == 1]['score'].values
        ctrl_scores = pat[pat['y'] == 0]['score'].values

        thresholds = np.linspace(0.005, 0.995, 500)
        sens = np.array([np.mean(case_scores >= t) for t in thresholds])
        spec = np.array([np.mean(ctrl_scores < t) for t in thresholds])

        ppv = (sens * prevalence) / (
            sens * prevalence + (1 - spec) * (1 - prevalence))
        nnt = np.where(ppv > 0, 1.0 / ppv, np.nan)

        ax1 = axes[col_i]
        add_panel_label(ax1, chr(ord('A') + col_i))
        ax1_r = ax1.twinx()
        # Keep right spine visible for dual axis
        ax1_r.spines['right'].set_visible(True)

        valid = ~np.isnan(nnt) & np.isfinite(nnt) & (nnt > 0)
        ax1.semilogy(thresholds[valid], nnt[valid], color=NNT_COLOR,
                     linewidth=LINE_WIDTH, label='NNT')
        ax1_r.plot(thresholds, sens, color=SENS_COLOR, linewidth=LINE_WIDTH,
                   linestyle='--', label='Sensitivity')

        ax1.set_yticks([10, 20, 50, 100, 200, 500, 1000])
        ax1.set_yticklabels(['10', '20', '50', '100', '200', '500', '1,000'])
        ax1.set_ylim(5, 2000)

        # Annotate NNT = 10 and NNT = 20 operating points
        op_points = []
        for target_nnt in [10, 20]:
            valid_idx = np.where(valid)[0]
            nnt_valid = nnt[valid_idx]
            closest = valid_idx[np.argmin(np.abs(nnt_valid - target_nnt))]
            op_points.append((thresholds[closest], sens[closest], nnt[closest]))

        for i, (t_val, s_val, n_val) in enumerate(op_points):
            # Horizontal NNT dashed line (blue, matching NNT curve)
            ax1.axhline(n_val, color=NNT_COLOR, linestyle=':', linewidth=0.5,
                        alpha=0.5)
            # Vertical dashed line at the threshold
            ax1.axvline(t_val, color='gray', linestyle=':', linewidth=0.5,
                        alpha=0.6)
            # Blue dot on NNT curve
            ax1.plot(t_val, n_val, 'o', color=NNT_COLOR, markersize=4, zorder=10)
            # NNT label near the dot, offset left to avoid overlapping curve
            ax1.text(t_val - 0.08, n_val, f'NNT = {n_val:.0f}',
                     fontsize=FONT_SIZE_ANNOTATION, color=NNT_COLOR,
                     fontweight='bold', va='bottom', ha='right')

            # Horizontal sensitivity dashed line (matching sens curve color)
            ax1_r.axhline(s_val, color=SENS_COLOR, linestyle=':', linewidth=0.5,
                          alpha=0.5)
            # Dot on sensitivity curve
            ax1_r.plot(t_val, s_val, 'o', color=SENS_COLOR, markersize=4, zorder=10)

        # Sensitivity labels: lower (NNT=10) to left, upper (NNT=20) slightly right
        s_vals = [p[1] for p in op_points]
        if len(s_vals) == 2 and abs(s_vals[0] - s_vals[1]) < 0.08:
            y_offsets = [-0.055, 0.03]
        else:
            y_offsets = [0.0, 0.0]
        # op_points[0] = NNT=10 (lower sens), op_points[1] = NNT=20 (higher sens)
        x_nudges = [-0.03, 0.005]   # left for lower, slightly right for upper
        h_aligns = ['right', 'left']
        for i, (t_val, s_val, n_val) in enumerate(op_points):
            ax1_r.text(t_val + x_nudges[i], s_val + y_offsets[i], f'{s_val:.0%}',
                       fontsize=FONT_SIZE_ANNOTATION, color=SENS_COLOR,
                       fontweight='bold', va='bottom', ha=h_aligns[i])

        ax1.set_xlabel('Score threshold')
        ax1.set_ylabel('NNT (log scale)', color=NNT_COLOR)
        ax1_r.set_ylabel('Sensitivity', color=SENS_COLOR)
        ax1_r.set_ylim(0, 1.05)
        ax1.set_title(f'{outcome_labels[outcome]}')

        # Tufte-style inline labels on the curves (no legend box)
        # Place NNT label at ~30% along the x-axis
        mid_idx = len(thresholds) // 3
        if valid[mid_idx]:
            ax1.text(thresholds[mid_idx], nnt[mid_idx] * 1.3, 'NNT',
                     fontsize=FONT_SIZE_LEGEND, color=NNT_COLOR,
                     fontweight='bold', va='bottom', ha='center')
        # Place Sensitivity label near the start of the curve
        ax1_r.text(thresholds[mid_idx], sens[mid_idx] + 0.03, 'Sensitivity',
                   fontsize=FONT_SIZE_LEGEND, color=SENS_COLOR,
                   fontweight='bold', va='bottom', ha='center')

    fig.suptitle(f'NNT analysis  (prevalence: {prevalence*100:.2f}%, '
                 f'1 in {1/prevalence:,.0f})',
                 fontsize=FONT_SIZE_SUPTITLE, y=1.02)
    plt.tight_layout()
    pub_savefig(fig, os.path.join(MANUSCRIPT_FIG_DIR, 'figure3_nnt_analysis.png'))
    plt.close()
    print("Saved: figure3_nnt_analysis.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_outcome(df_all, feat_names, pat_info, outcome):
    """Run training, evaluation, and per-outcome plots for one outcome."""
    _outcome_labels = {
        'any_narcolepsy': 'Any Narcolepsy (NT1 + NT2/IH)',
        'nt1': 'NT1',
        'nt2ih': 'NT2/IH',
    }
    outcome_label = _outcome_labels[outcome]
    h = 0.5
    results = {}

    print(f"\n{'=' * 60}")
    print(f"  {outcome_label} — Horizon = {h} yr")
    print(f"  Training window: [-{MAX_YEARS_BEFORE_DIAG_TRAIN}yr, -{h}yr]")
    print(f"  Testing window:  [-{MAX_YEARS_BEFORE_DIAG_TEST}yr, 0yr]")
    print(f"{'=' * 60}")

    # Training dataset: [-2.5yr, -0.5yr]
    df_train = prepare_dataset(df_all, pat_info, outcome,
                               horizon_years=h, max_years=MAX_YEARS_BEFORE_DIAG_TRAIN)
    results[h] = {}

    # Primary: pooled CV (for performance estimation)
    scores, coefs, fnames, perf, sids, y, T, fold_map, cv_artifacts = \
        train_pooled_cv(df_train, feat_names)
    results[h]['pooled'] = {
        'scores': scores, 'coefs': coefs, 'feat_names': fnames,
        'perf': perf, 'sids': sids, 'y': y, 'T': T,
    }

    # Secondary: LOSO CV
    scores_l, coefs_l, fnames_l, perf_l, sids_l, y_l, T_l = \
        train_loso_cv(df_train, feat_names)
    results[h]['loso'] = {
        'scores': scores_l, 'coefs': coefs_l, 'feat_names': fnames_l,
        'perf': perf_l, 'sids': sids_l, 'y': y_l, 'T': T_l,
    }

    # Select best alpha: modal (most common) alpha across CV folds
    from collections import Counter
    fold_alphas = [cv_artifacts[fi]['clf'].alpha for fi in cv_artifacts]
    best_alpha = Counter(fold_alphas).most_common(1)[0][0]
    print(f"\n  Per-fold alphas: {fold_alphas} -> modal alpha: {best_alpha}")

    # Train final model on ALL training data
    final_artifacts = train_final_model(df_train, feat_names, best_alpha)

    # Score full test dataset [-5yr, 0yr] with final model
    df_test = prepare_dataset(df_all, pat_info, outcome,
                              horizon_years=0, max_years=MAX_YEARS_BEFORE_DIAG_TEST)
    ft_scores, ft_sids, ft_y, ft_T = score_with_final_model(
        df_test, feat_names, final_artifacts)
    results[h]['pooled']['traj_scores'] = ft_scores
    results[h]['pooled']['traj_sids'] = ft_sids
    results[h]['pooled']['traj_y'] = ft_y
    results[h]['pooled']['traj_T'] = ft_T

    # Resubstitution: final model evaluated on training data
    print("\n  --- Resubstitution (final model on training data) ---")
    resub_scores, resub_sids, resub_y, resub_T = score_with_final_model(
        df_train, feat_names, final_artifacts)
    resub_df = pd.DataFrame({'sid': resub_sids, 'score': resub_scores, 'y': resub_y})
    resub_pat = resub_df.groupby('sid').agg(
        score=('score', 'mean'), y=('y', 'max')).reset_index()
    resub_auc = roc_auc_score(resub_pat['y'], resub_pat['score'])
    resub_auprc = average_precision_score(resub_pat['y'], resub_pat['score'])
    print(f"    AUC={resub_auc:.3f}, AUPRC={resub_auprc:.3f}")
    results[h]['resubstitution'] = {
        'perf': pd.DataFrame([{
            'AUC': resub_auc, 'AUPRC': resub_auprc,
            'N_diag': int(resub_pat['y'].sum()),
            'N_ctrl': int((resub_pat['y'] == 0).sum()),
        }]),
    }

    # Save results WITH final model artifacts
    save_data = {
        'results': results,
        'final_model': final_artifacts,
        'feat_names': feat_names,
    }
    with open(f'v2_results_{outcome}.pickle', 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved results + final model: v2_results_{outcome}.pickle")

    # Summary table
    print(f"\n\n{'=' * 60}")
    print(f"  SUMMARY: {outcome_label}")
    print(f"{'=' * 60}")
    rows = []
    for cv_type in ['pooled', 'loso', 'resubstitution']:
        p = results[h][cv_type]['perf']
        n_cases = int(p['N_diag'].sum())
        rows.append({
            'Horizon': h, 'CV': cv_type,
            'Mean AUC': f"{p['AUC'].mean():.3f}",
            'Mean AUPRC': f"{p['AUPRC'].mean():.3f}",
            'N cases': n_cases,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(f'v2_summary_{outcome}.csv', index=False)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python risk_score_v2.py <any_narcolepsy|nt1|nt2ih|both|all>")
        sys.exit(1)

    outcome = sys.argv[1].strip().lower()
    valid = ['any_narcolepsy', 'nt1', 'nt2ih', 'both', 'all']
    assert outcome in valid, \
        f"outcome must be one of {valid}, got '{outcome}'"

    # Load all data once
    df_all, feat_names, pat_info = load_all_data()

    if outcome == 'all':
        outcomes = ['any_narcolepsy', 'nt1', 'nt2ih']
    elif outcome == 'both':
        outcomes = ['any_narcolepsy', 'nt1']
    else:
        outcomes = [outcome]

    all_results = {}
    for oc in outcomes:
        all_results[oc] = run_one_outcome(df_all, feat_names, pat_info, oc)

    # Combined plots (requires all 3 outcomes)
    if len(all_results) >= 2:
        print("\nGenerating combined plots...")
        plot_performance_combined(all_results)
        plot_score_distributions_combined(all_results, cv_type='pooled')
        plot_feature_importance_combined(all_results, cv_type='pooled')
        plot_trajectories_combined(all_results, pat_info, cv_type='pooled')
        plot_nnt_analysis(all_results, prevalence=0.0008)
        save_loso_table(all_results)

    print("\nDone.")


if __name__ == '__main__':
    main()
