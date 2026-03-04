"""
Feature evolution heatmap: shows how selected model features evolve over time
leading up to diagnosis in cases vs controls.

Layout: 2 panels side by side
  [Cases]  [Controls]

Each heatmap:
  - Rows (y-axis): features with non-zero L1 coefficients, sorted by coefficient
  - Columns (x-axis): time bins from -2.5 years to 0 (diagnosis)
  - Color: mean cumulative feature count per time bin, z-scored per feature
  - Red rows = positive coefficient, Blue rows = negative coefficient

Usage:
    python feature_heatmap.py                  # generates both outcomes
    python feature_heatmap.py any_narcolepsy   # any narcolepsy only
    python feature_heatmap.py nt1              # NT1 only
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

sns.set_style('ticks')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RISK_DIR = os.path.join(BASE, '..', 'predictive-modeling', 'risk_score_v2')
FEAT_DIR = os.path.join(BASE, '..', 'predictive-modeling', 'features_update')
MANUSCRIPT_FIG_DIR = os.path.join(BASE, '..', 'manuscript', 'figures')

MAX_YEARS = 2.5
N_BINS = 10
MIN_VISITS = 5

# Colormaps
cmap_pos = plt.cm.Reds
cmap_neg = plt.cm.Blues


# ── Helper functions ───────────────────────────────────────────────────────

def filter_by_min_visits(df, patient_ids, ref_times, max_years, min_visits):
    """Keep only patients with >= min_visits in the [-max_years, 0] window."""
    kept = {}
    for sid in patient_ids:
        if sid not in ref_times:
            continue
        ref_t = ref_times[sid]
        sub = df[df['bdsp_patient_id'] == sid].sort_values('days_since_first_visit')
        t_rel = (sub['days_since_first_visit'].values - ref_t) / 365.25
        n_in_window = ((t_rel >= -max_years) & (t_rel <= 0)).sum()
        if n_in_window >= min_visits:
            kept[sid] = ref_times[sid]
    return kept


def compute_cumulative_features(df, patient_ids, ref_times,
                                feat_col_names, n_bins, max_years):
    """
    Compute mean cumulative feature values in time bins relative to reference.
    Forward-fills missing bins with last known value.
    """
    bin_edges = np.linspace(-max_years, 0, n_bins + 1)
    n_feats = len(feat_col_names)
    feat_sums = np.zeros((n_feats, n_bins))
    feat_counts = np.zeros((n_feats, n_bins))

    for sid in patient_ids:
        if sid not in ref_times:
            continue
        ref_t = ref_times[sid]
        sub = df[df['bdsp_patient_id'] == sid].sort_values('days_since_first_visit')
        if len(sub) == 0:
            continue

        t_rel = (sub['days_since_first_visit'].values - ref_t) / 365.25
        mask = (t_rel >= -max_years) & (t_rel <= 0)
        if mask.sum() == 0:
            continue

        sub_masked = sub[mask]
        t_rel_masked = t_rel[mask]
        feat_vals = sub_masked[feat_col_names].values

        bin_idx = np.digitize(t_rel_masked, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_last = {}
        for vi in range(len(t_rel_masked)):
            bi = bin_idx[vi]
            bin_last[bi] = feat_vals[vi]

        # Forward-fill
        first_bin = min(bin_last.keys())
        last_val = None
        for bi in range(n_bins):
            if bi in bin_last:
                last_val = bin_last[bi]
            elif last_val is not None and bi > first_bin:
                bin_last[bi] = last_val

        for bi, vals in bin_last.items():
            feat_sums[:, bi] += vals
            feat_counts[:, bi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        feat_means = np.where(feat_counts > 0, feat_sums / feat_counts, 0)
    return feat_means


def make_rgba(z_matrix, coefs_positive, vmin, vmax):
    """Build an RGBA image using Reds for + coef rows and Blues for - coef rows."""
    n_rows, n_cols = z_matrix.shape
    rgba = np.zeros((n_rows, n_cols, 4))
    norm_vals = np.clip((z_matrix - vmin) / (vmax - vmin), 0, 1)
    for i in range(n_rows):
        cmap = cmap_pos if coefs_positive[i] else cmap_neg
        for j in range(n_cols):
            rgba[i, j] = cmap(norm_vals[i, j])
    return rgba


def generate_heatmap(outcome, df, nt1_case_ids, nt2_case_ids, ctrl_ids, rng):
    """Generate a feature evolution heatmap for a given outcome."""
    outcome_labels = {
        'any_narcolepsy': 'Any Narcolepsy (NT1 + NT2/IH)',
        'nt1': 'NT1 Only',
    }

    pickle_file = os.path.join(RISK_DIR, f'v2_results_{outcome}.pickle')
    ms_fig_map = {'any_narcolepsy': 'figure10_feature_heatmap_any_narcolepsy.png',
                  'nt1': 'figure11_feature_heatmap_nt1.png'}
    out_file = os.path.join(MANUSCRIPT_FIG_DIR, ms_fig_map[outcome])

    # ── Load final model ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generating heatmap for: {outcome_labels[outcome]}")
    print(f"{'='*60}")

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    fm = data['final_model']
    clf = fm['clf']
    sel_idx = fm['sel_idx']
    feat_names_all = data['feat_names']

    coefs = clf.coef_.ravel()
    nonzero_idx = np.where(coefs != 0)[0]
    final_feat_idx = sel_idx[nonzero_idx]
    final_feat_names = [feat_names_all[i] for i in final_feat_idx]
    final_coefs = coefs[nonzero_idx]

    # Sort by coefficient descending
    order = np.argsort(-final_coefs)
    final_feat_idx = final_feat_idx[order]
    final_feat_names = [final_feat_names[i] for i in order]
    final_coefs = final_coefs[order]

    print(f"  {len(final_feat_names)} features with non-zero L1 coefficients")

    # ── Define cases for this outcome ─────────────────────────────────────
    if outcome == 'any_narcolepsy':
        case_ids = nt1_case_ids | nt2_case_ids
    else:  # nt1
        case_ids = nt1_case_ids

    # ── Compute reference times ───────────────────────────────────────────
    case_diag_t = {}
    for sid in case_ids:
        sub = df[df['bdsp_patient_id'] == sid].sort_values('days_since_first_visit')
        pos_visits = sub[sub['n+_state'] == 1]
        if len(pos_visits) > 0:
            case_diag_t[sid] = pos_visits['days_since_first_visit'].iloc[0]

    ctrl_pseudo_t = {}
    for sid in ctrl_ids:
        sub = df[df['bdsp_patient_id'] == sid]
        if len(sub) > 0:
            ctrl_pseudo_t[sid] = sub['days_since_first_visit'].max()

    # ── Filter by minimum visits ──────────────────────────────────────────
    case_diag_t = filter_by_min_visits(df, case_ids, case_diag_t,
                                        MAX_YEARS, MIN_VISITS)
    ctrl_pseudo_t_filtered = filter_by_min_visits(df, ctrl_ids, ctrl_pseudo_t,
                                                   MAX_YEARS, MIN_VISITS)

    # Match control count to case count
    n_cases = len(case_diag_t)
    ctrl_list = list(ctrl_pseudo_t_filtered.keys())
    if len(ctrl_list) > n_cases:
        ctrl_sample = list(rng.choice(ctrl_list, size=n_cases, replace=False))
    else:
        ctrl_sample = ctrl_list
    ctrl_pseudo_sampled = {sid: ctrl_pseudo_t_filtered[sid] for sid in ctrl_sample}

    print(f"  Using {n_cases} cases and {len(ctrl_pseudo_sampled)} matched controls")

    # ── Map feature names ─────────────────────────────────────────────────
    feat_col_names = [feat_names_all[i] for i in final_feat_idx]
    missing = [f for f in feat_col_names if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features not in parquet")
        keep = [i for i, f in enumerate(feat_col_names) if f in df.columns]
        feat_col_names = [feat_col_names[i] for i in keep]
        final_feat_names = [final_feat_names[i] for i in keep]
        final_coefs = final_coefs[keep]

    # ── Compute matrices ──────────────────────────────────────────────────
    print("  Computing case features...")
    case_matrix = compute_cumulative_features(
        df, case_diag_t.keys(), case_diag_t, feat_col_names, N_BINS, MAX_YEARS)

    print("  Computing control features...")
    ctrl_matrix = compute_cumulative_features(
        df, ctrl_pseudo_sampled.keys(), ctrl_pseudo_sampled,
        feat_col_names, N_BINS, MAX_YEARS)

    # Z-score normalize
    combined = np.concatenate([case_matrix, ctrl_matrix], axis=1)
    feat_mean = combined.mean(axis=1, keepdims=True)
    feat_std = combined.std(axis=1, keepdims=True)
    feat_std[feat_std == 0] = 1
    case_z = (case_matrix - feat_mean) / feat_std
    ctrl_z = (ctrl_matrix - feat_mean) / feat_std

    # ── Plot ──────────────────────────────────────────────────────────────
    print("  Plotting...")
    display_names = []
    for name, coef in zip(final_feat_names, final_coefs):
        short = name[:35] if len(name) > 35 else name
        sign = '+' if coef > 0 else '\u2212'
        display_names.append(f'{short} ({sign})')

    is_positive = np.array([c > 0 for c in final_coefs])

    fig = plt.figure(figsize=(16, max(12, len(display_names) * 0.22)))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)
    ax_cases = fig.add_subplot(gs[0, 0])
    ax_ctrls = fig.add_subplot(gs[0, 1], sharey=ax_cases)
    ax_cb = fig.add_subplot(gs[0, 2])

    vmax_cc = min(max(np.abs(case_z).max(), np.abs(ctrl_z).max()), 3.0)

    rgba_cases = make_rgba(case_z, is_positive, 0, vmax_cc)
    ax_cases.imshow(rgba_cases, aspect='auto',
                    extent=[-MAX_YEARS, 0, len(display_names) - 0.5, -0.5])
    ax_cases.set_title('Cases', fontsize=14, fontweight='bold')
    ax_cases.set_xlabel('Years relative to diagnosis')
    ax_cases.set_yticks(range(len(display_names)))
    ax_cases.set_yticklabels(display_names, fontsize=6)

    rgba_ctrls = make_rgba(ctrl_z, is_positive, 0, vmax_cc)
    ax_ctrls.imshow(rgba_ctrls, aspect='auto',
                    extent=[-MAX_YEARS, 0, len(display_names) - 0.5, -0.5])
    ax_ctrls.set_title('Controls', fontsize=14, fontweight='bold')
    ax_ctrls.set_xlabel('Years relative to last visit')
    plt.setp(ax_ctrls.get_yticklabels(), visible=False)

    # Colorbars
    norm_cc = Normalize(vmin=0, vmax=vmax_cc)
    ax_cb.set_axis_off()
    cb_ax_pos = fig.add_axes([ax_cb.get_position().x0, 0.55,
                               ax_cb.get_position().width, 0.3])
    cb_ax_neg = fig.add_axes([ax_cb.get_position().x0, 0.15,
                               ax_cb.get_position().width, 0.3])
    fig.colorbar(ScalarMappable(norm=norm_cc, cmap=cmap_pos), cax=cb_ax_pos)
    cb_ax_pos.set_ylabel('+ coef', fontsize=9)
    fig.colorbar(ScalarMappable(norm=norm_cc, cmap=cmap_neg), cax=cb_ax_neg)
    cb_ax_neg.set_ylabel('\u2212 coef', fontsize=9)

    fig.suptitle(f'Feature Evolution: {outcome_labels[outcome]}\n'
                 f'({len(display_names)} features with non-zero L1 coefficients, '
                 f'{n_cases} cases vs {len(ctrl_pseudo_sampled)} matched controls, '
                 f'\u2265{MIN_VISITS} visits required)',
                 fontsize=13)

    os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_file}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Determine which outcomes to generate
    if len(sys.argv) > 1:
        outcomes = [sys.argv[1]]
    else:
        outcomes = ['any_narcolepsy', 'nt1']

    # Load parquet data (shared across outcomes)
    print("Loading parquet data...")
    df_nt1 = pd.read_parquet(os.path.join(FEAT_DIR, 'nt1', 'features_3.parquet'))
    df_nt2 = pd.read_parquet(os.path.join(FEAT_DIR, 'nt2ih', 'features_3.parquet'))
    for dfp in [df_nt1, df_nt2]:
        if 'cohort' in dfp.columns:
            dfp.rename(columns={'cohort': 'site'}, inplace=True)

    nt1_label = df_nt1.groupby('bdsp_patient_id')['n+_state'].max()
    nt1_case_ids = set(nt1_label[nt1_label == 1].index)
    nt1_ctrl_ids = set(nt1_label[nt1_label == 0].index)

    nt2_label = df_nt2.groupby('bdsp_patient_id')['n+_state'].max()
    nt2_case_ids = set(nt2_label[nt2_label == 1].index)
    nt2_ctrl_ids = set(nt2_label[nt2_label == 0].index)

    ctrl_ids = (nt1_ctrl_ids | nt2_ctrl_ids) - (nt1_case_ids | nt2_case_ids)

    meta_cols = {'bdsp_patient_id', 'site', 'cohort', 'date', 'n+_state',
                 'days_since_first_visit', 'num_visits_since_first_visit',
                 'filename'}
    feat_cols = sorted(set(df_nt1.columns) - meta_cols)
    keep_cols = ['bdsp_patient_id', 'n+_state', 'days_since_first_visit'] + feat_cols

    df = pd.concat([
        df_nt1[[c for c in keep_cols if c in df_nt1.columns]],
        df_nt2[[c for c in keep_cols if c in df_nt2.columns]],
    ], ignore_index=True)
    df = df.drop_duplicates(subset=['bdsp_patient_id', 'days_since_first_visit'])
    df = df.sort_values(['bdsp_patient_id', 'days_since_first_visit']).reset_index(drop=True)

    print(f"  Total: {df['bdsp_patient_id'].nunique()} patients, {len(df)} visits")

    rng = np.random.RandomState(42)

    for outcome in outcomes:
        generate_heatmap(outcome, df, nt1_case_ids, nt2_case_ids, ctrl_ids, rng)

    print("\nDone!")
