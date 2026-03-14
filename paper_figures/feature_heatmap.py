"""
Feature evolution heatmap: shows the difference between case and control
cumulative feature values over the 5-year pre-diagnostic window.

Each figure is a single heatmap (cases minus controls):
  - Rows (y-axis): features with non-zero L1 coefficients, sorted by coefficient
  - Columns (x-axis): time bins from -5 years to 0 (diagnosis)
  - Color: z-scored difference in patient-normalized cumulative counts
  - Red rows = positive coefficient, Blue rows = negative coefficient

Patient normalization: each patient's cumulative feature trajectory is divided
by their own final (maximum) cumulative value, placing all patients on a [0, 1]
scale.  This handles left censoring — patients who only appear in later bins
don't dilute earlier bins with low absolute counts.

Three separate figures:
  eFigure 12: Any Narcolepsy (NT1 + NT2/IH)
  eFigure 13: NT1 Only
  eFigure 14: NT2/IH Only

Usage:
    python feature_heatmap.py                  # generates all 3 outcomes
    python feature_heatmap.py any_narcolepsy   # any narcolepsy only
    python feature_heatmap.py nt1              # NT1 only
    python feature_heatmap.py nt2ih            # NT2/IH only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, add_panel_label, savefig as pub_savefig
apply_style()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RISK_DIR = os.path.join(BASE, '..', 'predictive-modeling', 'risk_score_v2')
FEAT_DIR = os.path.join(BASE, '..', 'data', 'predictive-modeling')
MANUSCRIPT_FIG_DIR = os.path.join(BASE, '..', 'manuscript', 'figures')

MAX_YEARS = 5.0
N_BINS = 20
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


def compute_normalized_cumulative(df, patient_ids, ref_times,
                                  feat_col_names, n_bins, max_years):
    """
    Compute mean *patient-normalized* cumulative feature values in time bins.

    For each patient, cumulative feature values are divided by the patient's
    own final cumulative value (at diagnosis / last visit in window), placing
    all patients on a [0, 1] scale.  This handles left censoring: patients
    entering in later bins contribute their normalized fraction rather than
    their raw (low) absolute count.

    Forward-fills missing bins with last known value.
    Only patients with non-zero final value for a given feature contribute
    to that feature's average (features never mentioned are skipped).
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
        feat_vals = sub_masked[feat_col_names].values  # (n_visits, n_feats)

        # Final cumulative value in window (for normalization)
        final_vals = feat_vals[-1]  # last visit in window

        bin_idx = np.digitize(t_rel_masked, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # Record last value per bin
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

        # Normalize per feature and accumulate
        for bi, vals in bin_last.items():
            for fi in range(n_feats):
                if final_vals[fi] > 0:
                    feat_sums[fi, bi] += vals[fi] / final_vals[fi]
                    feat_counts[fi, bi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        feat_means = np.where(feat_counts > 0, feat_sums / feat_counts, 0)
    return feat_means


def compute_pooled_normalized(df, patient_ids, feat_col_names, n_bins):
    """
    Compute mean patient-normalized cumulative feature values for controls,
    pooled across all visits and replicated across time bins.

    Controls have no diagnosis date so all visits are treated equivalently.
    For each control, the mean normalized value across all visits is computed
    (each visit's cumulative value / final cumulative value).  This is then
    averaged across patients and tiled across bins.
    """
    n_feats = len(feat_col_names)
    # Per-feature: accumulate (sum of per-patient means, count of patients)
    feat_sum = np.zeros(n_feats)
    feat_count = np.zeros(n_feats)

    for sid in patient_ids:
        sub = df[df['bdsp_patient_id'] == sid].sort_values('days_since_first_visit')
        if len(sub) == 0:
            continue
        feat_vals = sub[feat_col_names].values  # (n_visits, n_feats)
        final_vals = feat_vals[-1]

        for fi in range(n_feats):
            if final_vals[fi] > 0:
                # Mean normalized value across all visits
                mean_norm = feat_vals[:, fi].mean() / final_vals[fi]
                feat_sum[fi] += mean_norm
                feat_count[fi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        grand_mean = np.where(feat_count > 0, feat_sum / feat_count, 0)
    return np.tile(grand_mean[:, np.newaxis], (1, n_bins))


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


OUTCOME_LABELS = {
    'any_narcolepsy': 'Any Narcolepsy (NT1 + NT2/IH)',
    'nt1': 'NT1 Only',
    'nt2ih': 'NT2/IH Only',
}

MS_FIG_MAP = {
    'any_narcolepsy': 'efigure12_feature_heatmap_any_narcolepsy.png',
    'nt1': 'efigure13_feature_heatmap_nt1.png',
    'nt2ih': 'efigure14_feature_heatmap_nt2ih.png',
}


def generate_heatmap(outcome, df, nt1_case_ids, nt2_case_ids, ctrl_ids, rng):
    """Generate a single-panel difference heatmap (cases - controls)."""

    pickle_file = os.path.join(RISK_DIR, f'v2_results_{outcome}.pickle')
    out_file = os.path.join(MANUSCRIPT_FIG_DIR, MS_FIG_MAP[outcome])

    # ── Load final model ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generating heatmap for: {OUTCOME_LABELS[outcome]}")
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
    elif outcome == 'nt1':
        case_ids = nt1_case_ids
    else:  # nt2ih
        case_ids = nt2_case_ids

    # ── Compute reference times ───────────────────────────────────────────
    case_diag_t = {}
    for sid in case_ids:
        sub = df[df['bdsp_patient_id'] == sid].sort_values('days_since_first_visit')
        pos_visits = sub[sub['n+_state'] == 1]
        if len(pos_visits) > 0:
            case_diag_t[sid] = pos_visits['days_since_first_visit'].iloc[0]

    # ── Filter cases by minimum visits ───────────────────────────────────
    case_diag_t = filter_by_min_visits(df, case_ids, case_diag_t,
                                        MAX_YEARS, MIN_VISITS)

    # ── Select controls (min visits filter, then match count to cases) ──
    ctrl_eligible = []
    for sid in ctrl_ids:
        sub = df[df['bdsp_patient_id'] == sid]
        if len(sub) >= MIN_VISITS:
            ctrl_eligible.append(sid)

    n_cases = len(case_diag_t)
    if len(ctrl_eligible) > n_cases:
        ctrl_sample = list(rng.choice(ctrl_eligible, size=n_cases, replace=False))
    else:
        ctrl_sample = ctrl_eligible

    print(f"  Using {n_cases} cases and {len(ctrl_sample)} matched controls")

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
    case_matrix = compute_normalized_cumulative(
        df, case_diag_t.keys(), case_diag_t, feat_col_names, N_BINS, MAX_YEARS)

    print("  Computing control features...")
    ctrl_matrix = compute_pooled_normalized(
        df, ctrl_sample, feat_col_names, N_BINS)

    # Difference: cases - controls
    diff_matrix = case_matrix - ctrl_matrix

    # Z-score normalize the difference per feature
    feat_mean = diff_matrix.mean(axis=1, keepdims=True)
    feat_std = diff_matrix.std(axis=1, keepdims=True)
    feat_std[feat_std == 0] = 1
    diff_z = (diff_matrix - feat_mean) / feat_std

    # ── Plot ──────────────────────────────────────────────────────────────
    print("  Plotting...")
    display_names = []
    for name, coef in zip(final_feat_names, final_coefs):
        short = name[:35] if len(name) > 35 else name
        sign = '+' if coef > 0 else '\u2212'
        display_names.append(f'{short} ({sign})')

    is_positive = np.array([c > 0 for c in final_coefs])

    fig = plt.figure(figsize=(10, max(12, len(display_names) * 0.22)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.04], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])

    vmax = min(np.abs(diff_z).max(), 3.0)

    rgba = make_rgba(diff_z, is_positive, 0, vmax)
    ax.imshow(rgba, aspect='auto',
              extent=[-MAX_YEARS, 0, len(display_names) - 0.5, -0.5])
    ax.set_xlabel('Years relative to diagnosis')
    ax.set_xticks(range(-int(MAX_YEARS), 1))
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=6)

    # Colorbars
    norm = Normalize(vmin=0, vmax=vmax)
    ax_cb.set_axis_off()
    cb_ax_pos = fig.add_axes([ax_cb.get_position().x0, 0.55,
                               ax_cb.get_position().width, 0.3])
    cb_ax_neg = fig.add_axes([ax_cb.get_position().x0, 0.15,
                               ax_cb.get_position().width, 0.3])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_pos), cax=cb_ax_pos)
    cb_ax_pos.set_ylabel('+ coef (case > ctrl)', fontsize=8)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_neg), cax=cb_ax_neg)
    cb_ax_neg.set_ylabel('\u2212 coef (ctrl > case)', fontsize=8)

    fig.suptitle(f'Feature Difference (Cases \u2212 Controls): '
                 f'{OUTCOME_LABELS[outcome]}\n'
                 f'({len(display_names)} features, '
                 f'{n_cases} cases vs {len(ctrl_sample)} matched controls, '
                 f'\u2265{MIN_VISITS} visits)',
                 fontsize=12)

    os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)
    pub_savefig(fig, out_file)
    plt.close()
    print(f"  Saved to {out_file}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Determine which outcomes to generate
    if len(sys.argv) > 1:
        outcomes = [sys.argv[1]]
    else:
        outcomes = ['any_narcolepsy', 'nt1', 'nt2ih']

    # Load parquet data (shared across outcomes)
    # Only n+ (cases) and controls (general hospital population); n- excluded
    print("Loading parquet data...")

    def _load_task_data(task_dir):
        """Load n+ and controls parquets for a task; rename id->bdsp_patient_id."""
        parts = []
        for fname in ['n+_features_3.parquet', 'controls_features_3.parquet']:
            fpath = os.path.join(FEAT_DIR, task_dir, fname)
            if os.path.exists(fpath):
                dfp = pd.read_parquet(fpath)
                if 'id' in dfp.columns and 'bdsp_patient_id' not in dfp.columns:
                    dfp.rename(columns={'id': 'bdsp_patient_id'}, inplace=True)
                parts.append(dfp)
        return pd.concat(parts, ignore_index=True)

    df_nt1 = _load_task_data('nt1')
    df_nt2 = _load_task_data('nt2ih')

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
