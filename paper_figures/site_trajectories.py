"""
eFigure 16: Site-stratified risk score trajectories (sensitivity analysis).

Three pairs of subplots (one per outcome). Each pair:
  - Top: risk score trajectories with one curve per contributing hospital
  - Bottom: time-dependent AUROC per site

Controls are always the pooled BIDMC + MGH general-population controls.

Usage:
    python site_trajectories.py
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (apply_style, savefig as pub_savefig, add_panel_label,
                       SITE_COLORS, CTRL_COLOR, DOUBLE_COL_IN,
                       LINE_WIDTH, LINE_WIDTH_THICK, LINE_WIDTH_THIN,
                       FONT_SIZE_ANNOTATION, FONT_SIZE_LEGEND)
apply_style()

BASE = os.path.dirname(os.path.abspath(__file__))
RISK_DIR = os.path.join(BASE, '..', 'predictive-modeling', 'risk_score_v2')
MANUSCRIPT_FIG_DIR = os.path.join(BASE, '..', 'manuscript', 'figures')

SITE_ORDER = ['BCH', 'BIDMC', 'Emory', 'MGH', 'Stanford']

OUTCOME_LABELS = {
    'any_narcolepsy': 'Any Narcolepsy',
    'nt1': 'NT1 Only',
    'nt2ih': 'NT2/IH Only',
}

PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

MAX_YEARS = 5.0


def _logit(p, eps=1e-3):
    """Logit transform with clipping to avoid infinities."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _derive_site(patient_id):
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


def _sliding_window_mean_ci(t_rel, values, sids, window=1.5, step=0.1,
                             min_patients=5, n_boot=200):
    """Patient-level mean with bootstrap 95% CI in sliding windows."""
    t_centers = np.arange(-MAX_YEARS + window / 2, 0.0 + step, step)
    mean_vals = np.full(len(t_centers), np.nan)
    ci_lo = np.full(len(t_centers), np.nan)
    ci_hi = np.full(len(t_centers), np.nan)
    rng = np.random.RandomState(42)

    for i, tc in enumerate(t_centers):
        mask = (t_rel >= tc - window / 2) & (t_rel < tc + window / 2)
        if mask.sum() == 0:
            continue
        df_win = pd.DataFrame({'sid': sids[mask], 'val': values[mask]})
        pat_means = df_win.groupby('sid')['val'].mean().values
        if len(pat_means) < min_patients:
            continue
        mean_vals[i] = np.mean(pat_means)
        boot_means = np.array([
            np.mean(rng.choice(pat_means, size=len(pat_means), replace=True))
            for _ in range(n_boot)
        ])
        ci_lo[i] = np.percentile(boot_means, 2.5)
        ci_hi[i] = np.percentile(boot_means, 97.5)

    return t_centers, mean_vals, ci_lo, ci_hi


def _sliding_window_auc_vs_flat_ctrl(case_t, case_scores, case_sids,
                                      ctrl_pat_scores,
                                      window=1.0, step=0.1, min_cases=5):
    """Time-varying AUROC: case scores per window vs all control patient means."""
    t_centers = np.arange(-MAX_YEARS + window / 2, 0.0 + step, step)
    auc_vals = np.full(len(t_centers), np.nan)
    n_cases = np.full(len(t_centers), 0)

    for i, tc in enumerate(t_centers):
        mask = (case_t >= tc - window / 2) & (case_t < tc + window / 2)
        if mask.sum() == 0:
            continue
        df_win = pd.DataFrame({'sid': case_sids[mask], 'score': case_scores[mask]})
        case_pat = df_win.groupby('sid')['score'].mean().values
        n_cases[i] = len(case_pat)
        if len(case_pat) < min_cases:
            continue
        all_scores = np.concatenate([case_pat, ctrl_pat_scores])
        all_labels = np.concatenate([np.ones(len(case_pat)),
                                     np.zeros(len(ctrl_pat_scores))])
        if len(np.unique(all_labels)) < 2:
            continue
        auc_vals[i] = roc_auc_score(all_labels, all_scores)

    return t_centers, auc_vals, n_cases


def _sigmoid(t, L, U, k, t0):
    """Sigmoid: L + (U-L) / (1 + exp(-k*(t-t0)))."""
    return L + (U - L) / (1.0 + np.exp(-k * (t - t0)))


def _fit_sigmoid_auroc(t_centers, auc_vals, n_cases):
    """Fit a sigmoid to empirical AUROC values. Returns (t_fine, auroc_fit) or None."""
    valid = ~np.isnan(auc_vals)
    if valid.sum() < 4:
        return None
    t_obs = t_centers[valid]
    y_obs = auc_vals[valid]
    w_obs = n_cases[valid]
    t_fine = np.linspace(-5.0, 0.0, 300)
    try:
        popt, _ = curve_fit(
            _sigmoid, t_obs, y_obs,
            p0=[0.7, 0.95, 1.5, -2.5],
            bounds=([0.5, 0.7, 0.1, -5.0], [0.9, 1.0, 10.0, 0.0]),
            sigma=1.0 / np.sqrt(np.clip(w_obs, 1, None)),
            maxfev=10000)
        return t_fine, _sigmoid(t_fine, *popt)
    except Exception:
        return None


def main():
    outcomes = ['any_narcolepsy', 'nt1', 'nt2ih']
    n_cols = len(outcomes)

    fig, axes = plt.subplots(2, n_cols, figsize=(DOUBLE_COL_IN * 1.3, 6.5),
                             sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1]})

    for col, outcome in enumerate(outcomes):
        # Load pickle
        pickle_file = os.path.join(RISK_DIR, f'v2_results_{outcome}.pickle')
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        results = data['results']
        pat_info = data['pat_info']
        r = results[0.5]['pooled']

        sids_arr = r.get('traj_sids', r['sids'])
        scores = r.get('traj_scores', r['scores'])
        raw_scores = r.get('traj_raw_scores')
        y_arr = r.get('traj_y', r['y'])
        T_arr = r.get('traj_T', r['T'])

        dtmp = pd.DataFrame({'sid': sids_arr, 'score': scores, 'y': y_arr, 'T': T_arr})
        if raw_scores is not None:
            dtmp['raw_score'] = raw_scores
        dtmp = dtmp.dropna(subset=['score'])
        has_raw = 'raw_score' in dtmp.columns
        dtmp['site'] = dtmp['sid'].apply(_derive_site)

        pat_label = dtmp.groupby('sid')['y'].max()
        case_sids_all = set(pat_label[pat_label == 1].index)
        ctrl_sids_all = set(pat_label[pat_label == 0].index)

        # Build case data aligned to diagnosis
        case_rows = []
        for sid in case_sids_all:
            sub = dtmp[dtmp['sid'] == sid].sort_values('T')
            info = pat_info.get(sid)
            if info is None or info['diag_t'] is None:
                continue
            diag_t_yr = info['diag_t'] / 365.25
            site = _derive_site(sid)
            for _, row in sub.iterrows():
                t_rel = row['T'] - diag_t_yr
                if -MAX_YEARS - 0.1 <= t_rel <= 0.1:
                    r_dict = {
                        't_rel': t_rel, 'score': row['score'],
                        'sid': sid, 'site': site
                    }
                    if has_raw:
                        r_dict['raw_score'] = row['raw_score']
                    case_rows.append(r_dict)
        case_df = pd.DataFrame(case_rows)

        # Controls: patient-level means (pooled BIDMC + MGH)
        ctrl_df = dtmp[dtmp['y'] == 0]
        ctrl_scores = ctrl_df.groupby('sid')['score'].mean().values
        ctrl_raw_scores = (ctrl_df.groupby('sid')['raw_score'].mean().values
                           if has_raw else None)
        n_ctrl = len(ctrl_scores)
        ctrl_mean = np.mean(ctrl_scores)
        rng_boot = np.random.RandomState(42)
        boot_means = np.array([
            np.mean(rng_boot.choice(ctrl_scores, size=n_ctrl, replace=True))
            for _ in range(200)
        ])
        ctrl_ci_lo = np.percentile(boot_means, 2.5)
        ctrl_ci_hi = np.percentile(boot_means, 97.5)

        ax_traj = axes[0, col]
        ax_auc = axes[1, col]

        # Logit-scale tick positions
        prob_ticks = [0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
        logit_ticks = [_logit(p) for p in prob_ticks]

        # Plot control band
        ax_traj.axhspan(_logit(ctrl_ci_lo), _logit(ctrl_ci_hi),
                         color=CTRL_COLOR, alpha=0.12)
        ax_traj.axhline(_logit(ctrl_mean), color=CTRL_COLOR, lw=LINE_WIDTH,
                         ls='--', alpha=0.7)

        # Plot each site's case trajectory
        sites_present = sorted(case_df['site'].unique(),
                                key=lambda s: SITE_ORDER.index(s) if s in SITE_ORDER else 99)

        for site in sites_present:
            site_df = case_df[case_df['site'] == site]
            n_site = site_df['sid'].nunique()
            color = SITE_COLORS.get(site, '#888888')

            t_rel = site_df['t_rel'].values
            vals = site_df['score'].values
            sids = site_df['sid'].values

            tc, mu, lo, hi = _sliding_window_mean_ci(
                t_rel, vals, sids, window=1.5, step=0.1, min_patients=5)

            valid = ~np.isnan(mu)
            if valid.sum() == 0:
                continue

            ax_traj.plot(tc[valid], _logit(mu[valid]), color=color,
                          lw=LINE_WIDTH_THICK, label=f'{site} (n={n_site})')
            ax_traj.fill_between(tc[valid], _logit(lo[valid]),
                                  _logit(hi[valid]), color=color, alpha=0.10)

            # AUROC per site (use raw decision_function scores if available)
            auc_vals_site = (site_df['raw_score'].values if has_raw else vals)
            auc_ctrl = ctrl_raw_scores if has_raw else ctrl_scores
            tc_a, auc, nc_a = _sliding_window_auc_vs_flat_ctrl(
                t_rel, auc_vals_site, sids, auc_ctrl,
                window=1.0, step=0.1, min_cases=5)
            valid_a = ~np.isnan(auc)
            if valid_a.sum() > 0:
                ax_auc.plot(tc_a[valid_a], auc[valid_a], color=color,
                             lw=LINE_WIDTH)

        # Control label
        ax_traj.text(-4.9, _logit(ctrl_mean) + 0.15,
                     f'Controls (n={n_ctrl})',
                     fontsize=FONT_SIZE_ANNOTATION, color=CTRL_COLOR,
                     va='bottom')

        # Formatting - trajectories (logit scale)
        ax_traj.set_ylim(_logit(0.08), _logit(0.995))
        ax_traj.set_yticks(logit_ticks)
        ax_traj.set_yticklabels([f'{p:.0%}' for p in prob_ticks])
        ax_traj.set_ylabel('Risk score' if col == 0 else '')
        ax_traj.set_title(OUTCOME_LABELS[outcome])
        ax_traj.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND - 1,
                        frameon=False)
        for yv in prob_ticks:
            ax_traj.axhline(_logit(yv), color='gray', lw=0.4, alpha=0.25)
        add_panel_label(ax_traj, PANEL_LABELS[col])

        # Formatting - AUROC
        ax_auc.set_ylim(0.45, 1.02)
        ax_auc.set_ylabel('AUROC' if col == 0 else '')
        ax_auc.set_xlabel('Years relative to diagnosis')
        ax_auc.set_xlim(-5.15, 0.15)
        ax_auc.set_xticks(range(-5, 1))
        ax_auc.axhline(0.5, color='gray', lw=0.6, ls=':', alpha=0.5)
        for yv in [0.8, 0.9]:
            ax_auc.axhline(yv, color='gray', lw=0.4, ls='--', alpha=0.3)
        add_panel_label(ax_auc, PANEL_LABELS[col + n_cols])

    fig.tight_layout()
    os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)
    out_path = os.path.join(MANUSCRIPT_FIG_DIR, 'efigure16_site_trajectories.png')
    pub_savefig(fig, out_path)
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
