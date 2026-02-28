import sys, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


outcome_of_interest = sys.argv[1].strip().lower()
assert outcome_of_interest in ['nt1', 'nt2ih']

with open(f'data_processed_updated_features_3_{outcome_of_interest}.pickle', 'rb') as f:
    res = pickle.load(f)
    sites = res['sites']
    sids = res['sids']
    X = res['X']  # features
    T = res['T']  # year from first visit
    Y = res['Y']  # 0 or 1 for the outcome
    C = res['C']
    names_feat = res['names_feat'] # feature names

site_names = ['mgb', 'bidmc', 'bch', 'emory', 'stan']

df = pd.DataFrame(data=X, columns=names_feat)
df.insert(0, 'Y', Y)
df.insert(0, 'T', T)
df.insert(0, 'sid', sids)
df.insert(0, 'site', sites)

# Assert no same sid across sites
sid_site = df[['sid', 'site']].drop_duplicates()
assert sid_site.groupby('sid')['site'].nunique().max() == 1, "Some subjects appear in multiple sites!"

sid2ids = df.groupby('sid').indices
unique_sids = np.array(sorted(sid2ids.keys()))
unique_sid_sites = np.array([df.site.iloc[sid2ids[sid][0]] for sid in unique_sids])

# Precompute subject-level statistics
subj_n_visits = {}
subj_max_T = {}
subj_outcome = {}
subj_first_feat = {}
subj_last_feat_div_T = {}

for sid in unique_sids:
    ids = sid2ids[sid]
    rows = df.iloc[ids]
    subj_n_visits[sid] = len(ids)
    max_T = rows['T'].max()
    subj_max_T[sid] = max_T
    subj_outcome[sid] = rows['Y'].iloc[-1]
    subj_first_feat[sid] = rows[names_feat].iloc[0].values
    subj_last_feat_div_T[sid] = rows[names_feat].iloc[-1].values / max_T if max_T > 0 else np.full(len(names_feat), np.nan)

# Group unique sids by site
site_to_sids = {}
for site in site_names:
    site_to_sids[site] = unique_sids[unique_sid_sites == site]


def compute_stats(sid_list):
    """Compute all comparators for a list of subject IDs."""
    n_visits = np.array([subj_n_visits[s] for s in sid_list])
    max_Ts = np.array([subj_max_T[s] for s in sid_list])
    outcomes = np.array([subj_outcome[s] for s in sid_list])

    outcome_mask = outcomes == 1
    censored_mask = outcomes == 0

    stats = {
        'Number of visits': np.median(n_visits),
        'Follow-up time (years)': np.median(max_Ts),
        'Follow-up time in outcome (years)': np.median(max_Ts[outcome_mask]) if outcome_mask.sum() > 0 else np.nan,
        'Follow-up time in censored (years)': np.median(max_Ts[censored_mask]) if censored_mask.sum() > 0 else np.nan,
        'Percent outcome (%)': np.mean(outcomes) * 100,
    }

    first_feats = np.array([subj_first_feat[s] for s in sid_list])
    last_feats_divT = np.array([subj_last_feat_div_T[s] for s in sid_list])

    for fi, feat in enumerate(names_feat):
        stats[f'First visit mean: {feat}'] = np.mean(first_feats[:, fi])
        stats[f'Last visit rate mean: {feat}'] = np.nanmean(last_feats_divT[:, fi])

    return stats


nbt = 1000
n_jobs = 14


def _bootstrap_site(site, site_sids, nbt):
    """Run all bootstrap iterations for one site."""
    pe = compute_stats(site_sids)
    bt_stats = []
    for bti in range(nbt):
        rng_bt = np.random.default_rng(2026 + bti)
        bt_sids = rng_bt.choice(site_sids, len(site_sids), replace=True)
        bt_stats.append(compute_stats(bt_sids))
    site_results = {}
    for comp in pe:
        bt_vals = [bt[comp] for bt in bt_stats]
        lb, ub = np.nanpercentile(bt_vals, [2.5, 97.5])
        site_results[comp] = (pe[comp], lb, ub)
    return site, site_results


# Parallelize across sites (5 jobs), each running all nbt iterations
site_results_list = Parallel(n_jobs=len(site_names))(
    delayed(_bootstrap_site)(site, site_to_sids[site], nbt) for site in site_names)
results = dict(site_results_list)

# Build output DataFrame
comparator_names = list(results[site_names[0]].keys())
rows = []
stan_idx = site_names.index('stan')
for comp in comparator_names:
    row = {'comparator': comp}
    point_ests = [results[site][comp][0] for site in site_names]
    for site in site_names:
        pe, lb, ub = results[site][comp]
        row[site] = f'{pe:.3f} ({lb:.3f}-{ub:.3f})'
    row['stan_extreme'] = np.argmin(point_ests) == stan_idx or np.argmax(point_ests) == stan_idx
    rows.append(row)

out_df = pd.DataFrame(rows)
print(out_df)
out_df.to_excel(f'result_site_comparison_{outcome_of_interest}.xlsx', index=False)
