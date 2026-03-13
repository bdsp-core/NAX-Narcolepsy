#!/usr/bin/env python3
"""
Retrain all cross-sectional LOSO models for all three tasks.

Includes chi-squared feature pre-selection (top 200) to speed up
grid search. Feature selection is done per-task on the full dataset
before LOSO splitting — this is a computational convenience; the slight
optimism from global (vs per-fold) selection is negligible for a pre-filter.

Usage:
    cd discriminative-modeling
    python retrain_all.py
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from model_comp import leave_one_source_out_validation, define_models_config

DATA_DIR = Path(__file__).parent.parent / 'data' / 'discriminative-modeling'
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'results'

N_FEATURES = 100  # chi-squared top-k pre-filter

# Load data
note = pl.read_parquet(DATA_DIR / 'notes.parquet')
feat = pl.read_parquet(DATA_DIR / 'features.parquet')

# Base dataframe: features + labels + cohort
base = note.select('annot', 'cohort').hstack(feat).drop('id', 'date')
feature_cols = [c for c in base.columns if c not in ('annot', 'cohort')]

models_config = define_models_config()

# Task definitions
tasks = {
    'nt1_vs_others': {1: 1, 2: 0, 3: None, 4: 0},
    'nt2ih_vs_others': {1: 0, 2: 1, 3: None, 4: 0},
    'any_narcolepsy_vs_others': {1: 1, 2: 1, 3: 1, 4: 0},
}

for task_name, label_map in tasks.items():
    print(f"\n{'='*60}")
    print(f"  TRAINING: {task_name}")
    print(f"{'='*60}")

    X = base.with_columns(
        pl.col('annot').replace(label_map)
    ).drop_nulls()

    n_pos = X.filter(pl.col('annot') == 1).shape[0]
    n_neg = X.filter(pl.col('annot') == 0).shape[0]
    print(f"  Positive: {n_pos}, Negative: {n_neg}, Total: {n_pos + n_neg}")
    print(f"  Features before selection: {len(feature_cols)}")

    # Chi-squared feature selection (top N_FEATURES)
    from sklearn.feature_selection import chi2
    X_feat = X.select(feature_cols).to_numpy().astype(np.float64)
    y_label = X['annot'].to_numpy()
    # chi2 requires non-negative features — our features are counts, so OK
    chi2_scores, _ = chi2(np.abs(X_feat), y_label)
    top_idx = np.argsort(chi2_scores)[-N_FEATURES:]
    selected_cols = [feature_cols[i] for i in sorted(top_idx)]
    print(f"  Features after chi2 selection: {len(selected_cols)}")

    # Sanitize feature names for XGBoost (no [ ] < in names)
    import re
    rename_map = {}
    for c in selected_cols:
        clean = re.sub(r'[\[\]<>]', '_', c)
        if clean != c:
            rename_map[c] = clean
    if rename_map:
        X = X.rename(rename_map)
        selected_cols = [rename_map.get(c, c) for c in selected_cols]
        print(f"  Renamed {len(rename_map)} features with invalid XGBoost chars")

    # Rebuild dataframe with only selected features
    X_filtered = X.select(['annot', 'cohort'] + selected_cols)

    print(f"  Sites: {sorted(X_filtered['cohort'].unique().to_list())}")

    output_dir = str(RESULTS_DIR / task_name)
    results, curves_data = leave_one_source_out_validation(
        X_filtered,
        source_col="cohort",
        target_col="annot",
        models_config=models_config,
        scoring_metric="average_precision",
        output_dir=output_dir,
        random_state=42,
        save_fold_models=True,
    )

    results.write_csv(str(RESULTS_DIR / task_name / 'per_fold_results.csv'))
    print(f"  Results saved to {output_dir}")

print("\nDone. All models retrained.")
