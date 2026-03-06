# Reproducibility Guide

## Data Sources

All data originates from the Brain Data Science Platform (BDSP), a multi-site
academic medical center EHR network (5 sites: BCH, BIDMC, Emory, MGB, Stanford).
Raw EHR data requires BDSP credentials and institutional data use agreements.

## Ground Truth Labels

**Single source of truth**: `data/discriminative-modeling/predictive_annotation.parquet`
- 596 confirmed narcolepsy cases (282 NT1, 314 NT2/IH)
- Each case has: `bdsp_patient_id`, `diagnosis` (NT1 or NT2IH), `date` (diagnosis date)
- This file was produced by physician review and is authoritative for both pipelines.

`data/reconcile_labels.py` synchronizes note-level annotations (`notes.parquet`)
with the patient-level ground truth. Run it after any label changes.

## Pipeline 1: Cross-Sectional Classification

### Feature Extraction (notes → features)
```
Input:  data/discriminative-modeling/notes.parquet (8,990 notes)
        data/discriminative-modeling/icd.parquet
        data/discriminative-modeling/med.parquet
Config: discriminative-modeling/config.yaml
Code:   discriminative-modeling/narcolepsy_model.py
Output: data/discriminative-modeling/features.parquet (8,990 × 926)
```

Features are extracted from note text (keyword counts with negation detection),
ICD codes (regex-matched narcolepsy patterns within ±6 months), and medications
(27 narcolepsy-related drugs within ±6 months). Feature extraction is
**label-independent** — changing annotations does not change features.

### Model Training & Evaluation
```
Code:   discriminative-modeling/model_comp.py
        paper_figures/roc_prc.ipynb (LOSO CV + figures)
Models: data/results/{nt1,nt2ih}_vs_others/fold_models_*/*.pkl
```

Four classifiers (LR, RF, GBT, XGB) are trained with LOSO cross-validation.
Labels come from the `annot` column in `notes.parquet` (synced with ground truth).

## Pipeline 2: Longitudinal Prediction

### Feature Extraction (visit-level → longitudinal cumulative)
```
Input:  data/discriminative-modeling/features.parquet
        data/discriminative-modeling/predictive_annotation.parquet
Code:   predictive-modeling/predictive-model.ipynb
Output: predictive-modeling/features_update/nt1/features_3.parquet
        predictive-modeling/features_update/nt2ih/features_3.parquet
```

Visit-level features are aggregated into cumulative counts per patient-visit.
The `n+_state` column marks pre- vs. post-diagnosis visits. Diagnosis dates
come from `predictive_annotation.parquet`.

### Model Training & Evaluation
```
Code:   predictive-modeling/risk_score_v2/risk_score_v2.py
Output: predictive-modeling/risk_score_v2/v2_results_*.pickle
        predictive-modeling/risk_score_v2/v2_summary_*.csv
```

SGDClassifier with L1 penalty, balanced minibatches, chi-squared feature
selection (top 100). 5-fold stratified CV + LOSO. Training window: [-2.5, -0.5]
years before diagnosis.

## Figure Generation
```
bash build_manuscript_figures.sh
```

This runs all figure-generating scripts and notebooks in sequence.
Individual figures can be regenerated independently (see script for details).

## Full Rebuild Sequence

1. `python data/reconcile_labels.py`              # Sync labels
2. Run `discriminative-modeling/` notebook          # Retrain cross-sectional
3. `python predictive-modeling/risk_score_v2/risk_score_v2.py both`  # Retrain longitudinal
4. `bash build_manuscript_figures.sh`               # All figures
5. `python manuscript/verify_manuscript_numbers.py` # Validate numbers
