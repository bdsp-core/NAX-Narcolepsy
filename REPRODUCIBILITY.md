# Reproducibility Guide

## Data Sources

All data originates from the Brain Data Science Platform (BDSP), a multi-site
academic medical center EHR network (5 sites: BCH, BIDMC, Emory, MGH, Stanford).
Raw EHR data requires BDSP credentials and institutional data use agreements.

## Ground Truth Labels

**Single source of truth**: `data/discriminative-modeling/predictive_annotation.parquet`
- 596 confirmed narcolepsy cases (282 NT1, 314 NT2/IH)
- Each case has: `bdsp_patient_id`, `diagnosis` (NT1 or NT2IH), `date` (diagnosis date)
- This file was produced by physician review and is authoritative for both pipelines.

Note-level annotations in `notes.parquet` are the original manual annotations
(the `annot` column). These are used directly — no label reconciliation is applied.

## Pipeline 1: Cross-Sectional Classification

### Feature Extraction (notes → features)
```
Input:  data/discriminative-modeling/notes.parquet (9,356 notes)
        data/discriminative-modeling/icd.parquet
        data/discriminative-modeling/med.parquet
Config: discriminative-modeling/config.yaml
Code:   discriminative-modeling/narcolepsy_model.py
Output: data/discriminative-modeling/features.parquet (9,356 × 926)
```

Features are extracted from note text (keyword counts with negation detection),
ICD codes (regex-matched narcolepsy patterns within ±6 months), and medications
(27 narcolepsy-related drugs within ±6 months). Feature extraction is
**label-independent** — changing annotations does not change features.

### Model Training & Evaluation
```
Code:   discriminative-modeling/retrain_all.py
        discriminative-modeling/model_comp.py
        paper_figures/roc_prc.ipynb (LOSO CV + figures)
Models: data/results/{nt1,nt2ih,any_narcolepsy}_vs_others/fold_models_*/*.pkl
```

Three classification tasks:
- **NT1 vs others**: NT1 notes vs Absent notes (best: RandomForest, AUROC=0.997)
- **NT2/IH vs others**: NT2/IH notes vs Absent notes (best: XGBoost, AUROC=0.988)
- **Any narcolepsy vs others**: NT1 + NT2/IH + Unclear notes vs Absent notes (best: GradientBoosting, AUROC=0.990)

Four classifiers (LR, RF, GBT, XGB) are trained with LOSO cross-validation.
Chi-squared feature selection (top 100 of 924 features) is applied before training.
Labels come from the `annot` column in `notes.parquet`.

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
        predictive-modeling/risk_score_v2/v2_loso_by_site.csv
```

Three outcome models: any_narcolepsy, NT1, NT2/IH.

SGDClassifier with L1 penalty, balanced minibatches, chi-squared feature
selection (top 100). 5-fold stratified CV (primary) + LOSO CV (secondary,
restricted to BIDMC and MGH with ≥50 controls). Training window: [-2.5, -0.5]
years before diagnosis. Testing/scoring window: [-5, 0] years.

## Figure Generation
```
bash build_manuscript_figures.sh
```

This runs all figure-generating scripts and notebooks in sequence.
Individual figures can be regenerated independently (see script for details).

### Main text figures (3)
| Figure | Description | Source |
|--------|-------------|--------|
| 1 | ROC & PRC curves (best model, 3 tasks) | `paper_figures/roc_prc.ipynb` |
| 2 | Risk score trajectories | `predictive-modeling/risk_score_v2/risk_score_v2.py` |
| 3 | NNT analysis | `predictive-modeling/risk_score_v2/risk_score_v2.py` |

### Supplementary figures (16)
| eFigure | Description | Source |
|---------|-------------|--------|
| 1 | CONSORT cross-sectional | `paper_figures/consort_diagrams.py` |
| 2 | CONSORT longitudinal | `paper_figures/consort_diagrams.py` |
| 3 | ROC & PRC NT1 (all models) | `paper_figures/roc_prc.ipynb` |
| 4 | ROC & PRC NT2/IH (all models) | `paper_figures/roc_prc.ipynb` |
| 5 | ROC & PRC Any Narcolepsy (all models) | `paper_figures/roc_prc.ipynb` |
| 6 | Confusion matrices NT1 | `paper_figures/confusion_matrices.ipynb` |
| 7 | Confusion matrices NT2/IH | `paper_figures/confusion_matrices.ipynb` |
| 8 | Confusion matrices Any Narcolepsy | `paper_figures/confusion_matrices.ipynb` |
| 9 | Predictive model performance | `predictive-modeling/risk_score_v2/risk_score_v2.py` |
| 10 | Risk score distributions | `predictive-modeling/risk_score_v2/risk_score_v2.py` |
| 11 | Top predictive features | `predictive-modeling/risk_score_v2/risk_score_v2.py` |
| 12 | Feature heatmap (any narcolepsy) | `paper_figures/feature_heatmap.py` |
| 13 | Feature heatmap (NT1) | `paper_figures/feature_heatmap.py` |
| 14 | Feature heatmap (NT2/IH) | `paper_figures/feature_heatmap.py` |
| 15 | Swimmer plot | `paper_figures/swimmer_plot.py` * |
| 16 | Site-stratified trajectories | `paper_figures/site_trajectories.py` |

\* eFigure 15 (swimmer plot) requires `bdsp_narco_swimmer.parquet`, which is
derived from the full EHR dataset on [bdsp.io](https://bdsp.io) and is not
included in this repository. To regenerate from scratch, query the BDSP database for
hospital record spans, diagnosis dates, and death dates for the narcolepsy
cohort, export as `data/discriminative-modeling/bdsp_narco_swimmer.parquet`,
then run `python paper_figures/swimmer_plot.py`.

## Verification

```
python manuscript/verify_manuscript_numbers.py
```

Validates 117 quantitative claims in the manuscript against computed values
from the data and model outputs (cohort counts, annotation counts, model
performance metrics, feature counts, NNT operating points, per-site LOSO).

## Full Rebuild Sequence

1. Run `discriminative-modeling/retrain_all.py`                          # Retrain cross-sectional
2. `python predictive-modeling/risk_score_v2/risk_score_v2.py all`       # Retrain longitudinal
3. `bash build_manuscript_figures.sh`                                     # All figures
4. `python manuscript/verify_manuscript_numbers.py`                       # Validate numbers
