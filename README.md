# NAX-Narcolepsy

NLP-based narcolepsy detection from electronic health record (EHR) clinical notes. This repository contains two complementary approaches:

1. **Discriminative Modeling** -- Classifies individual clinical notes as belonging to patients with or without narcolepsy (NT1, NT2/IH).
2. **Predictive Modeling** -- Computes a longitudinal risk score from pre-diagnostic clinical notes to identify patients likely to be diagnosed with narcolepsy in the future.

## Repository Structure

```
NAX-Narcolepsy/
├── discriminative-modeling/       # Note-level classification
│   ├── narcolepsy_model.py        # Feature extraction & prediction pipeline
│   ├── model_comp.py              # Model training & evaluation framework
│   ├── discriminative-model.ipynb # Usage instructions and example workflow
│   ├── config.yaml                # Feature definitions and model paths
│   ├── env.toml                   # Environment configuration
│   └── models/                    # Pre-trained classifiers
│       ├── nt1_vs_not.joblib
│       ├── nt2_vs_not.joblib
│       └── nt12_vs_not.joblib
│
├── predictive-modeling/           # Pre-diagnostic risk scores
│   ├── features_update/           # Input feature data (parquet)
│   │   ├── nt1/
│   │   └── nt2ih/
│   ├── risk_score_v2/             # Active risk score pipeline
│   │   ├── risk_score_v2.py       # Main training/evaluation/plotting script
│   │   └── METHODS.md             # Detailed methodology
│   └── pooled-logistic-regression/  # Alternative PLR approach (archived)
│
├── paper_figures/                 # Notebooks & scripts to reproduce paper figures
│   ├── confusion_matrices.ipynb   # Confusion matrix plots
│   ├── roc_prc.ipynb              # ROC and precision-recall curves
│   ├── swimmer_plot.ipynb         # Swimmer plot of patient timelines
│   └── feature_heatmap.py         # Feature evolution heatmaps (cases vs controls)
│
├── timeline-viewer/               # Annotation tool (git submodule)
│
└── LICENSE
```

## Discriminative Modeling

Classifies whether a clinical note belongs to a patient with narcolepsy. Uses keyword matching with negation detection, ICD code features, and medication features extracted from clinical notes.

### Models

Three pre-trained Random Forest classifiers:
- `nt1_vs_not` -- NT1 vs. non-narcolepsy
- `nt2_vs_not` -- NT2/IH vs. non-narcolepsy
- `nt12_vs_not` -- Any narcolepsy (NT1 + NT2/IH) vs. non-narcolepsy

### Features

924 features per visit, including:
- **Clinical keywords** (~446 stemmed terms with negation detection, e.g., `cataplexi_`, `sleepi attack_`, `narcolepsi_neg_`)
- **ICD codes** (regex matching for narcolepsy diagnosis codes, e.g., G47.41, G47.42)
- **Medications** (27 narcolepsy-relevant drugs: modafinil, Xyrem, stimulants, antidepressants, etc.)

### Usage

See `discriminative-modeling/discriminative-model.ipynb` for a complete walkthrough including:
- Loading clinical data (notes, ICD codes, medications) from parquet files
- Feature extraction using the `NarcolepsyModel` class
- Model inference for all three classification tasks
- Leave-one-source-out cross-validation and model comparison

See `predictive-modeling/predictive-model.ipynb` for additional feature extraction code for predictive modeling

### Dependencies

polars, pandas, NLTK, Ray, scikit-learn, joblib

## Predictive Modeling

Computes a longitudinal risk score from clinical notes written **before** a narcolepsy diagnosis is made, to support earlier referral for diagnostic testing.

### Data

- **Source**: BDSP (5 academic medical centers: BCH, BIDMC, Emory, MGB, Stanford)
- **Cohort**: 196 any-narcolepsy training cases (66 NT1), 11,049 controls
- **Features**: Same 924 NLP features as the discriminative models, extracted per visit

### Method

- SGD logistic regression with L1 penalty and balanced minibatches
- Chi-squared feature prefiltering (top 100 features)
- Training window: [-2.5yr, -0.5yr] before diagnosis (0.5yr horizon exclusion prevents learning from diagnostic-workup visits)
- Alpha (regularization) selected via modal cross-validation across folds
- Validation: stratified 5-fold CV (primary), leave-one-site-out CV (secondary)

### Results

| Outcome | 5-fold CV AUC | LOSO AUC | Training Cases |
|---------|---------------|----------|----------------|
| Any Narcolepsy (NT1 + NT2/IH) | 0.835 | 0.797 | 196 |
| NT1 Only | 0.838 | 0.788 | 66 |

### Running

```bash
cd predictive-modeling/risk_score_v2
python risk_score_v2.py both
```

This trains both outcome models (any_narcolepsy, NT1), runs all cross-validation, trains the final model, and generates all figures and tables:

- `v2_performance_combined.png` -- CV / LOSO / resubstitution AUC and AUPRC
- `v2_distributions_combined.png` -- Risk score distributions for cases vs. controls
- `v2_features_combined.png` -- Top predictive features by coefficient magnitude
- `v2_trajectories_combined.png` -- Risk score trajectories aligned to diagnosis (logit scale)
- `v2_nnt_analysis.png` -- Number Needed to Test analysis (assumed prevalence 0.08%)
- `v2_loso_by_site.csv` -- LOSO performance broken down by site

### Dependencies

numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, pyarrow

## Paper Figures

Jupyter notebooks and scripts in `paper_figures/` reproduce the main figures from the manuscript. Each notebook loads data from `../data` (relative to the notebook) and requires the discriminative-modeling data files (features, notes, models).

`feature_heatmap.py` generates feature evolution heatmaps showing how selected predictive model features (non-zero L1 coefficients) evolve over time leading up to diagnosis in cases vs. matched controls. Run with `python feature_heatmap.py` to generate both outcome heatmaps, or pass `any_narcolepsy` or `nt1` for a single outcome.

## Annotation Tool

The `timeline-viewer/` directory is a git submodule pointing to [bdsp-core/timeline-viewer](https://github.com/bdsp-core/timeline-viewer), a web application for reviewing and annotating patient clinical timelines. See its own README for setup instructions.

## License

CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International). Commercial use is prohibited. See [LICENSE](LICENSE) for details.
