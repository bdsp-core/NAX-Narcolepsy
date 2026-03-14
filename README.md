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
│   ├── retrain_all.py             # Retrain all cross-sectional models
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
├── paper_figures/                 # Scripts to reproduce all manuscript figures
│   ├── pub_style.py               # Shared publication style (colors, fonts, sizes)
│   ├── consort_diagrams.py        # CONSORT flow diagrams
│   ├── roc_prc.ipynb              # ROC and precision-recall curves
│   ├── confusion_matrices.ipynb   # Confusion matrix plots
│   ├── feature_heatmap.py         # Feature evolution heatmaps (cases vs controls)
│   ├── swimmer_plot.py            # Swimmer plot of patient timelines
│   └── site_trajectories.py       # Site-stratified trajectory sensitivity analysis
│
├── manuscript/                    # Manuscript verification and figures
│   ├── verify_manuscript_numbers.py
│   └── figures/                   # All generated figures (PNG + TIFF)
│
├── build_manuscript_figures.sh    # Regenerate all figures end-to-end
├── REPRODUCIBILITY.md             # Full reproducibility guide
├── timeline-viewer/               # Annotation tool (git submodule)
│
└── LICENSE
```

## Discriminative Modeling

Classifies whether a clinical note belongs to a patient with narcolepsy. Uses keyword matching with negation detection, ICD code features, and medication features extracted from clinical notes.

### Models

Three classification tasks, each trained with four classifiers (LR, RF, GBT, XGB) via LOSO cross-validation:
- `nt1_vs_others` -- NT1 vs. non-narcolepsy (best: RandomForest, AUROC=0.997)
- `nt2ih_vs_others` -- NT2/IH vs. non-narcolepsy (best: XGBoost, AUROC=0.988)
- `any_narcolepsy_vs_others` -- Any narcolepsy (NT1 + NT2/IH + Unclear) vs. non-narcolepsy (best: GradientBoosting, AUROC=0.990)

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

- **Source**: BDSP (5 academic medical centers: BCH, BIDMC, Emory, MGH, Stanford)
- **Cohort**: 181 any-narcolepsy training cases (68 NT1, 113 NT2/IH), 9,858 controls (BIDMC and MGH general population)
- **Features**: Same NLP features as the discriminative models, transformed to running means (cumulative count / number of visits) to normalize for visit frequency

### Method

- SGD logistic regression with L1 penalty and balanced minibatches
- Chi-squared feature prefiltering (top 100 features)
- Training window: [-2.5yr, -0.5yr] before diagnosis (0.5yr horizon exclusion prevents learning from diagnostic-workup visits)
- Testing/scoring window: [-5yr, 0yr] before diagnosis
- Alpha (regularization) selected via modal cross-validation across folds
- Validation: stratified 5-fold CV (primary), leave-one-site-out CV (secondary, BIDMC and MGH only)

### Results

AUC evaluated at t = −1.5 years relative to diagnosis (1-year window):

| Outcome | 5-fold CV AUC | LOSO AUC | Training Cases |
|---------|---------------|----------|----------------|
| Any Narcolepsy (NT1 + NT2/IH) | 0.788 | 0.739 | 181 |
| NT1 Only | 0.798 | 0.722 | 68 |
| NT2/IH Only | 0.757 | 0.630 | 113 |

### Running

```bash
cd predictive-modeling/risk_score_v2
python risk_score_v2.py all
```

This trains all three outcome models (any_narcolepsy, NT1, NT2/IH), runs all cross-validation, trains the final models, and generates all figures and tables:

- `v2_summary_*.csv` -- CV / LOSO / resubstitution AUC and AUPRC per outcome
- `v2_loso_by_site.csv` -- LOSO performance broken down by site
- `v2_results_*.pickle` -- Full results including model artifacts and trajectory data
- Manuscript figures saved to output directory

### Dependencies

numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, pyarrow

## Paper Figures

Scripts and notebooks in `paper_figures/` reproduce all manuscript figures. A shared publication style (`pub_style.py`) ensures consistent formatting across all figures (colorblind-safe palette, JAMA Neurology specs).

Run all figures at once:
```bash
bash build_manuscript_figures.sh
```

Or generate individual figures:
- `python consort_diagrams.py` -- CONSORT flow diagrams (eFigures 1-2)
- `roc_prc.ipynb` -- ROC and precision-recall curves (Figure 1; eFigures 3-5)
- `confusion_matrices.ipynb` -- Confusion matrices (eFigures 6-8)
- `python feature_heatmap.py` -- Feature evolution heatmaps (eFigures 12-14)
- `python swimmer_plot.py` -- Swimmer plot of patient timelines (eFigure 15)
- `python site_trajectories.py` -- Site-stratified trajectory sensitivity analysis (eFigure 16)

## Annotation Tool

The `timeline-viewer/` directory is a git submodule pointing to [bdsp-core/timeline-viewer](https://github.com/bdsp-core/timeline-viewer), a web application for reviewing and annotating patient clinical timelines. See its own README for setup instructions.

## License

CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International). Commercial use is prohibited. See [LICENSE](LICENSE) for details.
