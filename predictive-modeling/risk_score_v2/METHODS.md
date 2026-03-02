# Risk Score Model for Pre-Diagnostic Narcolepsy Detection

## Objective

Develop a risk score that can be applied to clinical notes in an electronic health record (EHR) to identify patients who are likely to eventually be diagnosed with narcolepsy, **before** the diagnosis is formally made. The goal is to enable earlier referral for diagnostic testing.

## Data

### Sources

| Dataset | Patients | Visits | Cases |
|---------|----------|--------|-------|
| NT1 (narcolepsy type 1) | 282 cases | 23,818 | 282 |
| NT2/IH (narcolepsy type 2 / idiopathic hypersomnia) | 314 cases | 41,870 | 314 |
| Controls (no narcolepsy) | 9,929 | 632,449 | 0 |

- 5 sites: Stanford (stan), Boston Children's (bch), Beth Israel Deaconess (bidmc), Emory (emory), Mass General Brigham (mgb)
- Controls come from bidmc and mgb only

### Features

924 features derived from clinical notes using NLP (`features_3` = cumulative counts):
- **ICD codes**: Regex-matched patterns for narcolepsy-related diagnoses (e.g., G47.41, G47.42)
- **Medications**: 27 narcolepsy-relevant medications (Xyrem, modafinil, Adderall, stimulants, antidepressants, etc.)
- **Clinical keywords**: ~446 stemmed keyword/phrase features with negation detection (e.g., `cataplexi_`, `sleepi attack_`, `narcolepsi_neg_`)

Each feature is a **cumulative count** of how many times it appeared in the patient's notes up to and including the current visit.

### Data Processing

1. **Visit subsampling**: Capped at 20 visits per patient (random subsample preserving first and last visits) to reduce dataset from ~588K to ~119K rows
2. **Sparse feature removal**: Dropped features with fewer than 50 non-zero values across all visits
3. **Exclusions**: Removed patients with only 1 visit or with gaps >5 years between consecutive visits

## Models

Two outcome definitions were tested:

| Model | Cases | Definition |
|-------|-------|------------|
| **Any Narcolepsy** | NT1 + NT2/IH combined (596 total) | Diagnosed with any type of narcolepsy |
| **NT1 only** | NT1 only (282 total) | Diagnosed with narcolepsy type 1 specifically |

Both models use the same 9,929 control patients. For the NT1 model, NT2/IH patients are excluded entirely (neither cases nor controls).

## Temporal Design

### Case visit truncation

For diagnosed patients, only visits within **2 years before diagnosis** are used for training and evaluation. Visits from earlier in the patient's history are excluded. Rationale: notes from >2 years before diagnosis are unlikely to contain narcolepsy-relevant signal and add noise to the training data.

### Horizon exclusion

Two exclusion horizons were tested:

| Horizon (h) | Case visits used | Purpose |
|-------------|-----------------|---------|
| h = 0 yr | [diag - 2yr, diag] (includes diagnosis visit) | Baseline — detects diagnosis, not pre-diagnostic signal |
| h = 0.5 yr | [diag - 2yr, diag - 0.5yr] (excludes 6 months before diagnosis) | **Primary model** — forces learning from pre-diagnostic features |

With h = 0.5, visits from the 6 months immediately preceding diagnosis are excluded. This prevents the model from learning features that only appear at or very near diagnosis (e.g., the narcolepsy ICD code itself, Xyrem prescriptions written at diagnosis). The effective training window for each case is [diagnosis - 2 years, diagnosis - 6 months].

## Training Algorithm

### SGD Logistic Regression with Balanced Minibatches

- **Model**: `SGDClassifier` with log loss and L1 penalty (lasso regularization)
- **Minibatch construction**: Each training batch contains one randomly selected visit per diagnosed patient, plus an equal number of randomly selected visits from distinct control patients. This simultaneously:
  - Handles **class imbalance** (equalizes cases and controls in each batch)
  - Handles **patient-level deduplication** (one visit per patient per batch, preventing patients with many visits from dominating)
- **Training**: 200 epochs of balanced minibatches via `partial_fit()`

### Feature Selection

**Chi-squared pre-filter**: Before training each fold, the top 100 features (by chi-squared statistic on non-negative feature values vs. labels) are selected from the training data only. This reduces dimensionality from 924 to 100, improving speed and reducing overfitting.

### Regularization Tuning

The L1 penalty strength (alpha) is selected via inner cross-validation from candidates [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]. The inner CV uses stratified 3-fold splits within the training set.

## Evaluation

### Primary analysis: Pooled stratified 5-fold CV

Patients are randomly split into 5 folds, stratified by case/control status. Each fold trains on 80% of patients and tests on 20%. This is the primary analysis because:
- All sites contribute to both training and testing in each fold
- More training data per fold than LOSO
- Avoids the problem of 3 sites (bch, emory, stan) having zero controls

### Secondary analysis: Leave-one-site-out (LOSO) CV

Each site is held out in turn. This tests generalizability across institutions. Note: only 2 of 5 sites (bidmc, mgb) produce valid metrics because the other 3 sites have no control patients.

### Metrics

- **AUC (area under ROC curve)**: Patient-level — each patient's score is the mean of their visit-level predicted probabilities; the patient-level binary label is "ever diagnosed"
- **AUPRC (area under precision-recall curve)**: Same patient-level aggregation. More informative than AUC in this class-imbalanced setting

## Results

### Any Narcolepsy (NT1 + NT2/IH combined)

| Horizon | CV method | Mean AUC | Mean AUPRC | N cases |
|---------|-----------|----------|------------|---------|
| h = 0 yr | Pooled 5-fold | 0.964 | 0.759 | 244 |
| h = 0 yr | LOSO | 0.905 | 0.571 | 244 |
| **h = 0.5 yr** | **Pooled 5-fold** | **0.830** | **0.517** | **120** |
| h = 0.5 yr | LOSO | 0.841 | 0.275 | 120 |

### NT1 Only

| Horizon | CV method | Mean AUC | Mean AUPRC | N cases |
|---------|-----------|----------|------------|---------|
| h = 0 yr | Pooled 5-fold | 0.962 | 0.741 | 93 |
| h = 0 yr | LOSO | 0.948 | 0.588 | 93 |
| **h = 0.5 yr** | **Pooled 5-fold** | **0.828** | **0.297** | **39** |
| h = 0.5 yr | LOSO | 0.889 | 0.154 | 39 |

### Recommended model

**Any Narcolepsy, h = 0.5 yr, pooled CV**: AUC = 0.830, AUPRC = 0.517

This model:
- Detects pre-diagnostic signal from clinical notes written 6 months to 2 years before diagnosis
- Has enough training cases (120 after horizon exclusion) for stable learning
- Produces clinically interpretable features (stimulant trials, hypersomnia mentions, antidepressant use)
- Shows gradually elevated risk scores across the pre-diagnostic period, rather than a sudden spike at diagnosis

### Feature importance (h = 0.5, Any Narcolepsy)

Top features driving a positive risk score:
1. idiopathic hypersomnia/hypersomnolence mentions
2. Venlafaxine (SNRI antidepressant, used off-label for cataplexy)
3. Concerta (methylphenidate, stimulant)
4. Cymbalta (duloxetine, SNRI)
5. Sertraline (SSRI, used for cataplexy)
6. Dextroamphetamine (stimulant)
7. Provigil / modafinil (wakefulness-promoting agents)
8. MSLT mentions (multiple sleep latency test)
9. Sleepiness attack mentions

These features represent the clinical breadcrumbs that appear in the EHR before a formal narcolepsy diagnosis: stimulant prescriptions, antidepressant trials (for cataplexy), and mentions of excessive daytime sleepiness.

## Output Files

| File | Description |
|------|-------------|
| `risk_score_v2.py` | Model training and evaluation code |
| `v2_performance_*.png` | AUC/AUPRC bar charts comparing h=0 vs h=0.5, pooled vs LOSO |
| `v2_features_*.png` | Feature importance (top 20 coefficients) for h=0 vs h=0.5 |
| `v2_trajectories_*.png` | Case risk score trajectories aligned to diagnosis, limited to [-2yr, 0] |
| `v2_distributions_*.png` | Score distributions for cases vs controls |
| `v2_summary_*.csv` | Summary performance tables |
| `v2_results_*.pickle` | Full results (scores, coefficients, per-fold metrics) |

## Reproducibility

```bash
# Run from the longitudinal-time-to-event-model/ directory
python risk_score_v2/risk_score_v2.py any_narcolepsy
python risk_score_v2/risk_score_v2.py nt1
```

Dependencies: numpy, pandas, pyarrow, scikit-learn, scipy, matplotlib, seaborn, tqdm
