**Cross-Sectional and Longitudinal Phenotyping of Narcolepsy from Electronic Health Records**

**Abstract**

**Background.** Narcolepsy is a chronic neurological disorder affecting approximately 1 in 2,000 individuals, yet the average diagnostic delay from symptom onset to diagnosis ranges from 8 to 15 years. During this prolonged interval, patients accumulate clinical encounters that may contain identifiable signals of undiagnosed narcolepsy. We sought to determine whether natural language processing (NLP) of electronic health record (EHR) clinical notes could identify patients with narcolepsy, both cross-sectionally and in the pre-diagnostic period.

**Methods.** We used EHR data from 5 academic medical centers participating in the Brain Data Science Platform (BDSP). We developed two complementary modeling approaches: (1) cross-sectional classifiers to identify narcolepsy type 1 (NT1) and narcolepsy type 2/idiopathic hypersomnia (NT2/IH) from individual clinical notes (6,498 patients, 8,990 annotated notes), and (2) longitudinal predictive models to estimate pre-diagnostic narcolepsy risk from cumulative NLP features (initial cohort of 596 narcolepsy cases and 12,746 controls; after filtering, 196 cases with any narcolepsy diagnosis, 66 with confirmed NT1, and 11,049 controls). Both approaches used 924 NLP-derived features extracted from unstructured clinical notes, including stemmed keywords with negation detection, ICD codes, and medication mentions. Cross-sectional models were evaluated using leave-one-site-out (LOSO) cross-validation. Longitudinal models excluded data from within 6 months of diagnosis and were evaluated using both stratified 5-fold and LOSO cross-validation.

**Results.** For cross-sectional classification, gradient boosting models achieved mean AUROC of 0.994 and mean sensitivity of 0.876 for NT1, and mean AUROC of 0.984 and mean sensitivity of 0.570 for NT2/IH across sites. For longitudinal prediction, the any-narcolepsy model achieved a mean AUC of 0.835 (5-fold CV) and 0.797 (LOSO), and the NT1-only model achieved a mean AUC of 0.838 (5-fold CV) and 0.788 (LOSO). Risk score trajectories showed progressive elevation beginning approximately 2 years before diagnosis. Number-needed-to-test analysis demonstrated 62.5- to 125-fold enrichment over the population base rate at clinically practical operating thresholds.

**Conclusions.** NLP-based analysis of routine clinical notes can identify patients with narcolepsy cross-sectionally and detect elevated risk months to years before formal diagnosis. These findings support the feasibility of EHR-based screening to reduce narcolepsy's prolonged diagnostic delay.

**Introduction**

Narcolepsy is a chronic neurological disorder characterized by excessive daytime sleepiness, with narcolepsy type 1 (NT1) additionally defined by cataplexy or cerebrospinal fluid hypocretin deficiency. The disorder affects approximately 1 in 2,000 individuals, with a combined prevalence of narcolepsy types 1 and 2 estimated at 38-56 per 100,000 in the United States and Europe [1,2]. Despite its significant impact on quality of life, safety, and occupational functioning, narcolepsy remains substantially underdiagnosed: the average delay from symptom onset to diagnosis ranges from 8 to 15 years, with nearly 60% of patients receiving at least one misdiagnosis before the correct diagnosis is established [3,4]. During this prolonged diagnostic odyssey, patients face a 3- to 4-fold increased risk of motor vehicle accidents, elevated rates of depression and anxiety, significant economic burden, and years of accumulated psychosocial harm [5,6,7].

The persistence of this diagnostic gap is notable because patients with undiagnosed narcolepsy are not absent from the healthcare system. They are seeking care from primary care physicians, neurologists, psychiatrists, and other specialists, generating clinical notes that may contain subtle but identifiable signals of the underlying disorder. These "clinical breadcrumbs" -- mentions of excessive sleepiness, empiric trials of stimulant medications, referrals for sleep studies, complaints of difficulty waking, and sleep-related symptoms -- accumulate progressively in the medical record in the years preceding diagnosis. Natural language processing (NLP) offers the potential to systematically extract and quantify these signals from unstructured clinical notes at scale across large healthcare systems.

We hypothesized that NLP-derived features from routine EHR clinical notes contain sufficient information to (1) identify clinical notes documenting narcolepsy and its subtypes, and (2) detect patients at elevated risk for narcolepsy in the pre-diagnostic period, months to years before formal diagnosis. To test this hypothesis, we developed and validated two complementary machine learning approaches using data from 5 academic medical centers participating in the Brain Data Science Platform (BDSP). First, we trained cross-sectional classifiers to distinguish clinical notes indicating NT1 or NT2/idiopathic hypersomnia (NT2/IH) from notes of patients without narcolepsy. Second, we developed a longitudinal predictive model that estimates narcolepsy risk from cumulative clinical features over time, explicitly excluding data from the 6-month period immediately preceding diagnosis to ensure the model captures pre-diagnostic signals rather than the diagnostic workup itself.

If successful, such an approach could be deployed as an EHR-integrated screening tool to flag patients at high risk for narcolepsy, enabling expedited referral to sleep medicine for confirmatory testing and, ultimately, reducing the years of unnecessary diagnostic delay that many patients currently endure.

**Methods**

***Study Design and Data Sources***

We used electronic health record data from 5 academic medical centers participating in the Brain Data Science Platform (BDSP): Boston Children's Hospital (BCH), Beth Israel Deaconess Medical Center (BIDMC), Emory University Hospital (Emory), Massachusetts General Brigham (MGB), and Stanford University Medical Center (Stanford). Each site contributed comprehensive EHR data including demographics, ICD diagnosis codes, medication orders, and unstructured clinical notes. Stanford and Emory cohorts include only patients who have visited their respective sleep clinics, whereas BCH, BIDMC, and MGB include broader patient populations. A swimmer plot illustrating the temporal coverage of the narcolepsy patient cohort across sites is provided in eFigure 10.

This study was conducted under IRB protocols approved and overseen by the BIDMC ethics committee (protocols 2024P000807, 2022P000417, 2024P000804), which granted a waiver of consent for retrospective analysis of de-identified EHR data.

We developed two distinct analytic approaches using overlapping but differently constructed cohorts, as described below and illustrated in the CONSORT diagrams (eFigures 1 and 2).

***Feature Extraction***

For both the cross-sectional and longitudinal analyses, we extracted a shared set of 924 NLP-derived features from clinical notes. Features were drawn from three categories:

1. **ICD diagnosis codes** (3 features): Binary indicators for narcolepsy-associated ICD codes identified through regular expression matching, grouped as narcolepsy with cataplexy (ICD-9: 347.01, 347.11; ICD-10: G47.411, G47.421), narcolepsy without cataplexy (ICD-9: 347.00, 347.10; ICD-10: G47.419, G47.429), and hypersomnia (ICD-9: 780.53, 780.54; ICD-10: G47.1x). A code was counted if it appeared within 6 months before or after the clinical visit.

2. **Medication features** (27 features): Binary indicators for 27 narcolepsy-relevant medications including narcolepsy-specific treatments (sodium oxybate/Xyrem, Xywav, Lumryz, modafinil/Provigil, armodafinil/Nuvigil, solriamfetol/Sunosi, pitolisant/Wakix), stimulants (Adderall, Concerta, Dexedrine, dextroamphetamine, Ritalin, methylphenidate, Desoxyn, Evekeo), and antidepressants used off-label for cataplexy (venlafaxine/Effexor, duloxetine/Cymbalta, fluoxetine/Prozac, sertraline, imipramine, clomipramine, protriptyline/Vivactil, paroxetine). A medication was counted if prescribed within 6 months before or after the visit.

3. **Textual features** (894 features): Stemmed clinical keywords and phrases with negation detection, covering narcolepsy-related terminology. Clinical notes underwent preprocessing (lowercasing, special character removal, sentence tokenization, word tokenization, and Snowball stemming). Each stemmed sentence was matched against a curated keyword list. If a negation term ("no," "not," "absent") appeared in the same sentence as a matched keyword, a separate negation feature was recorded instead. The keyword list encompassed terms for sleep disorders (e.g., "narcolepsi_," "cataplexi_," "hypersomnia_"), diagnostic testing (e.g., "mslt_," "polysomnogram_"), symptoms (e.g., "excess daytim sleepi_," "sleep paralysi_"), and related clinical concepts (Supplementary Material 3).

The initial candidate feature set included 1,204 features. After filtering for features with 10 or more occurrences in the cross-sectional dataset, 924 features were retained for both analyses.

***Cohort Construction: Cross-Sectional Classification***

For the cross-sectional analysis, we constructed an enriched dataset optimized for training note-level classifiers. Because random sampling from the full EHR would yield extremely low narcolepsy prevalence and severe class imbalance, we used a stratified enrichment strategy.

We defined three sampling groups for both NT1 and NT2/IH based on ICD codes and medication patterns (Supplementary Material 1): (1) "almost certainly positive" (NT1+ or NT2/IH+): patients with 3 or more disease-specific ICD codes, no ICD codes for the other narcolepsy subtype, and at least 1 narcolepsy-relevant medication; (2) "almost certainly negative" (NT1- or NT2/IH-): patients with no narcolepsy ICD codes and no narcolepsy medications; and (3) "maybe" (NT1? or NT2/IH?): patients not meeting positive or negative criteria, with at least 1 relevant ICD code and no codes for the other subtype. Approximately 250 patients were selected from each site per classification task, dependent on data availability.

For each selected patient, we sampled clinical notes filtering to those with more than 500 words. For patients in the "almost certainly positive" group, we additionally required at least 1 narcolepsy-related keyword in the note. Each site contributed approximately 1,800 notes, with roughly 300 notes per classification category (6 categories: NT1+, NT1-, NT1?, NT2/IH+, NT2/IH-, NT2/IH?). The resulting cross-sectional dataset comprised 6,498 patients and 8,990 clinical notes across all 5 sites (Table 1).

***Narcolepsy Ascertainment (Ground Truth Labeling)***

We performed manual chart annotation of the selected clinical notes using a custom web-based annotation tool. A standard operating procedure (SOP) was developed to define diagnostic criteria for NT1 and NT2/IH, incorporating CSF hypocretin levels, MSLT results, PSG findings, HLA typing, physician assessments, and clinical history (Supplementary Material 2).

Six physician annotators were recruited. Each annotator reviewed an initial calibration batch of 100 notes. After adjudicating discrepancies and refining the SOP, each annotator was assigned their own batch of notes. The annotation tool highlighted narcolepsy-relevant terms and allowed annotators to classify each note as indicating: (1) NT1 (>80% confidence the note indicates NT1), (2) NT2/IH (>80% confidence the note indicates NT2/IH), (3) Unclear (50-80% confidence the note indicates narcolepsy of either type), or (4) Absent (>80% confidence the note does not indicate narcolepsy). Of the 8,990 annotated notes, 296 were classified as "Unclear" and excluded from model training, leaving 8,694 notes with definitive labels (Table 2).

Of the 8,694 notes with definitive labels, 620 (7.1%) were classified as NT1, 360 (4.1%) as NT2/IH, and 7,714 (88.7%) as Absent. The distribution of NT1-positive notes varied across sites: BCH contributed 194, BIDMC 265, Emory 56, MGB 77, and Stanford 28. NT2/IH-positive notes were similarly distributed: BCH 46, BIDMC 126, Emory 71, MGB 61, and Stanford 56 (Table 1).

***Cross-Sectional Classification Model Development***

We trained two binary classification models: NT1 vs. others (NT2/IH and Absent combined) and NT2/IH vs. others (NT1 and Absent combined). We performed nested cross-validation: the outer loop used leave-one-site-out (LOSO) cross-validation to estimate cross-site generalization, while the inner loop used 5-fold patient-stratified cross-validation with Bayesian optimization for hyperparameter selection. In each inner loop, features were standardized based on training fold statistics; the model was trained with the hyperparameters maximizing averaged performance across held-out folds.

Four classifier types were evaluated:

- **Logistic regression (LR)**: regularization strength C (0.01, 0.1, 1.0, 10.0), L1 ratio (0.0, 0.25, 0.5, 0.75, 1.0), saga solver, class weight (None, balanced), max iterations 20,000.
- **Random forest (RF)**: estimators (100, 200, 300), max depth (None, 10, 20, 30), min samples split (2, 5, 10), class weight (None, balanced).
- **Gradient boosting tree (GBT)**: estimators (100, 200), learning rate (0.01, 0.1, 0.2), max depth (3, 5, 7), subsample (0.8, 1.0).
- **XGBoost (XGB)**: estimators (100, 200), learning rate (0.01, 0.1, 0.2), max depth (3, 5, 7), subsample (0.8, 1.0), column sample by tree (0.8, 1.0).

All models used random state 42. Models were implemented using scikit-learn and XGBoost.

***Cross-Sectional Classification Model Evaluation***

Each clinical note received a predicted probability of being positive for narcolepsy. Performance was assessed using AUROC, AUPRC, sensitivity, specificity, and F1 score. Model interpretability was assessed through feature importances: coefficients for LR and purity-based importance for tree-based models.

***Cohort Construction: Longitudinal Prediction***

For the longitudinal analysis, we used a broader cohort encompassing all available patients with sufficient EHR data, without the enrichment sampling used for the cross-sectional analysis. Data were loaded from two disease-specific parquet files (NT1 cohort and NT2/IH cohort), each containing both confirmed cases and non-case patients from the same institutional EHR systems.

The initial cohort comprised 13,342 patients (1,022,458 clinical visits): 282 confirmed NT1 cases, 314 confirmed NT2/IH cases (596 total narcolepsy cases), and 12,746 non-narcolepsy controls from both disease-specific datasets (with no overlapping patient IDs between datasets).

The following filtering steps were applied sequentially (eTable 1):

1. **Gap exclusion**: Patients with gaps exceeding 5 years between consecutive visits were excluded, removing 1,754 patients (remaining: 11,588 patients, 876,318 visits).
2. **Visit subsampling**: To limit computational burden and prevent overrepresentation of frequently-seen patients, visits were subsampled to a maximum of 20 per patient, preserving the first and last encounters (remaining: 164,383 visits).
3. **Sparse feature removal**: Features with fewer than 50 non-zero values across the dataset were excluded to improve model stability.

***Longitudinal Predictive Model: Temporal Design***

To evaluate the model's ability to identify narcolepsy risk from pre-diagnostic clinical data, we implemented a temporal exclusion design. For diagnosed patients, training data were restricted to visits occurring within a pre-diagnostic window of 2.5 years to 6 months before the date of diagnosis (horizon exclusion h = 0.5 years). This exclusion window ensures the model learns from clinical features present before the diagnostic workup period, during which narcolepsy-specific ICD codes, diagnostic test orders (e.g., MSLT), and narcolepsy-targeted medications would be expected to appear. All visits from control patients were included without temporal restriction. Patients with fewer than 2 visits after temporal filtering were excluded.

After applying all filtering steps, 196 narcolepsy cases remained for the any-narcolepsy outcome model (from 539 after gap exclusion), and 66 NT1 cases remained for the NT1-only outcome model (from 258 after gap exclusion), with 11,049 controls in both analyses. The substantial case attrition reflects the requirement that cases have sufficient clinical documentation in the narrow 2-year pre-diagnostic training window (eTable 1).

Two outcome models were developed: (1) any narcolepsy (NT1 combined with NT2/IH) and (2) NT1 only. For the NT1-only model, NT2/IH patients were excluded entirely from both case and control groups.

***Longitudinal Predictive Model: Model Development***

We trained a logistic regression classifier using stochastic gradient descent (SGD) with L1 (lasso) regularization. To address the substantial class imbalance (1.7% case prevalence in the training cohort), we employed a balanced minibatch training strategy: at each training iteration, one visit was randomly sampled from each diagnosed patient, paired with an equal number of visits from distinct control patients. This approach simultaneously addresses class imbalance and prevents patients with more frequent encounters from disproportionately influencing the model.

Feature selection was performed independently within each cross-validation fold to prevent information leakage. Within each training fold, a chi-squared test was applied to rank features by their association with the outcome, and the top 100 features were retained. Selected features were standardized to zero mean and unit variance using parameters estimated from the training data only.

The regularization parameter alpha was selected from 5 candidate values (5 x 10^-5, 1 x 10^-4, 5 x 10^-4, 1 x 10^-3, 5 x 10^-3) via an inner cross-validation loop within each training fold. The model was trained for 200 epochs using balanced minibatches at each candidate alpha value, and the alpha yielding the highest inner-fold AUC was selected.

***Longitudinal Predictive Model: Evaluation***

We evaluated model performance using 2 complementary cross-validation strategies. The primary analysis used patient-stratified 5-fold cross-validation, in which patients were randomly partitioned into 5 folds with stratification by case-control status. The secondary analysis used leave-one-site-out (LOSO) cross-validation, in which the model was trained on data from 4 sites and evaluated on the held-out site, cycling through all 5 sites.

For both validation strategies, visit-level predicted probabilities were aggregated to patient-level scores by computing the mean predicted probability across all of a patient's visits. Discrimination was assessed using the area under the receiver operating characteristic curve (AUC) and the area under the precision-recall curve (AUPRC).

A final model was trained on all available data using the optimal hyperparameters identified during cross-validation. This model was used for risk score trajectory analysis, feature importance assessment, and clinical utility evaluation.

***Risk Score Trajectory Analysis***

To characterize the temporal evolution of the risk score relative to diagnosis, we scored all visits within a 5-year pre-diagnostic window using the final model. For diagnosed patients, time was aligned to the date of diagnosis; for controls, a pseudo-diagnosis date was randomly assigned from the empirical distribution of case diagnosis times. Risk score trajectories were visualized on the logit scale, with sliding-window percentile bands (25th, 50th, and 75th percentiles; 1-year window) computed separately for cases and controls. Time-varying AUC was computed within 1-year sliding windows to assess how discriminative performance evolves as patients approach diagnosis.

***Feature Evolution Analysis***

To examine how individual model features accumulate over time in cases versus controls, we generated feature evolution heatmaps for each outcome model. For each feature retained after L1 regularization (i.e., features with non-zero model coefficients), we computed the mean cumulative feature count in 10 equally spaced time bins spanning the 2.5-year pre-diagnostic window. Cases were aligned to diagnosis date; controls were aligned to their last visit. To ensure adequate longitudinal coverage, only patients with 5 or more visits within the window were included. For time bins in which a patient had no visit, the last known cumulative value was carried forward. Feature values were z-score normalized across both groups to enable comparison across features with different scales. Features were ordered by model coefficient value, and rows were color-coded by coefficient sign (warm colors for positive coefficients indicating increased narcolepsy risk; cool colors for negative coefficients indicating decreased risk).

***Clinical Utility Analysis***

To evaluate the potential clinical utility of the risk score as a screening tool, we performed a number-needed-to-test (NNT) analysis. Using Bayes' theorem, we estimated the positive predictive value (PPV) at each score threshold under an assumed narcolepsy population prevalence of 0.08% (1 in 1,250). The NNT -- defined as the reciprocal of the PPV -- represents the number of patients who would need to undergo confirmatory diagnostic testing (e.g., polysomnography followed by MSLT) to identify one true case. We annotated clinically relevant operating points at NNT = 10 and NNT = 20.

**Results**

***Cohort Characteristics***

The cross-sectional classification cohort comprised 6,498 patients and 8,990 annotated clinical notes across 5 sites (Table 1). Patients were 52.8% female with a mean age of 44.0 years (SD 23.5); the cohort was 64.4% White, 14.4% Black or African American, and 5.6% Asian. The site-level distribution was: BIDMC 1,549 patients (1,921 notes), Stanford 1,454 patients (1,477 notes), Emory 1,294 patients (1,858 notes), BCH 1,141 patients (1,877 notes), and MGB 1,060 patients (1,857 notes). Of 8,990 annotated notes, 296 were classified as "Unclear" and excluded, yielding 8,694 notes with definitive labels for model training (Table 1).

The longitudinal prediction cohort initially comprised 13,342 patients with 1,022,458 clinical visits: 596 narcolepsy cases (282 NT1, 314 NT2/IH) and 12,746 controls. After sequential filtering (gap exclusion removing 1,754 patients with >5-year visit gaps, visit subsampling to max 20 per patient, and temporal windowing with h = 0.5 year horizon exclusion), 196 cases remained for the any-narcolepsy model and 66 for the NT1-only model, with 11,049 controls (eTable 1). The per-site distribution of cases in the final predictive model cohort was: BCH 29 any-narcolepsy cases (16 NT1), BIDMC 48 (13 NT1), Emory 33 (9 NT1), MGB 54 (21 NT1), and Stanford 32 (7 NT1) (eTable 2).

eFigures 1 and 2 present the CONSORT diagrams illustrating patient flow through the cross-sectional classification and longitudinal prediction pipelines, respectively.

***Cross-Sectional Classification: NT1***

All four classifiers achieved strong discriminative performance for NT1 detection, with mean AUROC values near 0.99 across all models and sites (Table 2). XGBoost (XGB) and Gradient Boosting (GBT) demonstrated the best overall balance of sensitivity and specificity. GBT achieved mean sensitivity of 0.876 (SD 0.082), mean specificity of 0.987 (SD 0.018), mean F1 of 0.850 (SD 0.070), mean AUROC of 0.994 (SD 0.003), and mean AUPRC of 0.935 (SD 0.039). XGB performed comparably, with mean sensitivity of 0.869 (SD 0.075), mean specificity of 0.990 (SD 0.011), mean F1 of 0.855 (SD 0.068), mean AUROC of 0.993 (SD 0.005), and mean AUPRC of 0.924 (SD 0.050). LR and RF achieved similarly high AUROC values (0.991 and 0.994, respectively) with mean AUPRCs of 0.906 (SD 0.054) and 0.922 (SD 0.075), but exhibited lower sensitivity, particularly RF (mean 0.722, SD 0.168). Figure 1 shows the receiver operating characteristic and precision-recall curves for all NT1 models.

Site-level performance was largely consistent across BCH, BIDMC, and Emory, where sensitivity ranged from 0.73 to 0.97 depending on the model. MGB demonstrated the most variable sensitivity across classifiers, with XGB achieving 0.766 and RF achieving only 0.442, a pattern consistent with its smaller NT1-positive cohort. Stanford showed similarly variable sensitivity (0.786-0.921). AUPRC ranged from 0.817-0.976 across sites and models, with MGB and Stanford consistently yielding the lowest AUPRC values, likely attributable to their smaller NT1-positive test cohorts. Confusion matrices confirmed that false negative rates were lowest for GBT and XGB, particularly at BCH (12.4% and 10.8%, respectively) and BIDMC (9.1% and 14.3%), with MGB remaining the most challenging site (41.6% miss rate for XGB) (eFigure 3).

***Cross-Sectional Classification: NT2/IH***

Classification of NT2/IH presented a substantially more challenging task compared to NT1, with all models demonstrating lower sensitivity despite maintaining high specificity. AUROC remained high across models and sites (range: 0.950-0.992), reflecting strong rank-order discrimination. However, AUPRC was reduced relative to NT1, with mean values ranging from 0.692 (SD 0.085, RF) to 0.778 (SD 0.064, XGB). This divergence between AUROC and AUPRC is expected under low NT2/IH prevalence (approximately 4-8% of the study population): AUROC is relatively insensitive to class imbalance, whereas AUPRC directly captures the precision-recall tradeoff and is more informative when positives are rare (Table 2).

XGB achieved the highest overall balance for NT2/IH, with mean sensitivity of 0.570 (SD 0.117), mean specificity of 0.995 (SD 0.003), mean F1 of 0.675 (SD 0.104), mean AUROC of 0.984 (SD 0.007), and mean AUPRC of 0.778 (SD 0.064). GBT performed comparably (sensitivity 0.621, SD 0.056; F1 0.667, SD 0.039; AUROC 0.976, SD 0.011; AUPRC 0.718, SD 0.071). LR demonstrated lower sensitivity (0.462, SD 0.096) and F1 (0.575, SD 0.075). RF showed the most severe sensitivity deficit (mean 0.216, SD 0.143), including complete failure at MGB (0% sensitivity), where all NT2/IH cases were classified as negative. Figure 1 shows the ROC and precision-recall curves for all NT2/IH models.

Site-level heterogeneity was more pronounced for NT2/IH than for NT1. MGB was consistently the most challenging site, with sensitivity ranging from 0.00 (RF) to 0.574 (GBT) and AUPRC values of 0.543-0.805. Stanford also showed variable performance, with sensitivity ranging from 0.386 (RF) to 0.596 (GBT and XGB). BCH, BIDMC, and Emory demonstrated more moderate but still lower sensitivity than observed for NT1. AUPRC values across sites ranged from 0.543-0.860 (eTable 3). eFigure 4 shows confusion matrices for the best-performing NT2/IH model (XGB).

***Longitudinal Prediction: Model Discrimination***

The longitudinal predictive model demonstrated robust discrimination for both outcomes using pre-diagnostic clinical data (eFigure 5). For the any-narcolepsy model (196 cases, 11,049 controls), stratified 5-fold cross-validation yielded a mean AUC of 0.835 (range across folds: 0.771-0.893) and mean AUPRC of 0.377 (range: 0.280-0.451). LOSO cross-validation produced a mean AUC of 0.797 (range across sites: 0.740-0.894) and mean AUPRC of 0.428 (range: 0.157-0.691), confirming generalizability across institutions (eTable 2).

For the NT1-only model (66 cases, 11,049 controls), 5-fold cross-validation yielded a mean AUC of 0.838 (range: 0.782-0.889) and mean AUPRC of 0.298 (range: 0.232-0.479). LOSO cross-validation yielded a mean AUC of 0.788 (range: 0.628-0.941) and mean AUPRC of 0.285 (range: 0.040-0.618). Performance was more variable across sites for the NT1-only model, consistent with the smaller number of cases per site (range: 7-21 NT1 cases per site). Emory showed the highest LOSO performance (AUC 0.894 for any-narcolepsy; 0.764 for NT1), while MGB showed the weakest (AUC 0.773 and 0.628, respectively), reflecting differences in cohort composition and clinical documentation practices.

***Longitudinal Prediction: Risk Score Distributions and Trajectories***

The distributions of patient-level risk scores showed clear separation between cases and controls (eFigure 6). Control patients' scores were concentrated near zero (median < 0.05), whereas diagnosed patients' scores were broadly distributed, with a substantial proportion receiving scores above 0.9. This separation was observed for both models.

Risk score trajectories, aligned to the time of diagnosis, revealed a progressive increase in model-assigned risk among cases beginning approximately 2 years before diagnosis (Figure 2). The median case risk score (logit scale) rose steadily across the pre-diagnostic window, diverging from the relatively stable control trajectory. The time-varying AUC, computed within 1-year sliding windows, exceeded 0.80 from approximately 1.5 years before diagnosis onward, indicating that the pre-diagnostic signal is detectable well in advance of formal diagnosis.

***Longitudinal Prediction: Predictive Features and Feature Evolution***

The features most strongly associated with narcolepsy risk reflected clinically meaningful pre-diagnostic patterns (eFigure 7). For the any-narcolepsy model, the top positive-coefficient features included mentions of hypocretin, multiple sleep latency testing, modafinil, dextroamphetamine, and idiopathic hypersomnia -- reflecting the clinical workup and empiric treatment that often precedes a formal narcolepsy diagnosis. ICD codes for narcolepsy (G47.41, G47.42) retained non-zero coefficients despite the 6-month horizon exclusion, suggesting that some diagnostic coding occurs more than 6 months before the definitive diagnosis date recorded in the EHR. For the NT1-only model, Dexedrine (dextroamphetamine), hypocretin, and orexin mentions were the strongest predictors, consistent with features distinguishing NT1 from other hypersomnias.

Feature evolution heatmaps revealed distinct temporal accumulation patterns differentiating cases from controls (eFigure 8). Among the 82 features retained after L1 regularization in the any-narcolepsy model (234 cases, 234 matched controls with 5 or more visits), features with positive model coefficients -- including stimulant medications, narcolepsy-related keywords, and sleep study references -- showed progressive accumulation in cases approaching diagnosis, while remaining near-absent in controls. Features with negative coefficients, representing general medical terms more prevalent in the broader clinical population, accumulated more rapidly in controls. These patterns were consistent in the NT1-only model (84 features; 81 cases, 81 matched controls), with cataplexy-related features and NT1-specific medications showing particularly strong case-control divergence (eFigure 9).

***Longitudinal Prediction: Clinical Utility***

The NNT analysis demonstrated that the risk score can meaningfully enrich a screened population for narcolepsy cases (Figure 3). At an assumed prevalence of 0.08%, the baseline NNT without any screening is 1,250 (i.e., 1 in 1,250 individuals in the general population has narcolepsy). For the any-narcolepsy model, applying a score threshold of 0.95 yielded an NNT of 20 with a sensitivity of 68%, representing a 62.5-fold enrichment over the population base rate. At a more stringent threshold of 0.99, the NNT decreased to 10 (sensitivity 67%; 125-fold enrichment). For the NT1-only model, a threshold of 0.85 achieved an NNT of 20 with a sensitivity of 84%, and a threshold of 0.95 achieved an NNT of 10 with a sensitivity of 79%. These results indicate that the model can identify a high-risk subpopulation in which confirmatory diagnostic testing would have a substantially higher yield than unselected screening.

**Discussion**

***Summary of Principal Findings***

Narcolepsy's prolonged diagnostic delay -- averaging 8 to 15 years from symptom onset -- represents a major clinical challenge, with patients suffering preventable morbidity while remaining undiagnosed within the healthcare systems that could identify them. In this study, we developed and validated two complementary NLP-based approaches for narcolepsy phenotyping using EHR data from 5 academic medical centers. Our cross-sectional classifiers achieved AUROC values near 0.99 for NT1 and 0.98 for NT2/IH, demonstrating that narcolepsy-related content in clinical notes can be reliably identified. Our longitudinal predictive model detected elevated narcolepsy risk months to years before formal diagnosis, with AUC of 0.835 (any narcolepsy) and 0.838 (NT1) in cross-validation, and a pre-diagnostic signal detectable from approximately 1.5 years before diagnosis. NNT analysis showed that the model achieves 62.5- to 125-fold enrichment over the population base rate at clinically practical thresholds. Together, these findings represent the first multi-site demonstration that routine clinical notes contain sufficient signal for automated, pre-diagnostic narcolepsy screening, offering a potential path to reducing the diagnostic delay that affects the majority of narcolepsy patients.

***Cross-Sectional Model Performance***

Across both classification tasks, gradient boosting methods -- particularly GBT and XGB -- consistently outperformed LR and RF in terms of sensitivity, F1 score, and cross-site AUPRC stability. Both boosting approaches demonstrated greater robustness to class imbalance than RF, likely because gradient-based learning with shallow trees provides more calibrated treatment of minority-class errors during training. LR achieved competitive AUROC and AUPRC for NT1 but lagged in sensitivity for NT2/IH, suggesting its linear decision boundary is insufficient to capture the more complex, overlapping feature distributions associated with NT2/IH. RF, while achieving the highest specificity (mean 0.996 for NT1, 0.998 for NT2/IH), suffered from poor sensitivity, particularly for NT2/IH.

The contrast in performance between NT1 and NT2/IH classification is clinically interpretable. NT1 is defined by more objective, measurable criteria -- hypocretin deficiency, sleep-onset REM periods, and cataplexy -- that are reliably encoded in ICD codes, medication prescriptions (e.g., sodium oxybate, pitolisant), and clinical note language. NT2/IH, by contrast, is a diagnosis of exclusion with heterogeneous clinical presentation and overlapping symptomatology with other hypersomnolence disorders, resulting in a less discriminative feature landscape. The lower AUPRC values observed for NT2/IH across all models indicate that this represents a fundamentally harder classification problem and motivate further investigation into NT2/IH-specific features and diagnostic biomarkers. The high AUROC values maintained for NT2/IH (0.950-0.992) show that meaningful rank-order discrimination is achievable, but translating this into clinically actionable precision at realistic operating thresholds remains challenging at the prevalence levels observed in this cohort.

***Longitudinal Model Performance and Clinical Implications***

The longitudinal predictive model represents a distinct contribution from the cross-sectional classifiers. While the cross-sectional models identify whether a given clinical note documents narcolepsy, the longitudinal model estimates whether a patient is at elevated risk for eventually receiving a narcolepsy diagnosis, based on cumulative clinical features accrued over time.

Several aspects of these findings merit emphasis. First, the model relies exclusively on NLP features extracted from unstructured clinical notes -- keyword and phrase mentions, medication references, and ICD codes -- rather than structured laboratory values, polysomnographic data, or patient-reported outcomes. This design choice makes the approach broadly deployable in any EHR system containing clinical notes, without requiring specialized data feeds. Second, the 0.5-year horizon exclusion is a deliberately conservative design that forces the model to learn from features present before the typical diagnostic workup period. The resulting model captures the clinical "breadcrumbs" of narcolepsy -- empiric stimulant trials, mentions of excessive sleepiness, antidepressant prescriptions for possible cataplexy -- rather than the diagnostic evaluation itself. Third, the NNT analysis demonstrates practical clinical utility: at a score threshold achieving 68% sensitivity, the model reduces the number of patients requiring confirmatory testing by more than 60-fold compared with unselected screening. For NT1 specifically, the model achieves 84% sensitivity at an NNT of 20, reflecting the more distinctive pre-diagnostic clinical phenotype of NT1 compared with NT2/IH.

The feature evolution analysis provides additional insight into the model's clinical validity. The progressive accumulation of narcolepsy-associated features in cases -- but not controls -- over the 2.5 years preceding diagnosis suggests that the model detects a genuine, gradually emerging clinical signal rather than a sudden diagnostic event. This temporal pattern is consistent with the known diagnostic delay in narcolepsy, which averages 8 to 15 years from symptom onset to diagnosis [3,4].

***Cross-Site Generalizability***

The LOSO framework enabled direct assessment of cross-site generalizability, revealing that model performance -- while generally strong -- exhibited meaningful site-level variability, particularly in sensitivity. For NT1 cross-sectional classification, sensitivity standard deviations ranged from 0.075 (XGB) to 0.168 (RF) across sites, with MGB and Stanford consistently underperforming relative to BCH, BIDMC, and Emory. This pattern partially reflects smaller NT1-positive test cohorts at these sites, where stochastic variation in prediction outcomes has a proportionally greater effect on sensitivity estimates. For NT2/IH, sensitivity standard deviation was highest for RF (0.143) and XGB (0.117), underscoring greater site-to-site instability.

For the longitudinal predictive model, LOSO performance varied from AUC 0.740 (Stanford) to 0.894 (Emory) for any-narcolepsy, and from 0.628 (MGB) to 0.941 (BCH) for NT1. This variability reflects differences in cohort size (e.g., only 7 NT1 cases at Stanford vs. 21 at MGB), patient populations (Stanford and Emory include only sleep clinic patients, whereas other sites draw from broader populations), and institutional documentation practices. These results highlight the importance of multi-site validation and suggest that deployment in new clinical settings should be accompanied by local calibration or prospective evaluation.

***Limitations***

Several limitations should be considered when interpreting these results. First, the model was developed and validated within academic medical centers participating in a single research network, and generalizability to community practice settings, smaller healthcare systems, or non-English-speaking populations requires further study. Second, the class imbalance inherent to rare disease detection (1.7% case prevalence in our cohort, substantially higher than the estimated population prevalence of 0.08%) means that precision-recall metrics are more informative than AUC alone, and real-world positive predictive value will depend on the prevalence in the target screening population. Third, the use of cumulative NLP features means that the model's performance improves with longitudinal data availability and may be less informative for patients with limited clinical documentation. Fourth, annotation quality relies on the judgment of 6 physician annotators with calibration but without formal inter-rater reliability assessment beyond the initial batch; future work should quantify inter-rater agreement more rigorously. Fifth, this study is retrospective in design; prospective validation in a clinical workflow, including assessment of alert fatigue and provider response to model-generated flags, is needed before deployment. Sixth, the feature set is limited to text-derived signals; incorporating structured data (laboratory values, vital signs, polysomnographic results) could potentially improve performance, particularly for NT2/IH.

***Conclusions***

Automated NLP-based analysis of clinical notes can identify patients with narcolepsy cross-sectionally and detect elevated risk before formal diagnosis, with sufficient discrimination and clinical utility to support deployment as an EHR-based screening tool. Cross-sectional classifiers achieved near-perfect discrimination for NT1 and strong performance for NT2/IH, while the longitudinal model demonstrated that pre-diagnostic clinical signals are detectable 1.5 to 2 years before diagnosis at clinically meaningful sensitivity thresholds. Such a system could reduce diagnostic delay by flagging high-risk patients for expedited referral to sleep medicine, potentially shortening the prolonged diagnostic odyssey experienced by many individuals with narcolepsy. Future work should focus on prospective validation, integration with EHR clinical decision support systems, and expansion to community practice settings.

**References**

1. Silber MH, Krahn LE, Olson EJ, Pankratz VS. The epidemiology of narcolepsy in Olmsted County, Minnesota: a population-based study. *Sleep.* 2002;25(2):197-202.
2. Ohayon MM, et al. Prevalence and incidence of narcolepsy symptoms in the US general population. *Sleep.* 2023;47(1).
3. Thorpy MJ, Krieger AC. Delayed diagnosis of narcolepsy: characterization and impact. *Sleep Medicine.* 2014;15(5):502-507.
4. Maurovich-Horvat E, et al. Idling for decades: a European study on risk factors associated with the delay before a narcolepsy diagnosis. *Nature and Science of Sleep.* 2022;14:909-926.
5. Philip P. Therapeutic strategies for mitigating driving risk in patients with narcolepsy. *Nature and Science of Sleep.* 2020;12:1093-1101.
6. Dodel R, et al. The economic consequences of narcolepsy. *Journal of Clinical Sleep Medicine.* 2007;3(7):735.
7. Villa KF, et al. The humanistic and economic burden of narcolepsy. *Journal of Clinical Sleep Medicine.* 2016;12(3):401-407.
8. Carter LP, et al. Listening to the patient voice in narcolepsy: diagnostic delay, disease burden, and treatment efficacy. *Journal of Clinical Sleep Medicine.* 2016;12(12):1635-1646.
9. Bassetti CLA, et al. Narcolepsy -- clinical spectrum, aetiopathophysiology, diagnosis and treatment. *Nature Reviews Neurology.* 2019;15(9):519-539.

**Acknowledgments**

This study was sponsored by Takeda.

**Tables**

Table 1. Cross-Sectional Classification Cohort Characteristics

| | | N | % |
| ----- | ----- | ----- | ----- |
| **Overall** |  |  |  |
|  | Total Patients | 6,498 |  |
|  | Total Notes | 8,990 |  |
|  | Usable Notes (excl. Unclear) | 8,694 |  |
| **Sex** |  |  |  |
|  | Female | 3,428 | 52.8 |
|  | Male | 3,069 | 47.2 |
|  | Unknown | 1 | 0.0 |
| **Race** |  |  |  |
|  | White | 4,187 | 64.4 |
|  | Black or African American | 935 | 14.4 |
|  | Unknown/Declined | 392 | 6.0 |
|  | Other Race | 389 | 6.0 |
|  | Asian | 364 | 5.6 |
|  | Hispanic/Latino | 154 | 2.4 |
|  | Multiracial | 38 | 0.6 |
|  | Native Hawaiian/Pacific Islander | 20 | 0.3 |
|  | American Indian/Alaska Native | 19 | 0.3 |
| **Ethnicity** |  |  |  |
|  | Not Hispanic | 5,132 | 79.0 |
|  | Unknown | 845 | 13.0 |
|  | Hispanic | 521 | 8.0 |
| **Age (years)** |  |  |  |
|  | Mean (SD) | 44.0 (23.5) |  |
|  | Median [IQR] | 46.2 [22.5-63.9] |  |
|  | Range | 0.0-98.3 |  |
| **Site** |  |  |  |
|  | BIDMC -- Patients (Notes) | 1,549 (1,921) | 23.8 (21.4) |
|  | Stanford -- Patients (Notes) | 1,454 (1,477) | 22.4 (16.4) |
|  | Emory -- Patients (Notes) | 1,294 (1,858) | 19.9 (20.7) |
|  | BCH -- Patients (Notes) | 1,141 (1,877) | 17.6 (20.9) |
|  | MGB -- Patients (Notes) | 1,060 (1,857) | 16.3 (20.7) |
| **Annotation** |  |  |  |
|  | NT1 | 620 | 6.9 |
|  | NT2/IH | 360 | 4.0 |
|  | Absent (no narcolepsy) | 7,714 | 85.8 |
|  | Unclear (excluded) | 296 | 3.3 |
| **Annotations by Site** | **NT1 / NT2/IH / Unclear / Absent** | **Total** |  |
|  | BCH: 194 / 46 / 74 / 1,563 | 1,877 |  |
|  | BIDMC: 265 / 126 / 77 / 1,453 | 1,921 |  |
|  | Emory: 56 / 71 / 33 / 1,698 | 1,858 |  |
|  | MGB: 77 / 61 / 73 / 1,646 | 1,857 |  |
|  | Stanford: 28 / 56 / 39 / 1,354 | 1,477 |  |

Table 2. Cross-Sectional Classification -- Average LOSO Cross-Validation Performance

**Panel A: NT1 vs. Others**

| Model | Sensitivity | Specificity | F1 Score | AUROC | AUPRC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.777 +/- 0.119 | 0.989 +/- 0.011 | 0.804 +/- 0.079 | 0.991 +/- 0.005 | 0.906 +/- 0.054 |
| **Random Forest** | 0.722 +/- 0.168 | 0.996 +/- 0.004 | 0.800 +/- 0.118 | 0.994 +/- 0.004 | 0.922 +/- 0.075 |
| **Gradient Boosting** | 0.876 +/- 0.082 | 0.987 +/- 0.018 | 0.850 +/- 0.070 | 0.994 +/- 0.003 | 0.935 +/- 0.039 |
| **XGBoost** | 0.869 +/- 0.075 | 0.990 +/- 0.011 | 0.855 +/- 0.068 | 0.993 +/- 0.005 | 0.924 +/- 0.050 |

**Panel B: NT2/IH vs. Others**

| Model | Sensitivity | Specificity | F1 Score | AUROC | AUPRC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.462 +/- 0.096 | 0.995 +/- 0.002 | 0.575 +/- 0.075 | 0.967 +/- 0.011 | 0.699 +/- 0.056 |
| **Random Forest** | 0.216 +/- 0.143 | 0.998 +/- 0.001 | 0.325 +/- 0.203 | 0.977 +/- 0.008 | 0.692 +/- 0.085 |
| **Gradient Boosting** | 0.621 +/- 0.056 | 0.989 +/- 0.007 | 0.667 +/- 0.039 | 0.976 +/- 0.011 | 0.718 +/- 0.071 |
| **XGBoost** | 0.570 +/- 0.117 | 0.995 +/- 0.003 | 0.675 +/- 0.104 | 0.984 +/- 0.007 | 0.778 +/- 0.064 |

*All values reported as mean +/- SD across five leave-one-site-out folds.*

**Figures**

Figure 1. Receiver Operating Characteristic and Precision-Recall Curves for Cross-Sectional Classification. (A) NT1 vs. Others: ROC curves (left) and precision-recall curves (right) for all four classifier types (LR, RF, GBT, XGB). Each curve represents performance on one LOSO test site. (B) NT2/IH vs. Others: same layout. GBT and XGB consistently achieved the best balance of sensitivity and specificity for both classification tasks. (See figures/figure1a_nt1_roc_prc.png and figures/figure1b_nt2ih_roc_prc.png)

Figure 2. Risk Score Trajectories. Risk score trajectories on the logit scale for cases (blue) and controls (orange), aligned to the time of diagnosis (cases) or pseudo-diagnosis (controls), spanning the 5-year pre-diagnostic window. Left panel: any-narcolepsy model. Right panel: NT1-only model. Individual patient trajectories are shown as thin lines; bold lines represent the 25th, 50th, and 75th percentile trajectories computed with a 1-year sliding window. The dashed vertical line at -0.5 years marks the horizon exclusion boundary. The bottom subpanels show the time-varying AUC within 1-year sliding windows. The green shaded region represents the training window (-2.5 to -0.5 years). (See figures/figure2_risk_score_trajectories.png)

Figure 3. Number Needed to Test (NNT) Analysis. NNT (blue, left y-axis, log scale) and sensitivity (red dashed, right y-axis) as a function of risk score threshold for the any-narcolepsy model (left) and NT1-only model (right). NNT was computed under an assumed population prevalence of 0.08% (1 in 1,250). Annotated operating points: for the any-narcolepsy model, threshold 0.95 yields NNT = 20 (sensitivity 68%; 62.5-fold enrichment) and threshold 0.99 yields NNT = 10 (sensitivity 67%; 125-fold enrichment). For the NT1-only model, threshold 0.85 yields NNT = 20 (sensitivity 84%) and threshold 0.95 yields NNT = 10 (sensitivity 79%). (See figures/figure3_nnt_analysis.png)

**Supplementary Materials**

eFigure 1. CONSORT Diagram -- Cross-Sectional Classification Pipeline. EHR data from 5 BDSP sites underwent stratified enrichment sampling, note selection (>500 words), and manual annotation by 6 physician annotators. Of 8,990 annotated notes, 296 with "Unclear" labels were excluded, yielding 8,694 notes with definitive labels (620 NT1, 360 NT2/IH, 7,714 Absent). Two binary classification tasks were evaluated using LOSO cross-validation with 4 classifier types. (See figures/efigure1_consort_cross_sectional.png)

eFigure 2. CONSORT Diagram -- Longitudinal Prediction Pipeline. The initial cohort of 13,342 patients (596 narcolepsy cases, 12,746 controls) was filtered through gap exclusion (removing 1,754 patients with >5-year visit gaps), visit subsampling (max 20 per patient), sparse feature removal, and temporal windowing (training window -2.5 to -0.5 years before diagnosis). Final cohorts comprised 196 any-narcolepsy cases and 66 NT1 cases with 11,049 controls. Both outcome models were evaluated using 5-fold and LOSO cross-validation. (See figures/efigure2_consort_longitudinal.png)

eFigure 3. Confusion Matrices -- NT1 vs. Others. Confusion matrices for the best-performing Gradient Boosting (GBT) model. Each matrix shows model predictions and true labels for one LOSO test site. (See figures/efigure3_nt1_confusion_matrices.png)

eFigure 4. Confusion Matrices -- NT2/IH vs. Others. Confusion matrices for the best-performing XGBoost (XGB) model. Each matrix shows model predictions and true labels for one LOSO test site. (See figures/efigure4_nt2ih_confusion_matrices.png)

eFigure 5. Predictive Model Performance. AUC (left column) and AUPRC (right column) for the any-narcolepsy model (top row) and NT1-only model (bottom row). Results are shown for stratified 5-fold cross-validation (blue), leave-one-site-out cross-validation (orange), and resubstitution on the final model (green). Individual dots represent per-fold (5-fold CV) or per-site (LOSO) performance. All models used a 0.5-year horizon exclusion, restricting training data to visits occurring 6 months to 2.5 years before diagnosis. The any-narcolepsy model included 196 cases and 11,049 controls; the NT1-only model included 66 cases and 11,049 controls. (See figures/efigure5_predictive_performance.png)

eFigure 6. Risk Score Distributions. Density histograms of patient-level mean risk scores from the final model, shown separately for cases (blue) and controls (orange). Top panel: any-narcolepsy model. Bottom panel: NT1-only model. Controls cluster near zero, while cases are broadly distributed with a prominent peak near 1.0. (See figures/efigure6_risk_score_distributions.png)

eFigure 7. Top Predictive Features. Horizontal bar charts showing the 20 features with the largest mean absolute coefficients for the any-narcolepsy model (left) and NT1-only model (right). Blue bars indicate positive coefficients (increased narcolepsy risk); orange bars indicate negative coefficients (decreased risk). Feature names reflect stemmed clinical keywords, medication names, or ICD code regex patterns; the suffix "_neg_" denotes negated mentions. (See figures/efigure7_top_predictive_features.png)

eFigure 8. Feature Evolution Heatmaps -- Any Narcolepsy. Heatmaps showing mean cumulative feature values over the 2.5-year pre-diagnostic window for the 82 features with non-zero L1 coefficients in the any-narcolepsy model. Left panel: cases (n = 234); right panel: matched controls (n = 234). Only patients with 5 or more visits were included. Feature values were z-score normalized. Rows are ordered by model coefficient value; red rows indicate positive coefficients and blue rows indicate negative coefficients. (See figures/efigure8_feature_heatmap_any_narcolepsy.png)

eFigure 9. Feature Evolution Heatmaps -- NT1 Only. Same as eFigure 8, for the 84 features with non-zero L1 coefficients in the NT1-only model (81 cases, 81 matched controls with 5 or more visits). Cataplexy-related features, NT1-specific medications, and hypocretin/orexin mentions show particularly strong case-control divergence. (See figures/efigure9_feature_heatmap_nt1.png)

eFigure 10. Swimmer Plot of Narcolepsy Patient Timelines. Each horizontal line represents one patient (n = 6,447). Light gray bars indicate the span of hospital records; orange bars indicate periods with narcolepsy-related clinical notes; dark gray bars indicate death. Patients are sorted by the date of their first hospital record. The plot illustrates the temporal coverage of the cohort, spanning from the early 1990s to 2025, with most patients entering the dataset after 2010. (See figures/efigure10_swimmer_plot.png)

eTable 1. Longitudinal Prediction Cohort -- Patient Flow

| Filtering Step | Patients | Visits | Cases (Any Narcolepsy) | Cases (NT1) | Controls |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Initial cohort | 13,342 | 1,022,458 | 596 | 282 | 12,746 |
| After gap exclusion (>5 yr) | 11,588 | 876,318 | 539 | 258 | 11,049 |
| After visit subsampling (max 20) | 11,588 | 164,383 | 539 | 258 | 11,049 |
| After temporal window (h = 0.5 yr) | 11,245 | 155,613 | **196** | **66** | **11,049** |

*Note: Gap exclusion removed 1,754 patients (24 NT1 cases, 33 NT2/IH cases, 1,697 controls) with >5-year gaps between consecutive visits. Case attrition from 539 to 196 (any narcolepsy) and from 258 to 66 (NT1) at the temporal windowing step reflects the requirement that cases have clinical visits within the pre-diagnostic training window (2.5 to 0.5 years before diagnosis). Cases without sufficient documentation in this window were excluded. Controls were not affected by temporal windowing.*

eTable 2. LOSO Cross-Validation Performance by Site -- Longitudinal Predictive Model

| Site | Any-Narcolepsy Cases | NT1 Cases | Controls | AUC (Any-Narc) | AUPRC (Any-Narc) | AUC (NT1) | AUPRC (NT1) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| BCH | 29 | 16 | 605 | 0.797 | 0.454 | 0.941 | 0.311 |
| BIDMC | 48 | 13 | 5,133 | 0.779 | 0.376 | 0.828 | 0.321 |
| Emory | 33 | 9 | 483 | 0.894 | 0.691 | 0.764 | 0.618 |
| MGB | 54 | 21 | 4,182 | 0.773 | 0.157 | 0.628 | 0.133 |
| Stanford | 32 | 7 | 646 | 0.740 | 0.462 | 0.779 | 0.040 |
| **Mean** | **196 total** | **66 total** | **11,049 total** | **0.797** | **0.428** | **0.788** | **0.285** |

eTable 3. LOSO Cross-Validation Performance by Site -- Cross-Sectional Classification

NT1 vs. Others -- Per-Site LOSO AUROC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.992 | 0.984 | 0.997 | 0.989 | 0.991 |
| **Random Forest** | 0.994 | 0.994 | 0.999 | 0.987 | 0.993 |
| **Gradient Boosting** | 0.993 | 0.995 | 0.999 | 0.996 | 0.990 |
| **XGBoost** | 0.994 | 0.995 | 0.998 | 0.996 | 0.984 |

NT1 vs. Others -- Per-Site LOSO AUPRC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.925 | 0.953 | 0.940 | 0.817 | 0.892 |
| **Random Forest** | 0.954 | 0.966 | 0.976 | 0.793 | 0.919 |
| **Gradient Boosting** | 0.948 | 0.973 | 0.963 | 0.899 | 0.887 |
| **XGBoost** | 0.939 | 0.976 | 0.959 | 0.894 | 0.852 |

NT2/IH vs. Others -- Per-Site LOSO AUROC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.978 | 0.976 | 0.965 | 0.965 | 0.949 |
| **Random Forest** | 0.986 | 0.976 | 0.983 | 0.972 | 0.965 |
| **Gradient Boosting** | 0.992 | 0.980 | 0.968 | 0.978 | 0.962 |
| **XGBoost** | 0.992 | 0.988 | 0.986 | 0.976 | 0.977 |

NT2/IH vs. Others -- Per-Site LOSO AUPRC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.681 | 0.784 | 0.627 | 0.683 | 0.705 |
| **Random Forest** | 0.751 | 0.748 | 0.659 | 0.543 | 0.725 |
| **Gradient Boosting** | 0.805 | 0.780 | 0.656 | 0.654 | 0.654 |
| **XGBoost** | 0.799 | 0.860 | 0.719 | 0.696 | 0.785 |

Supplementary Material 1: Sampling cohort definitions

* Narcolepsy Type 1 (NT1)
  * Positive (NT1+)
    * Has at least 3 NT1 ICD codes
    * Has no NT2IH ICD codes
    * Has at least 1 narcolepsy medication
  * Negative (NT1-)
    * No narcolepsy medications
    * No NT1 ICD codes
  * Maybe (NT1?)
    * Is not in NT1 positive or negative groups
    * Has no NT2IH ICD codes
    * Has at least 1 NT1 ICD code
* Narcolepsy Type 2 / Idiopathic Hypersomnia
  * Positive (NT2IH+)
    * Has at least 3 NT2IH ICD codes
    * Has no NT1 ICD codes
    * Has at least 1 narcolepsy medication
  * Negative (NT2IH-)
    * No narcolepsy medications
    * No NT2 ICD codes
  * Maybe (NT2IH?)
    * Is not in NT2IH almost certainly positive or negative groups
    * Has no NT1 ICD codes
    * Has at least 1 NT2IH ICD code

Supplementary Material 2: Standard operating procedure for NT1 and NT2/IH annotation

* Annotation Criteria
  * NT1 = you are >80% confident that the note indicates NT1
  * NT2/IH = you are >80% confident that the note indicates NT2/IH
  * Unclear = you are >50% but <80% confident that the note indicates narcolepsy of either type
  * Absent = you are >80% confident that the note does not indicate narcolepsy of either type
* CSF/hypocretin results (if present in the note):
  * If a patient has a CSF/hypocretin test result less than or equal to 110, then the patient definitely has NT1.
  * Corollarily, if a patient has a CSF/hypocretin lab result greater than 110, this is strong evidence against narcolepsy, but defers to clinical judgment if the doctor explicitly overrides it.
  * The CSF/hypocretin lab results are absolute rules.
  * If the patient does not have a CSF/hypocretin lab (which most patients will not), look for MSLT results, PSG results, HLA results, the doctor's assessment, and/or the patient's illness history.
* HLA/DQB1*06:02 (if mentioned in the note):
  * Most NT1 patients will have a positive HLA/DQB1*06:02, though this is supportive not diagnostic.
* MSLT results:
  * For an MSLT, you would expect the patient to reach REM sleep in 15 minutes or less (this is called a SOREMP).
  * Two or more SOREMPs on the MSLT or PSG as well as a mean sleep latency (MSL) of 8 minutes or less suggest NT1.
* PSG results:
  * For a PSG, you would expect the patient to have a short sleep latency, abnormal REM periods (ex: short REM latency or REM stage occurs first) and fragmented sleep (this might mean a slightly lower sleep efficiency).
  * A patient with NT2/IH would have a short sleep latency and high sleep efficiency. They could also have an overall longer total sleep time (TST). Make sure the short sleep latency (8 min or less) is not just a side effect of sleep disordered breathing (though it is possible for a patient to have both disordered breathing and NT2/IH).
* Doctor's notes:
  * Cataplexy is very important in defining NT1. Check if the doctor mentions cataplexy or the patient having cataplectic attacks. Cataplexy is usually caused by strong emotions, though they can vary. But, if the note only mentions cataplexy and not narcolepsy, then it should be labeled as Unsure.
  * You could also check if the doctor mentions sleep attacks as a synonym for cataplectic attacks (but this is less clear terminology).
  * Cataplexy is a loss of muscle tone (atonia) that might result in a physical collapse.
  * If the doctor says that it is narcolepsy without cataplexy then it is not NT1. That would be NT2/IH.
  * If the doctor only says narcolepsy but does not specify the type or if there is cataplexy, then the patient may or may not have NT1.
  * You can also consider the patient's history if the clinical note mentions a previous diagnosis.
  * Note that NT2 and IH are often confused with NT1.
  * Do not rely on a patient's self reported sleep latency as those are unreliable.
* Edge cases:
  * If CSF/Hypocretin is >110, but other exams indicated the patient has NT1 and the doctor decides to reorder the CSF mark as NT1
  * If the doctor is still waiting on the sleep study results and has no past patient information to make an inference on mark Absent
  * If most of the sleep study results are scrubbed and the diagnosis the doctor makes is not leaning towards narcolepsy mark as Absent
  * If waiting on sleep study and no previous NT1 diagnosis indicated and doctor leans towards hypersomnia or NT2 mark as NT2/IH, or if leans towards apnea mark as Absent
  * If doctor does not mention any kind of narcolepsy, hypersomnia, or excessive sleepiness mark as Absent
  * If a patient has a biological sibling with NT1, most likely not NT1
  * If doctor or tech only talk about mask (ex: CPAP) for a psg, mark as Absent

Supplementary Material 3: Features for classification models

* ICD codes: ICD codes were grouped into three categories using regular expressions (regex) to capture all ICD 9 and ICD 10 pertinent to the group. If a patient had the respective code 6 months before or after their clinical visit, the binary feature was marked as positive. The categories are:
  * Narcolepsy, with cataplexy: ^347\.?[0|1]1|^G47\.?4[1|2]1
  * Narcolepsy, without cataplexy: ^347\.?[0|1]0|^G47\.?4[1|2]9
  * Hypersomnia: ^780\.?5[3|4]|^G47\.?1
* Medications: If a patient had a medication order within 6 months before or after their clinical visit, a binary feature for that medication was marked as positive. Medications included: adderall, ambien, clomipramine, concerta, cymbalta, dexedrine, dextroamphetamine, duloxetine, effexor, fluoxetine, imipramine, modafinil, nuvigil, paroxetine, pitolisant, protriptyline, provigil, prozac, ritalin, sertraline, sodium oxybate, solriamfetol, sunosi, venlafaxine, wakix, xyrem, and xywav.
* Textual features: We created a list of keywords or phrases relevant for narcolepsy by consulting physicians, reviewing published articles, and finding commonly used words in clinical notes, for a total of 587 features. To extract features, each clinical note underwent these preprocessing steps:
  * All special characters and extra whitespace were removed
  * Lowercasing
  * Sentence-level tokenization
  * Word-level tokenization
  * Stemming of each word using SnowballStemmer, which reduces each word to its base or stem (e.g., "denies", "denied", and "deny" all reduce to "deni")

  Each word-stemmed sentence was checked against the list of keywords. If the keyword or set of keywords was found in the sentence, the binary feature was marked positive. However, if a negation keyword (i.e., "no," "not," "absent") was found in the sentence with the keyword(s), a different negating feature was marked as positive.


