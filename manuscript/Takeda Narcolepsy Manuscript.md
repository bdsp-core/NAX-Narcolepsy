**Cross-Sectional and Longitudinal Phenotyping of Narcolepsy from Electronic Health Records**

**Abstract**

**Background.** Narcolepsy affects approximately 1 in 2,000 individuals, yet diagnostic delay averages 8 to 15 years from symptom onset. During this interval, patients accumulate clinical encounters containing identifiable signals — "clinical breadcrumbs" — of undiagnosed narcolepsy. We tested whether natural language processing (NLP) of electronic health record (EHR) clinical notes could identify patients with narcolepsy — not only after diagnosis but in the pre-diagnostic period, months to years before formal recognition.

**Methods.** We used EHR data from 5 academic medical centers participating in the Brain Data Science Platform (BDSP). We developed two complementary modeling approaches: (1) cross-sectional classifiers to identify narcolepsy type 1 (NT1), narcolepsy type 2/idiopathic hypersomnia (NT2/IH), and any narcolepsy (NT1 + NT2/IH combined) from individual clinical notes (6,498 patients, 8,990 annotated notes), and (2) longitudinal predictive models to estimate pre-diagnostic narcolepsy risk from cumulative NLP features for three outcomes — any narcolepsy, NT1 only, and NT2/IH only (initial cohort of 596 narcolepsy cases and 12,746 controls; after filtering, 196 cases with any narcolepsy diagnosis, 66 with confirmed NT1, 130 with confirmed NT2/IH, and 11,049 controls). Both approaches used NLP-derived features extracted from unstructured clinical notes, including stemmed keywords with negation detection, ICD codes, and medication mentions (924 features total; chi-squared selection reduced to 100 for cross-sectional models). Cross-sectional models were evaluated using leave-one-site-out (LOSO) cross-validation. Longitudinal models excluded data from within 6 months of diagnosis and were evaluated using both stratified 5-fold and LOSO cross-validation.

**Results.** For cross-sectional classification, the best-performing models achieved mean area under the receiver operating characteristic curve (AUROC) of 0.996 for NT1 (Random Forest), 0.977 for NT2/IH (XGBoost), and 0.992 for any narcolepsy (XGBoost) across sites. For longitudinal prediction, the any-narcolepsy model achieved a mean AUC of 0.835 (5-fold cross-validation [CV]) and 0.797 (LOSO), the NT1-only model achieved 0.838 (5-fold CV) and 0.788 (LOSO), and the NT2/IH-only model achieved 0.773 (5-fold CV) and 0.794 (LOSO). Beginning approximately 2 years before formal diagnosis, risk scores progressively elevated, diverging from controls. At the most stringent threshold, only 10 patients required confirmatory testing to identify one true case — a 125-fold improvement over unselected population screening.

**Conclusions.** NLP-based analysis of routine clinical notes can identify patients with narcolepsy cross-sectionally and detect elevated risk months to years before formal diagnosis. These findings establish the analytical evidence base for EHR-integrated narcolepsy screening, with specific operating thresholds yielding 62- to 125-fold enrichment over population base rates; prospective validation of clinical utility and provider uptake remains necessary before deployment.

**Introduction**

Narcolepsy is a chronic neurological disorder characterized by excessive daytime sleepiness, with narcolepsy type 1 (NT1) additionally defined by cataplexy or cerebrospinal fluid hypocretin deficiency. The disorder affects approximately 1 in 2,000 individuals, with a combined prevalence of narcolepsy types 1 and 2 estimated at 38-56 per 100,000 in the United States and Europe [1,2]. Despite its significant impact on quality of life, safety, and occupational functioning, narcolepsy remains substantially underdiagnosed: the average delay from symptom onset to diagnosis ranges from 8 to 15 years, with nearly 60% of patients receiving at least one misdiagnosis before the correct diagnosis is established [3,4]. During this prolonged diagnostic odyssey, patients suffer psychosocial harm, develop depression and anxiety at elevated rates, bear substantial economic costs, and — most critically — crash their vehicles three to four times more often than the general population [5,6,7].

This diagnostic gap persists not because patients are absent from the healthcare system, but because their symptoms are diffuse. Patients with undiagnosed narcolepsy seek care from primary care physicians, neurologists, psychiatrists, and other specialists, generating clinical notes that contain subtle but identifiable signals of the underlying disorder — "clinical breadcrumbs" that accumulate progressively in the medical record in the years preceding diagnosis: mentions of excessive sleepiness, empiric stimulant trials, sleep study referrals, and complaints of difficulty waking. Natural language processing (NLP) offers the potential to systematically harvest these signals from unstructured clinical notes at scale across large healthcare systems.

The central challenge is not merely classification but temporal prediction: can we detect narcolepsy risk from clinical notes generated before the diagnostic workup begins? Prior work has applied rule-based and machine-learning phenotyping algorithms to electronic health record (EHR) data for common sleep disorders such as obstructive sleep apnea and insomnia, and the eMERGE Network has demonstrated that EHR-based phenotyping can achieve high accuracy for a range of conditions [10,11]. Narcolepsy presents a harder phenotyping target than common sleep disorders because its rarity (0.05% prevalence) creates severe class imbalance, NT2/idiopathic hypersomnia (NT2/IH) lacks pathognomonic biomarkers, and the diagnostic delay means training data is concentrated years after the clinically interesting pre-diagnostic window. Single-site retrospective studies have identified candidate features predictive of narcolepsy [12], but none have demonstrated multi-site generalizability or pre-diagnostic prediction. Temporal prediction models have been developed for sepsis, autoimmune diseases, and other disorders with prolonged diagnostic delays [13,14] — yet none have addressed central hypersomnias. We chose an interpretable bag-of-words NLP approach rather than contextual language models (e.g., ClinicalBERT) to maximize transparency, facilitate clinical adoption, and enable feature-level analysis of the pre-diagnostic signal.

We tested whether NLP-derived features from routine EHR clinical notes could (1) identify clinical notes documenting narcolepsy and its subtypes, and (2) detect patients at elevated risk for narcolepsy in the pre-diagnostic period, months to years before formal diagnosis. We developed and validated two complementary machine learning approaches using data from 5 academic medical centers participating in BDSP. First, we trained cross-sectional classifiers to distinguish clinical notes indicating NT1, NT2/IH, or any narcolepsy from notes of patients without narcolepsy. Second, we developed a longitudinal predictive model that estimates narcolepsy risk from cumulative clinical features over time. Our key methodological contribution is a temporal exclusion design that enforces this pre-diagnostic constraint: by removing all data from within 6 months of diagnosis, we force the model to learn from the clinical breadcrumbs that accumulate years before diagnosis rather than the diagnostic evaluation itself. To our knowledge, this is the first study to develop and externally validate NLP-based narcolepsy classifiers across multiple institutions and to demonstrate that pre-diagnostic risk can be estimated years before formal diagnosis using routine clinical notes alone.

**Methods**

***Study Design and Data Sources***

We used electronic health record data from 5 academic medical centers participating in the Brain Data Science Platform (BDSP): Boston Children's Hospital (BCH), Beth Israel Deaconess Medical Center (BIDMC), Emory University Hospital (Emory), Massachusetts General Brigham (MGB), and Stanford University Medical Center (Stanford). Each site contributed comprehensive EHR data including demographics, International Classification of Diseases (ICD) diagnosis codes, medication orders, and unstructured clinical notes. Stanford and Emory cohorts include only patients who have visited their respective sleep clinics, whereas BCH, BIDMC, and MGB include broader patient populations. A swimmer plot illustrating the temporal coverage of the narcolepsy patient cohort across sites is provided in eFigure 14.

The BIDMC ethics committee approved and oversaw this study under Institutional Review Board (IRB) protocols (protocols 2024P000807, 2022P000417, 2024P000804), which granted a waiver of consent for retrospective analysis of de-identified EHR data.

We developed two distinct analytic approaches using overlapping but differently constructed cohorts, as described below and illustrated in the CONSORT diagrams (eFigures 1 and 2).

***Feature Extraction***

For both the cross-sectional and longitudinal analyses, we extracted a shared set of 924 NLP-derived features from clinical notes. Features were drawn from three categories:

1. **ICD diagnosis codes** (3 features): Binary indicators for narcolepsy-associated ICD codes identified through regular expression matching, grouped as narcolepsy with cataplexy (ICD-9: 347.01, 347.11; ICD-10: G47.411, G47.421), narcolepsy without cataplexy (ICD-9: 347.00, 347.10; ICD-10: G47.419, G47.429), and hypersomnia (ICD-9: 780.53, 780.54; ICD-10: G47.1x). A code was counted if it appeared within 6 months before or after the clinical visit.

2. **Medication features** (27 features): Binary indicators for 27 narcolepsy-relevant medications including narcolepsy-specific treatments (sodium oxybate/Xyrem, Xywav, Lumryz, modafinil/Provigil, armodafinil/Nuvigil, solriamfetol/Sunosi, pitolisant/Wakix), stimulants (Adderall, Concerta, Dexedrine, dextroamphetamine, Ritalin, methylphenidate, Desoxyn, Evekeo), and antidepressants used off-label for cataplexy (venlafaxine/Effexor, duloxetine/Cymbalta, fluoxetine/Prozac, sertraline, imipramine, clomipramine, protriptyline/Vivactil, paroxetine). A medication was counted if prescribed within 6 months before or after the visit.

3. **Textual features** (894 features): Stemmed clinical keywords and phrases with negation detection, covering narcolepsy-related terminology. We preprocessed clinical notes by lowercasing text, removing special characters, tokenizing sentences and words, and applying Snowball stemming. Each stemmed sentence was matched against a curated keyword list. If a negation term ("no," "not," "absent") appeared in the same sentence as a matched keyword, a separate negation feature was recorded instead. The keyword list encompassed terms for sleep disorders (e.g., "narcolepsi_," "cataplexi_," "hypersomnia_"), diagnostic testing (e.g., "mslt_," "polysomnogram_"), symptoms (e.g., "excess daytim sleepi_," "sleep paralysi_"), and related clinical concepts (Supplementary Material 3).

The initial candidate feature set included 1,204 features. After filtering for features with 10 or more occurrences in the cross-sectional dataset, 924 features were retained for both analyses.

***Cohort Construction: Cross-Sectional Classification***

For the cross-sectional analysis, we constructed an enriched dataset optimized for training note-level classifiers. We used a stratified enrichment strategy because random sampling from the full EHR would yield extremely low narcolepsy prevalence and severe class imbalance.

We defined three sampling groups for both NT1 and NT2/IH based on ICD codes and medication patterns (Supplementary Material 1): (1) "almost certainly positive" (NT1+ or NT2/IH+): patients with 3 or more disease-specific ICD codes, no ICD codes for the other narcolepsy subtype, and at least 1 narcolepsy-relevant medication; (2) "almost certainly negative" (NT1- or NT2/IH-): patients with no narcolepsy ICD codes and no narcolepsy medications; and (3) "maybe" (NT1? or NT2/IH?): patients not meeting positive or negative criteria, with at least 1 relevant ICD code and no codes for the other subtype. We selected approximately 250 patients from each site per classification task, dependent on data availability.

For each selected patient, we sampled clinical notes filtering to those with more than 500 words. For patients in the "almost certainly positive" group, we additionally required at least 1 narcolepsy-related keyword in the note. Each site contributed approximately 1,800 notes, with roughly 300 notes per classification category (6 categories: NT1+, NT1-, NT1?, NT2/IH+, NT2/IH-, NT2/IH?). The resulting cross-sectional dataset comprised 6,498 patients and 8,990 clinical notes across all 5 sites (Table 1).

***Narcolepsy Ascertainment (Ground Truth Labeling)***

Six physician annotators performed manual chart annotation using a custom web-based tool and a standard operating procedure (SOP) defining diagnostic criteria for NT1 and NT2/IH based on CSF hypocretin levels, MSLT results, PSG findings, HLA typing, physician assessments, and clinical history (Supplementary Material 2). Each annotator reviewed an initial calibration batch of 100 notes; after adjudicating discrepancies, each annotator was assigned their own batch. Annotators classified each note as: (1) NT1 (>80% confidence), (2) NT2/IH (>80% confidence), (3) Unclear (50-80% confidence), or (4) Absent (>80% confidence no narcolepsy). Notes classified as "Unclear" (296 notes) were excluded, leaving 8,694 notes with definitive labels: 620 (7.1%) NT1, 360 (4.1%) NT2/IH, and 7,714 (88.7%) Absent (Table 1).

***Cross-Sectional Classification Model Development***

We trained two binary classification models: NT1 vs. others (NT2/IH and Absent combined) and NT2/IH vs. others (NT1 and Absent combined). Prior to model training, chi-squared feature selection was applied to retain the top 100 features per task, reducing the feature space from 924 to 100 to improve computational efficiency while retaining the most discriminative features. We performed nested cross-validation: the outer loop used leave-one-site-out (LOSO) cross-validation to estimate cross-site generalization, while the inner loop used 5-fold patient-stratified cross-validation with grid search for hyperparameter selection. In each inner loop, features were standardized based on training fold statistics; the model was trained with the hyperparameters maximizing averaged performance across held-out folds.

Four classifier types were evaluated:

- **Logistic regression (LR)**: regularization strength C (0.01, 0.1, 1.0), L1 ratio (0.0, 0.5, 1.0), saga solver, class weight (None, balanced), max iterations 1,000.
- **Random forest (RF)**: estimators (100, 300), max depth (None, 20), min samples split (2, 5), class weight (None, balanced).
- **Gradient boosting tree (GBT)**: estimators (100, 200), learning rate (0.1), max depth (5, 7), subsample (0.8).
- **XGBoost (XGB)**: estimators (100, 200), learning rate (0.1), max depth (5, 7), subsample (0.8, 1.0), column sample by tree (0.8, 1.0).

All models used random state 42. We implemented all models using scikit-learn and XGBoost.

***Cross-Sectional Classification Model Evaluation***

Each clinical note received a predicted probability of being positive for narcolepsy. We assessed performance using AUROC, AUPRC, sensitivity, specificity, and F1 score. We assessed model interpretability through feature importances: coefficients for LR and purity-based importance for tree-based models.

***Cohort Construction: Longitudinal Prediction***

For the longitudinal analysis, we used a broader cohort encompassing all available patients with sufficient EHR data, without enrichment sampling. The initial cohort comprised 13,342 patients (1,022,458 clinical visits): 282 confirmed NT1 cases, 314 confirmed NT2/IH cases (596 total), and 12,746 controls.

We applied the following filtering steps sequentially (eTable 1):

1. **Gap exclusion**: Patients with gaps exceeding 5 years between consecutive visits were excluded, removing 1,754 patients (remaining: 11,588 patients, 876,318 visits).
2. **Visit subsampling**: To limit computational burden and prevent overrepresentation of frequently-seen patients, visits were subsampled to a maximum of 20 per patient, preserving the first and last encounters (remaining: 164,383 visits).
3. **Sparse feature removal**: Features with fewer than 50 non-zero values across the dataset were excluded to improve model stability.

***Longitudinal Predictive Model: Temporal Design***

We implemented a temporal exclusion design to evaluate the model's ability to identify narcolepsy risk from pre-diagnostic clinical data. For diagnosed patients, training data were restricted to visits occurring within a pre-diagnostic window of 2.5 years to 6 months before the date of diagnosis (horizon exclusion h = 0.5 years). This exclusion window forces the model to learn from pre-workup clinical features — before narcolepsy-specific ICD codes, diagnostic test orders, and targeted medications appear. All control visits were included without temporal restriction. Patients with fewer than 2 visits after filtering were excluded.

After all filtering, 196 cases remained for the any-narcolepsy model (from 539 after gap exclusion), 66 for the NT1-only model (from 258), and 130 for the NT2/IH-only model (from 281), with 11,049 controls. The substantial case attrition reflects the requirement for sufficient clinical documentation in the narrow 2-year pre-diagnostic training window (eTable 1).

Three outcome models were developed: (1) any narcolepsy (NT1 combined with NT2/IH), (2) NT1 only, and (3) NT2/IH only. For the NT1-only model, NT2/IH patients were excluded entirely from both case and control groups; for the NT2/IH-only model, NT1 patients were excluded entirely.

***Longitudinal Predictive Model: Model Development***

We trained a logistic regression classifier using stochastic gradient descent (SGD) with L1 (lasso) regularization. To address the substantial class imbalance (1.7% case prevalence in the training cohort), we employed a balanced minibatch training strategy: at each training iteration, one visit was randomly sampled from each diagnosed patient, paired with an equal number of visits from distinct control patients. This approach simultaneously addresses class imbalance and prevents patients with more frequent encounters from disproportionately influencing the model.

We performed feature selection independently within each cross-validation fold to prevent information leakage. Within each training fold, we applied a chi-squared test to rank features by their association with the outcome, and the top 100 features were retained. Selected features were standardized to zero mean and unit variance using parameters estimated from the training data only.

The regularization parameter alpha was selected from 5 candidate values (5 x 10^-5, 1 x 10^-4, 5 x 10^-4, 1 x 10^-3, 5 x 10^-3) via an inner cross-validation loop within each training fold. The model was trained for 200 epochs using balanced minibatches at each candidate alpha value, and the alpha yielding the highest inner-fold AUC was selected.

***Longitudinal Predictive Model: Evaluation***

We evaluated performance using two complementary strategies: patient-stratified 5-fold cross-validation (primary) and LOSO cross-validation (secondary), in which the model was trained on 4 sites and evaluated on the held-out site. Visit-level predicted probabilities were aggregated to patient-level scores by computing the mean across all visits. Discrimination was assessed using AUROC (abbreviated as AUC for the longitudinal models) and area under the precision-recall curve (AUPRC).

A final model was trained on all available data using the optimal hyperparameters identified during cross-validation. This model was used for risk score trajectory analysis, feature importance assessment, and clinical utility evaluation.

***Risk Score Trajectory Analysis***

To characterize the temporal evolution of the risk score relative to diagnosis, we scored all visits within a 5-year pre-diagnostic window using the final model. For diagnosed patients, time was aligned to the date of diagnosis; for controls, a pseudo-diagnosis date was randomly assigned from the empirical distribution of case diagnosis times. Risk score trajectories were visualized on the logit scale, with sliding-window percentile bands (25th, 50th, and 75th percentiles; 1-year window) computed separately for cases and controls. Time-varying AUC was computed within 1-year sliding windows to assess how discriminative performance evolves as patients approach diagnosis.

***Feature Evolution Analysis***

To examine how individual model features accumulate over time in cases versus controls, we generated feature evolution heatmaps for each outcome model. For each feature retained after L1 regularization (i.e., features with non-zero model coefficients), we computed the mean cumulative feature count in 10 equally spaced time bins spanning the 2.5-year pre-diagnostic window. Cases were aligned to diagnosis date; controls were aligned to their last visit. To ensure adequate longitudinal coverage, only patients with 5 or more visits within the window were included. For time bins in which a patient had no visit, the last known cumulative value was carried forward. Feature values were z-score normalized across both groups to enable comparison across features with different scales. Features were ordered by model coefficient value, and rows were color-coded by coefficient sign (warm colors for positive coefficients indicating increased narcolepsy risk; cool colors for negative coefficients indicating decreased risk).

***Clinical Utility Analysis***

To evaluate the potential clinical utility of the risk score as a screening tool, we performed a number-needed-to-test (NNT) analysis. Using Bayes' theorem, we estimated the positive predictive value (PPV) at each score threshold under an assumed narcolepsy population prevalence of 0.08% (1 in 1,250). The NNT -- defined as the reciprocal of the PPV -- represents the number of patients who would need to undergo confirmatory diagnostic testing (e.g., polysomnography followed by MSLT) to identify one true case. We annotated clinically relevant operating points at NNT = 10 and NNT = 20.

**Results**

***Cohort Characteristics***

The cross-sectional classification cohort comprised 6,498 patients and 8,990 annotated clinical notes across 5 sites (Table 1). Patients were 52.8% female with a mean age of 44.0 years (SD 23.5); the cohort was 64.4% White, 14.4% Black or African American, and 5.6% Asian. The site-level distribution was: BIDMC 1,549 patients (1,921 notes), Stanford 1,454 patients (1,477 notes), Emory 1,294 patients (1,858 notes), BCH 1,141 patients (1,877 notes), and MGB 1,060 patients (1,857 notes). Of these, 296 notes were classified as "Unclear" and excluded, yielding 8,694 notes with definitive labels for model training (Table 1).

The longitudinal prediction cohort initially comprised 13,342 patients with 1,022,458 clinical visits: 596 narcolepsy cases (282 NT1, 314 NT2/IH) and 12,746 controls. After sequential filtering (gap exclusion removing 1,754 patients with >5-year visit gaps, visit subsampling to max 20 per patient, and temporal windowing with h = 0.5 year horizon exclusion), 196 cases remained for the any-narcolepsy model, 66 for the NT1-only model, and 130 for the NT2/IH-only model, with 11,049 controls (eTable 1). The per-site distribution of cases in the final predictive model cohort was: BCH 29 any-narcolepsy cases (16 NT1, 13 NT2/IH), BIDMC 48 (13 NT1, 35 NT2/IH), Emory 33 (9 NT1, 24 NT2/IH), MGB 54 (21 NT1, 33 NT2/IH), and Stanford 32 (7 NT1, 25 NT2/IH) (eTable 2).

eFigures 1 and 2 present the CONSORT diagrams illustrating patient flow through the cross-sectional classification and longitudinal prediction pipelines, respectively.

***Cross-Sectional Classification: NT1***

All four classifiers achieved mean AUROC exceeding 0.99 for NT1 detection (Table 2). Random Forest (RF) achieved the best balance of discrimination and precision: AUROC 0.996 (SD 0.002), AUPRC 0.935 (SD 0.034), sensitivity 0.878 (SD 0.031), specificity 0.991 (SD 0.007), and F1 0.869 (SD 0.040). XGBoost (XGB) achieved comparable performance (AUROC 0.994, AUPRC 0.911, sensitivity 0.872). Gradient Boosting (GBT) and Logistic Regression (LR) performed similarly, with AUROCs of 0.995 and 0.993, respectively. Figure 1 shows the receiver operating characteristic and precision-recall curves for the best-performing NT1 model (RF); ROC and PRC curves for all four classifiers are shown in eFigure 3.

AUROC was uniformly high across sites (0.988-0.998), and AUPRC ranged from 0.773-0.974 across sites and models (eTable 3). Confusion matrices for the best-performing RF model are shown in eFigure 6.

***Cross-Sectional Classification: NT2/IH***

NT2/IH classification was substantially more challenging, with all models demonstrating lower sensitivity despite high specificity (Table 2). AUROC ranged from 0.962 to 0.978, reflecting strong rank-order discrimination, but AUPRC was reduced (0.632-0.676) given the heterogeneous clinical presentation of NT2/IH and its overlap with other hypersomnolence disorders.

XGBoost achieved the best balance for NT2/IH: AUROC 0.977 (SD 0.006), AUPRC 0.676 (SD 0.043), sensitivity 0.604 (SD 0.109), specificity 0.987 (SD 0.002), and F1 0.619 (SD 0.051). RF achieved the highest AUROC (0.978) but lower sensitivity (0.450). GBT and LR performed comparably (AUROCs 0.975 and 0.962). Figure 1 shows the ROC and precision-recall curves for the best-performing NT2/IH model (XGB); ROC and PRC curves for all four classifiers are shown in eFigure 4.

Site-level heterogeneity was more pronounced for NT2/IH than for NT1 (eTable 3). eFigure 7 shows confusion matrices for the best-performing NT2/IH model (XGB).

***Cross-Sectional Classification: Any Narcolepsy***

For the combined any-narcolepsy classification (NT1 + NT2/IH vs. controls), XGBoost achieved the best overall performance: mean sensitivity of 0.896 (SD 0.069), mean specificity of 0.981 (SD 0.014), mean F1 of 0.853 (SD 0.098), mean AUROC of 0.992 (SD 0.006), and mean AUPRC of 0.934 (SD 0.061) (Table 2). Performance was comparable to the NT1-only model, reflecting that the combined positive class includes the highly discriminable NT1 cases. Figure 1 shows the ROC and PRC curves for the best-performing any-narcolepsy model (XGB); all four classifiers are shown in eFigure 5. Confusion matrices for the best-performing XGB model are shown in eFigure 8.

***Longitudinal Prediction: Model Discrimination***

The longitudinal predictive model demonstrated robust discrimination for all three outcomes using pre-diagnostic clinical data (eFigure 9). For the any-narcolepsy model (196 cases, 11,049 controls), stratified 5-fold cross-validation yielded a mean AUC of 0.835 (range across folds: 0.771-0.893) and mean AUPRC of 0.377 (range: 0.280-0.451). LOSO cross-validation produced a mean AUC of 0.797 (range across sites: 0.740-0.894) and mean AUPRC of 0.428 (range: 0.157-0.691), confirming generalizability across institutions (eTable 2).

For the NT1-only model (66 cases, 11,049 controls), 5-fold cross-validation yielded a mean AUC of 0.838 (range: 0.782-0.889) and mean AUPRC of 0.298 (range: 0.232-0.479). LOSO cross-validation yielded a mean AUC of 0.788 (range: 0.628-0.941) and mean AUPRC of 0.285 (range: 0.040-0.618). Performance was more variable across sites for the NT1-only model, consistent with the smaller number of cases per site (range: 7-21 NT1 cases per site).

For the NT2/IH-only model (130 cases, 11,049 controls), 5-fold cross-validation yielded a mean AUC of 0.773 (range: 0.721-0.869) and mean AUPRC of 0.265 (range: 0.200-0.348). LOSO cross-validation yielded a mean AUC of 0.794 (range: 0.677-0.870) and mean AUPRC of 0.271 (range: 0.142-0.481). The lower AUC for NT2/IH compared with NT1 mirrors the cross-sectional findings and reflects the more heterogeneous pre-diagnostic clinical phenotype of NT2/IH.

Across all three outcomes, Emory showed the highest LOSO performance (AUC 0.894 for any-narcolepsy, 0.764 for NT1, 0.870 for NT2/IH), while MGB showed the weakest (AUC 0.773, 0.628, and 0.677, respectively), reflecting differences in cohort composition and clinical documentation practices.

***Longitudinal Prediction: Risk Score Distributions and Trajectories***

The distributions of patient-level risk scores showed clear separation between cases and controls (eFigure 10). Control patients' scores were concentrated near zero (median < 0.05), whereas diagnosed patients' scores were broadly distributed, with a substantial proportion receiving scores above 0.9. This separation was observed for all three models.

Risk score trajectories, aligned to the time of diagnosis, revealed a progressive increase in model-assigned risk among cases beginning approximately 2 years before diagnosis (Figure 2). The median case risk score (logit scale) rose steadily across the pre-diagnostic window, diverging from the relatively stable control trajectory. The time-varying AUC, computed within 1-year sliding windows, exceeded 0.80 from approximately 1.5 years before diagnosis onward, indicating that the pre-diagnostic signal is detectable well in advance of formal diagnosis.

***Longitudinal Prediction: Predictive Features and Feature Evolution***

The features most strongly associated with narcolepsy risk reflected clinically meaningful pre-diagnostic patterns (eFigure 11). For the any-narcolepsy model, the top positive-coefficient features included mentions of hypocretin, multiple sleep latency testing, modafinil, dextroamphetamine, and idiopathic hypersomnia -- reflecting the clinical workup and empiric treatment that often precedes a formal narcolepsy diagnosis. ICD codes for narcolepsy (G47.41, G47.42) retained non-zero coefficients despite the 6-month horizon exclusion, suggesting that some diagnostic coding occurs more than 6 months before the definitive diagnosis date recorded in the EHR. For the NT1-only model, Dexedrine (dextroamphetamine), hypocretin, and orexin mentions were the strongest predictors, consistent with features distinguishing NT1 from other hypersomnias.

Feature evolution heatmaps revealed distinct temporal accumulation patterns differentiating cases from controls (eFigure 12). Features with positive coefficients — stimulant medications, narcolepsy-related keywords, and sleep study references — showed progressive accumulation in cases approaching diagnosis while remaining near-absent in controls (82 features retained after L1 regularization; 234 cases, 234 matched controls with 5 or more visits). Features with negative coefficients, representing general medical terms more prevalent in the broader clinical population, accumulated more rapidly in controls. These patterns were consistent in the NT1-only model (84 features; 81 cases, 81 matched controls), with cataplexy-related features and NT1-specific medications showing particularly strong case-control divergence (eFigure 13).

***Longitudinal Prediction: Clinical Utility***

The risk score enriches screened populations for narcolepsy cases by 62- to 125-fold over the population base rate (Figure 3). At an assumed prevalence of 0.08%, the baseline NNT without any screening is 1,250 (i.e., 1 in 1,250 individuals in the general population has narcolepsy). For the any-narcolepsy model, applying a score threshold of 0.95 yielded an NNT of 20 with a sensitivity of 68%, representing a 62.5-fold enrichment over the population base rate. At a more stringent threshold of 0.99, the NNT decreased to 10 (sensitivity 67%; 125-fold enrichment). For the NT1-only model, a threshold of 0.85 achieved an NNT of 20 with a sensitivity of 84%, and a threshold of 0.95 achieved an NNT of 10 with a sensitivity of 79%. These results indicate that the model can identify a high-risk subpopulation in which confirmatory diagnostic testing would have a substantially higher yield than unselected screening.

**Discussion**

***Summary of Principal Findings***

These results represent, to our knowledge, the first multi-site demonstration that routine clinical notes contain sufficient signal for automated, pre-diagnostic narcolepsy screening. Using EHR data from 5 academic medical centers, our cross-sectional classifiers achieved AUROC values of 0.996 for NT1, 0.977 for NT2/IH, and 0.992 for any narcolepsy, demonstrating that narcolepsy-related content in clinical notes can be reliably identified across all three classification tasks. Our longitudinal predictive models detected elevated narcolepsy risk months to years before formal diagnosis, with AUC of 0.835 (any narcolepsy), 0.838 (NT1), and 0.773 (NT2/IH) in cross-validation, and a pre-diagnostic signal detectable from approximately 1.5 years before diagnosis. NNT analysis showed that the model achieves 62.5- to 125-fold enrichment over the population base rate at clinically practical thresholds, providing the operating parameters needed for EHR-integrated narcolepsy screening.

***Cross-Sectional Model Performance***

Random Forest achieved the highest AUROC (0.996) and AUPRC (0.935) for NT1, while XGBoost performed best for NT2/IH (AUROC 0.977, AUPRC 0.676) and any narcolepsy (AUROC 0.992, AUPRC 0.934). All four classifiers achieved comparable overall performance, with differences in sensitivity-specificity tradeoffs reflecting their distinct inductive biases.

The contrast between NT1 and NT2/IH performance reflects underlying clinical biology. NT1 is defined by objective criteria — hypocretin deficiency, sleep-onset REM periods, and cataplexy — reliably encoded in ICD codes, medication prescriptions, and clinical note language. NT2/IH, by contrast, is a diagnosis of exclusion with heterogeneous presentation and overlapping symptomatology, resulting in a less discriminative feature landscape. The lower AUPRC for NT2/IH (0.632-0.676) indicates a fundamentally harder classification problem, though AUROC values (0.962-0.978) confirm that strong rank-order discrimination is achievable.

***Longitudinal Model Performance and Clinical Implications***

The longitudinal predictive model represents a distinct contribution from the cross-sectional classifiers. While the cross-sectional models identify whether a given clinical note documents narcolepsy, the longitudinal model estimates whether a patient is at elevated risk for eventually receiving a narcolepsy diagnosis, based on cumulative clinical features accrued over time.

Three aspects distinguish this model from the cross-sectional classifiers. First, it relies exclusively on NLP features from unstructured clinical notes — keyword mentions, medication references, and ICD codes — rather than structured laboratory values or polysomnographic data, making it broadly deployable in any EHR system. Second, the 0.5-year horizon exclusion forces the model to learn from features present before the diagnostic workup, capturing the clinical breadcrumbs of narcolepsy — empiric stimulant trials, mentions of excessive sleepiness, antidepressant prescriptions for possible cataplexy — rather than the diagnostic evaluation itself. Third, the NNT analysis demonstrates practical utility: at 68% sensitivity, the model reduces the number of patients requiring confirmatory testing by more than 60-fold. For NT1 specifically, the model achieves 84% sensitivity at an NNT of 20.

The feature evolution analysis provides additional insight into the model's clinical validity. The progressive accumulation of narcolepsy-associated features in cases -- but not controls -- over the 2.5 years preceding diagnosis suggests that the model detects a genuine, gradually emerging clinical signal rather than a sudden diagnostic event. This temporal pattern is consistent with the known diagnostic delay in narcolepsy, which averages 8 to 15 years from symptom onset to diagnosis [3,4].

***Cross-Site Generalizability***

The LOSO framework revealed that performance, while generally strong, exhibited meaningful site-level variability. For NT1 cross-sectional classification, sensitivity SDs ranged from 0.031 (RF) to 0.090 (LR); for NT2/IH, variability was greater (SD up to 0.127 for RF).

For the longitudinal model, LOSO AUC varied from 0.740 (Stanford) to 0.894 (Emory) for any-narcolepsy, and from 0.628 (MGB) to 0.941 (BCH) for NT1. This variability reflects differences in cohort size (7-21 NT1 cases per site), patient populations (Stanford and Emory include only sleep clinic patients), and documentation practices. Deployment in new settings should be accompanied by local calibration or prospective evaluation.

***Limitations***

Several limitations should be considered when interpreting these results.

*Generalizability.* The model was developed within academic medical centers in a single research network. Generalizability to community settings, smaller systems, or non-English-speaking populations remains untested. Stanford and Emory cohorts included only sleep clinic patients, potentially inflating performance. Substantial case attrition in the longitudinal cohort (539 to 196 cases) may introduce selection bias.

*Methodological constraints.* Real-world positive predictive value will depend on the prevalence in the target screening population (1.7% in our cohort vs. 0.08% estimated population prevalence). Performance improves with longitudinal data availability and may be less informative for patients with limited documentation. Incorporating structured data or contextual language models (e.g., ClinicalBERT) could improve NT2/IH performance at increased computational and interpretability cost.

*Annotation and study design.* Six physician annotators classified notes after calibration, but we did not assess formal inter-rater reliability; future work should quantify agreement more rigorously. This study is retrospective; prospective validation — including assessment of alert fatigue and provider uptake — is needed before deployment.

***Conclusions***

Automated NLP-based analysis of clinical notes can identify patients with narcolepsy cross-sectionally and detect elevated risk before formal diagnosis, with sufficient discrimination and clinical utility to support deployment as an EHR-based screening tool. Cross-sectional classifiers achieved near-perfect discrimination for NT1 and strong performance for NT2/IH, while the longitudinal model demonstrated that pre-diagnostic clinical signals are detectable 1.5 to 2 years before diagnosis at practical sensitivity thresholds. Deployment of such a system could meaningfully reduce the years of diagnostic delay that most narcolepsy patients currently experience, by flagging high-risk patients for expedited referral to sleep medicine. Several avenues merit further investigation: prospective deployment studies should assess clinician response to model-generated alerts, integration of structured data may improve NT2/IH discrimination, and external validation in community healthcare settings is essential before broad deployment.

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
10. Kirby JC, et al. PheKB: a catalog and workflow for creating electronic phenotype algorithms for transportability. *Journal of the American Medical Informatics Association.* 2016;23(6):1046-1052.
11. Gottesman O, et al. The Electronic Medical Records and Genomics (eMERGE) Network: past, present, and future. *Genetics in Medicine.* 2013;15(10):761-771.
12. Ramachandran A, Bhatt P. Machine learning approaches for narcolepsy diagnosis using electronic health records. *Sleep Medicine Reviews.* 2023;71:101838.
13. Rajkomar A, et al. Scalable and accurate deep learning with electronic health records. *npj Digital Medicine.* 2018;1(1):18.
14. Norgeot B, et al. Assessment of a deep learning model based on electronic health record data to forecast clinical outcomes in patients with rheumatic diseases. *JAMA Network Open.* 2019;2(3):e190606.

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
| **Logistic Regression** | 0.790 +/- 0.090 | 0.989 +/- 0.008 | 0.803 +/- 0.051 | 0.993 +/- 0.003 | 0.896 +/- 0.064 |
| **Random Forest** | 0.878 +/- 0.031 | 0.991 +/- 0.007 | 0.869 +/- 0.040 | 0.996 +/- 0.002 | 0.935 +/- 0.034 |
| **Gradient Boosting** | 0.844 +/- 0.052 | 0.994 +/- 0.003 | 0.870 +/- 0.030 | 0.995 +/- 0.002 | 0.918 +/- 0.036 |
| **XGBoost** | 0.872 +/- 0.045 | 0.992 +/- 0.003 | 0.854 +/- 0.082 | 0.994 +/- 0.003 | 0.911 +/- 0.054 |

**Panel B: NT2/IH vs. Others**

| Model | Sensitivity | Specificity | F1 Score | AUROC | AUPRC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.515 +/- 0.072 | 0.991 +/- 0.003 | 0.589 +/- 0.049 | 0.962 +/- 0.011 | 0.653 +/- 0.037 |
| **Random Forest** | 0.450 +/- 0.127 | 0.991 +/- 0.003 | 0.525 +/- 0.091 | 0.978 +/- 0.005 | 0.672 +/- 0.047 |
| **Gradient Boosting** | 0.612 +/- 0.106 | 0.985 +/- 0.003 | 0.612 +/- 0.054 | 0.975 +/- 0.008 | 0.632 +/- 0.050 |
| **XGBoost** | 0.604 +/- 0.109 | 0.987 +/- 0.002 | 0.619 +/- 0.051 | 0.977 +/- 0.006 | 0.676 +/- 0.043 |

**Panel C: Any Narcolepsy (NT1 + NT2/IH) vs. Others**

| Model | Sensitivity | Specificity | F1 Score | AUROC | AUPRC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.840 +/- 0.085 | 0.983 +/- 0.015 | 0.831 +/- 0.103 | 0.988 +/- 0.011 | 0.918 +/- 0.069 |
| **Random Forest** | 0.908 +/- 0.055 | 0.974 +/- 0.028 | 0.838 +/- 0.139 | 0.992 +/- 0.006 | 0.934 +/- 0.056 |
| **Gradient Boosting** | 0.895 +/- 0.054 | 0.979 +/- 0.020 | 0.843 +/- 0.123 | 0.989 +/- 0.009 | 0.938 +/- 0.051 |
| **XGBoost** | 0.896 +/- 0.069 | 0.981 +/- 0.014 | 0.853 +/- 0.098 | 0.992 +/- 0.005 | 0.934 +/- 0.054 |

*All values reported as mean +/- SD across five leave-one-site-out folds.*

**Figures**

Figure 1. Cross-Sectional Classifiers Achieve Near-Perfect Discrimination for NT1 (AUROC 0.996), NT2/IH (AUROC 0.977), and Any Narcolepsy (AUROC 0.992). (A, B) NT1 vs. Others (620 NT1 notes, 8,074 non-NT1 notes): ROC and precision-recall curves for the best-performing Random Forest (RF) model. (C, D) NT2/IH vs. Others (360 NT2/IH notes, 8,334 non-NT2/IH notes): ROC and precision-recall curves for the best-performing XGBoost (XGB) model. (E, F) Any Narcolepsy vs. Others (980 narcolepsy notes, 7,714 non-narcolepsy notes): ROC and precision-recall curves for the best-performing XGBoost (XGB) model. Each curve represents performance on one LOSO test site. Full comparisons across all four classifier types are shown in eFigures 3-5. (See figures/figure1_roc_prc.png)

Figure 2. Pre-Diagnostic Risk Scores Diverge From Controls Approximately 2 Years Before Narcolepsy Diagnosis. Risk score trajectories on the logit scale for cases (blue) and controls (orange), aligned to the time of diagnosis (cases) or pseudo-diagnosis (controls). (A) Any-narcolepsy model (196 cases, 11,049 controls). (B) NT1-only model (66 cases, 11,049 controls). (C) NT2/IH-only model (130 cases, 11,049 controls). Individual patient trajectories are shown as thin lines; bold lines represent the median (solid) and 25th/75th percentile (dashed) trajectories computed with a 1-year sliding window. (D-F) Time-varying AUROC within 1-year sliding windows for the corresponding models. Horizontal dashed lines mark AUROC = 0.8 and 0.9. (See figures/figure2_risk_score_trajectories.png)

Figure 3. Risk Score Thresholds Achieve 62.5- to 125-Fold Enrichment Over Population Base Rate. NNT (blue, left y-axis, log scale) and sensitivity (red dashed, right y-axis) as a function of risk score threshold for the any-narcolepsy model (A; 196 cases, 11,049 controls), NT1-only model (B; 66 cases, 11,049 controls), and NT2/IH-only model (C; 130 cases, 11,049 controls). NNT was computed under an assumed population prevalence of 0.08% (1 in 1,250) using sensitivity and specificity from the final model. Annotated operating points are shown at NNT = 10 and NNT = 20 for each model. (See figures/figure3_nnt_analysis.png)

**Supplementary Materials**

eFigure 1. CONSORT Diagram -- Cross-Sectional Classification Pipeline. EHR data from 5 BDSP sites underwent stratified enrichment sampling, note selection (>500 words), and manual annotation by 6 physician annotators. Of 8,990 annotated notes, 296 "Unclear" notes were excluded, yielding 8,694 notes with definitive labels (620 NT1, 360 NT2/IH, 7,714 Absent). Three binary classification tasks (NT1 vs. others, NT2/IH vs. others, any narcolepsy vs. others) were evaluated using LOSO cross-validation with 4 classifier types. (See figures/efigure1_consort_cross_sectional.png)

eFigure 2. CONSORT Diagram -- Longitudinal Prediction Pipeline. The initial cohort of 13,342 patients (596 narcolepsy cases, 12,746 controls) was filtered through gap exclusion (removing 1,754 patients with >5-year visit gaps), visit subsampling (max 20 per patient), sparse feature removal, and temporal windowing (training window -2.5 to -0.5 years before diagnosis). Final cohorts comprised 196 any-narcolepsy cases, 66 NT1 cases, and 130 NT2/IH cases, with 11,049 controls. All three outcome models were evaluated using 5-fold and LOSO cross-validation. (See figures/efigure2_consort_longitudinal.png)

eFigure 3. ROC and PRC Curves -- NT1 vs. Others (All Models). ROC curves (left) and precision-recall curves (right) for all four classifier types (LR, RF, GBT, XGB) in the NT1 classification task. Each curve represents performance on one LOSO test site. (See figures/efigure3_nt1_all_models_roc_prc.png)

eFigure 4. ROC and PRC Curves -- NT2/IH vs. Others (All Models). ROC curves (left) and precision-recall curves (right) for all four classifier types (LR, RF, GBT, XGB) in the NT2/IH classification task. Each curve represents performance on one LOSO test site. (See figures/efigure4_nt2ih_all_models_roc_prc.png)

eFigure 5. ROC and PRC Curves -- Any Narcolepsy vs. Others (All Models). ROC curves (left) and precision-recall curves (right) for all four classifier types (LR, RF, GBT, XGB) in the any-narcolepsy classification task. Each curve represents performance on one LOSO test site. (See figures/efigure5_any_narcolepsy_all_models_roc_prc.png)

eFigure 6. Confusion Matrices -- NT1 vs. Others. Confusion matrices for the best-performing Random Forest (RF) model. Each matrix shows model predictions and true labels for one LOSO test site. (See figures/efigure6_nt1_confusion_matrices.png)

eFigure 7. Confusion Matrices -- NT2/IH vs. Others. Confusion matrices for the best-performing XGBoost (XGB) model. Each matrix shows model predictions and true labels for one LOSO test site. (See figures/efigure7_nt2ih_confusion_matrices.png)

eFigure 8. Confusion Matrices -- Any Narcolepsy vs. Others. Confusion matrices for the best-performing XGBoost (XGB) model. Each matrix shows model predictions and true labels for one LOSO test site. (See figures/efigure8_any_narcolepsy_confusion_matrices.png)

eFigure 9. Longitudinal Models Achieve AUC 0.77-0.84 Across Three Narcolepsy Outcomes. AUC (left column) and AUPRC (right column) for the any-narcolepsy model, NT1-only model, and NT2/IH-only model. Results are shown for stratified 5-fold cross-validation (blue), leave-one-site-out cross-validation (orange), and resubstitution on the final model (green). Individual dots represent per-fold (5-fold CV) or per-site (LOSO) performance. All models used a 0.5-year horizon exclusion, restricting training data to visits occurring 6 months to 2.5 years before diagnosis. The any-narcolepsy model included 196 cases, the NT1-only model 66 cases, and the NT2/IH-only model 130 cases, each with 11,049 controls. (See figures/efigure9_predictive_performance.png)

eFigure 10. Risk Scores Separate Cases From Controls Across All Three Models. Density histograms of patient-level mean risk scores from the final model, shown separately for cases (blue) and controls (orange) for each of the three outcome models. Controls cluster near zero, while cases are broadly distributed with a prominent peak near 1.0. (See figures/efigure10_risk_score_distributions.png)

eFigure 11. Stimulant Medications, Sleep Study Terms, and Hypocretin Mentions Drive Pre-Diagnostic Risk Scores. Horizontal bar charts showing the 20 features with the largest mean absolute coefficients for each outcome model. Blue bars indicate positive coefficients (increased narcolepsy risk); orange bars indicate negative coefficients (decreased risk). Feature names reflect stemmed clinical keywords, medication names, or ICD code regex patterns; the suffix "_neg_" denotes negated mentions. (See figures/efigure11_top_predictive_features.png)

eFigure 12. Narcolepsy-Associated Features Accumulate Progressively in Cases But Not Controls. Heatmaps showing mean cumulative feature values over the 2.5-year pre-diagnostic window for the 82 features with non-zero L1 coefficients in the any-narcolepsy model. Left panel: cases (n = 234); right panel: matched controls (n = 234). Only patients with 5 or more visits were included. Feature values were z-score normalized. Rows are ordered by model coefficient value; red rows indicate positive coefficients and blue rows indicate negative coefficients. (See figures/efigure12_feature_heatmap_any_narcolepsy.png)

eFigure 13. Cataplexy-Related Features and NT1-Specific Medications Show Strong Case-Control Divergence. Same as eFigure 12, for the 84 features with non-zero L1 coefficients in the NT1-only model (81 cases, 81 matched controls with 5 or more visits). Cataplexy-related features, NT1-specific medications, and hypocretin/orexin mentions show particularly strong case-control divergence. (See figures/efigure13_feature_heatmap_nt1.png)

eFigure 14. Swimmer Plot of Narcolepsy Patient Timelines. Each horizontal line represents one patient (n = 6,447). Light gray bars indicate the span of hospital records; orange bars indicate periods with narcolepsy-related clinical notes; dark gray bars indicate death. Patients are sorted by the date of their first hospital record. The plot illustrates the temporal coverage of the cohort, spanning from the early 1990s to 2025, with most patients entering the dataset after 2010. (See figures/efigure14_swimmer_plot.png)

eTable 1. Longitudinal Prediction Cohort -- Patient Flow

| Filtering Step | Patients | Visits | Cases (Any Narcolepsy) | Cases (NT1) | Cases (NT2/IH) | Controls |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Initial cohort | 13,342 | 1,022,458 | 596 | 282 | 314 | 12,746 |
| After gap exclusion (>5 yr) | 11,588 | 876,318 | 539 | 258 | 281 | 11,049 |
| After visit subsampling (max 20) | 11,588 | 164,383 | 539 | 258 | 281 | 11,049 |
| After temporal window (h = 0.5 yr) | 11,245 | 155,613 | **196** | **66** | **130** | **11,049** |

*Note: Gap exclusion removed 1,754 patients (24 NT1 cases, 33 NT2/IH cases, 1,697 controls) with >5-year gaps between consecutive visits. Case attrition from 539 to 196 (any narcolepsy), from 258 to 66 (NT1), and from 281 to 130 (NT2/IH) at the temporal windowing step reflects the requirement that cases have clinical visits within the pre-diagnostic training window (2.5 to 0.5 years before diagnosis). Cases without sufficient documentation in this window were excluded. Controls were not affected by temporal windowing.*

eTable 2. LOSO Cross-Validation Performance by Site -- Longitudinal Predictive Model

| Site | Any-Narc Cases | NT1 Cases | NT2/IH Cases | Controls | AUC (Any-Narc) | AUPRC (Any-Narc) | AUC (NT1) | AUPRC (NT1) | AUC (NT2/IH) | AUPRC (NT2/IH) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| BCH | 29 | 16 | 13 | 605 | 0.797 | 0.454 | 0.941 | 0.311 | 0.808 | 0.142 |
| BIDMC | 48 | 13 | 35 | 5,133 | 0.779 | 0.376 | 0.828 | 0.321 | 0.790 | 0.301 |
| Emory | 33 | 9 | 24 | 483 | 0.894 | 0.691 | 0.764 | 0.618 | 0.870 | 0.481 |
| MGB | 54 | 21 | 33 | 4,182 | 0.773 | 0.157 | 0.628 | 0.133 | 0.677 | 0.192 |
| Stanford | 32 | 7 | 25 | 646 | 0.740 | 0.462 | 0.779 | 0.040 | 0.822 | 0.238 |
| **Mean** | **196** | **66** | **130** | **11,049** | **0.797** | **0.428** | **0.788** | **0.285** | **0.794** | **0.271** |

eTable 3. LOSO Cross-Validation Performance by Site -- Cross-Sectional Classification

NT1 vs. Others -- Per-Site LOSO AUROC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.993 | 0.991 | 0.996 | 0.988 | 0.996 |
| **Random Forest** | 0.994 | 0.995 | 0.998 | 0.994 | 0.997 |
| **Gradient Boosting** | 0.993 | 0.994 | 0.998 | 0.995 | 0.994 |
| **XGBoost** | 0.994 | 0.995 | 0.998 | 0.994 | 0.989 |

NT1 vs. Others -- Per-Site LOSO AUPRC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.944 | 0.949 | 0.917 | 0.773 | 0.897 |
| **Random Forest** | 0.953 | 0.972 | 0.951 | 0.875 | 0.923 |
| **Gradient Boosting** | 0.918 | 0.961 | 0.948 | 0.858 | 0.903 |
| **XGBoost** | 0.950 | 0.974 | 0.937 | 0.832 | 0.863 |

NT2/IH vs. Others -- Per-Site LOSO AUROC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.976 | 0.966 | 0.965 | 0.962 | 0.943 |
| **Random Forest** | 0.984 | 0.977 | 0.984 | 0.973 | 0.971 |
| **Gradient Boosting** | 0.984 | 0.972 | 0.981 | 0.975 | 0.961 |
| **XGBoost** | 0.985 | 0.978 | 0.982 | 0.975 | 0.967 |

NT2/IH vs. Others -- Per-Site LOSO AUPRC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.626 | 0.691 | 0.597 | 0.663 | 0.690 |
| **Random Forest** | 0.655 | 0.712 | 0.669 | 0.597 | 0.730 |
| **Gradient Boosting** | 0.613 | 0.668 | 0.632 | 0.551 | 0.697 |
| **XGBoost** | 0.632 | 0.732 | 0.673 | 0.626 | 0.715 |

Any Narcolepsy vs. Others -- Per-Site LOSO AUROC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.995 | 0.996 | 0.992 | 0.991 | 0.967 |
| **Random Forest** | 0.996 | 0.997 | 0.995 | 0.990 | 0.980 |
| **Gradient Boosting** | 0.997 | 0.991 | 0.995 | 0.992 | 0.971 |
| **XGBoost** | 0.997 | 0.996 | 0.995 | 0.993 | 0.982 |

Any Narcolepsy vs. Others -- Per-Site LOSO AUPRC

| Model | BCH | BIDMC | Emory | MGB | Stanford |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.971 | 0.983 | 0.915 | 0.931 | 0.790 |
| **Random Forest** | 0.968 | 0.984 | 0.939 | 0.954 | 0.827 |
| **Gradient Boosting** | 0.979 | 0.977 | 0.938 | 0.956 | 0.839 |
| **XGBoost** | 0.974 | 0.982 | 0.934 | 0.951 | 0.831 |

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
* Textual features: We created a list of keywords or phrases relevant for narcolepsy by consulting physicians, reviewing published articles, and finding commonly used words in clinical notes, totaling 587 features. To extract features, each clinical note underwent these preprocessing steps:
  * All special characters and extra whitespace were removed
  * Lowercasing
  * Sentence-level tokenization
  * Word-level tokenization
  * Stemming of each word using SnowballStemmer, which reduces each word to its base or stem (e.g., "denies", "denied", and "deny" all reduce to "deni")

  Each word-stemmed sentence was checked against the list of keywords. If the keyword or set of keywords was found in the sentence, the binary feature was marked positive. However, if a negation keyword (i.e., "no," "not," "absent") was found in the sentence with the keyword(s), a different negating feature was marked as positive.


