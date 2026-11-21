# COGITATE MEG functional-connectivity estimator audit V0

**Status:** Literature audit and reproduction design. No MEG data were downloaded or analysed by this work item.  
**Date:** 2026-11-21

## 1. Why this changes CETB

Bryant and Whyte (2026) provide the closest direct comparator for a COGITATE MEG feature benchmark:

- Article: *A data-driven approach to identifying and evaluating connectivity-based neural correlates of conscious visual perception*.
- Journal: *Neuroscience of Consciousness* 2026(1), niag029.
- DOI: 10.1093/nc/niag029.
- Authors: Annie G. Bryant and Christopher J. Whyte.
- Correction: niag071, published 2026-10-15, DOI 10.1093/nc/niag071.

The study used COGITATE MEG BIDS data for 100 participants; 94 had the anatomical scans required for source localization. It used pyspi v1.1.0 to calculate 262 pairwise statistics; 246 remained after a declared QC criterion requiring nonzero variance and real/non-NaN values across region-pair outputs in at least 90% of participants (N≥85). The primary analysis screened 35,280 models. Barycenter-derived statistics were among the strongest stimulus-type decoders and generalized across several theory-relevant region pairs. Many measures showed cross-validated stimulus decoding in the 60–67% range, while none exceeded 56% for domain-general task relevance in the reported supplementary analysis.

This is a high-value same-dataset methodological comparator, not an independent replication of COGITATE and not a direct measurement of phenomenal consciousness. The authors themselves state that the results do not bear strongly on whether the proposed mechanisms are necessary or sufficient for perceptual consciousness.

## 2. What the study did and did not establish

### Useful contribution

The article demonstrates that Pearson correlation is not the only reasonable inter-areal coupling measure. It compared a broad family of 246 candidate statistics and identified barycenter-derived measures as promising for stimulus-type decoding. Its neural-mass models examined dynamics associated with theory-specific commitments.

### Limits CETB must preserve

1. **Same source cohort:** the MEG dataset originates from COGITATE and overlaps the same experimental programme. It is not an independent participant cohort or external replication.
2. **Candidate multiplicity:** 246 measures and 35,280 candidate-model combinations were examined. The winning statistic/family is therefore a model-selection result. Its selected maximum must not be treated as an unbiased estimate of future performance unless selection is repeated inside training data and evaluated on untouched outer data.
3. **Grouped CV is not the whole story:** the article reports grouped cross-validation for participant-wise classification. CETB should preserve that when reproducing the analysis, but must separately audit how feature-family and region-pair selection are nested. Cross-task transfer within the same people is not participant-held-out generalization to new people.
4. **Fold variability:** the authors report marked variability across cross-validation folds and emphasize relative performance trends rather than treating a single peak accuracy as a stable effect.
5. **Construct distinction:** stimulus-type decoding, task relevance, report/target detection and phenomenal experience remain different outcomes.
6. **Theory limits:** the article explicitly cautions that its results do not strongly decide whether theoretical mechanisms are necessary or sufficient. The models are theory-informed explanatory probes, not a definitive theory adjudication.
7. **Correction lock:** the correction dated 2026-10-15 replaced an earlier supplementary PDF, removed an extraneous supplement, corrected Eq. 1 formatting/content and corrected Table 1 and Table 2 presentation. Reproduction must use the corrected supplementary material.

## 3. Author-code implementation audit (source snapshot 21 November 2026)

The public author repository was inspected at main commit `cab648bbe2eab7093f9ab7b9a5fcddabf28e60a0` (commit date 2026-01-27). The relevant files at this snapshot are:
- `README.md`, blob `3639b61917dda7e903dc26f845478001e64c2a39`;
- `requirements.txt`, blob `5406f118265796a1a91a48f11dce2ffd81930837`;
- `classification/fit_pyspi_classifiers.py`, blob `bff12517d61fbab8b77272222074a2cb67736816`;
- `functional_connectivity_analysis/run_pyspi_for_subject_averaged_epochs.py`, blob `ef8d2b2c6d8037e39a9fd655a576933b41bf6c07`.

This is a pinned intake snapshot, not proof that it is identical to the archival Zenodo software record or the exact source state used to generate every published table. The code repository README badge links a software DOI `10.5281/zenodo.15122629`; the final paper cites preprocessed empirical/model archive `10.5281/zenodo.18294156`; the README still points to the older 2025 data archive `10.5281/zenodo.15315238`. CETB must retain these as distinct archival objects. The final paper's cited Zenodo record contains `empirical.zip` and `model.zip`; the older preprint-linked record has a different `data.zip`. Do not assume archive equivalence by title.

### Environment/version discrepancy

The final paper's methods state pyspi **1.1.0**, whereas the author repository's pinned `requirements.txt` states **pyspi==1.1.1**. The 1.1.1 release corrected directionality for time-lagged mutual-information measures (including `tlmi_gaussian`, `tlmi_kraskov_NN-4`, `tlmi_kraskov_NN-4_DCE` and `tlmi_kernel_W-0.25`), changing those outputs from symmetric to directed. This is material for directed ROI comparisons. The requirements file and paper text are both preserved as evidence; do not silently select one.

Required version-parity lanes:
1. Reproduce from the paper-stated pyspi 1.1.0 in a locked environment.
2. Re-run with repository-pinned pyspi 1.1.1.
3. Compare the four affected TLMI measures' directionality, matrices and classification results; do not pool the two versions.
4. Compare both against the archived empirical feature CSVs and the corrected paper/supplement. If the archival data match only one version, document that match and its exact digest.

The same repository requirements pin scikit-learn 1.6.1, MNE 0.23.0, MNE-Connectivity 0.7.0, NumPy 2.2.4, SciPy 1.15.2, pandas 2.2.3 and other dependencies. Its README mentions Python 3.9.17 but its conda command creates Python 3.9.21. Preserve the complete requirements file and record the Python minor/patch version, system packages and platform. The paper's analysis should use the declared matching dependency set, not an unpinned current environment.

### Exact classifier and split settings

The author script defines:
- `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=127)` for across-participant stimulus-type and task-relevance classification;
- grouping by `subject_ID`, with all rows for one participant assigned to one fold;
- a scikit-learn pipeline using `MixedSigmoidScaler(unit_variance=True)` followed by logistic regression;
- logistic regression `penalty='l1'`, `C=1`, `solver='liblinear'`, `class_weight='balanced'`, `random_state=127`;
- accuracy, balanced accuracy and AUC scoring; the paper emphasizes mean CV accuracy ± 1 SD and reports AUC as a robustness/complementary score;
- linear and RBF SVM robustness variants at `C=1`.

The paper's grouped 10-fold evaluation is a participant-level split. CETB must reproduce it exactly before adding new outer-site or nested-selection analyses. The current paper's cross-task analysis is different: train on one task-relevance context and test on the other using the same 94 participants. That is within-participant cross-context transfer, not unseen-participant validation.

### Preprocessing audit finding: fold-global imputation

In the current author script, when a statistic has between 0% and 10% missing values, missing values are mean-imputed on the full selected participant table before the grouped folds are passed to `cross_validate`. The same pattern appears in the cross-task helper before fitting on one context and testing on the other. This is a potential unsupervised transductive leakage path because held-out participants (and, in cross-task analysis, the test context) can influence the imputation mean. It is not label leakage, and the magnitude of any effect is unknown without the participant-level data.

Preserve the author path as the historical reproduction lane. Add a separate corrected sensitivity lane in which:
- missingness eligibility is predeclared;
- the imputer is fitted only on each training fold and applied to that fold's test data;
- all feature transforms and calibrators remain inside the scikit-learn pipeline;
- the cross-task context-transfer test uses imputation fit on the training context only;
- the paired change in accuracy/AUC and the affected participant/measure counts are reported.

Do not silently "fix" the historical pipeline and then claim exact reproduction. Do not claim that this issue invalidates the published findings before it has been quantified.

## 3. Data and code provenance

- COGITATE dataset DOI cited by the article: 10.17617/1.wqa3-wk71.
- Preprocessed empirical data cited by the article: Zenodo DOI 10.5281/zenodo.18294156.
- Analysis code: https://github.com/anniegbryant/MEG_functional_connectivity
- Pairwise-statistic package: pyspi v1.1.0.
- Source cohort used: 100 BIDS MEG participants.
- Source-localization subset: 94 participants with requisite anatomical scans.
- Candidate metrics: 262 evaluated, 246 retained after the described QC criterion.
- Candidate model fits: 35,280 in the primary aim.

At intake, freeze the exact repository commit, environment, pyspi version, preprocessed Zenodo archive manifest and all COGITATE participant mappings. Do not infer an unchanged code version from a mutable main branch. Confirm the corrected supplementary PDF and correction record before comparing parameters or figures.

## 5. CETB estimator-tournament design

### Phase 0 — Reference replay

Re-run the published workflow with the exact data, preprocessing, participant exclusions, region definitions, event windows, task conditions, pyspi version, classifier and grouped folds. First reproduce the original results. No HDC/CfC/LTC features are added during this phase.

Record every one of the 262 candidate statistics, including candidates rejected by the stated QC filter, with failure reason and per-participant coverage. The 246 retained metrics must be selected by the declared quality rule only—not by their accuracy.

### Phase 1 — Selection-aware nested evaluation

Use an outer participant-grouped split. All trials, sessions and rows from one participant stay in one outer fold.

Inside each outer training partition:
1. Apply candidate-statistic availability/QC checks using training participants only, unless the original fixed QC rule is explicitly part of the historical reproduction.
2. Compare the predeclared estimator library and theory-relevant region pairs in inner grouped folds.
3. Perform all model selection, scaling, feature selection, hyperparameter tuning and probability calibration inside inner folds.
4. Select one configuration under a frozen selection objective.
5. Evaluate it once on the outer test participants.
6. Preserve all outer-fold predictions and selection histories.

This estimates the performance of the *selection procedure*, not the apparent performance of the single best candidate selected using all participants.

Use repeated nested grouped CV only when fold design and sample size permit it; report fold and participant uncertainty. If the primary article's original grouped fold assignment can be recovered, reproduce it as a separate historical lane rather than silently changing the reference result.

### Phase 2 — Frozen estimator-family confirmation

If a family (for example, barycenter-derived measures) wins the nested selection procedure, freeze:
- exact statistic and implementation;
- directed/undirected convention;
- source-region pair;
- event window;
- normalization and detrending;
- classifier and hyperparameters;
- calibration;
- outer split and scoring rules.

Evaluate once on a genuinely untouched cohort, site or preregistered holdout. The COGITATE source cohort and later same-dataset papers are not independent holdouts. A different modality or experiment may be a useful transport test, but it is not a matched same-endpoint replication.

### Phase 3 — Objective-specific comparison

Do not rank measures on one composite accuracy number. Evaluate separately:
- stimulus identity decoding;
- task-relevance decoding;
- cross-task transfer within participant;
- participant-held-out generalization;
- site/device transfer where supported;
- temporal onset/offset sensitivity;
- measurement repeatability;
- motion/artifact susceptibility;
- calibration and selective prediction.

### Candidate benchmark families

At minimum:
- Pearson and Spearman correlation;
- covariance and regularized precision/inverse covariance;
- pairwise phase consistency and amplitude-envelope measures where applicable;
- barycenter-derived statistics from the pinned pyspi implementation;
- Weighted Jaccard only after an appropriate nonnegative representation is declared;
- HDC-only, CfC/LTC-only and hybrid HDC–CfC variants as candidate representations after the conventional replay.

Do not force measures with incompatible mathematical domains into one distance metric. Each representation must declare sign, scaling, stationarity, directionality and missing-value semantics.

## 6. Selection optimism and multiplicity ledger

Every run must publish:
- total candidate count;
- candidates that passed the frozen QC filter;
- candidates rejected and reasons;
- number of classifier/region/task/time-window combinations;
- all inner-fold selection winners;
- all outer-fold scores;
- selected measure frequency across outer folds;
- distribution of winning-minus-runner-up margins;
- calibration and fold variability;
- negative-control and nuisance-only results.

Never report only the single best candidate. Report the distribution across the full candidate family and the selection frequency/stability of the winning family.

The central quantity is the nested outer-fold score of the **selection procedure**. A peak selected from thousands of model combinations is descriptive, not a generalization estimate.

## 7. Baselines and metrics

Baselines:
- prevalence/chance;
- logistic regression on conventional Pearson/Spearman features;
- the original published grouped-CV classifier;
- regularized covariance/precision features;
- nuisance-only model for run, site, motion, event timing and data quality;
- shuffled-label and temporal-shift negative controls.

Metrics:
- balanced accuracy, AUROC, AUPRC, MCC, sensitivity and specificity;
- log loss and Brier score with explicitly declared convention;
- calibration slope/intercept and reliability plots;
- participant-clustered bootstrap confidence intervals;
- risk-versus-coverage and abstention for quality-aware predictions;
- compute and preprocessing cost per participant and per model.

Accuracy in the published study is not directly comparable to an unrelated cohort, metric or split. CETB claims superiority only under matched data, endpoint, split, preprocessing and compute budget, with a nested selection protocol and independent confirmation.

## 8. Immediate go/no-go

**Go:** exact BIDS MEG source archive and terms verified; N=100/94 participant flow reconciled; preprocessing and code commit pinned; corrected supplement acquired; event/region definitions frozen.

**Exploratory only:** selection is performed after viewing all participants; no untouched cohort exists; source localization or participant eligibility cannot be reproduced; nested evaluation differs materially from the historical reference.

**Blocked:** source bundle/terms are unclear, participant-phase mapping is unavailable, or critical anatomical/event files are missing.

## 9. Sources

- Bryant & Whyte (2026), *Neuroscience of Consciousness*, niag029, DOI: 10.1093/nc/niag029.
- Correction (2026-10-15), niag071, DOI: 10.1093/nc/niag071.
- COGITATE data release documentation: https://github.com/Cogitate-consortium/cogitate-data/blob/main/docs/01_intro.md
- MEG data DOI: 10.17617/1.wqa3-wk71.
- Preprocessed MEG data DOI: 10.5281/zenodo.18294156.
- Analysis repository: https://github.com/anniegbryant/MEG_functional_connectivity
