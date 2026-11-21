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

## 4. CETB estimator-tournament design

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

## 5. Selection optimism and multiplicity ledger

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

## 6. Baselines and metrics

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

## 7. Immediate go/no-go

**Go:** exact BIDS MEG source archive and terms verified; N=100/94 participant flow reconciled; preprocessing and code commit pinned; corrected supplement acquired; event/region definitions frozen.

**Exploratory only:** selection is performed after viewing all participants; no untouched cohort exists; source localization or participant eligibility cannot be reproduced; nested evaluation differs materially from the historical reference.

**Blocked:** source bundle/terms are unclear, participant-phase mapping is unavailable, or critical anatomical/event files are missing.

## 8. Sources

- Bryant & Whyte (2026), *Neuroscience of Consciousness*, niag029, DOI: 10.1093/nc/niag029.
- Correction (2026-10-15), niag071, DOI: 10.1093/nc/niag071.
- COGITATE data release documentation: https://github.com/Cogitate-consortium/cogitate-data/blob/main/docs/01_intro.md
- MEG data DOI: 10.17617/1.wqa3-wk71.
- Preprocessed MEG data DOI: 10.5281/zenodo.18294156.
- Analysis repository: https://github.com/anniegbryant/MEG_functional_connectivity
