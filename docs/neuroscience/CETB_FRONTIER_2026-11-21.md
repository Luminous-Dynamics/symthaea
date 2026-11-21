# CETB frontier update — 21 November 2026

**Status:** Research and benchmark-design update. No COGITATE or ds006623 signals were downloaded or analysed in this increment.

## Executive change

The opportunity is to make CETB a measurement-assurance and theory-prediction benchmark, rather than another generic consciousness classifier. The benchmark should assess whether an inference survives repeatability limits, data-quality variation, estimator choice, participant/site shift, and independent replay.

## 1. New public benchmark assets

### COGITATE Experiment 1 is now the best near-term external theory-testing target

The 2025 Nature adversarial study preregistered competing IIT/GNWT predictions and analysed multiple modalities. The acquisition/cohort counts in the article are fMRI 120, MEG 102 and iEEG 34. The quality-controlled main-study samples are not those same counts:

- fMRI: 108 in the reported analysis, split into 35 optimization and 73 replication participants.
- MEG: 97, split into 32 optimization and 65 replication participants.
- iEEG: 34 recruited; two participants were excluded for incomplete data. Specific endpoints can have further denominators (for example, one identity-RDM analysis reports 29 after excluding incomplete experiments).

The current public COGITATE data catalogue lists 118 fMRI, 100 MEG/EEG and 38 iEEG records. These are release counts, not the same thing as the paper's quality-controlled analysis samples. All three denominators must be retained in the data registry; the participant-level release manifest and the supplement must determine each endpoint's actual eligible set.

The fMRI release spans Yale and Donders; the MEG/EEG release spans Birmingham and Peking University. The fMRI descriptor reports 118 released participants and includes raw and BIDS data, quality reports, demographics, behavioural records and eye tracking. The MEG/EEG descriptor reports 100 released participants and offers a sample bundle and full bundle. The iEEG dataset is a distinct clinical epilepsy cohort; never present its participants as healthy controls.

The official release states CC BY 4.0, but portal download requires an account and acceptance of the terms. Record the exact bundle DOI, version, terms, file manifest and checksums before processing or redistributing derivatives. Public availability does not mean files have been retrieved or analysed.

### Dataset priority

1. **COGITATE fMRI / MEG-EEG / iEEG:** best for preregistered visual-content, duration, task-relevance and theory-disagreement tests. Not a direct binary subjective-experience label; the primary task measures stimulus processing and target detection.
2. **OpenNeuro ds006623:** best for the narrow propofol state-transition and repeatability reproduction. Its 2026 WJ analysis is a second analysis of the same cohort, not external replication.
3. **NEMAR on005620:** repeated awakening/report-aligned EEG and perturbation lane, after exact version and licence reconciliation.
4. **Sleep-EDF / Hypnos:** sleep-stage signal classification only, with an explicit ceiling against consciousness claims.

## 2. New external SOTA references CETB must incorporate

### Network Similarity Index (NSI), 2026

A 2026 Neuron paper introduces a Network Similarity Index for assessing the suitability of precision fMRI data for individual-level functional-connectivity mapping. The index fits seed-to-whole-brain connectivity maps as weighted combinations of canonical network templates using ridge regression, with model fit (R²) as the per-seed quality measure. The work links data quality to reliability growth and expected value of additional scan time.

CETB implication: add NSI as an external comparator and quality covariate where its assumptions fit. Do not impose it as a universal gate: it was developed for individual-level precision mapping, not automatically for every task-fMRI, sedation, EEG or clinical dataset. Its original code, parcellation and network templates must be pinned and licensed before adoption.

### 239-statistic functional-connectivity benchmark

A 2025 Nature Methods study compared 239 pairwise statistics for functional-connectivity network construction against multiple objectives: distance/weight relations, structure-function coupling, biological network correspondence, individual fingerprinting, brain-behaviour prediction and motion susceptibility. It found large method-dependent changes and no single statistic that dominates every objective; inverse-covariance methods performed well on several composite criteria.

CETB implication: Weighted Jaccard is one similarity estimator, not the answer. The RCAF robustness battery should compare WJ with Pearson/Spearman correlation, covariance/precision-derived representations and predeclared graph/network summaries, using objective-matched metrics. Do not optimise the estimator on the final test set.

### Existing test-retest reliability evidence

A widely cited meta-analysis found mean edge-level functional-connectivity ICC around 0.29, illustrating why repeatability must be measured rather than assumed. A large two-day sample (N=833) found that only a small portion of the edge-wise connectome had good-to-excellent reliability and that atlas, GSR and thresholding choices mattered. Reliability must be considered alongside validity: stable artifact can be reliable but uninformative.

## 3. RCAF v1.1 additions

The kernel now adds:

- An exact participant-level sign test for paired within-minus-between gaps. Ties are excluded from the binomial denominator. This is a distribution-free complement to the paired effect distribution and participant bootstrap; it is not a randomization test or causal test.
- Edge-sign transition accounting across aligned matrices. This separately counts polarity reversals, transitions to/from a neutral zone, stable positive/negative/neutral edges, and corresponding fractions.
- A neutral-band parameter that must be selected before looking at the effect and evaluated over a predeclared sensitivity grid.

This addresses a limitation of sign-filtered WJ: when only edges with the same sign in both matrices are selected, actual sign reversals are excluded from that WJ denominator. The WJ result must therefore be accompanied by direct sign-transition metrics. Also report edge counts and the ROI-order digest; equal matrix dimensions do not prove parcel alignment.

### Required RCAF output bundle

- Input dataset DOI/version and immutable file-manifest digest.
- Participant-level inclusion status, exclusion reason, modality/site, run/session and phase.
- Frozen ROI label list/order digest, parcellation, preprocessing and estimator.
- Within- and between-state values at participant level, with repeated segment placement nested within participants.
- Within-state repeatability by condition, duration and temporal gap.
- Absolute, shared-positive, shared-negative and signed-shift WJ variants.
- Direct polarity transitions over a predeclared neutral-threshold grid.
- Participant-paired gap distribution, exact sign-test summary, bootstrap interval and leave-one-participant-out sensitivity.
- Full QC covariates, failed/unsupported runs and negative controls.
- Reproducibility receipt: code revision, configuration digest, environment digest and output digest.

## 4. Paradigm shifts and concrete research targets

| From | To |
|---|---|
| One binary consciousness label | Separate behaviour, stimulus processing, report, memory, physiology and latent-theory constructs |
| Accuracy-only leaderboards | Proper predictive scores, calibration, uncertainty, abstention and resource cost |
| One preferred connectivity estimator | Objective-matched estimator tournament with a frozen multiverse |
| Perfect-match similarity ceiling | Same-condition repeatability surfaces at matched data quantity |
| Group averages only | Participant-level inference with nested session/run/segment uncertainty |
| Random train/test splits | Frozen subject, site/device and dataset holdouts |
| Post-hoc theory narratives | Preregistered theory prediction cards and adversarial scoring |
| Forced prediction | Qualified abstention when quality, domain or evidence is inadequate |
| Single-modality claims | Cross-modal convergence, while keeping distinct cohorts and outcome semantics separate |
| Hand-authored conclusions | Independently replayable, evidence-carrying analysis receipts |

### Benchmark scorecard

Do not collapse these dimensions into one consciousness score.

| Lane | Primary measures | Proposed gate |
|---|---|---|
| Measurement reliability | within-state similarity, ICC, fingerprinting/identifiability, reliability-vs-duration | Report per-condition distributions and uncertainty; compare matched-duration baselines |
| State/content discrimination | AUROC, AUPRC, balanced accuracy, MCC, sensitivity/specificity | Subject-held-out and site-held-out results with confidence intervals |
| Probabilistic theory prediction | log score, Brier/CRPS under explicit convention, calibration slope/intercept | Score frozen prediction cards on untouched outcomes |
| Temporal transitions | detection delay, lead time, hazard log score, interval coverage, false alarms/hour | No future-data leakage; compare against reactive and prevalence baselines |
| Robustness | effect-direction agreement, effect-size envelope, sign transitions, multiverse stability | Target ≥90% directional agreement across the *predeclared defensible* variants; report failures |
| External validity | leave-site/device-out, independent cohort and modality transfer | At least one independent dataset before calling a result generalizable |
| Safety/uncertainty | risk-coverage, abstention rate, selective error, subgroup calibration | Unknown/inadequate quality remains unknown; no forced negative label |
| Reproducibility | exact input/code/config/environment/output digests, clean replay | 100% of confirmatory results have complete receipts and zero split leakage |

The proposed thresholds are engineering targets, not achieved results. A result should be called better than a published baseline only when data release, cohort, endpoint, preprocessing, split, tuning budget and scoring convention match. A different dataset or outcome is a separate experiment, not a leaderboard comparison.

## 5. Immediate next execution ladder

1. Freeze COGITATE EXP1 exact release bundle, terms, file manifest and participant-level phase mapping.
2. Rebuild the published analysis participant flow from paper/supplement plus release metadata. Resolve any participant that appears in the release but not the published sample.
3. Select the fMRI replication phase as the first theory-prediction test; the optimization phase is development-only. Do not use a random split as a substitute for the original phase mapping.
4. Reproduce one published primary endpoint exactly before adding Symthaea features.
5. In parallel, reproduce the ds006623 ISD and WJ analyses independently; only then evaluate RCAF extensions.
6. Compare multiple estimators against objective-specific published baselines; include NSI only where its precision-mapping assumptions are valid.
7. Add cross-modal prediction as a separate transfer task, preserving the different participant cohorts and outcome semantics.
8. Only after exact replay should HDC/CfC/LTC be added as candidate predictors, with matched compute budgets and frozen test data.

## 6. Claim limits and execution state

- This update adds research protocol and kernel source only.
- No fMRI, EEG, MEG or iEEG signal data were downloaded or analysed.
- No published COGITATE or ds006623 results were reproduced.
- No SOTA superiority, consciousness detection or clinical readiness is claimed.
- Rust compilation, tests and Clippy remain pending until a supported Rust toolchain executes the standalone crate.
