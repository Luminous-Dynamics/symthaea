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

The current public COGITATE data catalogue lists cohort counts of 118 fMRI, 100 MEG and 38 iEEG. These are catalogue cohort counts, not a guarantee that every modality's complete raw/BIDS files are currently available, and not the same thing as the paper's quality-controlled analysis samples. All denominators must be retained in the data registry.

The versioned data-release introduction (V1.3) explicitly says raw/unprocessed and BIDS fMRI for Experiment 1 are released. For M-EEG, V1.3 lists raw/unprocessed batch 1 and batch 2 among future releases, but earlier V1.0 and V1.2 M-EEG release documents exist. Bryant and Whyte (2026) confirm that BIDS MEG data were used for 100 participants, with 94 having the anatomical scans needed for source localization. Therefore BIDS MEG availability and raw/unprocessed MEG availability must be recorded separately: BIDS release is available through earlier versioned release/data access, while V1.3 still lists the raw files as future. The iEEG release documentation is V1.1. Experiment 2 fMRI and M-EEG formats are listed as future releases. fMRI participants were recruited at Yale and Donders; M-EEG cohort counts span Birmingham and Peking University. The iEEG dataset is a distinct clinical epilepsy cohort; never present its participants as healthy controls.

The release states CC BY 4.0, but portal download requires an account and acceptance of the terms. Record the exact bundle DOI, version, modality, format, terms, file manifest and checksums before processing or redistributing derivatives. Public catalogue presence does not prove a specific data bundle has been retrieved, checksummed or analysed.

### Dataset priority

1. **COGITATE fMRI / MEG-EEG / iEEG:** best for preregistered visual-content, duration, task-relevance and theory-disagreement tests. Not a direct binary subjective-experience label; the primary task measures stimulus processing and target detection.
2. **OpenNeuro ds006623:** best for the narrow propofol state-transition and repeatability reproduction. Its 2026 WJ analysis is a second analysis of the same cohort, not external replication.
3. **NEMAR on005620:** repeated awakening/report-aligned EEG and perturbation lane, after exact version and licence reconciliation.
4. **Sleep-EDF / Hypnos:** sleep-stage signal classification only, with an explicit ceiling against consciousness claims.

## 2. New external SOTA references CETB must incorporate

### Network Similarity Index (NSI), 2026

A 2026 Neuron paper (Lynch et al., DOI 10.1016/j.neuron.2026.05.020; issue date 11 November 2026) introduces the Network Similarity Index for assessing precision-fMRI data suitability for individual-level functional-connectivity mapping. It fits each seed-to-whole-brain connectivity map as a weighted combination of canonical network templates using ridge regression; per-seed R² is the NSI, and the dataset-level NSI is the median across sampled seeds. The authors provide code in https://github.com/cjl2007/pfm-nsi and processing workflows in https://github.com/cjl2007/pfm-mefmri, with related data/statistics at https://osf.io/qx2et/.

CETB implication: add NSI as an external comparator and possible quality covariate where its assumptions fit. Do not impose it as a universal gate: it was developed for large-scale network structure and individual-level FC data quality, not automatically for every task-fMRI, sedation, EEG or clinical dataset. Importantly, it measures recoverable network-template structure and has limitations around atypical network topography; pair it with motion/artifact controls and independent repeatability rather than interpreting a high NSI as proof of biological validity or consciousness.

### COGITATE MEG 246-measure estimator tournament (Bryant & Whyte, 2026)

A highly relevant 2026 paper compared 262 pyspi v1.1.0 measures on COGITATE BIDS MEG data from 100 participants; 94 had anatomical MRI for source localization. After a stated nonzero-variance / real-value availability filter requiring at least 90% coverage (N≥85), 246 measures remained. The primary analysis examined 35,280 model configurations. Barycenter-derived measures were among the strongest stimulus-type decoders, with many configurations in the 60–67% accuracy range; no candidate measure exceeded 56% for the study's domain-general task-relevance classification. The authors report substantial fold variability and treat peak-accuracy trends comparatively rather than as stable absolute scores. Cross-task transfer in this paper tests training on one task-relevance condition and evaluating on the other within the same 94 participants; it is not by itself unseen-participant generalization.

A public author-code audit additionally fixes the source snapshot at main commit `cab648bbe2eab7093f9ab7b9a5fcddabf28e60a0` and captures the README, requirements, classifier and feature-extraction file blob digests. The author classifier uses 10-fold `StratifiedGroupKFold` with `subject_ID` groups, shuffle enabled and random state 127; the pipeline is `MixedSigmoidScaler(unit_variance=True)` plus L1 logistic regression (C=1, liblinear, balanced class weights). It reports fold accuracy, balanced accuracy and AUC, with mean accuracy ± SD emphasized in the paper. Linear and RBF SVM variants at C=1 are listed as robustness checks.

There are two source-version mismatches to resolve at intake:
- The paper methods specify pyspi 1.1.0, while the author's pinned requirements specify 1.1.1. The 1.1.1 release corrects directionality of time-lagged mutual-information measures. Reproduce both and compare to archived outputs; directed TLMI is affected.
- The final paper cites preprocessed empirical/model archive DOI 10.5281/zenodo.18294156 (empirical.zip and model.zip), while the README still links older preprint support data DOI 10.5281/zenodo.15315238 (data.zip). Keep both file manifests distinct.

The classifier code also mean-imputes under-10%-missing SPI values before applying grouped folds. This is a potential unsupervised transductive preprocessing leak, not label leakage; the magnitude is unknown without participant files. CETB will preserve the historical author script as the exact-reproduction lane, then compare a separately labelled training-fold-only imputation sensitivity. It is not justified to claim that this issue invalidates the published results without measuring the actual missingness and score difference.

This is now CETB's closest direct MEG comparator, but it reuses the COGITATE source cohort. It is not independent replication. Because 246 measures and 35,280 configurations were screened, CETB must evaluate the *selection procedure* in nested participant-grouped validation: statistic family, region pair, event window, model, scaling, tuning and calibration must all be selected inside the inner training folds; outer participant folds evaluate the whole selection procedure. A winning measure chosen across all participants and scored on those same folds is hypothesis-generating, not an independent performance estimate.

The article's correction (niag071, 2026-10-15) replaced an earlier supplementary PDF and corrected Eq. 1 and table presentation. CETB locks the corrected supplement and correction record for reproduction. Both theory-informed neural-mass models were reported to approximate aspects of the observed connectivity patterns; the authors caution that these analyses do not strongly establish that either theory's proposed mechanisms are necessary or sufficient for perception.

The new audit and machine-readable tournament contract are:
- `COGITATE_MEG_ESTIMATOR_TOURNAMENT_V0.md`
- `COGITATE_MEG_ESTIMATOR_TOURNAMENT_V0.json`

### 239-statistic functional-connectivity benchmark

A 2025 Nature Methods study compared 239 pairwise statistics for functional-connectivity network construction against multiple objectives: distance/weight relations, structure-function coupling, biological network correspondence, individual fingerprinting, brain-behaviour prediction and motion susceptibility. It found large method-dependent changes and no single statistic that dominates every objective; inverse-covariance methods performed well on several composite criteria.

CETB implication: Weighted Jaccard is one similarity estimator, not the answer. The RCAF robustness battery should compare WJ with Pearson/Spearman correlation, covariance/precision-derived representations and predeclared graph/network summaries, using objective-matched metrics. Do not optimise the estimator on the final test set.

### Existing test-retest reliability evidence

A widely cited meta-analysis found mean edge-level functional-connectivity ICC around 0.29, illustrating why repeatability must be measured rather than assumed. A large two-day sample (N=833) found that only a small portion of the edge-wise connectome had good-to-excellent reliability and that atlas, GSR and thresholding choices mattered. Reliability must be considered alongside validity: stable artifact can be reliable but uninformative.

## 3. RCAF v1.2 additions

The kernel now adds:

- An exact participant-level sign test for paired within-minus-between gaps. Ties are excluded from the binomial denominator. This is a distribution-free complement to the paired effect distribution and participant bootstrap; it is not a randomization test or causal test.
- Edge-sign transition accounting across aligned matrices. This separately counts polarity reversals, transitions to/from a neutral zone, stable positive/negative/neutral edges, and corresponding fractions.
- A neutral-band parameter that must be selected before looking at the effect and evaluated over a predeclared sensitivity grid.
- Participant-keyed summary inputs reject empty or duplicate participant identifiers; label-aware matrix APIs reject duplicate, empty or misordered ROI labels even when matrix dimensions agree.

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


## 7. Measurement theory extension: repeatability is not validity

RCAF v1.1 is being hardened around three separate questions:

1. **Reliability:** would repeated acquisitions under the same declared condition yield similar estimates?
2. **Validity / construct sensitivity:** does the representation track the operational construct or a preregistered discriminating prediction, rather than stable artifact?
3. **Identifiability / transport:** can the measurement distinguish individuals or states on held-out participants/sites without leaking session identity or acquisition signatures?

A reliable artifact can score highly on repeatability. A network-quality index can be high while a task-specific hypothesis is still wrong. Consequently, do not collapse NSI, ICC, RCAF, motion robustness and theory-prediction scores into a single scalar.

### Proposed hierarchical repeatability model

Represent each measurement as Y(participant, condition, session, run, segment, estimator). Condition and estimator are declared analysis factors; participant, session-within-participant, run-within-session and segment-within-run induce nested dependence. Estimate participant-, session-, run- and segment-level variance components with a hierarchical model or generalizability-theory design. Report which components are estimable from the actual sampling design; do not manufacture precision from repeated windows.

Use separate relative-reliability and absolute-agreement quantities where applicable (for example, a predeclared ICC form with confidence intervals, within-participant standard error and minimum detectable change). Pair these with criterion validity and artifact negative controls. An ICC alone is not evidence that a biomarker is meaningful.

### Repeatability surface

Estimate R(duration, temporal_gap, quality) rather than one pooled floor. Duration and temporal gap must be matched when comparing conditions. Quality covariates should include at least motion/censoring and acquisition/site metadata where available. Fit the surface only within observed duration/quality support; do not extrapolate beyond data without an explicit predictive interval and validation.

### Evidence-aware acquisition

Choose an additional scan segment or modality by expected reduction in uncertainty about a specified decision or theory contrast per unit of acquisition cost. Report the acquisition policy, stopping rule, minutes/energy/device cost, achieved calibration and whether the acquired data remain temporally independent of the evaluation set. More data is not automatically more evidence when the same correlated run is repeatedly subdivided.

### Machine-checkable integrity invariants

A report validator must check, in addition to JSON Schema:
- unique, non-empty participant keys;
- participant row count equals summary.participant_count;
- each paired_gap equals within_similarity minus between_similarity within a declared numeric tolerance;
- positive-gap fraction and sign-test positive/negative/tie counts recompute from participant rows;
- bootstrap interval ordering and confidence level are valid;
- acquisition ≥ released ≥ QC-eligible ≥ analysis-included, with exclusion flow fully explained;
- ROI labels/order digest is the same for every matrix in a comparison;
- each run uses frozen participant-level membership, phase and modality assignments;
- no same-participant session/segment crosses train, calibration and test partitions.

JSON Schema validates structure; it does not prove these cross-record and derived-value invariants by itself.

The preferred empirical kernel entry points implement participant-key and ROI-order checks before comparison. The report validator remains responsible for cross-row arithmetic and source-to-output provenance.

The committed `research/rcaf/tools/validate_report.py` checks the Draft 2020-12 report schema when supplied and recomputes semantic invariants (participant count, paired-gap arithmetic, medians, positive-gap fraction, exact sign-test counts/p-value, dataset flow and claim ceilings). It is an isolated research helper, not a production dependency. Nine Python unit tests were run locally and passed; this is separate from the Rust kernel tests, which remain uncompiled here.


## 8. Power and cohort design

The new exact sign-test power note (`RCAF_SIGN_TEST_POWER_V0.md`) quantifies how weak small cohorts can be for moderate participant-level consistency. Under a two-sided alpha of 0.05 and no ties, N=23 yields only about 44% power if 70% of participant gaps are positive; approximately 49 non-tied participants are required for 80% power under that alternative. For q=0.65, the approximate requirement is 90; for q=0.60, about 199.

These are exact-binomial planning calculations for a sign test only. They do not power the median-gap effect estimate or account for hierarchical variance, missingness, site structure or effect magnitude. Use design-specific simulation before new acquisition; do not use repeated segments to inflate participant N.


## 9. Intake source lock updates

The MEG tournament JSON and dataset registry now record the author repository commit/blob IDs, the two pyspi versions, the two distinct Zenodo data archives, the grouped 10-fold seed, model/scaler settings, and the pre-fold mean-imputation finding. These are intake audits only. No preprocessed data file, BIDS bundle or raw MEG file has been downloaded or analysed by this branch.
