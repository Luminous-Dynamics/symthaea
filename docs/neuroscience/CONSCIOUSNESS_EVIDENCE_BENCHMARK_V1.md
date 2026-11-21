# Consciousness Evidence & Theory Benchmark (CETB) — Constitution V1

**Status:** Research protocol proposal; no new empirical result is reported here.  
**Date:** 2026-11-21  
**Scope:** External biological-data evaluation, theory-discriminating predictions, and evidence provenance for consciousness-relevant research.

## 1. Purpose

CETB is an evaluation layer connecting explicit theory predictions to independently collected observations. It is not a new consciousness theory, a clinical diagnostic instrument, or a universal consciousness classifier.

The central question is:

> Can competing theories make different, calibrated predictions about well-defined neural, behavioural, report and perturbation outcomes, and can those predictions survive subject-held-out and dataset-held-out tests?

The benchmark must preserve differences between what a participant does, what neural systems process, what the participant reports, what is later remembered, and what we infer about experience. There is no single gold-standard label that collapses all of these.

## 2. Non-equivalent outcome constructs

| Construct | Operational observation | What it does not establish |
|---|---|---|
| Behavioural responsiveness | Correct motor or verbal response to a defined command | Absence of response does not prove absence of experience |
| Task-specific neural processing | Time-locked neural response or above-chance decoding for a specified task/content | Neural processing alone does not prove subjective experience |
| Immediate subjective report | Participant's report when safely obtainable | Reportability is not identical to experience |
| Later memory/report | Report after awakening or recovery | Failure to remember is not proof that no experience occurred |
| Pharmacological / sleep state | Dose, effect-site concentration, sleep stage, or clinical state | These are context variables, not direct experience labels |
| Perturbational complexity | Response to a physical perturbation measured with a declared instrument and algorithm | A synthetic network complexity score is not clinical PCI |
| Phenomenal experience | Latent theoretical construct inferred from multiple observations | Cannot be treated as directly observed ground truth in the proposed datasets |

Every dataset adapter must declare which constructs are actually measured, which are proxies, and which remain unobserved. Unknown must remain unknown; it must not be silently recoded as unconscious.

## 3. Position relative to existing Symthaea work

CETB is deliberately an extension, not a replacement:

- The decision-relevant predictability benchmark at docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md already specifies proper scores, grouped splits, decision regret, active sensing, and full resource accounting. Reuse its contract and receipt concepts; do not duplicate its generic forecasting harness.
- symthaea-psych-bench already contains a large synthetic benchmark suite. Treat those results as architecture and implementation tests unless a benchmark is explicitly grounded in independent human or animal data.
- symthaea-futures-calibration and symthaea-futures-core already provide typed forecast distributions and proper scoring rules. Reuse them where appropriate; do not add another in-workspace scoring crate merely to recompute Brier or log scores.
- The Butlin probe qualification work already distinguishes structural wiring evidence from functional support. CETB should preserve that separation for every new indicator.
- Symthaea's independent evidence plane is a candidate integration point for frozen protocol, dataset, run and result receipts after its current schema is inspected.

A binary Brier convention must be declared. The standard single-probability binary Brier score is (p - y)^2. The existing futures-calibration BrierScore uses a multiclass sum across forecast branches; with two complementary Boolean branches its value is twice the single-probability convention. Scores are not comparable until this convention is normalized or explicitly reported.

## 4. Benchmark lanes

### Lane A — Adversarial theory prediction

The COGITATE IIT–GNWT collaboration is the reference case for preregistered theory disagreement. The Nature paper describes acquisition/cohort counts of 256 overall (fMRI 120, MEG 102, iEEG 34), but its quality-controlled main-study samples differ: fMRI 108 (35 optimization, 73 replication), MEG 97 (32 optimization, 65 replication), and iEEG 34 recruited with two excluded for incomplete data. Some iEEG endpoints have further exclusions; a specific identity-RDM analysis reports 29 participants. The current public data catalogue separately lists 118 fMRI, 100 MEG/EEG and 38 iEEG records. These are distinct denominators, not interchangeable sample sizes. Freeze endpoint-level inclusion, QC and optimization/replication membership from participant metadata and the exact supplement before reproducing any result. The study aligned with some commitments and challenged important predictions from both theories. It is a published benchmark and reproduction target, not a blind holdout.

CETB extension: serialize prediction cards, analysis windows, expected distributions, primary outcomes, failure conditions and provenance; compare theory predictions only on outcomes the theories actually committed to before analysis. Reconcile the participant-level dataset-release and paper-analysis denominators before reproducing modality counts.

### Lane B — Propofol state transitions and integration–segregation

Use OpenNeuro ds006623 v1.0.0 as the first bounded reproducibility target. It contains MRI/fMRI and behavioural data from 26 healthy volunteers under graded propofol sedation, including task/rest runs, hand-squeeze traces and infusion information.

The dataset descriptor reports a decrease in integration–segregation difference (ISD) during loss of responsiveness and no task-evoked mental-imagery responses in this 26-person release cohort. Do not conflate this cohort with the separate 2018 Scientific Reports study (n=5), which reported time-locked imagery in one participant during LOR. The descriptor provides the published processing details and points to the authors' analysis code and Zenodo archive. This target is a reproduction of published work, not a novel discovery or a direct readout of phenomenal experience.

A separate 2026 NeuroImage: Reports paper reanalysed the same dataset with Weighted Jaccard and a within-condition repeatability floor (DOI 10.1016/j.ynirp.2026.100417). Track this as a second, distinct same-dataset reproduction target under RCAF. Do not merge its WJ estimand with the ISD pipeline or describe it as independent replication.

### Lane C — Repeated-awakening EEG and experience reports

NEMAR on005620 contains 21 participants and 202 EEG recordings, including awake and propofol-sedation sessions, repeated awakenings and some TMS sessions. It is potentially valuable because awakening reports provide a different outcome channel from behavioural command following.

The NEMAR catalogue and dataset metadata display CC0, while the mirrored README states CC-BY-4.0. Preserve that discrepancy in the registry. Before any redistribution, inspect the exact v1.0.0 archive metadata and Zarr root attributes; do not relicense the source data. NEMAR documents remote Zarr slicing, so selective retrieval is preferable to downloading the full 77.3 GB archive.

### Lane D — Sleep-stage classification

Project Hypnos and Sleep-EDF belong in a sleep-stage classification lane. Sleep-stage labels are not ground truth for phenomenal consciousness. Wake/N1/N2/N3/REM classification can be a useful signal-processing task, but it must not be reported as consciousness detection without independent experience-sensitive outcomes.

### Lane E — Synthetic architecture probes

Keep digital PCI, HDC/CfC dynamics, global-workspace simulations, and synthetic ablations in a separate lane. Their role is to test whether a mechanism is implemented and to generate predictions for biological tests. Synthetic results cannot satisfy empirical or clinical evidence gates.

### Lane F — Anomalous information acquisition

This is a later, separate programme. Use preregistered, blinded, target-randomized protocols, independent target custody, fixed stopping rules, complete null-result reporting, and replication by independent laboratories. Do not pool NDE reports, childhood-memory cases, Ganzfeld studies, and psychophysical device tests into one undifferentiated anomaly score.

## 5. Prediction-card contract

Every confirmatory prediction must be registered before access to its confirmatory outcome and contain:

1. Theory name, version, source passage and prediction owner.
2. Exact construct and operational outcome.
3. Population, modality, task, state and analysis window.
4. Predictive distribution, not only a directional narrative.
5. Primary contrast and clinically or scientifically meaningful effect scale.
6. Predicted result under competing theories.
7. Failure condition and interpretation if the result is null or ambiguous.
8. Baselines, preprocessing, exclusions, multiplicity plan and stopping rule.
9. Subject/site split and whether the data are genuinely held out.
10. Dataset DOI/version, source-code revision and analysis-environment digest.

A prediction written after the data are visible is exploratory, even if its prose sounds specific.

## 6. Required baselines and metrics

Use matched baselines for the exact endpoint and dataset:

- prevalence / majority baseline;
- logistic or linear mixed-effects model where appropriate;
- regularized linear model;
- published ISD, integration-only and segregation-only measures for the fMRI lane;
- published modality-specific complexity / PCI pipeline for the TMS–EEG lane;
- established EEG or fMRI feature baselines;
- HDC-only, CfC/LTC-only and HDC–CfC variants as model candidates, not presumed winners.

For estimator/model tournaments with many candidate representations, all statistic-family selection, region-pair selection, feature selection, scaling, hyperparameter tuning and calibration must occur within inner participant-grouped training folds. The outer participant-grouped fold evaluates the entire selection procedure. Report every candidate and selection frequency; never report only the peak among thousands of combinations as an unbiased generalization estimate.

Report metrics by outcome shape:

- Binary / categorical forecasts: Brier score, log score, AUROC, AUPRC, balanced accuracy, sensitivity, specificity and Matthews correlation.
- Ordinal / continuous outcomes: appropriate likelihood, CRPS or other proper score, calibration and residual diagnostics.
- Transition timing: event-detection delay, time-to-transition error and prediction-interval coverage.
- Selective prediction: risk-versus-coverage, abstention rate and accepted-case error.
- Every metric: participant-clustered confidence intervals, subgroup/site breakdowns where powered, and missingness/exclusion accounting.

Do not make expected calibration error the only calibration metric. Include reliability diagrams, calibration slope/intercept and the declared binning convention. Do not collapse heterogeneous metrics into a single consciousness score.

### Measurement-repeatability and analysis-choice layer

For any neural-state similarity/distance claim, report a within-condition repeatability reference estimated at matched recording length and with the same estimator. Report per-condition floors where possible; a pooled floor is a separate estimand and must not be assumed universal. Also report a predeclared analysis-variant envelope across defensible preprocessing, estimator, segment duration and signal-quality choices. The between-state effect must be shown beside these uncertainty sources. Do not compare only against a perfect-match ceiling or select the best-performing pipeline after inspecting outcomes. See RCAF_BENCHMARK_CONSTITUTION_V1.md.

## 7. Split, leakage and statistical controls

- Group all epochs, trials and windows from a participant into one split.
- Use nested, participant-grouped validation for tuning and a frozen subject-held-out evaluation for confirmation.
- Where multiple sites/devices exist, add leave-site/device-out evaluation. If the cohort does not support it, explicitly say so.
- Do not split adjacent windows from the same recording across train/test.
- Fit normalization, feature selection, imputation and calibration only on training folds.
- Preserve timestamps, masks, event timing, dose, motion, recording quality and preprocessing versions.
- Use label permutation, temporal-shift, sham-event, artifact and negative-control checks.
- Treat participants, not epochs, as the primary independent sampling unit unless the statistical model explicitly handles within-participant dependence.
- Predefine primary endpoints and multiplicity correction; retain all exploratory analyses as exploratory.
- Report every failed, excluded, unsupported or timed-out run.

## 8. Comparison to the state of the art

CETB should not claim that it is a better consciousness detector than existing research groups. The credible competitive goal is a stronger, interoperable evidence and theory-evaluation layer.

- COGITATE is the adversarial-theory and multimodal preregistration reference.
- Published ISD work is the main fMRI network-state comparator; its paper reports an AUC of 0.77 for awake versus loss-of-responsiveness discrimination in a specified dataset/protocol. Replicate the exact task and split before comparing.
- Bryant & Whyte (2026), DOI 10.1093/nc/niag029, is a same-COGITATE-cohort MEG comparator: 246 pyspi statistics retained from 262 candidates, 35,280 model configurations, barycenter-derived measures among the strongest stimulus decoders. Its corrected supplement (niag071, 2026-10-15) is mandatory for reproduction. CETB adds a nested participant-grouped estimator tournament to qualify the measure-selection procedure; this secondary analysis is not independent replication.
- TMS–EEG PCI is a physical perturbation-based clinical research method. A 2026 clinical-translation review places it in an advanced translational phase while still identifying the need for larger multicentre validation and practical protocol qualification. It is not interchangeable with digital PCI.
- OpenNeuro/BIDS, NEMAR/DataLad, fMRIPrep/XCP-D, MNE, EEGLAB and FieldTrip already solve important data and signal-processing problems. CETB should interoperate with them rather than rebuild them.

A better-than-SOTA claim requires the exact same dataset version, cohort, endpoint, preprocessing, split, tuning budget and metric as the comparator. The candidate must beat the best eligible baseline on a frozen held-out cohort, with uncertainty intervals, calibration, negative controls and domain-shift results. If those conditions do not match, report a non-comparable result rather than a leaderboard rank.

## 9. Programme targets

These are proposed gates, not achieved results:

| Dimension | Minimum gate |
|---|---|
| Preregistration | 100% of confirmatory primary predictions frozen before outcome access |
| Provenance | 100% of confirmatory runs carry dataset, code, config, environment and result digests |
| Split integrity | Zero participant-level leakage in automated checks |
| Reproduction | At least one published, fully specified analysis reproduced within a preregistered tolerance |
| Generalization | At least one independent dataset or site held out, where suitable data exist |
| Calibration | Report proper scores and calibration by participant/site/state; no unqualified aggregate-only claim |
| Negative controls | Prespecified label/timing/artifact controls run and reported |
| Replication | Any new anomalous effect requires at least three independent laboratories before strong claims |
| Clinical translation | No diagnostic or clinical-performance claim without prospective, multicentre validation and appropriate regulatory/ethics review |

## 10. Go / no-go rules

- **Go to confirmatory evaluation:** exact dataset version and terms verified; outcomes and splits defined; preprocessing and analysis frozen; protocol validated; statistical plan reviewed.
- **Exploratory only:** public outcomes have been repeatedly inspected, sample size is too small, split provenance is incomplete, or the prediction was formulated after outcome access.
- **Blocked:** unclear licence, unavailable participant-level phase labels, missing critical metadata, incomplete ethics authority, or non-reproducible input files.
- **No scientific claim:** a synthetic test passes, a model self-reports a state, one indicator is present, or a single biomarker crosses a threshold.

## 11. Immediate implementation order

1. Correct the Project Hypnos evidence language and label it as a two-subject sleep-stage prototype.
2. Qualify digital PCI examples as synthetic perturbation-propagation tests, not clinical PCI.
3. Reproduce the published ds006623 ISD analysis from the pinned dataset and authors' archived code.
4. Independently reproduce the same-dataset WJ-within-state-floor paper, then run the separately preregistered RCAF guard-gap and repeatability-surface extensions.
5. Compare both network measures against their own matched baselines before attempting HDC/CfC/LTC additions.
6. Add subject-grouped prediction tasks only after the reproduction baselines are stable.
7. Add on005620 selectively after exact version metadata, report alignment and licence attribution are verified.
8. Build an external data adapter and evidence receipt only after the benchmark contract is stable; do not add a new root workspace member for the first documentation increment.

## References

- Melloni et al. (2025), COGITATE adversarial test, Nature, DOI: 10.1038/s41586-025-08888-1.
- Huang et al. (2026), ds006623 data descriptor, Scientific Data, DOI: 10.1038/s41597-025-06442-2.
- Huang et al. (2024), ISD and consciousness/anesthesia/sleep, Nature Communications, DOI: 10.1038/s41467-024-53299-x.
- Huang et al. (2018), imagery and behavioural unresponsiveness under propofol, Scientific Reports, DOI: 10.1038/s41598-018-31436-z.
- Bajwa et al. (2026), on005620 repeated-awakening EEG dataset, NEMAR DOI: 10.82901/nemar.on005620.
- Ziemann et al. (2026), clinical utility and translation of TMS–EEG, Clinical Neurophysiology, DOI: 10.1016/j.clinph.2025.2111487.
- Casali et al. (2013), PCI, Science Translational Medicine, DOI: 10.1126/scitranslmed.3006294.


## 12. COGITATE denominator and release correction (2026-11-21)

The machine-readable registry now distinguishes (a) acquisition/cohort counts in the Nature paper, (b) quality-controlled primary-analysis counts, (c) optimization versus replication phase counts, and (d) current public-release counts. Never quote one of these as another. For endpoint-specific iEEG analyses, retain the actual participant flow rather than reusing a modality-wide N.

The current COGITATE release documentation states CC BY 4.0 and provides raw/BIDS bundles, but account registration and agreement to the data terms are required to retrieve them. Freeze the exact bundle DOI/version and terms at intake. A metadata page or public DOI does not prove that the files have been downloaded, checksummed or analyzed.


## 13. Selection-aware estimator evaluation

The 2026 COGITATE MEG comparison of 246 pairwise statistics is a reason to treat estimator choice as part of the learning procedure, not as a preprocessing constant. The full candidate set, all exclusions, all region/condition/window combinations and all model fits must be logged. The inner grouped folds select the statistic family and model; outer grouped folds evaluate the selection policy on participants not used for that selection. A separate untouched dataset or cohort is required for independent confirmation.

A high peak among many candidate models is an exploratory observation until this nested and independent evaluation is complete. Cross-task transfer on the same participants is useful but is not the same as participant-held-out transfer. The corrected supplement and article correction must be pinned in any replay.
