# COGITATE Experiment 1 fMRI — CETB replay protocol V0

**Status:** Intake and reproduction protocol. Not a new preregistration, not a blind holdout, and not an empirical result.  
**Date:** 2026-11-21  
**Target:** COGITATE Experiment 1 fMRI, with the published replication phase treated as a reproduction cohort.  
**Primary paper:** Cogitate Consortium et al. (2025), Nature 642, 133–142, DOI 10.1038/s41586-025-08888-1.  
**Current release documentation:** COGITATE Data Release V1.3.

## 1. Why this is the first external CETB task

COGITATE is a rare example of competing consciousness theories making preregistered, discriminating predictions within a coordinated multimodal experiment. The fMRI dataset provides a practical first reproduction route because the V1.3 release introduction explicitly confirms both raw/unprocessed and BIDS formats for Experiment 1 fMRI.

The published study separates an optimization phase from a later replication phase. Its fMRI optimization cohort is N=35 and its replication cohort is N=73, after earlier exclusions from the acquisition cohort. The full paper cohort and public catalogue counts are not the denominator for this primary replication analysis. Rebuild the participant flow from the paper, supplement and release metadata before any computation.

This is a reproduction of published results, not a new blind benchmark. The original replication phase has already been analysed and reported publicly. CETB can test replay fidelity, evidence provenance and additional predeclared sensitivity analyses, but cannot market it as untouched independent evaluation.

## 2. Construct boundaries

The experiment uses suprathreshold visual stimuli in four categories—faces, objects, letters and false fonts—shown at three orientations and three durations (0.5, 1.0 and 1.5 seconds). Participants perform target detection in blocks; some stimulus categories are task-relevant and others task-irrelevant.

These distinctions must remain explicit:

- **Stimulus category information:** whether neural data distinguish faces from objects or letters from false fonts.
- **Task relevance:** whether the stimulus belongs to a category relevant to the current block's target-detection task.
- **Target detection / behavioural report:** whether the participant responds to the rare target identity.
- **Stimulus orientation and duration:** controlled stimulus attributes.
- **Subjective experience:** inferred only through the experiment's validated design and theory-specific assumptions; not directly observed as a universal binary label by every trial.

The paper reports that participants were conscious of stimuli in both task-relevant and task-irrelevant conditions in a separate experiment with a surprise memory test. Therefore:

> Task-irrelevant is not an unconscious label. Correct decoding of task-irrelevant content is evidence about stimulus processing across attentional/task contexts, not a direct consciousness detector.

## 3. Frozen population and phase contract

The reproduction must distinguish these denominators:

| Denominator | fMRI N | Meaning |
|---|---:|---|
| Acquisition / paper cohort | 120 | Participants recruited/collected for the paper's fMRI cohort |
| Quality-controlled main analysis | 108 | Eligible after published quality/exclusion decisions |
| Optimization phase | 35 | Pipeline quality and analysis optimization |
| Replication phase | 73 | Novel dataset used for the published main replication results |
| Current public catalogue | 118 | fMRI records listed by the current COGITATE data catalogue |

The 118 catalogue count must not replace the 73-person published replication denominator. The 35 optimization participants are not to be merged into the replication phase simply to increase sample size.

At intake, create a pseudonymous participant-flow manifest containing:
- catalogue participant key;
- paper inclusion / exclusion status and reason;
- site;
- optimization or replication phase;
- BIDS session/run identifiers;
- exact source file path and SHA-256;
- missing modality/task/run flags.

Never place direct identifiers in the public repository. The participant flow must reconcile to the exact release and published supplement. If it cannot, stop and mark the affected endpoint blocked.

## 4. Primary reproduction endpoints

Run the published analysis before any Symthaea model or new feature is introduced.

### Endpoint A — Cross-task category decoding

Reproduce the paper's fMRI searchlight decoding of stimulus category (faces versus objects) trained on one task-relevance condition and tested on the other, in both directions:

1. Train on task-relevant trials; test on task-irrelevant trials.
2. Train on task-irrelevant trials; test on task-relevant trials.

The published fMRI replication analysis uses leave-one-run-out cross-validation within participant for this task, followed by the published group-level inference. Reproduce the original feature selection, normalization, searchlight, trial inclusion, run handling, and inference exactly before changing any step.

### Endpoint B — Category generalization across content families

Where supported by the exact source task structure, reproduce the published letter-versus-false-font analysis separately from faces-versus-objects. Do not merge category families or trial types if the original analysis kept them separate.

### Endpoint C — Orientation decoding

Reproduce the three-class orientation classification (front, left, right) under the published trial aggregation and evaluation convention. The chance reference is 33%; use balanced accuracy for the imbalanced class counts, as in the paper.

### Endpoint D — Behavioural target detection

Analyse target detection and response timing separately from neural decoding. It is an access/report/behavioural outcome and must not be used to overwrite the task-irrelevant neural-processing label.

## 5. Comparison baselines

All comparisons use identical participants, trials, run splits, preprocessing and compute budgets.

- Chance / class-prevalence baseline.
- Original published analysis implementation.
- Simple regularized linear classifier on the same input features.
- A conventional SVM / searchlight baseline matching the paper's primary model family.
- A nuisance-only model using motion, run, site, timing and acquisition metadata. If nuisance-only prediction is high, investigate leakage before interpreting neural decoding.
- Symthaea/HDC, CfC/LTC and hybrid models only after the published baseline is reproduced; no architecture is presumed superior.

For binary classification report balanced accuracy, AUROC, AUPRC, sensitivity, specificity, MCC, Brier score and log score where calibrated probabilities are available. For orientation report balanced accuracy and class-wise errors. Report participant-level uncertainty, not trial-count-only confidence intervals.

## 6. Nested evaluation and leakage controls

The primary historical reproduction should preserve the paper's run-wise cross-validation exactly. CETB must add distinct generalization analyses without replacing the primary reproduction:

- **Within-participant run generalization:** leave-one-run-out, following the paper.
- **Participant generalization:** subject-held-out evaluation, with all trials/runs from a participant in one fold.
- **Site transfer:** train on one site and test on the other, then reverse. Report this as exploratory if the sample size or acquisition differences make inference unstable.
- **Phase discipline:** use optimization participants only to reproduce the declared pipeline-development role; use the published replication cohort for the historical confirmatory reproduction. Do not retune on the replication outcomes.
- **Preprocessing discipline:** fit scalers, feature selection and calibration inside the training fold. Preserve all censoring, nuisance regression and filtering settings from the pinned original pipeline.
- **Leakage controls:** permute labels within valid blocks; test a nuisance-only model; assess run/site decodability; verify stimulus timing and event alignment; test whether shuffled or temporally displaced event labels destroy the expected effect.
- **Multiplicity:** freeze primary outcomes and ROI/time-window families before the reproduction run. Report exploratory additions separately.

Leave-site-out results are not interchangeable with within-subject leave-run-out results. They answer different generalization questions.

## 7. RCAF and measurement-quality extensions

RCAF is an extension lane, not a replacement for the original COGITATE endpoints.

For any connectivity or representational-similarity comparison:
- match segment length, run and task context;
- estimate within-condition repeatability separately by task condition;
- preserve participant pairing;
- report reliability as a function of segment duration and temporal gap;
- compare WJ with correlation, covariance/precision and task-appropriate representation metrics;
- include motion, run, site and temporal-autocorrelation controls;
- report ROI/feature identity ordering and digest;
- report sign transitions separately from pair-aware sign-filtered WJ;
- do not pool repeated segments as independent people.

NSI may be evaluated only where its seed-to-network-template assumptions are valid for the specific fMRI representation. It is a quality-assessment comparator, not an outcome or consciousness score.

## 8. Confirmatory reporting contract

Every run must produce:
- exact source DOI, release-document version and bundle identifier;
- data-use terms and access date;
- source-file manifest and checksums;
- participant-flow table and phase/site labels;
- preprocessing pipeline/version, configuration and environment digest;
- exact code revision and command;
- fold assignment manifest and leakage-control results;
- per-participant predictions and metric contributions;
- group-level uncertainty and multiplicity correction;
- all failed, excluded, unsupported and missing inputs;
- output digest and a clean-environment replay receipt.

A result is not qualified merely because a script exits zero. Qualification also requires source completeness, denominator reconciliation, correct splits, negative controls and independent replay.

## 9. Go / no-go gates

**Go to exact reproduction** only after:
1. the V1.3 fMRI Experiment 1 bundle and CC BY 4.0 terms are verified;
2. data files are downloaded by an authorized user, manifests and SHA-256 hashes are frozen;
3. paper and supplement participant flow is reconciled with the release;
4. the authors' pipeline, trial definitions, cross-validation and group inference are pinned;
5. run/site/participant assignment is audited.

**Exploratory only** if:
- participant phase membership cannot be reconstructed;
- a preprocessing step differs from the published protocol;
- a metric or representation is selected after inspecting replication outcomes;
- site split is underpowered or confounded with protocol differences.

**Blocked** if:
- the exact source bundle or licence cannot be verified;
- subject-level mappings are missing;
- file checksums cannot be reproduced;
- required event or behavioural timing is incomplete.

## 10. Explicit claim ceiling

A successful reproduction would establish that the public data and code support a replay of the published cross-task stimulus-processing results under the declared protocol. It would not establish:
- that CETB has detected consciousness;
- that one theory has been proved or falsified in general;
- that task-irrelevant means unconscious;
- that HDC/CfC/LTC is superior;
- clinical diagnostic utility;
- generalization beyond the studied stimuli, cohort, modality and acquisition sites.

## 11. Sources

- Cogitate Consortium et al. (2025), Nature 642, 133–142. DOI: 10.1038/s41586-025-08888-1.
- COGITATE Data Release documentation V1.3: https://github.com/Cogitate-consortium/cogitate-data/blob/main/docs/01_intro.md
- COGITATE data catalogue: https://cogitate-consortium.github.io/cogitate-data/03_cogdataset/
- Data release access and terms: https://www.arc-cogitate.com/data-release
