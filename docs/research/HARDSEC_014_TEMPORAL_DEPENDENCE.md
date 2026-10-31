# HARDSEC-014 — Temporal Dependence and Effective Evidence Support

**Status:** research protocol / pre-registration draft  
**Parent:** [HARDSEC-014 issue #9413](https://github.com/Luminous-Dynamics/symthaea/issues/9413)  
**Scope:** telemetry trace lineage, temporal dependence, effective evidence support and nested model selection. No data has been reprocessed and no new detector result is claimed.

## 1. Thesis: count independent evidence, not rows

A telemetry corpus can contain hundreds of thousands of windows while representing only a few dozen independent programs and a smaller number of actual attack implementations. Adjacent windows, repeated trace segments, replays and executions of the same binary are not automatically independent observations.

The benchmark must state the independent unit appropriate to each claim. It must report raw sample counts alongside program, execution, attack-family, device and configuration support. An effective-sample estimate is conditional on a dependence model; it must never be presented as though it creates new independent attacks or programs.

This protocol extends, rather than duplicates:
- HARDSEC-001 / #8992: source provenance and split audit;
- HARDSEC-003 / #9045: deterministic corpus acquisition/import/sharding;
- HARDSEC-004 / #9060 and PR #9063: operational detector metrics and matched baselines;
- HARDSEC-012 / #9346 and PR #9410: adaptive-evasion campaigns.

## 2. Literature motivation: HARPY-V-GEM5

Pujol et al., “Lightweight detection and classification of RISC-V hardware exploits using machine learning techniques,” *Neural Computing and Applications* (published 27 October 2026), DOI [10.1007/s00521-026-12467-4](https://doi.org/10.1007/s00521-026-12467-4), describe a GEM5 simulator corpus with:
- 32 programs: 16 benign and 16 malicious;
- a 10,000-committed-instruction sampling interval;
- 12,890 feature windows per program after truncating each trace from the beginning to a common length;
- GhostWrite naturally producing 2,340 windows, then being extended by repeated iterations to 12,890;
- 412,480 total windows;
- preprocessing including variance thresholding, Min-Max normalization and ANOVA SelectKBest;
- Leave-One-Program-Out (LOPO) folds, each holding out one whole program and therefore one class only.

The authors explicitly state that LOPO folds are single-class and conventional per-fold precision, F1 and MCC are undefined or degenerate; they report per-program accuracy. That should be preserved as the paper-faithful endpoint. For operational interpretation, additionally aggregate held-out malicious-program sensitivity and held-out benign-program specificity as separate class-conditional strata. Do not compute mixed-set precision or device-hour alert burden from single-class folds.

The article reports known-workload detection above 99% for HPC and instruction modalities, instruction-only accuracy above 96% on its unseen attack-program cases, but substantially weaker instruction-based attack-family classification (around 60%). The instruction-only arm is therefore a required binary detection baseline in the simulator evidence lane, not proof of physical-HARPY-V modality availability or attack-family attribution.

The paper describes the learned preprocessing transforms and LOPO separately, but its text does not make the fitting boundary for every transform fully explicit. This is not proof that the published results leaked held-out data. A clean reproduction must fit variance filters, normalization, feature selection, imputation, hyperparameters, calibration and thresholds inside the outer-training groups only.

The paper's methods do not invalidate its results. They identify the lineage and dependence metadata our reproduction must preserve.

Source: https://link.springer.com/article/10.1007/s00521-026-12467-4

## 3. Trace/window lineage contract

Every imported or derived window must bind:
- corpus DOI/version, raw artifact ID/path/SHA-256;
- source trace ID and source program/binary ID;
- execution/run ID and device/configuration ID when known;
- attack-family and observed-effect identity as separate fields;
- original sample index or byte/record offset;
- committed-instruction range or capture-clock interval only when the source supports it;
- window length, stride, sampling policy and ordering semantics;
- replay/iteration ID and parent trace ID;
- origin status: original, replayed, repeated extension, synthetic fixture or unknown;
- extractor/preprocessor revision, exact configuration digest and feature order;
- derived shard/window identity and SHA-256.

Unknown identities and time fields remain unknown. Never infer an exact onset, duration, offset or independent execution from row order, equal-sized windows or approximate timestamps.

For replayed or repeated segments, preserve the parent trace identity. A new iteration ID does not automatically create a new independent program, attack family or device trial.

## 4. Dependence diagnostics

Apply only to pinned traces for which temporal ordering and source semantics are defensible.

Required diagnostics:
- adjacent-window exact and near-duplicate detection;
- repeated-segment and periodicity detection;
- per-feature autocorrelation and partial autocorrelation when meaningful;
- within-execution versus between-execution feature variance;
- window overlap/stride and instruction-range overlap;
- dependence sensitivity across alternative block lengths;
- count of independent programs, executions, attack families, device/configuration groups and observed attack-effect episodes.

Effective sample-size approaches may include:
- integrated autocorrelation time for approximately stationary segments;
- block bootstrap across ordered trace blocks;
- cluster bootstrap across independent executions or programs;
- hierarchical bootstrap where the nesting structure and sample counts support it.

Each report must state the estimator, stationarity/dependence assumptions, block-length or cluster choices, and sensitivity to those choices. If assumptions do not hold or ordering is unknown, mark the estimate unavailable. Keep raw counts visible.

Do not use one scalar effective sample size to imply that the corpus contains that many independent attacks.

## 5. Split ladder and claim limits

Report supported splits separately:

| Split | Claim tested | Limitation |
|---|---|---|
| Random-window | Interpolation within represented traces/programs | Diagnostic only; not unseen-program transfer |
| Contiguous-block holdout | Temporal extrapolation within a trace | Same program/execution may remain known |
| Execution holdout | New run/replay of a represented program | Not new program/family transfer |
| Program holdout | Unseen program/binary | Family may still be represented |
| Attack-family holdout | Unseen attack mechanism family | Requires sufficient independent attack families |
| Adaptation-family holdout | Unseen timing/interleaving strategy | Requires evaluator-controlled strategy lineage |
| Device/configuration holdout | New capture/device setting | Only where independent groups exist |
| Platform holdout | Cross-architecture transfer | Requires distinct real devices/platforms |

Never split overlapping or replayed windows from one claimed holdout execution across train/validation/test. If the source provides only one execution, execution holdout is unavailable.

Preserve GEM5 LOPO as a paper-faithful comparator. Each held-out fold has one class, so:
- show per-program accuracy, class and support;
- aggregate malicious-program sensitivity and benign-program specificity separately;
- macro-average over independent programs, not windows;
- mark mixed-fold precision, F1, MCC and deployment prevalence as unavailable for the single-class folds;
- do not treat the average of 32 program accuracies as a mixed-population operational precision estimate.

## 6. Nested model and preprocessing selection

For every outer holdout, the outer test group must not influence:
- missingness or variance filters;
- feature ranking/selection, including SelectKBest;
- Min-Max/standardization transforms;
- imputation or dimensionality reduction;
- feature and denominator choice;
- hyperparameter and model selection;
- probability calibration;
- operating threshold or alert deduplication/re-arm policy.

Fit every learned transform on outer-training groups only. Use inner group-aware validation for model/feature/threshold selection. Recompute and bind each fold's fitted transform/config digest to the fold receipt.

Report paper-faithful and fully nested reproductions as separate arms when their procedures differ. Preserve every failed fold and null result. Do not label a method leakage-free solely because classifier fitting excluded test rows if preprocessing or feature selection may have seen them.

## 7. Statistical and operational reporting

Every confirmatory report includes:
- raw windows, unique source traces, programs, executions, families, devices and configurations;
- original/repeated/replayed/synthetic window counts;
- duplicate and overlap diagnostics;
- dependence estimator/block settings and effective-sample sensitivity;
- per-program and per-family metrics plus group supports;
- group-level or block/cluster uncertainty, not a bootstrap over adjacent windows treated as IID;
- attack-effect-conditioned recall only where independent outcome labels exist;
- benign false-alert behavior separately from held-out-attack sensitivity;
- time-to-detection only with a defensible onset timestamp and aligned capture clock;
- false alerts per device-hour only with real monitored exposure time;
- explicit N/A reasons for unsupported metrics.

Small numbers of independent groups remain small even when each group contains thousands of windows. For example, 16 held-out malicious programs do not become hundreds of thousands of independent attacks because each program contributes 12,890 windows.

## 8. Negative controls and mutation tests

1. Duplicate/extend a source trace and ensure the independent program/execution count does not increase.
2. Repeat a deterministic loop; preserve its parent program and trace IDs.
3. Copy a window under a new sample ID; detect duplicate bytes or identical feature content and retain the source lineage.
4. Change window length/stride but preserve row count; derived manifest and shard digests must change.
5. Split adjacent windows from one execution across train/test; the grouped split auditor must reject the confirmatory split.
6. Fit preprocessing over the full dataset in an intentionally invalid fixture; the fold receipt must reject it as confirmatory evidence.
7. Fit normalization/feature selection on outer-training groups only; mutating the test group must not change fitted transform digests.
8. Attempt conventional F1/MCC on one-class LOPO; report N/A/degenerate, not a fabricated two-class score.
9. Add repeated GhostWrite-like iterations and verify that the independent trace/program count remains unchanged.
10. Permute labels within the correct independent group and inspect any above-chance score as a leakage/control failure.
11. Reorder rows without changing source identities; deterministic canonicalization or stable sample IDs must retain their declared semantics.
12. Change the source offset or replay parent ID while preserving feature values; provenance verification must detect the mutation.

## 9. Proposed project gates

These are project targets, not standards or achieved Symthaea results:
- 100% of imported windows bind source trace/program identity and derivation parameters, or explicitly state unknown where the raw source lacks them.
- 100% of repeated/replayed windows retain parent trace and iteration identity.
- Zero adjacent/replayed windows cross a claimed execution holdout.
- Zero full-dataset preprocessing fits are admitted into confirmatory outer-fold receipts.
- Every result reports raw window count and independent program/execution/family support together.
- Uncertainty is resampled at the independent group level, with block/cluster sensitivity where temporal dependence exists.
- Single-class folds cannot produce mixed-class confusion metrics or deployment precision.
- No effective-sample or generalization claim is derived from row count alone.

## 10. Implementation sequence

1. Extend HARDSEC-003 importer and shard metadata; do not create a parallel corpus importer.
2. Add a Rust trace/window lineage and split auditor, reusing IDs/digests from #8992 and #9045.
3. Add dependence diagnostics as a separate report with exact trace bytes/configuration bound.
4. Extend the HARDSEC-004 evaluator to accept hierarchical group metadata and block/cluster uncertainty summaries.
5. Apply first to HARPY-V-GEM5 only after the exact simulator archive and rights are pinned; apply to physical HARPY-V V2 only after the Dataverse inventory/acquisition semantics are resolved.
6. Compare paper-faithful versus fully nested LOPO procedures without silently conflating them.

## 11. Claim ceiling

This protocol can establish trace lineage, dependence diagnostics and statistical support for named datasets, trace sources and split policies. It does not authenticate labels, prove that a malicious effect occurred, establish causal detection, or certify hardware. Effective sample size estimates are conditional on their assumptions and cannot turn one program into thousands of independent programs.
