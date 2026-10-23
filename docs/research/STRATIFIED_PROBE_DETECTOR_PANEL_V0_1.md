# Stratified Self-Probe Detector Panel v0.1

**Status:** draft extension of the reference-free detector panel. Exact-head Rust/rustfmt results remain unverified. This is not DecayBench Alert reproduction and makes no no-regret claim.

## Why stratify self-perturbations?

A detector panel selected from one pooled perturbation distribution can omit specialists. For example, channel A may increase strongly when location shifts, while channel B decreases under another shift family. Pooling both perturbation types can make each detector look weak or can select only one direction. A good aggregate score cannot recover information from detectors excluded during selection.

The fit_multi_probe API takes:
- one clean selector-score matrix with frozen detector-column order;
- a list of separately represented perturbation-family matrices, each with the same detector-column order;
- a predeclared per-family top-M limit, detectability AUC threshold, clean-score scale floor and aggregation rule.

For each family independently, it computes the empirical AUC of every detector, orients each channel so larger standardized scores align with that probe family, and retains only candidates above the predeclared threshold. At most top-M candidates survive per family. Equal detector/orientation pairs selected by multiple families are merged while retaining every selecting family index. An opposite orientation remains a separate candidate. A ProbeFamilyCoverage receipt records the best oriented AUC, number of channels above threshold, number retained for that family and whether any channel survived selection.

The original fit API remains as a one-family wrapper so current callers keep the same signature. probe_family_coverage exposes the per-family coverage summary.

## What this solves—and does not

**Solved relative to pooled selection:** one strong probe family cannot directly dilute another family's AUC because each family is evaluated independently. Specialist channels can be retained for distinct declared families. An uncovered family is reported rather than silently lost. Selection direction is visible in the selected-detector receipt.

**Not solved:** the panel cannot discover a shift family that the probe suite never activates. If an operational shift changes only a detector that no self-probe excites, the channel can still be omitted. A panel's selected families are a claim about coverage of the declared probe suite, not a universal coverage guarantee.

The current score uses the positive oriented deviations of all retained detector/orientation entries. This means one detector may contribute more than once when different probes select opposite orientations; this is intentional to preserve separate directional responses, but may change weighting. The mean and max aggregate alternatives must be compared under identical outer calibration.

The whole panel is still a nonconformity score, not a p-value. Selection data, clean outer-calibration data and audit data must be split according to the declared protocol. Conformal calibration on independent clean outer batches is required after the panel has been frozen; it provides marginal false-alarm control only under exchangeability, not power or universal shift coverage.

## Tests added

Deterministic unit tests cover:
- a location-probe family selecting detector 0 with positive orientation;
- a second family selecting detector 1 with negative orientation;
- an uninformative third family being reported as uncovered, with best AUC 0.5;
- malformed families and insufficient per-family samples failing closed.

This is a contract fixture. It is not a false-alarm, power, or real-world benchmark.

## Benchmark protocol

Compare pooled single-family selection against stratified family-specific selection while holding the total probe-score budget fixed. For each method report:
- per-probe-family coverage and the fraction of required families uncovered;
- selected channels and detector/orientation duplication;
- clean-null alarm rate after separate outer calibration;
- power and detection delay by held-out shift family and severity;
- panel versus best individual detector on every world;
- sensitivity to probe/evaluation mismatch;
- CPU p50/p95 latency, frozen state and selector cost.

Required worlds include:
1. location and sparse marginal shifts;
2. scale and heteroscedastic changes;
3. correlation/low-rank covariance shift;
4. nonlinear dependence with identical first and second moments;
5. benign input shift with unchanged conditional prediction relation;
6. shifts that are deliberately absent from the self-probe suite;
7. a dominance world in which one detector alone carries nearly all signal;
8. abrupt and gradual changes, with delayed-label outcomes where available.

Use separate development and confirmatory structural worlds. Probe families and effect sizes, AUC thresholds, top-M, aggregation, calibration sizes, seeds and alpha should be frozen before confirmatory evaluation. Tune only on development worlds and use paired confidence intervals. A 5% marginal conformal guarantee should be assessed separately from empirical power; 10,000+ Monte Carlo null streams per confirmatory null family is a proposed simulation target, not a replacement for the exchangeability proof.

## Research basis and comparators

- Xu & Tian, [DecayBench: A Reference-Free Benchmark for Trustworthy Drift Detection](https://doi.org/10.3390/math14173045), Mathematics (24 August 2026), motivates per-detector self-detectability on clean-data perturbations and selective aggregation, while stating the limitations of probe mismatch and dominant-detector regimes. This implementation independently tests a stratified extension, not their algorithm.
- Cerqueira et al., [A Framework for Evaluating and Benchmarking Concept Drift Detection Methods](https://doi.org/10.1145/3770855.3819070), KDD 2026, motivates controlled perturbations on real data, time-aware metrics and leave-one-dataset-out tuning.
- ReDiST, [Knowledge-Based Systems, DOI 10.1016/j.knosys.2025.115247](https://doi.org/10.1016/j.knosys.2025.115247), is a lightweight discriminative residual-shift comparator.
- The Rust-native [driftwatch crate](https://docs.rs/crate/driftwatch/latest) is a practical comparator for per-feature statistical tests.

## Evidence boundary

The per-family selection code and tests are committed on a draft branch. The local execution environment lacks cargo, rustc and rustfmt, so no compile/test result is claimed. Workflow runs that are queued, pending, skipped or absent are not passing evidence.
