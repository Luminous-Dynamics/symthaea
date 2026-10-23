# Reference-Free Detector Panel v0.1

**Status:** draft Rust implementation stacked on the fixed-reference feature-shift scorer. Exact-head Rust/rustfmt results have not yet been observed. This is a reference implementation inspired by recent detector-selection research, not a reproduction of DecayBench's Alert algorithm and not a no-regret guarantee.

## Paradigm shift: select signals before combining them

No individual detector is uniformly strongest across all types of distribution change. DecayBench (Xu & Tian, *Mathematics*, 24 August 2026) evaluates ten detectors across ten NLP, vision, and multimodal datasets and proposes a label-free self-configuring combiner with finite-sample conformal false-alarm control. Its no-regret argument is conditional: when one detector dominates, averaging can dilute it; self-detectability under clean-data perturbations is used to prune weak channels. This result makes detector aggregation a more relevant next experiment than another isolated score. Source: https://doi.org/10.3390/math14173045.

The Rust prototype here implements only the core experimental pattern:
1. collect detector scores on a clean selector split;
2. make self-generated perturbations without access to deployment/evaluation labels;
3. compute each detector's empirical AUC separating clean and perturbed selector scores;
4. reverse detector orientation when lower values are more aligned with the declared perturbation;
5. keep at most top_m channels exceeding a predeclared minimum detectability AUC;
6. standardize using clean-selector means/scales and combine positive oriented z-scores by either mean or maximum;
7. calibrate the frozen aggregate on a separate, untouched clean outer-calibration split.

The selector deliberately returns NoInformativeDetectors instead of quietly falling back to unselected channels when no detector meets the threshold. The reportable selected-detector summary contains original channel index, orientation, clean mean/scale, and self-detectability AUC.

## Guarantees and limitations

The aggregate returned by DetectorPanelScorer is a nonconformity score, not a p-value. Finite-sample marginal false-alarm control comes only after calibrating it with ConformalAdequacyCalibrator on independent clean outer scores, assuming exchangeability under the declared batch protocol. The type cannot prove that selector, outer-calibration and audit data are genuinely disjoint.

The prototype is not Alert reproduction and makes no no-regret claim. It differs in implementation details and has not been compared empirically with Alert. Most importantly, self-generated probes may not resemble operational drift: selection can reject the one detector that would respond to a real shift. Selection and perturbation families must be frozen before held-out evaluation. Include dominance cases and perturbations that are intentionally poor proxies for operational changes.

The two aggregation rules are benchmark alternatives rather than universally ranked choices:
- MeanPositiveZ can reinforce several moderately responsive detectors, but may dilute a single strong response.
- MaxPositiveZ protects a single strong channel more directly, but can be driven by one noisy channel. Independent outer calibration is required for either rule.

The panel addresses input-score drift. It does not identify whether a shift is harmful, distinguish covariate shift from concept drift by itself, or guarantee detection of predictive failures. Those questions require outcomes/labels when available, domain constraints, or separately justified sequential evidence.

## Test fixtures

Deterministic unit tests cover:
- selecting two informative channels while rejecting an unresponsive third channel;
- orienting one channel whose score decreases under perturbation;
- comparing mean and max aggregation;
- malformed/non-finite input rejection and finite-value overflow rejection;
- returning a typed QueryDistributionShift assessment after outer conformal calibration;
- returning an explicit error when no detector meets a predeclared detectability threshold;
- an explicit probe-mismatch negative control where a real shift affects only a detector that the self-perturbations never activated, demonstrating the panel's selection blind spot.

These tests are committed but not considered passed until the focused workflow reports success for the exact head.

## Benchmark plan

### Baselines

Compare:
- each detector as a single channel, with the same clean outer-calibration and audit budget;
- fixed mean, max, median, Fisher/Simes/Bonferroni-style combination where score assumptions make the comparison valid;
- the self-configuring panel, with MeanPositiveZ and MaxPositiveZ;
- exact moment/correlation and random-projection hybrid shift scorers from adjacent research work;
- ReDiST's discriminative residual-shift approach and the Rust driftwatch crate's per-feature PSI, KL/JS, KS and chi-square baselines;
- WATCH and conformal sequential-martingale baselines when their assumptions are reproduced faithfully.

### Required worlds

Use at least six structural families with development/confirmatory separation:
1. diffuse location shifts;
2. sparse marginal and scale changes;
3. dependence/correlation and low-rank covariance changes;
4. nonlinear changes preserving the first two moments;
5. a benign covariate shift with unchanged conditional outcome relation;
6. consequential predictive degradation where a proxy probe may fail to excite the decisive detector.

Also test abrupt/gradual changes, subgroup and missingness changes, and at least one real feature dataset with controlled perturbations. Include detector-dominance worlds and probe-mismatch worlds. The confirmatory generator, effect sizes, seed list, selected-detector threshold, top_m, aggregation rule and alpha must be preregistered.

### Metrics and provisional gates

At equal calibration and audit cost, report clean false-alarm rates with confidence intervals, detection power by held-out world/severity, detection delay, missed-detection rate, consequential validity when labels are available, p50/p95 latency, memory, and the fraction of worlds in which the panel is statistically inferior to the best single detector. Use paired seeds and paired bootstrap confidence intervals rather than comparing unrelated averages.

Provisional goals for confirmatory tests:
- nominal clean false-alarm target 5%; use at least 10,000 null streams per confirmatory null family for simulation estimates, supplemented by the finite-sample calibration contract;
- seek at least 80% power at a preregistered medium shift and 95% at a large shift, measured separately for every world family;
- no significant degradation relative to the best constituent in the non-dominant regime, and publish every exception instead of averaging it away;
- show that the selector itself does not use held-out drift labels, and compare to the best constituent as an oracle only in offline analysis;
- no SOTA claim until reference implementations, splits, budgets, and tuning protocols are independently checked.

## Current external references

- Xu & Tian, [DecayBench: A Reference-Free Benchmark for Trustworthy Drift Detection](https://doi.org/10.3390/math14173045), 2026.
- Cerqueira et al., [A Framework for Evaluating and Benchmarking Concept Drift Detection Methods](https://doi.org/10.1145/3770855.3819070), KDD 2026; motivates diverse drift families, timing-aware criteria and leave-one-dataset-out tuning.
- Roostaee & Soheilian, [ReDiST: A lightweight residual analysis framework for fast and accurate unsupervised drift detection](https://doi.org/10.1016/j.knosys.2025.115247), *Knowledge-Based Systems*, published 28 February 2026. Its abstract reports up to 58× speed versus its most accurate supervised comparator; this is the paper's reported figure, not a benchmark result reproduced here.
- [Rust driftwatch crate](https://docs.rs/crate/driftwatch/latest), version 0.1.0 (August 2026), as a relevant Rust-native operational baseline.

## Evidence boundary

The module includes its scoring/selection implementation and tests, and the focused workflow now formats and runs its tests. There is no local Rust toolchain in the current execution environment, so neither compilation nor tests are claimed to pass. The GitHub workflow's queued, pending, skipped, or absent statuses are not treated as passing evidence.
