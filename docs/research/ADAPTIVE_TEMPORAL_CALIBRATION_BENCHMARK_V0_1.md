# Adaptive Temporal Calibration Benchmark (ATCB v0.1)

**Status:** protocol, synthetic regime registry and schema only. This is not an implementation of an online conformal method and reports no real-data score or guarantee.

**Parent evaluation:** #9954 / PR #9955. **Tracking:** #9980.

## 1. Research objective

Evaluate whether an online predictor can keep its uncertainty useful as the data-generating process changes, including when outcome labels arrive late or are later revised.

A single stream-average coverage number is inadequate. A method may under-cover badly for a long interval after a change, then compensate with over-coverage later. It may also attain high coverage by emitting unusably wide intervals. We therefore evaluate a **temporal reliability surface** over contiguous windows, regime transitions, feedback delay, interval width and computational cost.

This is a calibration benchmark. To isolate calibration behavior, all calibration methods receive the same forecast stream and the same outcome stream. A separate end-to-end lane may compare forecasters, but it must not conflate a better base forecast with a better calibration rule.

## 2. Literature and comparator taxonomy

### Strongly adaptive and online conformal methods

- Gibbs & Candès, *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*, JMLR 2024. It adapts the learning rate of ACI and develops regret control over local intervals without requiring the drift rate in advance. The theorem and exact update order must be reproduced from the paper: https://jmlr.org/beta/papers/v25/22-1218.html
- Bhatnagar et al., *Improved Online Conformal Prediction via Strongly Adaptive Online Learning*, ICML 2023. SAOCP/SF-OGD target strongly adaptive regret across intervals of multiple lengths and approximate coverage: https://proceedings.mlr.press/v202/bhatnagar23a.html
- Vaze, *Simultaneous Coverage and Efficiency Guarantee in Online Conformal Prediction*, 2026 preprint. It explicitly targets non-cancelling absolute coverage violation together with efficiency against dynamic benchmarks; its adversarial, stochastic and covariate-dependent results must be compared under their individual assumptions: https://arxiv.org/abs/2607.26577
- Amiri Shahbazi & Baheri, *Heterogeneous-Horizon Conformal Ensembles for Online Prediction Under Distribution Shift* (PACE), 2026. PACE is a useful multi-horizon empirical comparator, but the paper explicitly describes it as heuristic and does not claim an SAOCP-like formal guarantee: https://www.mdpi.com/2571-9394/8/5/77
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, 2021, remains the simple ACI reference; FACI is a further adaptive baseline.

Do not rank formal guarantees and empirical heuristics as if they were interchangeable. Reproduce method-specific assumptions, filtration, feedback timing, update ordering and objective. Do not claim arbitrary-shift conditional coverage from a long-run marginal or signed-error property.

### Benchmark coverage beyond irregularity

TIME (ICML 2026) introduces 50 fresh datasets and 98 forecasting tasks for strict zero-shot TSFM evaluation and pattern-level temporal analysis. It complements BITS and Time-IMM; their datasets, splits and task definitions must remain separate: https://proceedings.mlr.press/v306/qiao26g.html

## 3. Causality and feedback contract

Each prediction record must bind:
- decision ID, entity, target event time and forecast origin;
- model/evaluator/configuration digests and forecast distribution or raw interval;
- nominal coverage level and target horizon;
- the calibration state digest used before the prediction;
- prediction emission/availability time.

Each outcome record is evaluator-side and binds:
- decision ID and target value;
- target event time;
- first feedback availability time (when the calibration process could know the outcome);
- source and revision identity;
- later supersede/retract records with their own knowledge/availability times.

At decision time t, the calibration process may update only from outcomes with feedback availability <= t and revisions known by t. It must not backdate delayed labels to event time to make them appear available earlier. A later correction can affect future calibrator states only after the correction becomes known; it must not rewrite prior emitted intervals or prior calibration-state digests.

For public datasets with no true outcome-arrival metadata, use explicitly synthetic delay policies. Label those experiments synthetic; never present inferred delays as observed arrival times.

## 4. Synthetic regimes

The registry fixture pins ten deterministic generator scenarios: stable null, abrupt location shift, gradual drift, recurring regimes, variance shift, tail contamination, covariate-dependent drift, delayed feedback, revised feedback, and asynchronous source outage/recovery.

The generator contract is deliberately separate from a trained forecaster. It emits a fixed base forecast and a residual/nonconformity stream. The fixture now pins SplitMix64 with FNV-1a scenario/stream substream derivation, five fixed seeds, distinct residual/covariate/tail/feedback streams, integer parts-per-million mapping, integer residual arithmetic with ties away from zero, explicit tail-contamination draws and rejection-sampled integer feedback delays. This avoids floating-point distribution-library and platform drift in the synthetic generator. Change points and revision/availability schedules are known to the evaluator, not exposed to the calibration candidate unless the method is explicitly testing a change-detection signal available at that time.

Required controls:
- stable stationary / exchangeable stream for nominal baseline behavior;
- abrupt and gradual shift with exact change indices;
- recurring old regime to test whether a short-memory calibrator forgets useful historical structure;
- variance-only and tail-only changes where the mean is held fixed;
- covariate-dependent residual drift where unconditional coverage can conceal subgroup failure;
- feedback delay and out-of-order label arrival;
- supersede/retract after initial label publication;
- one or more asynchronous sources with outage and delayed recovery.

Run each scenario at multiple independent seeds and scale the horizons to the task's event-time cadence. The fixture uses step-based schedules for exact synthetic replay; real data must additionally report elapsed-time horizons.

## 5. Calibration methods to compare

At minimum, reproduce compatible versions of:

1. Static split conformal (stationary/exchangeability reference only; stress negative control under shift).
2. Adaptive Conformal Inference (ACI).
3. Fully Adaptive Conformal Inference (FACI).
4. Strongly Adaptive Online Conformal Prediction (SAOCP) / SF-OGD.
5. Gibbs–Candès local-interval-regret adaptive conformal.
6. Vaze's simultaneous coverage-and-efficiency approach, once its exact method and assumptions are implemented.
7. Fixed sliding-window quantile calibration with predeclared windows.
8. PACE heterogeneous-horizon ensemble as an empirical heuristic baseline.

### Symthaea candidate: Bitemporal Multi-Horizon Adaptive Calibration (BMAC)

BMAC is a candidate research architecture, not an established result:

- Keep the base forecaster fixed for calibration-isolation comparisons.
- Maintain calibration experts on both event-count horizons and elapsed-time horizons. The latter is important for irregular streams; a 128-observation window can represent seconds in one source and months in another.
- Process residual feedback in availability/knowledge-time order, preserving the original forecast origin and the revision history.
- Maintain a short, medium and long horizon plus dyadic/strongly-adaptive interval experts.
- Compare a strongly-adaptive expert aggregator against a simpler learned mixture. Do not claim a theorem for the mixture until its potential/update rule is proved under delayed feedback.
- Optimize coverage and sharpness jointly. A candidate must not win by making intervals arbitrarily wide.
- Keep per-entity and per-pattern diagnostics so aggregate coverage cannot conceal a failing stratum.

A clean first ablation isolates event-time windows vs feedback-count windows vs elapsed-feedback-time windows, followed by the multi-horizon aggregator. Any change detector or context gate is another separately preregistered arm; do not choose it after inspecting test outcomes.

## 6. Metrics

For nominal coverage 1-alpha, report all of the following.

### Global metrics
- empirical coverage and signed coverage error;
- absolute miscoverage error;
- average and median interval width;
- interval score / Winkler score and pinball loss (or CRPS for a compatible probabilistic forecast);
- average and worst-subgroup coverage;
- compute/update time, memory and bytes per calibration update.

### Temporal metrics
For every predeclared window length L (both number of feedback observations and elapsed time):
- minimum and maximum rolling coverage over all eligible contiguous windows;
- maximum absolute rolling coverage deviation;
- fraction of windows under-covering by more than epsilon;
- longest consecutive undercoverage duration;
- coverage deficit integral: sum over forecast origins of max(0, target_coverage - rolling_coverage);
- post-change recovery delay: first point after the known change where K consecutive non-overlapping evaluation windows meet the preregistered coverage floor;
- recovery width and integrated interval score during the same recovery period.

Rolling-window extrema are descriptive benchmark statistics. Overlapping windows are not independent replicates and must not be used as such for confidence intervals or multiplicity-free claims.

### Feedback and revision metrics
- feedback latency distribution (event-to-available and forecast-to-label);
- number/share of outcomes unavailable at each decision time;
- calibrator update lag after feedback arrival;
- revision count and correction lag;
- count of historical prediction/calibration receipts altered by later revisions (target zero);
- error from any accidental event-time-ordered replay compared with correct availability-time replay (negative control).

## 7. Statistical reporting

- Preregister the model, base forecast stream, nominal alpha, target horizons, window lengths, regimes, seeds, feedback delay policy, calibrator family and primary metrics.
- Use at least five independent synthetic seeds as an initial target; report every seed.
- Use stream/seed or entity/block as the uncertainty unit; overlapping windows are nested diagnostics, not independent samples.
- Publish paired method differences and uncertainty intervals for both coverage deficit and interval score/width.
- Apply multiplicity controls across methods, seeds, regimes, window lengths and selected metrics.
- Report best achievable and resource-matched comparisons separately.
- Preserve an exchangeable stationary control and negative controls for future-label leakage, delayed-label backdating and retroactive revision.
- Keep theoretical guarantee statements in a separate table that cites assumptions and exact theorem; empirical benchmark results never expand the guarantee.

## 8. Qualification targets (not results)

- 0 outcomes consumed before feedback availability.
- 0 historical prediction or calibration receipts changed by later label revisions.
- Stationary-control long-run coverage within 2.5 percentage points of nominal, with uncertainty intervals and sufficient sample size.
- For every preregistered window length, report worst-window coverage and undercoverage duration; do not hide failures in global averages.
- After a change, recover to within 5 percentage points of target coverage for three consecutive non-overlapping windows; publish recovery delay and interval width.
- No accepted calibration result if interval width or interval score exceeds the predeclared utility ceiling.
- At matched base forecasts, demonstrate a Pareto improvement in coverage deficit and interval efficiency against static, ACI/FACI, SAOCP, and rolling-window baselines on at least two distinct drift families before claiming a practical improvement.
- Formal guarantees are claimed only for the exact method/theorem assumptions actually satisfied by the run.

All numbers above are initial qualification targets. Freeze them on development streams before opening test results.

## 9. Execution order

1. Validate the registry JSON against its Draft 2020-12 schema and commit deterministic generator semantics.
2. Implement the synthetic residual and delayed-feedback replay runner in Rust as an evaluator-only crate.
3. Add exact negative controls for future feedback consumption, revision backdating and overlap-as-independent inference.
4. Reproduce static conformal, ACI/FACI, SAOCP and fixed sliding-window baselines.
5. Add the recent simultaneous coverage-and-efficiency comparator with a reviewed theorem-to-code map.
6. Implement BMAC as a separate candidate arm; initially mark it heuristic unless a complete proof supports the delayed-feedback aggregation rule.
7. Evaluate on TIME/BITS/Time-IMM only where compatible outcome feedback semantics are available; do not fabricate real label-arrival times.
8. Measure deployment cost and only then optimize the calibrator.

## 10. References

- Gibbs & Candès (2024), JMLR: https://jmlr.org/beta/papers/v25/22-1218.html
- Bhatnagar et al. (2023), ICML/PMLR: https://proceedings.mlr.press/v202/bhatnagar23a.html
- Vaze (2026), arXiv:2607.26577: https://arxiv.org/abs/2607.26577
- Shahbazi & Baheri (2026), PACE: https://www.mdpi.com/2571-9394/8/5/77
- Qiao et al. (2026), TIME, ICML/PMLR: https://proceedings.mlr.press/v306/qiao26g.html
- Parent Symthaea temporal benchmark: #9954 / PR #9955.
