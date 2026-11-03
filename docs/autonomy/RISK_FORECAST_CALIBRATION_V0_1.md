# Vehicle event-risk forecast calibration v0.1

**Status:** probability-calibration evaluator implemented; no risk forecasts or calibration results reported  
**Code:** `symthaea_vehicle::risk_calibration`  
**Related programme:** NIST Measurement Science for Automated Vehicles (2026), SAM counterfactual decision quality and RTA.

## Why this is a separate benchmark

A model's predicted risk is useful only if a forecast of 0.2 corresponds to an event frequency near 20% under the declared sampling conditions. Ranking scenarios well is not the same as assigning calibrated probabilities. A model can have high discrimination and poor calibration, or reasonable calibration with weak discrimination.

Do not derive event probabilities from a stress-search score, controller disagreement, confidence heuristic or adversarial search frequency. Stress testing deliberately oversamples unusual events. Calibration data need a separately declared sampling profile and a defensible observation process.

## Forecast unit and scope

Each report is restricted to:
- one event-definition ID (e.g. pedestrian geometric-envelope overlap under a named model);
- one forecast horizon in milliseconds;
- one predictor artifact digest;
- one outcome-evaluator artifact digest;
- one sampling-profile digest;
- one declared number of equal-width reliability bins.

Every observation binds a failure-family ID, scenario ID, predicted probability in [0,1], and outcome:
- `Occurred`: event observed within the declared horizon;
- `NotOccurred`: absence of the event observed through the full horizon;
- `Censored`: episode ended, was interrupted or lost observability before the horizon.

Censored episodes are retained and counted, but excluded from Brier/log loss/ECE. They are never silently recoded as negative examples. If no fully observed outcomes remain, the report is rejected.

## Metrics

- **Brier score:** mean squared probability error across fully observed binary outcomes.
- **Log loss:** Bernoulli negative log likelihood, probabilities clipped at 1e-12 for finite arithmetic.
- **Expected calibration error (ECE):** observation-weighted absolute gap between mean forecast probability and observed event frequency across fixed-width bins.
- **Maximum calibration error:** largest absolute non-empty-bin gap.
- **Family-macro Brier:** compute Brier inside each failure family, then average family scores with equal family weight.
- **Worst-family Brier:** highest family Brier, to expose weak tails hidden by a global average.
- **Reliability diagram data:** every bin is retained, including empty bins; bin counts are part of the report.

Brier and log loss are proper scoring rules. ECE is a useful descriptive diagnostic but is sensitive to bin count, bin edges and sample size. Always publish the reliability bins and counts; never report ECE by itself.

## Interpretation boundaries

The report can assess calibration conditional on the supplied event labels, scenario-family corpus, sampling profile and forecast horizon. It cannot establish:
- real-world event prevalence from an adversarial or convenience-sampled corpus;
- correctness of the event labels or evaluator;
- physical validity of the simulator or geometric event definition;
- calibration in a different geography, ODD, traffic population or horizon;
- safety or suitability of an RTA threshold.

A predictor should be evaluated on both a representative-sampling track and a separate stress track. The representative track estimates calibration under its declared sampling frame; the stress track diagnoses behavior under deliberately shifted exposure. Do not pool the two without an explicit target-distribution weighting method.

The family-macro score reduces sibling-variant domination but does not prove family independence. It is not a fleet-level confidence interval or a substitute for proper sampling.

## Controls

Nine unit tests are authored for:
- a correctly constructed calibrated forecast fixture (zero ECE);
- strongly overconfident forecasts and proper scoring loss;
- censoring retention without negative-label conversion;
- family macro aggregation under sibling-family imbalance;
- input-order-invariant report digest;
- invalid probability/digest/horizon/bin-count rejection;
- duplicate scenario and all-censored rejection;
- report mutation detection even after output digest recomputation;
- explicit empty reliability-bin retention.

Tests have not been compiled or executed in this session.

## Benchmark protocol

For each event definition, publish separate reports at fixed horizons (e.g. 250 ms, 1 s, 2 s, 5 s only where the forecast meaning supports those horizons). Evaluate:
1. calibration on a frozen representative/development sampling frame;
2. calibration under frozen, disjoint distribution shifts;
3. family-wise calibration and worst-family performance;
4. Brier/log loss and reliability diagrams alongside discrimination (AUROC/AUPRC), with class prevalence and censoring shown;
5. coverage of any abstention/uncertainty mechanism and downstream risk under retained cases.

Do not select RTA thresholds from a stress-set ECE. Use calibrated probability intervals and model-discrepancy bounds only after external label validity and sampling assumptions are established.

## Next steps

1. Bind the module to actual Symtropy execution receipts and per-horizon event observables.
2. Add uncertainty intervals or conformal sets as separate outputs; do not conflate set coverage with probability calibration.
3. Add family-cluster bootstrap intervals once a declared target family distribution is available.
4. Compare the raw predictor, post-hoc isotonic/Platt calibrators and calibration-aware training under identical frozen holdouts.
5. Measure calibration drift after changes to model, map, dynamics, sensor/actor policies and ODD assumptions.
6. Feed calibration evidence into RTA mode selection only through an independently reviewed policy, never by direct threshold self-tuning.

## Claim ceiling

The present module computes proper scoring and reliability-bin statistics for supplied forecast probabilities and supplied fully observed binary outcomes. It does not generate forecasts, validate event labels, model physical contact, estimate real-world probability or authorize vehicle behavior.
