# Aviation Distributionally Robust Recovery Benchmark (ADRB-001)

**Status:** implementation and specification committed to the AECB research branch; Rust tests defined but not executed here.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aviation_distributional_robustness.rs`  
**Related protocols:** [AECB](./AVIATION_ENTERPRISE_BENCHMARK.md), [network externalities](./AVIATION_NETWORK_EXTERNALITY_BENCHMARK.md), [finite-set robustness](./AVIATION_NETWORK_ROBUSTNESS_BENCHMARK.md), [public-data evaluation](./AVIATION_PUBLIC_DATA_EVALUATION_PROTOCOL.md).

## Research hypothesis

A policy that is beneficial under one nominal scenario distribution may stop being beneficial when disruption probabilities are misspecified. A distributionally robust gate should reveal the amount of probability-mass shift the decision can tolerate, reject candidates whose worst-case expected cost exceeds a predeclared threshold, and expose the cost of this robustness.

The tested hypothesis is **not** that a total-variation ambiguity set captures every real distribution shift. It is that an explicit uncertainty radius gives a reproducible sensitivity test that is stronger than reporting nominal expected cost alone.

## Current state of the art

- Zhang et al., *Distributionally robust optimisation approach for aircraft sequencing and scheduling with learning-driven arrival and departure time predictions*, Omega 2026. Uses real data from Xiamen Gaoqi International Airport and compares estimate-then-optimise (ETO) to estimate-then-distributionally-robust-optimise (ETDRO), including decomposition approaches for computational tractability. This is a close mathematical comparator; the present implementation is a small finite-support evaluation gate, not a reproduction of the runway sequencing model. https://doi.org/10.1016/j.omega.2025.103415
- Guardo-Martinez et al., *Robust airline scheduling with turnaround under uncertainty: towards collaborative airline scheduling*, Transportation Research Part E, January 2026. Highlights interdependence between flight planning, fleet, crew and turnaround, and warns that robustness must account for utilization/resource cost and may otherwise become over-conservative. https://doi.org/10.1016/j.tre.2025.104440
- Wu et al., *A collaborative decision making framework for airline irregular flight recovery with network manager disruption coordination*, Journal of Air Transport Management, 2026. Coordinates airline recovery alternatives with updated airport-slot capacity and a fairness mechanism, evaluated across 32 scenarios derived from realistic airline data. https://doi.org/10.1016/j.jairtraman.2026.103169 (verify final pagination/DOI from publisher version before formal citation)
- EUROCONTROL, *How Capacity and Weather-Based Operations are building network resilience*, 14 September 2026. Describes trigger-based network action for severe weather, capacity shortage and emerging instability; reports 3.1 million delay minutes saved from 1 May to 31 August and 14% higher network stability on challenging days. These programme-reported outcomes are motivation, not directly comparable targets for this prototype. https://www.eurocontrol.int/publication/how-capacity-and-weather-based-operations-cwbo-are-building-network-resilience-capacity

## Implemented mathematical contract

Each frozen scenario carries:
- a unique stable scenario ID;
- a nominal probability in integer parts per million, with all probabilities summing exactly to 1,000,000;
- candidate-minus-baseline objective cost in frozen integer units.

Let (d_i) denote cost delta in scenario (i) and (p_i) the nominal probability. Negative delta favors the candidate; positive delta means the candidate is worse. The nominal expected cost delta is (sum_i p_i d_i).

The evaluator computes the maximum expected delta over distributions (q) satisfying:
- (q_i \ge 0);
- (sum_i q_i=1);
- total variation distance (\tfrac{1}{2}\sum_i |q_i-p_i| \le \rho).

The implementation expresses (ho) as ppm of probability mass that may move. It obtains the exact worst-case distribution for this finite-support linear expectation problem by moving probability mass from the lowest delta scenarios to the highest delta scenarios until the declared radius, donor mass, or receiver capacity is exhausted. This is not a generic Wasserstein-DRO solver; it is the exact solution of the finite total-variation expected-loss subproblem.

To avoid floating-point decision thresholds, expected-cost gating uses an integer numerator scaled by 1,000,000. Floating-point expected-cost fields are reporting conveniences only. The gate binds the TV radius to the computed report and separately enforces a maximum single-scenario regression.

## Implementation path

File: `crates/domains/symthaea-operations-research/src/aviation_distributional_robustness.rs`

- `evaluate_tv_robust_cost_delta`: input validation, exact nominal expected delta, worst-case TV-ball expected delta, adversarial probabilities, worst single-scenario outcome and scenario-level counts.
- `assess_tv_robust_candidate`: predeclared worst-case expected-delta threshold plus single-scenario regression cap.
- Zero-probability scenarios are permitted as explicit modeled stress cases and can receive probability mass inside the ambiguity set. A zero-probability case must be justified by the scenario-generation protocol; it is not proof that an omitted real event is covered.
- The module does not estimate nominal probabilities, calibrate the radius, generate disruption scenarios, validate aircraft constraints, or prove generalization beyond the supplied support.

## Finite-sample radius calibration

The companion module `aviation_probability_calibration.rs` adds a transparent conservative baseline for selecting a TV radius from the Calibration partition, rather than choosing an arbitrary radius by inspecting final results.

Given a frozen nominal distribution `p`, empirical calibration frequencies `p_hat`, `K` finite-support categories, `n` independent calibration episodes and failure probability `delta`, use a union-Hoeffding bound:

```text
epsilon = sqrt(ln(2*K/delta) / (2*n))
rho = min(1, TV(p, p_hat) + (K/2)*epsilon)
```

The code reports empirical TV distance, the sampling allowance, failure probability, episode count and resulting radius separately. The new `calibrate_tv_radius_envelope` extends this to multiple declared airport/weather/season regimes: it allocates a family-wise failure budget by Bonferroni, calibrates each regime separately, and returns the maximum regime radius for a common robust policy. It uses integer ppm for probability inputs and rounds the empirical distance and final radius upward. When evidence is sparse or the support is large, the bound may saturate at `rho = 1`; that is a visible conservatism cost, not a reason to silently lower the confidence level.

**Assumption boundary:** this bound requires independent draws from one fixed finite-support distribution. Airline disruptions may be temporally dependent, season-dependent and nonstationary. Count complete event episodes rather than rows; calibrate separately across relevant regimes/time blocks and take a predeclared envelope where appropriate. This baseline does not guarantee coverage under arbitrary drift or for omitted categories. The split-integrity module can validate declared cluster separation and chronology, but cannot prove independence or data lineage.

A relevant comparator is Momtaz (2026), *Rolling-Calibrated Conformal Decision Support for Seasonal Airport Taxi-Time Planning under Distribution Shift*, which evaluates rolling conformal calibration across 17 IATA seasons and 722 EUROCONTROL CODA airports. It specifically illustrates the trade-off between under-planning and extra padding. https://doi.org/10.19139/soic-2310-5070-3782. A second comparator is Yang and Jin (ICML 2026), *Multi-Distribution Robust Conformal Prediction*, which considers finite-sample valid coverage across multiple heterogeneous source distributions. https://proceedings.mlr.press/v306/yang26ad.html. Neither paper validates this module; they define fairer calibration/reliability comparisons.

## Required benchmark

Compare the same candidate and incumbent using:
1. nominal expected objective;
2. the TV-robust objective at predeclared radii, e.g. 0%, 5%, 10%, 20%, and 50% of probability mass;
3. fixed conservative margin / separate interval gate;
4. finite-set worst-case and CVaR-style metrics already provided by the network-robustness module;
5. a conventional two-stage stochastic or distributionally robust optimizer when the same action space and model are available;
6. do-nothing/incumbent and hindsight reference (hindsight is non-deployable).

Use paired scenarios and freeze nominal weights, scenario support, ambiguity-radius grid, objective, stakeholder constraints, and compute budget before looking at final outcomes. Calibrate radii only on training/validation periods; report final results on temporally held-out and held-out-disruption-family tests.

### Required stress families

- weather/capacity regime transitions;
- probability calibration drift, including underweighting the rare severe regime;
- compound disruptions and correlated airport/airspace shocks;
- qualified parts shortages and maintenance-window changes;
- stale, contradictory or absent evidence;
- explicit zero-probability stress atoms with documented rationale;
- cases in which every non-incumbent candidate is worse;
- radii large enough to turn a nominal benefit into a robust expected loss.

## Metrics

Report nominal expected delta, worst-case expected delta across the declared TV ball, radius at which the sign changes, worst single-scenario delta, worst-actor regression, CVaR-style loss, regret, runtime, and candidate rejection/abstention counts. Report the robustness premium (robust objective minus nominal objective) separately from recovered benefit; conservative is not synonymous with better.

## Proposed acceptance gates (targets, not results)

- 100% agreement with hand-computable probability-shift fixtures and an independently implemented brute-force evaluator on tiny probability grids.
- Exact probability mass conservation and row-order invariance.
- Zero acceptance when the robust expected delta or single-scenario regression exceeds its predeclared threshold.
- Zero acceptance when the requested TV radius differs from the radius actually evaluated.
- On held-out disruption families, the robust gate reduces severe regressions without exceeding a preregistered mean-cost or utilization penalty.
- Report robustness premiums and all rejected candidates. Do not select the radius after inspecting the final test set.


## Robustness budget / sign-flip curve

The exported `evaluate_tv_robustness_curve` evaluates a caller-predeclared, strictly increasing radius grid. For every radius it reports nominal and worst-case expected cost numerators plus the robustness premium (worst-case minus nominal). It also reports the first tested radius where the candidate crosses the chosen maximum expected-cost threshold. Setting the threshold to zero yields the first evaluated radius where non-positive robust expected cost is lost.

For the hand-checkable fixture `routine = -100`, `capacity = 0`, `compound-weather = +20`, with nominal probabilities `[0.6, 0.3, 0.1]`, the nominal expected delta is -58 units. The worst-case delta is -34 at TV radius 20% and +2 at TV radius 50%; therefore the first non-beneficial radius on the grid `[0%, 20%, 40%, 50%, 100%]` is 50%. These are synthetic math fixtures, not operational estimates.

Do not optimize or select the radius against final test outcomes. The curve is a sensitivity analysis; the calibration protocol must choose radius using training/validation periods and then freeze it.

## Test status and claim ceiling

Nineteen Rust unit tests are defined, covering exact hand calculations, TV radius zero/full, identical costs, probability-mass validation, stable row ordering, radius-mismatch rejection, a scenario where nominal improvement turns into a positive worst-case expected cost delta, robustness-premium accounting and monotone sensitivity curves. Three integration-oracle tests independently enumerate tiny probability grids and compare the resulting maximum expected cost against the production greedy transfer method, including zero-probability stress atoms. These are defined but not run: the current environment has no `cargo` or `rustc`.

Any passing result is bounded to the finite support and the declared TV ambiguity set. It does not prove distribution-free robustness, airline savings, causal intervention effects, operational authority, airworthiness or regulatory certification.


## Tracking

Implementation and reproduction follow-up: [AVIATION-ENTERPRISE-008 #8752](https://github.com/Luminous-Dynamics/symthaea/issues/8752). Related implementation PR: [#8586](https://github.com/Luminous-Dynamics/symthaea/pull/8586).


## Executable calibration integration (28 October 2026)

The synthetic example `examples/aviation_calibration_protocol_benchmark.rs` now uses `calibrate_tv_radius` on explicitly labeled calibration frequencies, evaluates the resulting radius using `evaluate_tv_robust_cost_delta`, and separately reports a predeclared sensitivity curve. This demonstrates composition across the split auditor, tuning provenance gate, finite-sample radius calibration and DRO evaluator. It is not an operational dataset pipeline; in a real run, the frequencies must be derived from frozen Calibration event clusters, and the Test partition must not be used to choose the confidence level or other settings.


The finite-sample module also includes regime-envelope tests: deterministic regime ordering, per-regime Bonferroni allocation, maximum-radius selection, invalid/duplicate regime rejection and failure when the family-wise delta cannot be allocated as positive integer ppm. The calibration unit-test count is now nineteen, plus five public-API integration fixtures. Tests remain uncompiled in this environment.
