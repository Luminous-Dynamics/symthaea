# Vehicle trajectory prediction-set coverage v0.1

**Status:** evaluator contract implemented; no prediction sets, calibration thresholds from real data, or coverage results reported  
**Code:** `symthaea_vehicle::trajectory_coverage`  
**Issue:** [SAM-PRED-COV-001 #10256](https://github.com/Luminous-Dynamics/symthaea/issues/10256)  
**Related:** probability calibration #10247; model stability #10252; Symtropy interactive actors #3471 / execution #3387.

## Why this is separate

Point-forecast error, event-probability calibration and prediction-set coverage answer different questions:

- minADE/minFDE ask how close one or more predicted trajectories are to the observed trajectory under a benchmark's matching rule;
- Brier/log loss ask whether event probabilities correspond to observed event frequencies;
- set coverage asks whether the realised future falls inside a declared prediction set;
- set measure / sharpness asks how much space or uncertainty the predictor must reserve to obtain that coverage;
- downstream decision quality asks whether the planner makes better choices when it consumes those sets.

A system can have a good point forecast but poor coverage, or high coverage achieved only with enormous sets. These outcomes must not be collapsed into one number.

## Evaluator contract

The evaluator is deliberately offline. It receives caller-produced nonconformity scores and a scalar prediction-set measure. The caller fixes the score function, set construction, geometry/units and outcome evaluator. This module does not create forecasts, generate tubes, evaluate geometry, or authenticate any of those artifacts.

Each report binds:
- target / actor definition and forecast horizon;
- predictor, set-construction and outcome-evaluator artifact digests;
- calibration and evaluation sampling-profile digests;
- nominal coverage target and a named set-measure ID/unit;
- calibration rows and disjoint evaluation rows, each identified by scenario and failure family.

The evaluator sorts calibration scores and uses the finite-sample split-conformal order statistic:

`k = ceil((n_cal + 1) × nominal_coverage)`

The threshold is the `k`-th smallest calibration nonconformity score. If `k > n_cal`, the correct finite-sample threshold is unbounded; the evaluator rejects the input rather than pretending a finite threshold exists.

For the evaluation sample, the report computes empirical marginal coverage, signed coverage error, mean and nearest-rank p95 set measure, per-family coverage/size, macro-family coverage and worst-family coverage. Raw rows and derived coverage decisions are retained and revalidated before digest acceptance.

The group-size flag uses 20 rows as an engineering warning threshold only. It is not a statistical theorem or evidence that 20 rows are sufficient for a conditional claim.

## Assumptions and limits

- Split-conformal marginal coverage depends on a valid calibration/evaluation protocol and exchangeability of calibration and future examples under the chosen score function.
- Identical sampling-profile digests do not prove exchangeability.
- Different calibration/evaluation profile digests identify a shift diagnostic; they do not preserve the nominal guarantee.
- Marginal coverage does not imply coverage for each actor class, failure family, geography, weather state, sensor regime or individual scenario.
- Family metrics are descriptive unless an adequate independent-cluster sampling design supports uncertainty intervals.
- Frames from one episode and sibling variants from one scenario family are not independent samples.
- A scalar set measure is meaningful only under its explicitly named measure ID and unit. Width, radius, area and volume are not interchangeable.

No weighted, online, adaptive, covariate-conditional or distributionally robust conformal guarantee is implemented here. Those approaches require separate methods, assumptions, proofs or empirical qualification, and negative controls.

## Benchmark ladder

1. **Exact synthetic controls.** Test the finite-sample rank, exact membership boundary, input-order invariance, duplicate rejection, calibration/evaluation leakage rejection and insufficient-calibration abstention.
2. **Motion forecasting.** Use the official Waymo Open Motion and Argoverse 2 Motion splits where rights and protocol permit. Retain minADE/minFDE/miss rate as separate point/multimodal accuracy metrics; add coverage, set measure, actor class, horizon and failure-family slices.
3. **Long-tail closed-loop planning.** Use WOD-E2E as a long-tail planning track and Fail2Drive as a paired distribution-shift track. Their route/RFS or success metrics are not substitutes for trajectory-set coverage.
4. **Interactive closed loop.** Connect Symtropy #3387 and #3471 actor/ego event receipts to forecast evaluation; then compare planner outcomes and counterfactual regret under matched action histories.

## Comparison arms

- point prediction without an uncertainty set (report accuracy and misses; coverage/size not applicable);
- fixed Gaussian or heuristic tube;
- ensemble / multimodal set;
- split-conformal set;
- normalized split-conformal set when a local scale is available;
- weighted/online shift-adaptive methods only in a separately versioned protocol;
- evaluator-only oracle set, excluded from eligible model ranking.

Every eligible method must use identical model-visible histories, forecast horizons, actor sets, evaluation rows, and matched-compute limits. Also publish a best-attainable-compute track with hardware and resource details.

## Prospective targets (not results)

- On an adequately sized, family-disjoint exchangeable holdout, empirically assess nominal 90% and 95% marginal coverage with exact denominators and appropriate uncertainty intervals.
- Publish worst-family and macro-family coverage; do not let pooled coverage hide important tail failures.
- At matched empirical coverage, target at least 20% lower median and p95 set measure than a frozen heuristic-tube baseline, without degrading worst-family coverage. This is a research target, not a guarantee.
- Under shift, explicitly measure coverage collapse, set expansion, planner degradation and correct abstention. Do not reuse the exchangeable guarantee.
- Require zero calibration/evaluation ID overlap, zero duplicate acceptance and no silent conversion of unbounded thresholds.
- Keep model selection, calibration and final test sets disjoint by scenario family and, where possible, geography/time.

## Authored controls

Ten Rust test declarations cover:
1. finite-sample split-conformal order statistic;
2. separate coverage and set-sharpness metrics;
3. insufficient calibration rejection;
4. calibration/evaluation overlap rejection;
5. duplicate scenario rejection;
6. underpowered family warning;
7. differing profile disclosure;
8. input-order-invariant report digest;
9. report-tampering detection after digest recomputation;
10. invalid score and set-measure rejection.

These tests have not been compiled or executed in this session.

## Claim ceiling

The module measures supplied prediction-set membership and scalar set measure against supplied observations. It does not validate prediction geometry, establish exchangeability, prove conditional or real-world coverage, establish human behavioural fidelity, certify a planner, or authorize vehicle behaviour.
