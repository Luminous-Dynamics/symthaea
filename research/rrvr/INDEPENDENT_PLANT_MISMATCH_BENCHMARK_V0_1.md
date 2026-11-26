# RRVR Independent Plant-Mismatch Benchmark v0.1

**Status:** experimental design and process-model specification · 2026-11-26  
**Tracking:** [Symthaea #13835](https://github.com/Luminous-Dynamics/symthaea/issues/13835)  
**Related:** shadow observer [#13809](https://github.com/Luminous-Dynamics/symthaea/pull/13809), AERB [Symtropy #7631](https://github.com/Luminous-Dynamics/symtropy/issues/7631).

## 1. Why this benchmark exists

A predictive observer is only useful if its forecasts survive model mismatch. Existing manipulator simulator code can predict by cloning and advancing the same simulator instance that later produces the observed state. That comparison is valuable for deterministic software-consistency testing, but it is not an independent test of predictive accuracy.

This benchmark therefore separates three worlds:

| Component | Owns | Must not receive |
|---|---|---|
| Observer model | Current model parameters, history, prediction and uncertainty | Hidden plant parameters, future states, evaluator labels |
| Plant model | Independently implemented/parameterized evolution and disturbances | Observer predictions, candidate decisions |
| Evaluator oracle | Ground-truth plant state, hidden parameters, fault schedule, scoring | Candidate control over truth or scoring |

A passing software fixture demonstrates the interface contract only. A simulator result requires a separately pinned simulator and independent reset/action/state receipts. HIL and physical evidence are later tiers.

## 2. Paradigm shift: discrepancy is structured information

Do not treat all prediction residuals as white noise or immediately label them anomalies.

Decompose discrepancies into:

- parametric mismatch (inertia, mass, damping, friction, gravity);
- structural mismatch (missing backlash, compliance, contact or coupled dynamics);
- actuator mismatch (saturation, deadband, delay, missed/held commands);
- sensor mismatch (bias, noise, dropout, stale values, variable latency);
- context shift (different robot, tool, payload, temperature, surface, task);
- evidence mismatch (clock domain, out-of-order arrival, frame/unit/profile mismatch).

The report should preserve the observed discrepancy, its channel and source, forecast horizon, prior model profile, uncertainty and evidence cut. Any explanation such as "payload likely changed" remains a hypothesis until an independent observation identifies it.

## 3. Initial seven-DOF plant model

The initial reference lane uses a seven-joint manipulator state and torque/velocity control interface. It is a small, transparent process model intended for mutation and known-answer tests, not a high-fidelity robot digital twin.

A simplified per-joint model may be expressed as:

\[
M(q)\ddot q + C(q,\dot q)\dot q + g(q) + \tau_f(\dot q,T) =
\tau_{act} + \tau_{ext}
\]

The observer and plant must not share the same parameter object, integrator implementation or seeded disturbance generator.

### Plant-side parameters (hidden from candidate)

- link/joint inertia matrix or bounded diagonal approximation;
- viscous and Coulomb friction;
- gravity scaling;
- payload mass / centre-of-mass;
- backlash / compliance;
- actuation delay / saturation;
- external force/contact schedule;
- sensor bias, noise, dropout and delivery delay;
- clock-domain offset/drift.

The first independent plant can be a separately written scalar/coupled joint integrator with explicit parameter manifests. A future simulator bridge can add MuJoCo or another simulator after reset/action/effect receipts are independently available.

### Observer-side model

Begin with reproducible baselines and only then introduce learned dynamics:

- persistence;
- constant velocity and constant acceleration;
- AR/NARX;
- linear state-space and Kalman/LQG variants;
- reservoir / next-generation reservoir computing;
- GRU/LSTM, TCN and state-space sequence model;
- CfC/LTC temporal predictor;
- HDC-only representation (not a predictor);
- HDC + CfC/LTC;
- physics + learned residual hybrid.

HDC is used for a typed trajectory representation and similarity features. Numeric state, physical units, forecast means, covariance/intervals and quality flags remain typed and independently scored.

## 4. Mismatch campaign

Use a factorial seed schedule, with isolated single-factor tests before compound shifts. Do not train on the exact test schedules.

| Family | Training range / nominal | Held-out test shift |
|---|---|---|
| Inertia | nominal ±10% | ±25–40%, coupled-joint change |
| Payload | 0–1 kg | 2–3 kg and shifted centre of mass |
| Friction | fixed nominal Coulomb/viscous | temperature drift, stick-slip and surface change |
| Gravity | nominal | 5–15% effective scaling change |
| Backlash/compliance | disabled or small | nonzero dead zone, flex and hysteresis |
| Actuation | nominal control period | 1–5 step delay, saturation and held command |
| External forces | bounded isolated pushes | contact transition and compound disturbance |
| Sensing | low noise, in-order arrival | bias, dropout bursts, variable latency and delayed/reordered delivery |
| Clocking | one stable clock | offset, drift, reset and different acquisition/ingest clocks |
| Composition | isolated shifts | unseen pairs/triples, e.g. payload + friction + delay |

The hidden evaluator schedule is not serialized into candidate-visible artifacts. Candidate inputs bind to a decision cut and raw observation digests.

## 5. Calibration and selective prediction

Prediction error alone cannot establish reliable uncertainty.

For probabilistic/interval predictors:
- calibrate on a separate calibration partition;
- report empirical coverage and interval width by joint, channel, horizon and shift family;
- use proper scoring rules (CRPS or negative log-likelihood) only for a genuinely defined predictive distribution;
- show reliability plots and calibration drift after a shift;
- report risk-versus-coverage curves for abstention;
- retain out-of-domain and unknown samples in denominators.

Do not turn HDC cosine similarity, bits-saved diagnostics, raw prediction error or Mahalanobis distance into a probability without independent held-out calibration.

The observer may return a forecast with explicit uncalibrated status or abstain. It cannot authorize, dispatch, retry, fence, adjudicate Applied/NoEffect or modify evidence.

## 6. Metrics

### Forecast quality
- joint-angle, joint-velocity, end-effector and force prediction error by horizon;
- normalized RMSE/MAE as descriptive measures;
- CRPS/NLL and Brier only where outcome distributions/probabilities are actually produced and calibrated;
- interval coverage and width by operating regime.

### Selective and OOD quality
- risk-coverage curve;
- false confident forecast rate under held-out shift;
- abstention coverage and selective risk;
- OOD AUROC/AUPRC and false-alarm rate at frozen operating points;
- time-to-detect, lead time and detection delay.

### Temporal and operational quality
- source-capture vs receiver-ingest order handling;
- late-arrival count and source-clock regression false positives;
- p50/p95/p99 inference latency;
- CPU/GPU, memory, bytes, observation count and measured energy;
- adaptation cost, recalibration samples and post-shift stabilization time.

Do not collapse safety, prediction quality, calibration and runtime into one weighted score.

## 7. Required controls

- Same-model clone residual: explicit software-consistency control only.
- Oracle parameter model: non-deployable upper bound.
- Persistence and constant velocity: low-compute floor.
- Metadata-only classifier: shortcut leakage control.
- Shuffled temporal order: test whether chronology carries real predictive value.
- No-HDC / no-temporal-model ablations.
- Disturbance-only and sensor-only shifts.
- Compound-shift holdout.
- Delayed-arrival mutation: changing ingest order without changing source sequence must not alter source-ordered predictions.
- Future-leak mutation: any observation after the decision cut must be rejected.
- Profile substitution and unit/frame mismatch must be rejected or abstained.

## 8. Split and statistical protocol

- Partition by source trajectory, task, robot/profile, parameter family and disturbance schedule.
- Keep every timestep from one rollout in a single split.
- Reserve entire robot/controller/profile identities and compound-shift families for holdout.
- Freeze calibration data separately from training and final evaluation.
- Use paired episode inputs across baseline models; resample whole scenario/trajectory clusters.
- Report exact denominators for scheduled, valid, stale, reordered, disputed, unknown and scored episodes.
- Keep claims descriptive until the number of independent clusters and coverage of the estimator justify inferential statements.
- Publish negative/null results and failed operating regimes.

## 9. Target ladder

Proposed targets; not validated guarantees:

1. 100% of observer inputs are source/digest/profile/decision-cut bound.
2. Zero acceptance of post-cut, wrong-unit, wrong-frame, duplicate source-sequence or non-finite measurements in known-answer tests.
3. Delayed/reordered packets are retained and ordered by source sequence; no false clock regression solely from late arrival.
4. On nominal held-out cases, beat persistence and constant-velocity on preregistered multi-step forecast metrics.
5. On shifted cases, reduce confident-error rate versus uncalibrated neural-only baselines while reporting any increase in abstention.
6. Achieve empirical 90% prediction-interval coverage close to nominal on the held-out calibration protocol, with interval width reported; do not claim universal coverage.
7. Show an observer/forecast Pareto frontier against compute, latency, memory and energy.
8. Maintain separate reporting for process-model, simulator, HIL and physical tiers.

## 10. Evidence tiers and claim boundary

- Tier 0: deterministic known-answer and mutation tests.
- Tier 1: independent process-model plant/observer campaign.
- Tier 2: simulator with separate reset/action/state receipts and pinned model assets.
- Tier 3: HIL with measured actuator/sensor path and independent clock/fault injection evidence.
- Tier 4: physical robot with independent effect observations and approved risk controls.

This v0.1 document is an experiment plan. It does not claim a validated predictive model, physical dynamics accuracy, simulator qualification, safety certification or SOTA improvement.
