# PLASMA-DIAG-RECOVERY-001 — Synthetic diagnostic recovery under sensor faults

**Status:** Research benchmark contract v0.1; no measured result or solver validation claimed.  
**Issue:** [#12664](https://github.com/Luminous-Dynamics/symthaea/issues/12664)  
**Date:** 2026-11-17  
**Owners:** Symthaea (inference/evaluation); Symtropy (scenario, fault injection and replay)

## 1. Executive thesis

Move from filling missing values to **qualified diagnostic continuity**.

A reconstructed diagnostic channel can help interpret a plasma event, but it is not an instrument measurement. Every reconstructed sample must retain its origin, uncertainty, validity envelope and claim ceiling. The model must abstain when the remaining diagnostics do not make the requested quantity observable.

This benchmark complements:
- Symthaea [#12627 — measured-data evaluation on MIT PSFC DisruptionBench](https://github.com/Luminous-Dynamics/symthaea/issues/12627);
- Symthaea [#7665 — event-level multi-machine disruption benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7665);
- Symthaea [#7588 — source-pinned cold-plasma V&V and design discovery](https://github.com/Luminous-Dynamics/symthaea/issues/7588);
- Symtropy [#1685 — solver-neutral plasma scenario contract](https://github.com/Luminous-Dynamics/symtropy/issues/1685) and [#1703 — run receipts and capability gates](https://github.com/Luminous-Dynamics/symtropy/issues/1703).

It does not replace the physics authority of specialist solvers or measured diagnostics.

## 2. Industry and research anchor

DOE Office of Science's 2026 DIII-D highlight describes Diag2Diag, a cross-diagnostic AI model that reconstructs higher-resolution synthetic Thomson-scattering measurements and captures edge-localized-mode dynamics:
https://science.osti.gov/fes/Highlights/2026/6a

This is a useful research comparator, not yet a complete reproducible benchmark specification for this project. Before claiming reproduction, pin the underlying paper, source revision, dataset, case definitions, preprocessing, instrument uncertainty and usage rights.

The MIT PSFC DisruptionBench paper covers Alcator C-Mod, DIII-D and EAST across nine tasks and zero-shot/few-shot/many-shot transfer:
https://doi.org/10.1007/s10894-025-00495-2

Its reported AUC values are useful contextual comparators (CCNN up to 0.97 intra-machine on C-Mod; strongest reported transfer AUC 0.83 / F2 0.62), but are not diagnostic-reconstruction scores and must not be compared directly with this benchmark's metrics.

## 3. Evidence and state model

Keep these states distinct:

- `MEASURED`: value came from the named diagnostic and acquisition chain.
- `SYNTHETIC_DIAGNOSTIC`: value was inferred or reconstructed by a named model.
- `UNAVAILABLE`: no value is available.
- `INVALID`: a value exists but fails declared integrity, calibration, unit or timebase checks.
- `NOT_OBSERVABLE`: available inputs do not support the requested inference under the declared model profile.
- `OUT_OF_PROFILE`: scenario exceeds the model's qualified envelope.
- `ABSTAINED`: the model declines to produce a reconstruction.

A reconstructed record should bind:
- diagnostic/channel ID and physical quantity/unit;
- original source channels used, including their measured/synthetic/unavailable states;
- shot, machine, campaign, acquisition clock and exact time window;
- calibration identity, sensor profile and preprocessing revision;
- model revision, parameters, training-data identity and applicability envelope;
- point estimate plus predictive interval/distribution where supported;
- output digest, input digest and derivation lineage;
- missingness/fault class and reason for abstention, where applicable;
- evidence class and a non-promotable claim ceiling.

A reconstructed value must never be silently serialized or rendered as `MEASURED`. Synthetic diagnostic output must not become a protection input merely because a downstream schema accepts a number.

## 4. Ownership split

### Symtropy

Reuse its existing solver-neutral plasma scenario and run-receipt work. Own:
- immutable scenario identity and declared regime profile;
- deterministic fault schedules and timebase perturbations;
- diagnostic availability states and scenario replay;
- run receipts, raw artifact references, hashes and terminal dispositions;
- synthetic-vs-measured projection boundaries.

Symtropy's rigid-body engine is not a plasma solver. Fault injection is a test stimulus, not an estimate of real instrument failure rates.

### Symthaea

Own:
- cross-diagnostic reconstruction/inference;
- HDC/CfC and conventional model adapters;
- temporal alignment and uncertainty estimation;
- observability and out-of-distribution detection;
- event-preservation and downstream-utility evaluation.

Reuse existing statistics, split-integrity, calibration, experiment-design and evidence owners. Do not create a second generic data platform, units framework or uncertainty library.

## 5. Benchmark tasks

1. **Withheld-channel reconstruction:** hide a recorded diagnostic channel, reconstruct it from declared companion channels, and compare only against the withheld measured values.
2. **Temporal super-resolution:** downsample a channel, reconstruct at a finer temporal resolution, and measure amplitude, phase, event-onset and waveform fidelity.
3. **Contiguous dropout:** remove complete time blocks rather than only independent random samples.
4. **Correlated diagnostic failure:** degrade channels that share a diagnostic system, calibration, clock or environmental exposure.
5. **Calibration drift and bias:** inject declared drift, offset and gain errors with separate training and evaluation fault families.
6. **Acquisition faults:** test timebase shifts, jitter, irregular sampling, latency, saturation, aliasing and stale calibration.
7. **Cross-machine transfer:** hold out complete machines/campaigns. Compare only observables with genuinely compatible definitions and units.
8. **Event preservation:** test whether reconstructed signals preserve independently labelled event onset, amplitude, duration and ordering.
9. **Downstream utility:** compare disruption prediction with measured channels only, channels masked, channels reconstructed, and model abstention.

Do not infer a missing physical quantity merely because a neural encoder can produce a vector for it. Observability is a property of the model, measurements, operating regime and question.

## 6. Baselines

All methods receive the same information, windows, split and compute/evaluation budget.

- Last valid observation / nearest valid sample.
- Linear interpolation and domain-appropriate smoothing.
- State-space / Kalman-style reconstruction when its assumptions fit the signal.
- Random Forest or other conventional tabular baseline.
- Temporal convolution / CCNN or another reproducible sequence baseline.
- A calibrated temporal neural baseline with identical input channels.
- A Diag2Diag-compatible comparator if its method, data and evaluation can be reproduced.
- Symthaea HDC/CfC.

Retrospective interpolation and causal online reconstruction are separate tasks. A method using future samples cannot be credited with a live causal reconstruction result.

## 7. Splits, leakage and uncertainty

- Split by complete discharge/shot. All windows from a shot remain in one partition.
- Evaluate campaign and machine transfer separately from within-machine interpolation.
- Fit normalizers, encoders, thresholds, feature selection and uncertainty calibration on training/validation only.
- Freeze the final test set and reveal it only after model and threshold selection.
- Keep time-causal and retrospective modes explicitly separate.
- Include missing-not-at-random and correlated failure. Independent random dropout alone is inadequate.
- Keep instrument repeatability, calibration uncertainty, sensor noise, model uncertainty and temporal alignment error separate.
- If a reference value is unavailable or the instrument uncertainty is not known, mark the comparison unscoreable; do not treat missing uncertainty as zero.
- Keep measured raw payloads immutable and rights-cleared. A public data-access helper does not grant rights to redistribute raw measurements.

## 8. Metrics

Report separately by machine, campaign, channel, event and failure mode:

- MAE, RMSE and a predeclared normalized error;
- error divided by instrument repeatability/uncertainty where available;
- event onset/peak/duration and temporal-lag error;
- waveform, spectral and phase fidelity only where the observable supports those analyses;
- predictive interval coverage, interval width and proper distributional scores;
- false reconstruction of events not present in withheld measurements;
- fault-detection sensitivity, specificity and time to detect lost observability;
- abstention and scoreability rates;
- downstream event recall, false alarms per discharge and per plasma-hour, warning-time distribution and calibration;
- inference latency, memory and compute cost.

Never combine incompatible units or permit one well-reconstructed channel to conceal a poor one through aggregate averaging.

## 9. Frozen adversarial challenge

Include at minimum:
- independent and correlated channel dropout;
- complete diagnostic-system loss;
- clock offset/drift and timestamp reorder;
- stale calibration;
- saturation, stuck-at value, bias and gain shift;
- irregular sampling, duplicated samples and aliasing;
- corrupted units, non-finite values and malformed source identity;
- sensor disagreement beyond registered consistency limits;
- all-inputs-unavailable;
- out-of-profile operating conditions;
- future-window leakage in a causal task;
- train/test shot overlap;
- synthetic-to-measured type promotion;
- changed model revision with a stale output receipt.

Expected dispositions must be frozen before evaluation. A completed reconstruction is not automatically a qualified reconstruction.

## 10. Proposed acceptance gates

The first milestone is integrity and fair comparison, not a headline accuracy score.

- 100% of injected integrity faults in the finite frozen corpus cause refusal or an explicit non-success disposition.
- Zero synthetic diagnostic values are promoted to measured observations in serialization, replay or reports.
- Zero shot identity overlap between training, validation and test.
- Zero future-window leakage in the causal task.
- Prediction-interval coverage is reported against registered nominal levels with sample counts and uncertainty; uncalibrated intervals cannot pass.
- Reconstruction must outperform the strongest admissible interpolation/state-space baseline on a preregistered held-out task without worsening registered calibration or event-timing constraints.
- Downstream utility is compared on identical held-out discharges and operational alarm logic.
- Every attempted case, including failed, non-converged, unscoreable and abstained cases, has a terminal disposition.
- Exact model/data/config identity and raw-output hashes are replayable.

Any 95% TPR-at-5%-FPR target belongs to the separate disruption-prediction task, not the diagnostic-reconstruction task.

## 11. Staged execution

**S0 — Source and rights census**
Pin Diag2Diag primary sources, DisruptionBench commit and data task definitions. Confirm observables, instrument uncertainty, timebase and redistribution rights.

**S1 — Contract and synthetic fixtures**
Create a small, explicitly synthetic, rights-cleared fixture with measured/synthetic/unavailable/invalid states and deterministic fault schedules. Implement the refusal corpus.

**S2 — Baselines**
Run interpolation and state-space baselines on a frozen withheld-channel task before training new models.

**S3 — Symthaea evaluation**
Add HDC/CfC and matched temporal baselines. Report model uncertainty, OOD detection and abstention.

**S4 — Event and downstream evaluation**
Test event preservation and downstream disruption prediction on identical shot-level holdouts.

**S5 — Cross-machine and diagnostic transfer**
Only after compatible observables, data rights and machine-specific calibration have been qualified.

## 12. Claim ceiling

A PASS may establish only that the registered software contract, benchmark split, replay, output lineage and finite adversarial fixtures passed. It does not establish instrument equivalence, hidden-state observability, physical truth, diagnostic replacement, fusion controllability, protection-system fitness, safety certification or hardware readiness.

A model-generated signal remains synthetic, even when its numerical error is small.
