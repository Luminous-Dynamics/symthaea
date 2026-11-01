# Edge bridge for the HDC-LTC multi-timescale temporal competence suite v0.1

Tracking: #9804  
Implementation branch: `research/edge-temporal-bridge-v0.1`  
Parent temporal suite: [PR #9395](https://github.com/Luminous-Dynamics/symthaea/pull/9395)  
Related TinyML track: [PR #9778](https://github.com/Luminous-Dynamics/symthaea/pull/9778)

## Purpose

This bridge connects the ten preregistered temporal task families (T1–T10) to a consistent algorithm/system evaluation contract. It does not duplicate or replace the task generators in `research/temporal-bench`, and it does not modify Symthaea's neuron or model implementation.

The adapter separates three artifacts:

1. **Temporal benchmark manifest:** frozen task registry, task-specific metrics, baseline family requirements, candidate identities, source revisions, dependency-lock digests, source-family splits, and optional system target.
2. **Candidate output receipt:** candidate-visible public task digest, candidate artifact/configuration, prediction/output digest, action/model-state trace digests, and algorithm-level memory accounting. It contains no oracle digest or labels.
3. **Evaluator receipt:** candidate output digest, private oracle commitment, evaluator-code digest and task-specific metric observations with denominators/strata.

A fourth **system measurement receipt** is required for the System track. It binds the candidate and evaluator receipts to the exact board/core/accelerator, compiler, firmware, linker map, device session, raw latency samples, SRAM/flash, complete application-event energy and energy-uncertainty receipt.

The candidate receipt and evaluator receipt are different typed/hash domains. This is an API/data-contract separation, **not a sandbox**. Confirmatory qualification must run candidate code separately from evaluator code, with separate OS-level file/IPC permissions and an independently replayable evaluator receipt. The current temporal-suite prototype generators intentionally return both task and oracle in one process; that is acceptable for deterministic unit tests, not sealed confirmatory evaluation.

## Algorithm track and System track

### Algorithm track

Measures task capability and temporal integrity under a matched public stream:
- exact task metric and declared stratum;
- delay- and interference-conditioned outcomes;
- source-family disjoint development/calibration/confirmatory splits;
- persistent state, peak working set and memory read/write bytes;
- causal action/model-state traces;
- deterministic public task/output/evaluator digests.

It is hardware-independent. Do not label its modeled operation counts or state complexity as measured latency or power.

### System track

Measures the same task and candidate on a named physical target:
- board revision, CPU core and accelerator profile;
- compiler/runtime, firmware, linker map and device session;
- p50/p95/p99 recomputed from raw per-event latency samples;
- peak SRAM after a protected reserve, flash and p95 deadline;
- complete application-event energy per successful deadline-compliant event;
- successful-event count, energy replicate count, expanded uncertainty, uncertainty receipt digest and raw power-trace digest.

CPU-only, DSP/MVE, integrated signal-processor, NPU, heterogeneous and neuromorphic hardware tiers remain separate. An Algorithm-track result cannot be promoted to System status without a fresh physical-system receipt.

## T1–T10 registry

| Task | Core capability | Required metric families | Primary comparator families |
|---|---|---|---|
| T1 sparse event | Detect rare events and time decisions with long idle listening | target precision/recall, false alarms/hour, p95 decision latency, duplicates | threshold/EMA, static/adaptive TinyML, TCN, GRU, adaptive LIF, HDC/CfC |
| T2 delayed cue | Associative recall under delays and distractor similarity | overall/worst-delay accuracy, recall latency | GRU, LSTM, associative memory, HDC, HDC-LTC |
| T3 multi-timescale | Joint fast/medium/slow state and forecasting | state accuracy, forecast MAE, worst-timescale accuracy | GRU/LSTM, TCN, ODE-RNN/latent ODE, CfC, HDC-LTC |
| T4 online drift | Causal change detection, adaptation and retention | detection/recovery delay, false changes, prior-regime retention, regret | threshold/EMA, adaptive TinyML, GRU/LSTM/GRU-D, BrainTrace D-RTRL/pp-prop/e-prop, confidence-gated HDC-LTC |
| T5 irregular/missing | Irregular-time update semantics and missing data | accuracy by gap/missingness, invalid-input acceptance | adaptive TinyML, GRU-D, ODE-RNN/latent ODE, CfC, HDC-LTC |
| T6 retention/invalidation | Interference, selective retention and stale-memory rejection | valid recall before invalidation, stale recall after, invalidation response latency | adaptive TinyML, key/value overwrite, recency decay, associative memory, bitemporal ledger, HDC |
| T7 phase prediction | Phase/frequency tracking and crossing prediction | phase error, crossing-time error, prediction coverage | GRU/LSTM, TCN, ODE-RNN, CfC, HDC-LTC |
| T8 delayed match | Delayed match-to-sample and temporal credit | overall/worst-delay accuracy, invalid predictions | GRU/LSTM, adaptive LIF, BrainTrace D-RTRL/pp-prop/e-prop, HDC-LTC |
| T9 evidence accumulation | Sequential accumulation and delayed decisions | accuracy, false alarms, decision latency | BPTT, adaptive LIF, BrainTrace pp-prop/e-prop, HDC-LTC |
| T10 bitemporal knowledge | Exact valid-time/known-time corrections and retractions | exact bitemporal accuracy, future-knowledge leakage, stale use, historical as-known accuracy | key/value overwrite, recency decay, bitemporal ledger, RAG, HDC-LTC |

The required comparator families are task-specific. A baseline that is semantically inapplicable must be marked not applicable with a reason in the task run manifest; it must not receive a fabricated zero score. For T4, BrainTrace methods are online-learning comparators, not interchangeable model architectures. STDL is included where its spatial/temporal decoupled training semantics fit the matched vision/event task; it is not forced into every scalar time-series task.

## Metric contract

Metrics are not pooled into one hidden weighted score. The registry declares each metric's unit and maximize/minimize direction. Every required metric must have an overall `all` stratum plus any preregistered subgroup strata (delay, similarity, missingness, gap, regime, valid_at/known_at, etc.). The adapter validates:
- metric ID is allowed for the exact task;
- a positive sample denominator exists;
- the overall stratum is present;
- parts-per-million metrics remain within 0–1,000,000;
- metric observations are unique and sorted by ID/stratum;
- evaluator receipt binds the candidate receipt, oracle commitment and evaluator implementation.

Negative controls should report counts and exposures separately; never report false alarms/hour without the idle-listening exposure that generated them.

## Split and baseline provenance

The frozen manifest requires one assignment for every task × source-family × seed × split role. Source-family identity is the grouping unit. No source family may cross:
- training;
- development;
- calibration;
- confirmatory.

Multiple seeds from a source family remain in the same split. The manifest also pins one exact source revision and dependency-lock digest per candidate family, in addition to candidate artifact/configuration digests. A family name like “GRU” or “BrainTrace” is not sufficient baseline provenance.

The current fixture candidates in unit tests use synthetic digest placeholders. They are not baseline implementation pins and do not constitute executed comparisons. A real benchmark manifest must include exact commits/releases, dependency lock, build/configuration, data pipeline, model precision and training/update budget for every implementation.

## System receipt semantics

Energy in this bridge is **complete application-event energy**, not model-only inference energy. The receipt carries:
- successful, accepted, deadline-compliant event count;
- number of energy replicates (at least two and not more than successful-event observations);
- mean energy per successful event;
- expanded uncertainty;
- independent uncertainty-receipt digest;
- raw power trace digest.

This bridge checks that the digest is present and binds it into the system receipt. A production runner must also independently replay the phase-energy and event-uncertainty receipts, verify the calibration/voltage readback and make sure the measurand is identical for every comparator. It must not substitute the mean of failed attempts for energy per successful event.

The system validator recomputes latency percentiles using nearest-rank, enforces the protected SRAM reserve, flash limit and p95 deadline, and rejects CPU/accelerator target mismatch. It does not invent physical measurements.

## Benchmark programme and SOTA bar

The immediate confirmatory sequence should be:

1. Run the ten synthetic tasks as verifier/causality controls, with all negative controls and source-family split checks.
2. Add the matched strong algorithm baselines. Minimum core set across applicable tasks: GRU, LSTM, TCN, GRU-D, ODE-RNN/latent ODE, adaptive LIF, BrainTrace (D-RTRL and pp-prop/e-prop), STDL where applicable, static/adaptive TinyML, HDC-only, CfC-only and HDC-LTC.
3. Run the Algorithm track on at least 30 paired independent source families/seeds per confirmatory task or use a preregistered power-based alternative. Timesteps and windows are not independent replicates.
4. Add real held-out workload families after licensing/provenance is checked: inertial activity/gesture with subject/session splits; acoustic machine anomaly with device/operating-condition splits; continuous wake-word with long negative listening intervals and false-alarm exposure.
5. Select the strongest Pareto candidates and run the System track on the same MCU family, target, voltage, event contract, success policy, compiler boundary and workload duty cycle.
6. Compare complete-event energy and its expanded uncertainty; retain hardware tiers separately.
7. Report per-task wins and regressions, worst-source-family behaviour, quality, reliability, adaptation/recovery, memory/state complexity, latency distributions and full-event energy.

A SOTA claim requires a matched current baseline, exact pinned code/data, one-shot held-out test, preregistered thresholds, paired uncertainty, no leakage, hard feasibility and independent replay. Paper-reported BrainTrace/STDL numbers remain literature references until independently reproduced.

## Current implementation status

The Rust adapter currently has:
- the T1–T10 task registry, task-specific metrics and baseline family map;
- a strict manifest binding temporal suite commit, candidates, source revisions, dependency-lock digests and target;
- source-family disjoint train/development/calibration/confirmatory split validation;
- candidate output receipts with no oracle field;
- separate evaluator result receipts, pinned oracle/evaluator digests and required per-task overall metric strata;
- a separate system receipt binding candidate/evaluator, target, build materials, device session, raw latency, memory, complete-event energy and uncertainty;
- negative tests for source-family leakage, candidate output tampering, missing task metrics, missing overall strata, ppm range violations, missing baseline families, hardware budget failure and percentile reproduction.

Tests are authored but have not been compiled or executed. No real baseline, dataset, physical system measurement or SOTA score has been run. The task generators themselves remain in the parent temporal suite and still return oracle artifacts alongside tasks for local testing; sealed evaluator isolation remains future execution infrastructure.

## Claim ceiling

- A generator and manifest are not a model benchmark result.
- A family name is not an executed baseline.
- An API-level oracle omission is not process isolation.
- A System receipt is not a measurement unless its raw trace, voltage, calibration, device session and replay are independently checked.
- T1–T10 are a custom temporal competence suite; they are not MLPerf submissions.
- No SOTA improvement is claimed.

## References

- Temporal competence suite / PR #9395: https://github.com/Luminous-Dynamics/symthaea/pull/9395
- TinyML comparison / PR #9778: https://github.com/Luminous-Dynamics/symthaea/pull/9778
- BrainTrace source audit: `docs/research/BRAINTRACE_REFERENCE_AUDIT_V0_1.md`
- STDL source audit: `docs/research/STDL_REFERENCE_AUDIT_V0_1.md`
- MLPerf Tiny v1.4 task reference: https://mlcommons.org/benchmarks/inference-tiny/
- NeuroBench: https://neurobench.ai/
