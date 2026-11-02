# Hidden-state active-sensing policy simulator v0.1

Tracking: [#9844](https://github.com/Luminous-Dynamics/symthaea/issues/9844)  
Implementation branch: `research/edge-active-sensing-sim-v0.1`  
Parent: [decision-focused active-sensing planner PR #9821](https://github.com/Luminous-Dynamics/symthaea/pull/9821)

## Research question

Can an embedded policy jointly choose **whether to observe, which sensor to activate, when to wake next, and when to alarm** while maintaining a fixed event-quality target under sensor drift and scarce resources?

A model-only benchmark cannot answer this. A static model is evaluated on an observation stream supplied by the benchmark. An active system changes the observation stream it receives. We need to score the causal sensing policy as well as the classifier.

This direction is especially important after BoTTA (SenSys 2026), which evaluates periodic on-device test-time adaptation under few target samples, incomplete category exposure, diverse shifts and compound shifts, and reports system costs on Raspberry Pi 4B and Jetson Orin Nano. Adaptation overhead is not incidental: published SHOT results report material accuracy gains alongside higher memory, latency and energy. See [BoTTA](https://doi.org/10.1145/3774906.3800498).

## Candidate-facing contract

The policy implements `ActiveSensingPolicy::decide(&PublicObservation) -> PolicyAction`.

The public observation contains:
- current simulated step;
- most recent sensor measurement, if any;
- age of that measurement;
- total sample count and per-sensor sample counts.

The action contains:
- an alarm decision for the current step;
- optional LowPower or HighFidelity sample acquisition at the current step;
- next wake interval, bounded to 1–1,024 steps;
- candidate-declared compute-cycle, transmit-byte and memory-write counters.

The evaluator freezes an application decision deadline in the scenario. It is not passed to the candidate policy; the result distinguishes on-time from late detections.

The candidate-facing types contain no scenario seed, hidden event schedule, current hidden label, shift schedule, event count, or evaluator metrics. The scenario is supplied only to the simulator entry point and is not passed to policy code.

A policy receives the result of a measurement only at its next invocation. Thus sampling at step (t) does not let the policy decide using that measurement at the same step.

## Synthetic environment

The deterministic hidden event process produces non-overlapping event intervals with a configured event-start probability and duration range. Stable, abrupt sensor-error, gradual sensor-error, and compound sensor-error/event-rate shifts are supported.

Two synthetic sensor modes are defined:
- **LowPower:** 20 nJ modeled sample energy; 8% false-positive and 14% false-negative base rates.
- **HighFidelity:** 100 nJ modeled sample energy; 2.5% false-positive and 4% false-negative base rates.

Each policy invocation incurs 10 nJ of modeled wake energy, including a wake that elects not to sample. Shift profiles increase the sensor error rates; a compound shift also changes event frequency. All these costs and rates are synthetic fixture parameters, not measured hardware facts.

Included baselines:
- fixed-rate LowPower sampling;
- fixed-rate HighFidelity sampling;
- a periodic confirmation policy that uses LowPower while quiet and requests a HighFidelity sample after a positive LowPower sample.

The adaptive confirmation policy is deliberately modest. It is a control baseline, not a SOTA algorithm and not a substitute for BoTTA implementations.

## Evaluator metrics and receipts

Each episode retains:
- event count, detected events, on-time detections, late detections and missed events;
- all-event recall and on-time recall under the frozen application deadline;
- false alarms and duplicate alarms;
- event recall and alarm precision with explicit denominators;
- false alarms per 1,000 simulated steps;
- all detected-event delays and nearest-rank p95 delay;
- sample counts by sensor and wake count;
- separate synthetic sensor-acquisition and wake energy;
- combined modeled sensor-plus-wake energy per on-time detected event. This deliberately excludes compute, radio and memory energy;
- candidate-declared compute cycles, transmitted bytes and memory-write bytes;
- scenario, policy, causal trace and result SHA-256 commitments.

A result validator recomputes derived metrics from retained counts/delays and rejects summary tampering. The trace commitment covers each policy action and the acquired binary sensor result. The scenario commitment binds the seed and shift parameters for evaluator-side reproducibility.

## Comparison protocol

Every policy must receive:
- the same hidden episode seeds and scenario distribution;
- the same observation schema and legal action bounds;
- the same sensor models, event definitions and deadline semantics;
- the same calibration data and update budget where adaptation is evaluated;
- the same event-quality threshold and false-alarm exposure.

Required policy families for the first confirmatory run:
1. fixed-rate LowPower and HighFidelity;
2. threshold/EMA or hysteresis;
3. periodic test-time adaptation baselines compatible with the task;
4. entropy-driven sensing;
5. decision-focused expected value-of-information;
6. HDC-only, CfC-only and HDC-LTC policies;
7. a matched random or budget-matched sensing control;
8. an evaluator-only oracle ceiling, never passed to candidate code.

Report each scenario and subgroup separately. Do not collapse quality, latency, samples, energy, memory and communication into one score unless an application-specific utility function was preregistered. A recommended synthetic Pareto view is on-time event recall / alarm precision / p95 detection delay against sensor samples, wakes, and modeled sensor-plus-wake energy. A physical System-track Pareto view must instead use the independently measured complete application-event energy receipt.

## Claim and security ceilings

- The environment is synthetic and its sensor/energy values are modeled. It does not establish physical energy efficiency, battery life or superiority to TinyML.
- Candidate-declared compute cycles, transmit bytes and memory writes are not trusted as physical measurements.
- The Rust trait and module privacy are **not an adversarial security boundary**. A hostile candidate in the same process may exploit runtime or memory behavior. Confirmatory evaluation still requires OS-level process separation, candidate-only public inputs, evaluator-only oracle access, restricted IPC/filesystems, and an independently replayable evaluator receipt.
- The result digest is an integrity commitment, not a signature, hardware attestation, trusted timestamp or proof of honest acquisition.
- Synthetic policy simulation does not replace real HARTH/MIMII/SHD runs or physical policy-controlled sensing. High-rate trace replay must not claim to reveal measurements that were never acquired.

## Authored tests

The module authors tests for deterministic replay, paired sensor-noise consistency, sensor-cost monotonicity, sleeping without acquisition, adaptive-policy denominator integrity, on-time versus late event qualification, action-bound rejection, result-tamper detection including overflowing forged counters, scenario commitment changes, and invalid shift bounds.

**Execution status:** eleven tests are authored but have not been compiled or run. No performance or SOTA result is claimed.
