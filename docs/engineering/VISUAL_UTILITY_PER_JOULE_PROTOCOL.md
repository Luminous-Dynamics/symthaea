# Task-Conditioned Visual Utility per Joule

**Status:** research protocol, not a measured result  
**Issue:** [VIS-ENERGY-001 #7878](https://github.com/Luminous-Dynamics/symthaea/issues/7878)  
**Parent workload:** [VIS-BASE-001 #7853](https://github.com/Luminous-Dynamics/symthaea/issues/7853)  
**Program:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)

## 1. Research thesis

Optimize a display as a task-conditioned visual-information system, not as a contest to maximize a single headline specification. A system succeeds when it delivers correct, timely, readable information under declared viewing conditions while minimizing energy, latency, heat, cost, and waste.

This hypothesis is falsifiable: compare one bounded adaptive strategy against a fixed-output baseline under matched conditions; independently measure power and task outcomes; preserve minimum optical/perceptual constraints. If adaptation does not improve the measured energy/performance Pareto frontier, reject or narrow the strategy.

This protocol extends the existing conventional-display corpus. It does **not** create another general-purpose corpus, perceptual scoring engine, power-measurement stack, or optimizer.

## 2. Do not let one score hide a regression

Publish a vector of results before any aggregate ratio:

- task completion and error rates;
- response latency distribution (at least p50, p95 and p99);
- quality-floor measurements appropriate to the workload (e.g. readability, contrast/MTF, color/tone error, motion artifacts, clipping, flicker);
- display-only energy, rendering/GPU energy where measurable, and whole-system energy;
- temperature/thermal stability, calibration status and instrumentation uncertainty;
- robustness by predeclared ambient-light, content, viewing-angle and task strata.

The Pareto frontier is primary. Joules per successful task or a task-utility-per-joule ratio is a secondary, task-specific summary. Do not compare the ratios across unlike tasks without an explicit normalization and rationale. Never allow extra energy savings to compensate for a mandatory readability, color, motion, latency, accessibility or safety failure.

## 3. Measurement definitions

For each run, report these separately:

1. **Gross system energy:** integral of measured system power over the task window.
2. **Idle-subtracted energy:** gross energy less a separately measured idle baseline under the same setup; keep the gross figure so the subtraction is auditable.
3. **Display-rail energy:** only when a suitably calibrated instrument and safe measurement point exist.
4. **Rendering energy/time:** GPU or rendering subsystem telemetry with its source, update rate, and known limits recorded.
5. **Task success:** count of valid completed tasks according to a preregistered rubric; ambiguous/aborted trials are reported, not silently dropped.
6. **Energy per successful task:** task-window energy divided by successful-task count. If the denominator is zero, return undefined/not evaluated; do not smooth, impute or cherry-pick the denominator.
7. **Task non-inferiority:** compare raw success/error outcomes and confidence intervals against the fixed baseline. A ratio alone is insufficient.

Record the power instrument make/model/serial or stable ID, calibration record/date, accuracy range, sample rate, missing samples, clock synchronization, unit, sensor location, integration interval, and raw-trace digest. Store both raw and transformed traces when rights and size permit.

## 4. Matched-condition experiment

A valid A/B pair holds constant—or explicitly records any difference in—device and firmware, panel size, refresh rate, resolution, color/profile configuration, target luminance, content/task, viewing geometry, ambient illumination, run duration, thermal start state, operating-system power policy, and background workload.

Randomize or counterbalance run order. Include warm-up and repeatability runs. Use the same physical device for within-device comparisons; a cross-device comparison is a separate stratum, not an implicit equivalence. Calibrate optical measurement instruments and record calibration state before and after each experimental block.

Start with one display and a small, fixed subset of VIS-BASE workloads. Verify measurement repeatability on known loads before testing adaptive behavior. Human-subject claims require an appropriately reviewed protocol, informed participation where applicable, preregistered endpoints and randomized/blinded ordering. Synthetic and software-simulated outcomes must be labelled as such and cannot establish human perceptual non-inferiority.

## 5. Comparison ladder

Use the simplest valid baseline first, then add complexity one factor at a time:

- **B0 — fixed-output reference:** frozen display profile and fixed workload, with all power and optical settings recorded.
- **B1 — bounded content/region adaptation:** change only a declared luminance/local-dimming or region-of-interest policy, constrained by fixed quality floors.
- **B2 — device-aware rendering adaptation:** include the measured rendering/GPU work only where it can be isolated and compared reproducibly.
- **B3 — gaze-conditioned/foveated strategy:** only for hardware that supports it; add gaze failure, prediction error, calibration drift and tail-latency controls. A failed or uncertain gaze signal must trigger the declared safe fallback.

Do not combine B1–B3 into one treatment initially. Retain the unadapted fallback and include it as a negative-control condition.

## 6. Proposed target gates

The starting research targets are:

| Dimension | Proposed gate | Required interpretation |
|---|---|---|
| Energy | At least 10% less measured display-plus-rendering energy per successful task versus B0 | Report absolute joules, gross and idle-subtracted energy, instrument uncertainty and per-stratum results |
| Task outcome | No more than 1 percentage-point absolute drop in task success | Preregister test and confidence interval; an underpowered study is inconclusive, not a pass |
| Tail latency | No more than 5% regression in p95 task/render latency | Report p50/p95/p99, missed frames and deadline violations |
| Optical/perceptual floors | Zero mandatory-floor violations | No scalar score or energy gain can offset a floor failure |
| Integrity | 100% exact identity/trace/provenance fields present | Missing profile, calibration, denominator or raw trace means NOT EVALUATED |
| Robustness | Publish worst predeclared stratum and Pareto alternatives | No best-case-only headline |

These figures are hypotheses for preregistration, not measured capability or established universal standards. Amendments must be justified before examining the affected results. Any deviation from a proposed gate must be prominent in the report.

## 7. Mandatory negative controls

The benchmark must deliberately test:

- dropped, duplicated, delayed, clipped or unit-mismatched power samples;
- energy logs from a different device/profile/configuration;
- stale or absent photometric calibration;
- denominator manipulation, zero successful tasks, silently excluded trials and mismatched task-window boundaries;
- hidden optical-quality regression despite a lower energy total;
- profile/firmware change between baseline and treatment;
- thermal drift, background workload and incomplete warm-up;
- invalid or missing gaze signals (for B3);
- cherry-picked ambient-light/task strata and post-hoc threshold changes;
- missing raw trace or changed trace digest.

Each negative must either be rejected or explicitly marked NOT EVALUATED. Unknown evidence may not be imputed into a pass. The independent verifier should recompute energy integrals and per-task ratios from raw traces, rather than trusting the candidate strategy's own report.

## 8. Decision-aware experiment selection

Connect to [VIS-EXP-001 #7871](https://github.com/Luminous-Dynamics/symthaea/issues/7871) and the shared selector benchmark. Prefer the next measurement with highest expected decision value per cost, not merely highest model uncertainty. Examples include validating power meter bias, measuring a stratum near an optical-floor boundary, or collecting a repeated run that could reverse a Pareto ranking. Permit abstention when no affordable measurement can change the decision.

## 9. First reproducible milestone

1. Select one physical display and write its immutable profile/device record.
2. Select a small subset of VIS-BASE workloads and freeze the task rubric and exclusion rules.
3. Qualify the power-measurement method on known loads; inject a missing-sample negative control.
4. Capture repeated B0 runs and test repeatability, clock alignment and uncertainty.
5. Implement one bounded adaptive treatment and retain B0 as the control.
6. Recompute outcomes with an independent analysis; publish raw/derived digests, failures and uncertainty.
7. Report all Pareto dimensions and each gate. If instrumentation or human-study capacity is unavailable, stop at software/synthetic protocol tests and leave the physical claim NOT EVALUATED.

## 10. Evidence boundary

This document defines how a future study should be run. It contains no measured power saving, physical display result, human-subject result, or claim that adaptive output is universally better. Claims must be supported by exact device/configuration identity, calibrated instruments, raw traces, preregistered tasks, uncertainty estimates, and independent review.
