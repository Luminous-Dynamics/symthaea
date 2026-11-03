# Runtime assurance and safe-stop roadmap

Status: research roadmap, 3 November 2026. This is not a controller implementation, product safety case or permission to stop an actual dishwasher early. Runtime stop authority must remain disabled until the required physical validation is complete.

## Objective

Reduce water, energy and elapsed time by stopping a wash/rinse stage when the required outcome is already likely to be achieved—without increasing false-clean, false-rinse, cross-transfer, drying or hygiene failures.

This is **not** a universal camera detector and not an optimizer that trades cleaning quality for resource savings. It is a runtime-assurance problem: use cheap process observations only within an independently validated envelope, preserve hard endpoint constraints, and fall back safely whenever evidence is incomplete or contradictory.

## The four-state controller contract

Every decision must resolve to one of four states:

1. **Continue** — process remains in-envelope, but one or more endpoint conditions are not yet conservatively satisfied or the minimum validated stage duration is incomplete.
2. **StopCandidate** — the calibrated model's conservative prediction clears every required endpoint, the full sensor suite is healthy and current, all phase/time/temperature/chemistry interlocks pass, and the exact current job is inside the qualified scope. This is a candidate for a controlled release decision, not a certification.
3. **FallbackToValidatedProcess** — a proxy goes out of envelope, sensors disagree, a nozzle/filter/hydraulic diagnostic fails, or the predictor becomes unreliable. Finish the conservative qualified programme or execute the prescribed safe recovery.
4. **Abstain** — the exact scope, calibration, protocol, sensor identity, evidence or model version is unknown, missing, expired or unsupported. No early release is permitted.

A single turbidity threshold, conductivity threshold, temperature threshold, camera score or elapsed timer must never return StopCandidate on its own.

## Evidence stack

### Layer 1 — independent outcome ground truth

Continue using `quality::qualify_pilot_evidence`:
- item-level soil residual;
- final-rinse chemical residue;
- cross-contamination on clean witness surfaces;
- drying residual moisture where drying is claimed;
- relative microbiological reduction only under its separately defined method;
- exact criterion IDs, metric IDs, assay method IDs, uncertainty, batches, soil age and raw archive identity.

A process sensor does not become ground truth because it correlates with an assay in a training set. It must be qualified on held-out physical runs.

### Layer 2 — process proxy calibration

Potential sensor inputs:
- recirculated-water turbidity over each phase, not a single end-of-cycle value;
- conductivity and pH as ionic/process indicators, not complete detergent-residue assays;
- inlet and phase temperatures;
- branch flow and pressure, pump real power and pump ramp/diagnostics;
- water level, fill/drain timing, filter differential pressure and sump state;
- fluorescence imagery only where the exact soil/surface/optics are in the calibration envelope.

For every proxy, freeze sensor make/model, serial/channel identity, firmware, calibration ID, units, range, resolution, drift limits, sample cadence, signal-processing/model version and fault diagnostics. Store raw time series, not only a derived mean or class label.

### Layer 3 — outcome-calibrated predictor

Train/predict on independent physical runs; test on held-out:
- whole soil preparation batches;
- whole test days;
- whole machine units / nozzle geometries;
- surface materials and item geometries;
- detergent lots and water-hardness bands.

Avoid random image/frame splitting: frames from one wash belong to one physical run and must not leak across train/test sets. Report per-stratum sensitivity, specificity, calibration error, uncertainty coverage, false-pass rate and abstention rate. Aggregate accuracy must not hide a poor class or item geometry.

### Layer 4 — runtime safety supervisor

The model may propose Continue or StopCandidate; a deterministic supervisor enforces:
- exact model/programme/firmware, soil, material, load-set, soil batch and age, detergent/formulation batch, water batch/hardness, inlet and ambient scope;
- calibrated sensor identity, health, freshness and agreement;
- phase order and minimum dwell;
- validated temperature/chemistry constraints;
- hydraulic flow/pressure envelope and nozzle/filter fault state;
- no omitted endpoint or required final-rinse stage;
- no outstanding contradiction, missing sample or stale prediction;
- conservative uncertainty bounds for every required endpoint.

The supervisor must force FallbackToValidatedProcess or Abstain when any precondition is false. A model update or sensor replacement invalidates the old calibration until re-qualified.

## Statistical gates and denominators

The current Rust helper `plan_zero_failure_validation(alpha, maximum_upper_failure_probability)` computes the exact one-sided Clopper-Pearson sample floor for zero observed false passes under an IID physical-run assumption.

At one-sided 95% confidence:
- 299 independent runs with zero false passes give an upper bound below 1%;
- 2,995 independent runs with zero false passes give an upper bound below 0.1%.

These counts are per pre-registered scope/stratum. If a programme claims simultaneous 95% confidence across 20 soil/material strata and wants a 0.1% false-pass bound per stratum, a simple Bonferroni allocation uses alpha=0.05/20 and requires about 5,989 independent zero-failure runs per stratum. This is intentionally demanding: do not hide the burden by pooling easy and difficult soils.

One physical dishwasher cycle is one trial. Multiple dishes, images, video frames, sensor samples or assay repeats inside the same cycle are not independent trials. If runs are clustered by machine, day or soil batch, IID assumptions may be invalid; use an appropriate cluster-aware design and analysis.

Five runs remain only an initial pilot floor. Zero failures in five runs does not demonstrate a low false-pass rate.

## Proposed release criteria

These are research goals, not achieved results:

- Zero false-clean and zero false-rinse releases in the held-out challenge set; report the exact number of independent physical runs and a pre-registered one-sided upper confidence bound.
- At least 95% sensitivity and 95% specificity **in every claimed soil/material stratum** before a proxy is considered for routine assurance.
- At least 90% of truly acceptable runs released without unnecessary extra rinse, while all uncertain/out-of-scope cases abstain or use the qualified fallback.
- Empirical uncertainty coverage meets its preregistered level on held-out days, batches and machines.
- Safe-stop materially reduces fresh water, wall energy or elapsed time per verified successful clean without increasing any required endpoint failure or chemical carryover.
- A proxy is never used outside the calibration envelope; sensor fault, drift or disagreement must not authorize an early release.

The 86.7% fluorescence classification result for oatmeal residue on rice bowls is a replication baseline, not a deployment threshold or universal clean detector. Its lower intermediate-class accuracy motivates uncertainty-aware abstention rather than forcing a class label.

## Benchmark comparisons

Test the same exact load/soil/surface against:
- fixed-duration incumbent programme;
- existing turbidity/Auto/SenseClean process;
- camera or fluorescence observer where present;
- process-only proxy estimator;
- fused observer with calibrated uncertainty;
- the same process with early stop disabled as a negative control.

Report:
- false-clean and false-rinse rates, with exact denominators and confidence bounds;
- false-stop / unnecessary-rinse rate;
- indeterminate/abstention rate and recovery success;
- residual soil, final-rinse residue, cross-transfer and drying results;
- energy, fresh water, detergent mass, elapsed time and human intervention per verified successful clean;
- sensor-fault detection, fallback time, recovery-resource cost and cycle-completion reliability.

## Safety case and implementation sequence

1. Complete physical one-item, micro-batch and occupancy ladders with independent endpoints.
2. Calibrate proxy distributions against those outcomes, preserving failures and raw streams.
3. Freeze an outcome-to-proxy training dataset and held-out split by complete physical runs/batches/machines.
4. Publish predictor model card, uncertainty calibration, false-pass bounds, applicable scope and failure modes.
5. Implement the deterministic runtime supervisor; make StopCandidate impossible unless every required gate is present.
6. Run fault-injection tests for blocked nozzle, clogged filter, empty detergent cartridge, under/over-dose, disconnected/drifted sensor, stale telemetry, contradictory probes, unexpected load and changed soil age.
7. Conduct shadow mode first: compute decisions without controlling the real appliance. Compare every StopCandidate with independent ground truth.
8. Only after the pre-registered false-pass and reliability gates are met may a controlled physical early-stop trial begin. Maintain a hard fallback and an immediate user override.

## Non-claims

This roadmap does not claim:
- that turbidity measures all surface-bound soil;
- that conductivity captures all detergent components/residues;
- that fluorescence generalizes beyond its tested soil/material/optics;
- that a microbiological relative reduction is sanitization;
- that an algorithm can stop safely today;
- that any resource saving has been measured.

Related work:
- Outcome evidence: https://github.com/Luminous-Dynamics/symthaea/issues/10225
- Physical occupancy data: https://github.com/Luminous-Dynamics/symthaea/issues/10136
- Adaptive observer benchmark: https://github.com/Luminous-Dynamics/symthaea/issues/10157
- Runtime assurance implementation issue: https://github.com/Luminous-Dynamics/symthaea/issues/10226
