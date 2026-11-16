# Observation-process equivalence: exact paired-world controls

**Date:** 16 November 2026  
**Status:** executable synthetic integration-test fixture; not a capability result  
**Owners:** #9817 (active sensing and complete sensing cost), #12407 (event/mask/time semantics), #12194 (prospective versus realized information)

## Purpose

A forecasting model can score well while confusing the physical process with the mechanism that decides what gets measured. This small fixture supplies exact known-answer controls before introducing more realistic generators or learned policies.

It deliberately tests three different questions:

1. Can passive observations identify the physical mechanism?
2. Does an authorized independent measurement change the best downstream decision enough to justify its cost?
3. Can mask/policy information be informative while having zero value for the task decision?

The Rust integration tests are in `crates/core/symthaea-fep/tests/observation_process_equivalence.rs`.

## Fixture A — passive observational equivalence

Two evaluator-only worlds produce the same primary sensor history:

| Latent mechanism | Physical temperature | Sensor offset | Primary sensor |
|---|---:|---:|---:|
| Actual physical hazard | 10 | 0 | 10 |
| Cool environment, biased sensor | 0 | +10 | 10 |

The public history consists of three equally timed primary-thermal observations, all equal to 10, with the same sampling-policy identity. The candidate receives only this history. The hidden physical state and sensor offset are evaluator-only fields.

**Known answer:** the candidate histories are byte-for-byte equivalent under this finite fixture, while the correct physical decision differs. Passive prediction confidence cannot identify the cause. The appropriate claim is non-identifiability under the supplied observation policy.

## Fixture B — decision-focused reference probe

A separately declared, permitted independent-reference sensor reports the physical temperature rather than the primary sensor's biased value. It returns 10 in the hazard world and 0 in the offset world.

The task loss is intentionally asymmetric:

- Protect during a true hazard: 0.
- Protect when there is no hazard: 2.
- Continue during a true hazard: 10.
- Continue when there is no hazard: 0.

With an equal prior over the two worlds, the optimal pre-probe Bayes risk is 1. The independent measurement identifies the world in this deterministic fixture, making post-probe risk 0. Therefore:

- Expected value of sample information: 1 task-loss unit.
- Declared reference-probe cost: 0.25 task-loss units.
- Net value under this fixture's explicitly scalarized utility: 0.75.

This is an oracle calculation for a two-world deterministic control, not an empirical value or a general sensing policy. Real deployments must preserve the full resource vector (sensor energy, wake cost, compute, latency, transmission and memory writes) unless a task owner provides a preregistered utility conversion.

## Fixture C — mask-only negative control

Two candidate-visible histories differ in their declared sampling policy and whether the second measurement was scheduled. The physical condition remains the same and below the action threshold in both cases.

The mask/policy observation identifies which sampling policy was used (1 bit of information in this two-policy fixture), but it cannot change the correct task decision. Its task EVSI is exactly zero.

This control is designed to catch entropy-only samplers that pay to learn about sensor policy when no downstream decision can improve.

## Failure control — probe unavailable

When the reference sensor is unavailable, no diagnosis is returned. The failed request is not converted into a measurement, and no realized information is credited. Later versions should additionally bind the consumed action/resource budget and failure receipt.

## Test inventory

The initial Rust integration test has five deterministic cases:

1. Passive observations cannot identify opposite physical mechanisms.
2. Independent reference probe has positive decision value.
3. Mask-only policy information has zero value for an unchanged decision.
4. Unavailable reference probe yields no realized information.
5. Candidate-visible observation records are separate from evaluator-only latent truth.

Command: `rustc --edition=2024 --test crates/core/symthaea-fep/tests/observation_process_equivalence.rs -o /tmp/observation-process-equivalence-tests && /tmp/observation-process-equivalence-tests --nocapture`.

The hosted focused workflow compiles this dependency-free fixture directly with the pinned Rust compiler. It does not invoke Cargo because the current workspace glob includes `crates/core/symthaea`, which has no `Cargo.toml` (tracked separately by #12324/#9906). The test is kept under the FEP crate's `tests/` tree so it can also be picked up by Cargo once that workspace prerequisite is repaired. The fixture has not yet passed hosted execution. They do not establish runtime performance, calibration, sensor feasibility, physical energy savings, robustness to arbitrary MNAR mechanisms, or SOTA superiority.

## What this does not test yet

- Probabilistic/noisy sensors and calibrated likelihoods.
- Irregular and delayed timestamp distributions, clock conversion, duplicates and replay.
- Gradual/abrupt regime changes and rare-event deadlines.
- MAR/MNAR sensitivity regions, positivity/overlap, and policy-shift generalization.
- A learned FEP/HDC-LTC policy or the existing decision-focused planner.
- Full vector-resource Pareto optimization or physical sensor energy.
- Held-out real datasets and independent replication.

## Next benchmark progression

Keep the exact toy control as a permanent regression gate, then add:

1. Seeded noisy paired worlds with matched passive likelihoods and a safe probe whose outcomes separate the causes.
2. Identifiability-negative controls where no permitted action separates the world hypotheses.
3. Mask-policy shifts where mask predictiveness collapses but physical dynamics remain unchanged.
4. Rare, short-lived events where the cost of sleeping through a transition is measured against the full sampling/computation budget.
5. Official BITS and Time-IMM adapters through #9700, with dataset, code, preprocessing, split and evaluator digests frozen.
6. Decision-focused EVSI, risk-aware goal-oriented OED and exact small-instance oracle comparisons through #9817/#12194.

Do not create a second planner: use this fixture to test the observation process and oracle semantics around the existing active-sensing work. Do not claim a general missingness mechanism is identifiable from one passive stream.
