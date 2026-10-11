# Native Interoception v0.1 — Parameter Provenance

This document classifies the numerical assumptions in `symthaea-interoception`.
The purpose is to prevent normalized engineering parameters from being mistaken
for biological measurements or validated psychological constants.

## Classes

- **S — Structural**: representation or numerical-domain choice; not a behavioral claim.
- **H — Hypothesis**: deliberately chosen research default that must survive sensitivity analysis.
- **C — Calibrated**: fitted against a declared target dataset or benchmark.
- **E — Empirical**: directly derived from an external empirical measurement with stated mapping.

For v0.1, there are **no C or E behavioral parameters**. Behavioral magnitudes are
H-class research hypotheses. Normalization and schema choices are S-class.

## State-space defaults

| Parameter | Default | Class | Interpretation |
|---|---:|:---:|---|
| channel domain | `[0, 1]` | S | normalized artificial state space |
| standard initial value | `0.75` | H | neutral starting location for six generic channels |
| standard preferred interval | `[0.65, 0.85]` | H | initial acceptable operating band |
| standard viable interval | `[0.25, 1.00]` | H | initial failure-boundary hypothesis |
| standard precision | `1.0` | H | equal initial confidence weighting |
| standard importance | `1.0` | H | equal initial contribution to aggregate deviation |
| integrity initial value | `0.90` | H | conservative high-integrity starting condition |
| integrity preferred interval | `[0.80, 1.00]` | H | initial integrity operating band |
| integrity viable interval | `[0.40, 1.00]` | H | initial integrity-boundary hypothesis |
| integrity importance | `1.5` | H | initial asymmetric weighting hypothesis |
| novelty-balance initial value | `0.50` | H | centered starting condition |
| novelty-balance preferred interval | `[0.40, 0.60]` | H | centered operating-band hypothesis |
| novelty-balance viable interval | `[0.10, 0.90]` | H | symmetric boundary hypothesis |

The six channels using the standard profile are compute reserve, memory headroom,
model stability, epistemic resolution, action efficacy, and interaction reliability.

A standalone `ViabilityVariable` can represent arbitrary finite values for mathematical
or diagnostic tests. A runnable `NativeInteroceptiveModel`, however, enforces the
`InteroceptiveDynamicsConfig` numerical domain: every channel's current value,
preferred interval, and viable interval must lie inside `[min_value, max_value]`.
This prevents an invalid initial state from being silently collapsed to a clamp
boundary on its first transition. Being outside the *viable* interval remains a
valid model state as long as the value itself remains inside the declared numerical
domain.

## Dynamics defaults

| Parameter | Default | Class | Interpretation |
|---|---:|:---:|---|
| `step_dt` | `1.0` | S | one abstract model step; not seconds |
| `recovery_rate` | `0.05` | H | fraction of out-of-band distance restored per step |
| `min_value` | `0.0` | S | normalized lower bound |
| `max_value` | `1.0` | S | normalized upper bound |

Recovery is inactive inside a preferred interval. This prevents the midpoint of a
preferred band from becoming an undeclared setpoint.

`NativeInteroceptiveModel::try_new` validates both the dynamics configuration and the
state/domain contract before creating a runnable model. Preregistration performs the
same compatibility check for every arm, and snapshot validation fails closed on a
domain mismatch rather than constructing an invalid model during evidence loading.
These are invariant checks over the declared structural domain, not new behavioral
parameters.

## Allostatic defaults

| Parameter | Default | Class | Interpretation |
|---|---:|:---:|---|
| horizon | `16` steps | H | prospective horizon for the first experiments |
| forecast `dt` | `1.0` | S | must match model `step_dt` for dynamics-aware rollout |
| discount | `0.95` | H | weighting of more distant projected deviation |

Two forecast bases are preserved separately:

- **kinematic**: measured velocity is extrapolated linearly;
- **dynamics-aware constant drive**: the native transition law is rolled forward under an explicitly declared drive.

Neither forecast is asserted to be a model of biological allostasis. They are
competing computational hypotheses that later experiments can compare and ablate.

## Required sensitivity work before higher-level claims

Before any result is interpreted as evidence for higher-level affective organization,
we should vary at least:

- preferred-band width;
- viable-band width;
- per-channel precision and importance;
- recovery rate;
- forecast horizon;
- discount factor;
- drive magnitude and persistence;
- channel inclusion/exclusion.

A qualitative result should not depend on a narrow, hand-selected parameter point.
The target is recurrence across a defensible parameter region and across independent
initial conditions.

## Claim policy

A v0.1 parameter may be promoted from H to C only when the calibration target,
objective, dataset or benchmark, fitting procedure, held-out validation, and exact
source/evidence lineage are recorded. Promotion to E additionally requires an
explicit mapping from an external empirical quantity into this normalized artificial
state space.
