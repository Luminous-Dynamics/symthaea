# HUM-OBS-001B — Push-Recovery Observatory Adapter

Status: source-design candidate
Parent: HUM-OBS-001A / #4785
Authority: benchmark-result adaptation only; **no qualification or actuation authority**

## Purpose

Adapt the existing deterministic humanoid push-recovery matrix into the common capability-observation envelope without weakening the benchmark's native semantics.

## Mapping

The adapter records:

- capability domain = `BalanceRecovery`;
- substrate = `DeterministicSimulation`;
- exact subject identity supplied by the runner;
- internal benchmark identity `humanoid.push-recovery-matrix` / `v1`;
- exact caller-supplied `case_set_id` identifying the protocol parameters and selected force/direction matrix;
- environment, authority, and evidence profile identities;
- summary measurements;
- explicit per-case negative outcomes.

## Preserved measurements

V1 maps:

- case count;
- recovery rate;
- fall rate;
- worst uprightness;
- worst capture margin;
- directional asymmetry;
- mean recovery time when available.

These remain benchmark observations. They are not promoted to qualification or safety claims.

## Negative evidence

Each fall becomes an explicit `fall` failure event. Each case that neither falls nor recovers within the protocol window becomes an explicit `not-recovered` event.

A critical distinction is preserved:

```text
benchmark execution completed
+
capability failure observed
!=
benchmark execution failed
```

Therefore a matrix that executes normally but contains falls remains `RunDisposition::Completed` with explicit negative capability evidence. Infrastructure/execution failures must use their own disposition path.

## Case-set identity

HUM-OBS-001A now requires `case_set_id` for internal protocols. The push-recovery runner must bind that identity to the exact protocol parameters and selected disturbance cases. This tranche does not yet define the canonical hashing format for that identity.

## Tests

Source tests cover:

- a successful matrix becoming a completed observation with preserved recovery metric;
- a fall becoming explicit negative evidence without being mislabeled as execution failure;
- exact internal case-set identity surviving adaptation.

## Nonclaims

This adapter establishes no new push-recovery performance result, qualification PASS, physical-hardware result, safety certification, or deployment authority.
