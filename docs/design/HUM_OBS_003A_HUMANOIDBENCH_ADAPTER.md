# HUM-OBS-003A — Exact-Upstream HumanoidBench Observation Adapter

Status: source-design candidate
Issue: #4921
Parent: HUM-OBS-002B / #4903
Authority: external simulation evidence only; **no physical, safety, or deployment authority**

## Purpose

Import HumanoidBench episode evidence into the common Embodiment Observatory without embedding or reimplementing the upstream Python/MuJoCo benchmark inside the Rust humanoid crate.

## Exact upstream identity

The adapter pins:

`carlosferrazza/humanoid-bench@cb1189039151c8aadaaa987b442da54383c87fab`

No semantic release tag is invented. A different upstream commit is a different benchmark lineage and fails closed until its registry is audited explicitly.

## Registry binding

At the pinned source commit, upstream `humanoid_bench/env.py` defines 32 registered task keys and six registered robot keys. The Rust adapter carries the audited task vocabulary exactly and computes a domain-separated BLAKE3 task-registry commitment over:

```text
upstream repository
+ exact upstream commit
+ exact canonical task vocabulary
```

Robot identity is intentionally **not** part of the task-set commitment. It remains per-run evidence along with the upstream control mode.

## Imported episode evidence

`HumanoidBenchEpisodeResultV1` preserves:

- exact upstream commit;
- exact task ID;
- exact robot ID;
- position/torque control mode;
- episode identity;
- exact optional `u64` random seed;
- execution status;
- benchmark-native episode return;
- episode length;
- terminated/truncated flags;
- external runner artifact reference;
- environment, authority, and evidence profile identities.

The exact seed remains adapter metadata. It is deliberately not converted into an `f64` observatory measurement because sufficiently large `u64` values cannot be represented exactly as `f64`.

## Observation conversion

A validated completed episode lowers into `HumanoidCapabilityObservationV1` with:

- `ExecutionSubstrate::ExternalBenchmarkSimulation`;
- an external benchmark identity pinned to the upstream repository and commit;
- the exact task-registry commitment;
- explicit capability-domain mapping;
- benchmark-native numeric measurements;
- `RunDisposition::Completed` even when the return is poor.

A runner/infrastructure failure lowers to `InfrastructureIndeterminate` with explicit operational failure evidence. It is not rewritten as negative robot capability.

## Task domains

The first adapter maps balance tasks to `BalanceRecovery`, the clear locomotion/body-mobility tasks to `Locomotion`, and remaining manipulation-rich tasks to `MobileManipulation`.

This mapping is Symthaea observatory metadata, not an upstream taxonomy. It must remain explicit and reviewable rather than being mistaken for an upstream label.

## Important distinction

```text
benchmark executed
!= benchmark performed well
!= physical robot capability
!= sim-to-real transfer
!= safety qualification
!= deployment authority
```

No HumanoidBench-wide aggregate score is created by this tranche. Per-task/run evidence remains visible unless a separately specified upstream-compatible aggregation protocol is later adopted.

## Tests

Source tests cover:

- deterministic exact task-set identity;
- task-set identity independence from robot selection;
- unknown-task rejection;
- exact upstream-commit rejection;
- robot/control-mode binding;
- poor return remaining completed execution rather than infrastructure failure;
- exact seed remaining non-lossy adapter metadata;
- infrastructure failure remaining indeterminate;
- non-finite return rejection.

## Nonclaims

This adapter establishes no official upstream leaderboard standing, physical humanoid performance, sim-to-real transfer, safety qualification, household readiness, or product authority. The external runner artifact remains the detailed source evidence; the common observatory envelope is a claim-preserving summary, not a replacement for that artifact.
