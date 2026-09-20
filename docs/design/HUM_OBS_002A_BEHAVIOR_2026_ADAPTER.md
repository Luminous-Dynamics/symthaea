# HUM-OBS-002A — BEHAVIOR-1K 2026 Challenge Result Adapter

Status: source-design candidate
Issue: #4887
Parent: HUM-OBS-001C / #4849
Authority: external benchmark observation import only; **no physical or deployment authority**

## External profile pinned by this tranche

This adapter targets the BEHAVIOR-1K 2026 Challenge evaluation profile documented on 2026-09-20:

- repository/evaluator tag `v3.9.2`;
- 100 full-length household tasks;
- challenge-track policy observations limited to RGB + depth + proprioception;
- primary `q_score` equal to the fraction of BDDL goal predicates satisfied at episode end;
- public reporting instances 0..=9 per task;
- public development instances 10..=19;
- organizer-reserved final instances 20..=39;
- optional benchmark-native efficiency outputs such as simulator time, base travel, hand/end-effector travel, and normalized efficiency measures.

The Rust crate does not vendor or execute OmniGibson, BEHAVIOR datasets, Python evaluators, or model servers.

## Import envelope

`Behavior2026RolloutResultV1` binds:

- exact benchmark version;
- task-set identity;
- task identity;
- evaluation split and instance index;
- rollout identity;
- scene identity;
- challenge input profile;
- required `q_score`;
- optional native metrics;
- wrapper configuration reference;
- robot configuration reference;
- evaluator/result artifact reference.

`Behavior2026ObservationContextV1` separately binds the Symthaea source/model/morphology/sensor-actuator subject plus authority and evidence profiles.

## Population separation

Instance populations are explicit and fail closed:

```text
PublicReport       -> 0..=9
PublicDevelopment  -> 10..=19
HiddenFinal        -> 20..=39
```

A result cannot relabel an instance from one population as another.

## Capability-failure semantics

A valid imported rollout always represents a completed external benchmark execution. If `q_score < 1.0`, the observation remains `RunDisposition::Completed` and carries explicit `task-incomplete` negative capability evidence.

```text
completed evaluator run + partial task success
!= infrastructure failure
```

Malformed or incompatible external artifacts are rejected before an observation is created.

## Metric preservation

The required primary metric is emitted as:

- `behavior.q_score`, unit `ratio`, provenance `BenchmarkNative`.

Optional native metrics are preserved under `behavior.<native_metric_id>` with their supplied units. The reserved `q_score` name cannot be duplicated through the optional metric vector.

## Fail-closed checks

V1 rejects:

- benchmark versions other than `v3.9.2`;
- missing task/task-set/rollout/scene/config/artifact identities;
- split/instance mismatch;
- non-finite or out-of-range `q_score`;
- duplicate optional metric IDs;
- non-finite optional metrics;
- attempts to duplicate the reserved primary metric.

## Nonclaims

This adapter does not establish official challenge acceptance, leaderboard standing, full 100-task coverage, absence of cherry-picking, household deployment readiness, real-robot performance, safety qualification, or product authority. Those require separate evidence and qualification layers.
