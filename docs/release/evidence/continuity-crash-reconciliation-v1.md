# Continuity crash reconciliation v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent branch:

`architecture/continuity-known-good-execution-coordinator-v1`

Exact parent head:

`bd3c518a611677eb85711c60a9b7d9ee38f7b508`

Exact child branch:

`architecture/continuity-crash-reconciliation-v1`

The theorem under qualification is:

```text
anchored durable intent A -> B
+ exact journal says eligibility spent / no receipt
+ independent qualified physical observation
-> exact classification {A, B, other known, unreachable, unknown}
```

The classification grants no retry, health, recovery, promotion, or physical execution authority.

Required fail-closed cases include journal/eligibility mismatch, any receipt already present, wrong journal anchor, wrong attempt/subject/target/context observation, observation predating the transition/anchor world, source/target aliasing, missing identity/digest for known states, fabricated identity/digest for unknown states, and persisted record/live-proof mismatch.

No local or CI execution result is claimed by this record. Passing exact-head CI and all stacked parents remain required before qualification.
