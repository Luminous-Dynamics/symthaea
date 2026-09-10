# Crash-recovered active known good v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-health-auth-wire-v1`

Parent exact head at branch creation:

`2e7caf9c395a9bf9af1a9e23d5cf68b32e560c44`

Exact code subject frozen before this evidence-only commit:

`65b30acd5b40201492aa1794971e4868d65044c1`

This subject is intentionally distinct from the existing explicit recovery-attempt theorem.

The theorem under qualification is:

```text
durable original A -> B intent
+ exact supplied active-LKG selection/checkpoint == original source A
+ qualified crash reconciliation == exact A observed
+ QualifiedHealthyLocalSnapshot basis == CrashReconciledSource(exact reconciliation, exact attempt)
+ QualifiedExactDistributedHealthV2 == exact same healthy A snapshot and current distributed world
    -> QualifiedCrashRecoveryToActiveKnownGoodV1
```

No recovery execution is invented. This proof describes recovery **state** when crash reconciliation discovers that the machine is already at exact A and independent health/distributed evidence re-establishes the required conditions.

Required fail-closed properties include original LKG-lineage mismatch, reconciliation from another attempt or realization, non-source snapshot basis, local subject/realization/context mismatch, distributed proof substitution, absence of a current independently qualified recovery path, time ordering before the original transition/reconciliation world, record identity mismatch, and persisted-record/live-proof mismatch.

The persistent `CrashRecoveredActiveKnownGoodRecordV1` is audit evidence only. A self-consistent serialized record cannot recreate qualification; `rebind()` requires the exact live original intent, active selection, checkpoint, reconciliation, healthy-local snapshot, and distributed-health V2 proof.

## Important scope boundary

This theorem establishes continuity recovery for the exact subject and distributed state represented by its bound evidence. It does **not** prove that arbitrary external side effects caused before the crash were undone. Database writes, emitted messages, third-party API effects, irreversible device actions, or other external transactions must be included in the subject's health/distributed evidence contract or handled by a separate side-effect reconciliation/compensation theorem.

It also grants no execution, retry, checkpoint promotion, LKG-selection advancement, or bootstrap authority.

No test or CI pass is claimed here. Exact-head CI and all stacked parent qualification remain required.
