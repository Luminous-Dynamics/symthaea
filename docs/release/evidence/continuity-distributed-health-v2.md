# Exact distributed health v2 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-exact-local-health-snapshot-v1`

Parent exact head at branch creation:

`030fd79baa80e1174429546c0ec9a1e5f8b5bc9b`

Exact code subject frozen before evidence-only commits:

`e4ae38500d2988cd202c90b3d9a4c864bd01e081`

The theorem under qualification is:

```text
QualifiedHealthyLocalSnapshotV1
+ exact distributed context/currentness policies
+ complete authenticated participant state
+ complete authenticated failure-domain state
+ complete authenticated recovery-path state
    -> one shared closed-world distributed-health evaluation
    -> QualifiedExactDistributedHealthV2
```

The local healthy snapshot may retain either `PostExecutionTarget` or `CrashReconciledSource` identity provenance. V2 does not erase or reinterpret that basis.

The shared evaluator owns one implementation of participant completeness, candidate health, availability budget, minimum healthy count, mutual exclusions, failure-domain floors, approved verifier roots, freshness/future-skew checks, cross-evidence skew, exact recovery-class uniqueness, and at-least-one independently available recovery path.

Required fail-closed properties include context mismatch, evaluation-time mismatch, local subject outside the candidate set, incomplete/duplicate/unknown participant evidence, unavailable-budget overflow, minimum-health violation, mutual-exclusion violation, failure-domain policy/evidence substitution or incompleteness, stale/future evidence, cross-evidence skew, duplicate recovery classes regardless of outcome, recovery-class substitution, and absence of any independently available recovery path.

`QualifiedExactDistributedHealthV2` grants no execution, retry, recovery, bootstrap, checkpoint, LKG-selection, or promotion authority.

No test or CI pass is claimed here. Exact-head CI and all stacked parent qualification remain required.
