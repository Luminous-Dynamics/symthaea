# Exact local healthy snapshot v1

Different identity paths may establish the same downstream proposition only after their provenance-specific obligations are satisfied.

```text
normal transition:
  exact B observed
    -> exact B health
    -> QualifiedHealthyLocalSnapshot(B)

intent-only crash:
  durable A -> B intent
    -> independent observation says exact A
    -> crash reconciliation classifies SourceKnownGoodObserved
    -> independent source-A health
    -> QualifiedHealthyLocalSnapshot(A)
```

The common snapshot does not erase provenance. Its domain-separated identity commits the exact health basis.

## Crash-source health

`CrashSourceHealthClaimV1` can be created only when an exact `QualifiedCrashReconciliationV1` classifies the original active known-good source A as observed for the exact durable attempt.

The claim commits:

- exact reconciliation id;
- exact execution attempt id;
- exact continuity subject;
- exact source realization A;
- exact distributed context;
- exact verifier profile;
- exact adapter-defined health profile digest;
- health observation time;
- outcome and health-state digest;
- raw evidence digest.

A constructor-owned `CrashSourceHealthPolicyV1` pins verifier root/epoch, policy generation, health profile, maximum age and future skew.

`QualifiedCrashSourceHealthV1` is non-Serde and remains distinct from recovery. Health evidence must not predate the identity reconciliation and must satisfy the exact freshness policy.

## Common healthy snapshot

`QualifiedHealthyLocalSnapshotV1` can currently be derived from exactly two bases:

1. an existing `QualifiedPostExecutionHealthV1` for exact expected target B with outcome `Healthy`;
2. a `QualifiedCrashSourceHealthV1` for exact reconciled source A with outcome `Healthy`.

The basis remains explicit as `HealthyLocalSnapshotBasisV1` and participates in the snapshot identity.

```text
HealthyLocalSnapshot(B, PostExecutionTarget{...})
!=
HealthyLocalSnapshot(A, CrashReconciledSource{...})
```

## Why this exists

The existing post-execution health layer is correctly target-specific. It must not be weakened so a B-targeted attempt can treat `DifferentKnownRealization(A)` as if A were B.

This layer adds the missing source-health path without changing those semantics, while giving future distributed-health composition a common exact local Healthy proposition.

## Bootstrap relationship

Issue #1429 still needs its own independent baseline identity/admission proof. This V1 does not fabricate one. A later V2 may add an `InitialBaseline` basis only after baseline identity, health, distributed state, recovery path, bootstrap authority and trusted epoch are independently defined.

## Non-claims

A healthy local snapshot is not distributed health, recovery, LKG promotion, retry permission, or physical execution authority.
