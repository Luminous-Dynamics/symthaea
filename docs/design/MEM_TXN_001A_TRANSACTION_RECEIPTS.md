# MEM-TXN-001A — Transactional Memory-Graph Mutation Receipts

## Status

Source-only candidate contract. Qualification is separate and exact-head bound.

## Purpose

MEM-TXN-001A defines a content-blind transaction receipt for executing a predeclared memory/privacy graph mutation plan.

It exists because:

```text
plan accepted != mutation committed
some actions applied != transaction committed
abort requested != rollback proven
```

The contract deliberately does **not** decide whether a correction, privacy request, or other upstream plan is semantically correct. It only defines what must be true before execution evidence may be called committed, rolled back, or rollback-unproven.

## Separation from MEM-CORR

MEM-CORR-001A plans propagation from explicit correction through dependent memory. It is still a separate frozen candidate.

MEM-TXN-001A is independent and main-based. A later adapter may translate a qualified MEM-CORR plan into `MemoryMutationPlanRefV1`.

This avoids stacking execution authority directly on an unqualified planner.

## Plan reference

`MemoryMutationPlanRefV1` binds:

- operation ID;
- operation epoch;
- plan schema;
- upstream plan commitment;
- exact expected pre-mutation graph snapshot commitment;
- canonical planned artifact set;
- action per artifact;
- exact pre-artifact commitment;
- fresh-identity requirement;
- whether an external copy may exist.

The reference has its own domain-separated BLAKE3 commitment.

## Supported action vocabulary

The transaction layer preserves the planned action without deciding it:

- `NoEffectOutsideScope`
- `ShadowForCurrentScope`
- `Invalidate`
- `RecomputeRequired`
- `SupersedeWithFreshIdentity`
- `RetainIndependentBasis`
- `BlockedUnknownDependency`

## Committed transaction semantics

A `Committed` receipt requires:

1. the live pre-state snapshot exactly equals the plan's expected snapshot;
2. exactly one result exists for every planned artifact;
3. no unexpected artifact result exists;
4. action identity is unchanged;
5. pre-artifact commitment matches exactly;
6. each action-specific postcondition holds;
7. a mutating action changes the final graph snapshot;
8. the receipt validates its own commitment.

There is no successful partial-commit state.

## Action postconditions

### No effect / retained independent basis

The artifact identity and content commitment must be unchanged and the result disposition must be `VerifiedNoEffect`.

### Blocked unknown dependency

The artifact remains unchanged and the disposition is `Blocked`.

### Session/current-scope shadow

The governed durable artifact identity/content remain unchanged while transaction-visible graph state changes.

### Invalidate

No current post-artifact identity or content commitment may remain.

### Recompute / supersede

Both require:

- `Applied` disposition;
- a fresh artifact identity;
- a post-artifact content commitment;
- post-content commitment different from the old artifact commitment.

Fresh identity is therefore not merely documentation; it is an admission condition.

## External-copy limitation

For destructive/recompute actions where the plan says an external copy may exist, committed evidence must preserve:

```text
external_deletion_unproven = true
```

This does not claim an external copy definitely exists. It prevents local mutation evidence from being misrepresented as global deletion evidence.

## Abort semantics

### `AbortedRolledBack`

Requires the exact final graph snapshot commitment to equal the exact pre-state snapshot commitment.

The receipt records which planned artifact IDs were attempted and an opaque failure reason reference. Per-artifact applied results are intentionally absent from the final rolled-back receipt because they are not current committed state.

### `AbortedRollbackUnproven`

Represents execution failure where restoration cannot be established strongly enough to call rolled back.

It is never equivalent to success.

```text
AbortedRollbackUnproven != Committed
```

A later recovery/repair workflow should treat it as an explicit graph-integrity incident.

## Determinism and tamper resistance

Plan entries and applied results are canonicalized by artifact ID.

Changing any of the following changes a commitment or causes validation failure:

- operation epoch;
- plan/schema commitment;
- pre-state snapshot;
- planned artifact/action;
- pre-artifact commitment;
- final snapshot;
- transaction time window;
- result action/disposition;
- fresh replacement identity/content;
- external-deletion limitation;
- abort evidence.

## Tests

The source integration suite covers:

- exact committed happy path;
- result-order invariance;
- missing result rejection;
- duplicate result rejection;
- action substitution;
- pre-artifact commitment substitution;
- fresh-ID/content requirements;
- invalidate absence requirement;
- no-effect/retain rewrite rejection;
- mutating snapshot-change requirement;
- no-op-only unchanged snapshot;
- external-copy limitation preservation;
- rollback-proven exact snapshot restoration;
- rollback-unproven distinct state;
- unexpected attempted artifact rejection;
- plan-order invariance;
- plan self-tamper rejection;
- receipt self-tamper rejection;
- live pre-state mismatch rejection.

## Nonclaims

This tranche does not:

- mutate a production database or vector index;
- establish that a MEM-CORR plan is semantically valid;
- verify a correction receipt;
- execute recomputation logic;
- prove deletion of external copies;
- establish new user preference, consent, or physical authority;
- claim a qualification PASS before exact-head execution evidence exists.

## Next boundary

After both planner and transaction substrate are qualified enough for integration, introduce a narrow adapter/executor:

```text
verified correction receipt
+
qualified MEM-CORR plan
+
current live graph
+
MEM-TXN transaction backend
        ↓
committed mutation receipt
        ↓
index invalidation/rebuild
        ↓
new current retrieval snapshot
```

The adapter must refuse execution if the live graph snapshot no longer equals the snapshot used to produce the propagation plan.