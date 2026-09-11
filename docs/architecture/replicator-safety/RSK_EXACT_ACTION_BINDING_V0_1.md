# Replicator Safety Kernel — Exact Action Binding v0.1

Status: **normative design contract for Class A blocker #1335; implementation pending**

This document defines the exact intent → authorization → commit binding required before the Replicator Safety Kernel (RSK) can be considered for production admission.

It contains no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path.

## 1. Governing invariant

An authorization is valid only for the exact abstract action that was evaluated.

For every successful commit:

```text
committed_action == authorized_action == evaluated_action
```

No caller-controlled field may be reintroduced at commit time in a way that can change the action that was evaluated.

Hard ceilings remain necessary but are not sufficient. A commit that stays under a ceiling while substituting a different action is still an authority-integrity failure.

## 2. Current reference-model gap

The current reference ledger evaluates:

```text
ReplicationBudgetSnapshot.requested_resource_units
```

but `BoundedReplicationAuthorization` does not retain that exact value. `commit_descendant` later accepts a separate `resource_units` value.

Therefore the current reference semantics prove:

```text
committed_resource_units <= applicable ceilings
```

but do not yet prove:

```text
committed_resource_units == evaluated_resource_units
```

This is tracked as Class A blocker #1335.

## 3. Required typed binding

The bounded authorization should retain a typed action binding. Conceptually:

```text
ReplicationActionBinding {
    parent_subject
    output_lineage
    requested_capabilities
    requested_resource_units
    evaluated_cursor
    evaluated_at
    authorization_expiry
    grant_id
    grant_generation
    safety_case_digest
    containment_envelope_digest
}
```

Most of these facts are already present elsewhere in the current opaque authorization. v0.1 should not duplicate independent mutable sources of truth; it should gather the exact evaluated facts into one immutable commit boundary.

### 3.1 Required exact fields

At minimum, commit must be unable to substitute:

- parent subject;
- output lineage;
- effective/requested capability set;
- requested resource units;
- ledger epoch + sequence;
- grant ID + generation;
- safety-case digest;
- containment-envelope digest;
- evaluation time and expiry bound.

### 3.2 Child identity / creation-result binding

RSK must not assume a child identifier is always knowable before an abstract creation action is evaluated.

Two safe profiles are allowed:

**Profile A — child known before authorization**

Bind the exact `SubjectId` in the action commitment.

**Profile B — child determined by a later trusted creation result**

Bind an opaque, non-hazardous `creation_input_commitment` or equivalent abstract derivation commitment during authorization. The later child identity must be proven to derive from that committed input under a separately trusted boundary before lineage state is committed.

Production code must not silently switch between these profiles.

This document intentionally does not define how any physical child is made.

## 4. Typed enforcement beats hash-only enforcement

A canonical digest is required for durable evidence, but a digest alone must not become the runtime policy object.

Runtime enforcement should compare typed values directly:

```text
if commit.resource_units != authorization.action.requested_resource_units:
    deny before mutation
```

The evidence layer can then commit to the same typed action:

```text
RSK_ACTION_DIGEST =
    SHA256("symthaea.rsk.action-binding.v1\0" || canonical(action_binding))
```

This mirrors the Fabrication Kernel's pattern of domain-separated authorization-context evidence while preserving an explicit typed enforcement boundary.

## 5. Fail-before-mutation rule

Any action-binding mismatch must fail before:

- appending a ledger event;
- incrementing sequence;
- consuming a `MutationId`;
- changing direct-child counters;
- changing descendant counters;
- changing resource counters;
- creating a subject record;
- changing an authority scope;
- changing quarantine/revocation state.

The failure itself may be reported in a non-authoritative observability/audit stream, but it must not alter the authority ledger.

## 6. Error model

The reference ledger should expose explicit mismatch errors rather than collapsing them into a generic budget error.

Minimum v0.1 error:

```text
AuthorizedResourceMismatch {
    evaluated: u64,
    committed: u64,
}
```

Future action-binding fields should receive similarly specific mismatch errors or a typed mismatch enum.

Specific errors matter because an attempted action substitution is different evidence from ordinary budget exhaustion.

## 7. Cursor binding remains independent

Exact action binding does not replace the existing cursor/CAS rule.

Both must hold:

```text
commit.cursor == authorization.cursor
AND
commit.action == authorization.action
```

This prevents two distinct failure classes:

- stale/concurrent decisions attempting to consume the same ledger snapshot;
- a fresh caller attempting to substitute a different action under an otherwise valid token.

## 8. Runtime witness remains independent

Exact action binding also does not replace runtime assurance.

A commit must satisfy all three layers:

```text
snapshot still current
AND exact action still identical
AND runtime safety witness still valid
```

A perfectly bound action must still fail if monitoring becomes stale, containment drifts, evidence expires, or a negative authority fact appears.

## 9. Time continuity

The current reference model stores `evaluated_at_unix_secs` but accepts commit time from the caller.

As a minimum semantic invariant, future code must require:

```text
commit_time >= evaluated_at
commit_time < expires_at
```

Production admission requires more than this comparison: commit time must come from a trusted monotonic/continuity boundary rather than an arbitrary caller assertion. Clock rollback or inability to establish continuity freezes new replication authority.

Time-hardening is therefore related to exact action binding but remains a separate production-admission gate.

## 10. Verified evidence types

The current Rust structs are reference values. Production admission requires positive authority inputs to be introduced only through verification boundaries.

Target pattern:

```text
UnverifiedGrant -> verify(...) -> VerifiedReplicationGrant
UnverifiedQuorum -> verify(...) -> VerifiedIndependentQuorum
RawMonitorEvidence -> verify(...) -> VerifiedRuntimeSafetyWitness
RawTimeEvidence -> verify(...) -> VerifiedMonotonicTime
```

The hard evaluator should eventually consume the verified forms at the production boundary.

This prevents a controlled caller from turning an ordinary Rust field assignment into a positive trust fact.

## 11. Action digest canonicalization

When the durable-evidence implementation lands, action binding must use the canonical RSK encoding contract rather than Rust layout, `Debug`, `Hash`, or serializer defaults.

Minimum digest domain:

```text
symthaea.rsk.action-binding.v1\0
```

The canonical body should include fixed discriminants and fixed-width big-endian integers consistent with `RSK_DURABLE_EVIDENCE_V0_1`.

The action digest is evidence of exactly what was authorized. It is not an authority grant by itself.

## 12. Event binding

`DescendantCommitted` durable evidence should include either the complete canonical action binding or its action digest in addition to the fields needed for replay.

Replay must prove:

```text
recorded commit action == recorded authorization action
```

and must reject an event that would have been invalid under the exact-binding rule, even if its hash chain is internally intact.

## 13. Idempotence semantics

Retry behavior must distinguish:

**Same mutation, same action**

A durable implementation may return an idempotent already-committed result.

**Same mutation, different action**

Must be treated as a conflict/substitution attempt. It must never be normalized into the previous action or silently accepted.

**Different mutation, stale cursor**

Must fail as a cursor conflict.

## 14. Negative-authority precedence

Exact positive binding must never overpower a negative fact.

Even a byte-for-byte identical authorized action is denied if, before commit:

- subject or ancestor is quarantined;
- subject or ancestor is revoked;
- lineage or ancestor lineage is quarantined/revoked;
- grant/generation is revoked;
- runtime monitoring becomes unhealthy/stale;
- policy/evidence becomes stale;
- containment no longer matches;
- authorization expires;
- durable state is forked or non-operational.

## 15. Required reference-model tests

Before #1335's exact-resource item can close, tests must prove at least:

1. evaluated resource = committed resource → normal path unchanged;
2. evaluated resource < committed resource → explicit mismatch denial;
3. evaluated resource > committed resource → explicit mismatch denial;
4. mismatch does not append an event;
5. mismatch does not advance the cursor;
6. mismatch does not consume the mutation ID;
7. mismatch does not create the child subject;
8. mismatch does not change any ancestor subject counters;
9. mismatch does not change any lineage counters;
10. a later exact retry using the same previously unconsumed mutation may proceed if every other predicate remains valid;
11. stale cursor still fails independently;
12. runtime monitor failure still fails independently;
13. action binding cannot widen descendant capability ceiling;
14. action binding cannot bypass ancestor grant budgets.

## 16. Property/state-machine tests

The generated state-machine suite should include an abstract authorization map:

```text
AuthorizationToken -> (cursor, exact_action, expiry, evidence bindings)
```

For arbitrary interleavings of authorize, quarantine, revoke, branch, commit, duplicate, and stale-commit operations, assert:

```text
Committed(a) => previously AuthorizedExactly(a)
```

and:

```text
DeniedOrFailed(a) => authority_state_after == authority_state_before
```

except for separately modeled non-authoritative audit telemetry.

## 17. Relationship to Fabrication Kernel

RSK should reuse the architectural lesson already present in the Fabrication Kernel:

- authorization is bound to explicit operational context;
- context receives a domain-separated digest for evidence;
- verified trust/session/policy facts are retained through the authorized object;
- execution does not reconstruct positive authority from loose caller inputs.

RSK's action schema remains separate because descendant lineage, ancestral budgets, and replication authority have different invariants from ordinary fabrication jobs.

## 18. Promotion gate

Exact action binding is complete only when all of the following are true on one exact commit:

- typed binding implemented;
- mismatch errors implemented;
- pre-mutation ordering demonstrated;
- unit tests green;
- black-box integration tests green;
- generated/property tests green for the modeled transition set;
- focused RSK compile/test/Clippy green;
- rustfmt green;
- Class A ADR updated with executed evidence.

Until then, production admission remains denied.
