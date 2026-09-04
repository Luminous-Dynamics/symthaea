# Agency witness anchor runtime V0.1

## Purpose

This profile defines the transport-independent effect boundary for writing a guarded #456 witness frontier into an external chronology source.

It intentionally does not implement Xenia, TPM NV, SCITT, or another backend. Instead it freezes the rules that every backend must satisfy around effect ambiguity and returned evidence.

## Guarded input

Dispatch accepts `GuardedAnchorPermitV1` from #456.

The permit exists only while #456 holds its SQLite `BEGIN IMMEDIATE` writer barrier and only for a nonempty local state classified as anchorable:

- `InitialAnchorRequired`;
- `LocalAheadVerifiedDescendant`.

Divergent/rollback states and empty witness domains cannot reach this runtime through the reviewed permit API.

## Deterministic operation identity

`WitnessAnchorOperationV1` binds:

- source ID;
- source epoch;
- source-specific anchor-policy digest;
- witness ID;
- local high watermark;
- reservation head;
- exact frontier-statement digest.

The domain-separated operation ID is derived deterministically from those fields.

A concrete backend MUST use `operation_id` as its idempotency identity. The same source namespace + exact frontier therefore produces the same operation after process restart without requiring a new random append ID.

Changing source epoch or source-specific policy intentionally changes the operation ID.

## Closed-world dispatch outcome

The backend trait does not return a generic transport error after dispatch. It must classify every call as exactly one of:

```text
Applied(external claim)
ProvenNotDispatched(diagnostic commitment)
OutcomeUnknown(diagnostic commitment)
```

If a backend cannot prove that an error occurred before its external effect boundary, it MUST return `OutcomeUnknown`.

The runtime itself preserves this rule. After `backend.dispatch` is called it never returns a generic error.

## Applied is not automatically trusted

A backend's `Applied` result is only a claim.

The runtime requires the returned external claim to match the operation exactly:

- source ID/epoch;
- witness ID;
- high watermark;
- reservation head;
- frontier-statement digest.

It then passes that claim through #452's `ExternalWitnessFrontierVerifier`, which must independently authenticate the source and establish currentness/freshness.

Only then does the runtime return `VerifiedApplied`.

If the backend says Applied but returns another frontier, the result is `OutcomeUnknown / AppliedClaimMismatch`.

If the exact Applied claim fails external verification/freshness, the result is `OutcomeUnknown / AppliedClaimVerificationRejected`.

Both are retry-unsafe.

## Retry safety

The only dispatch result that permits another append attempt without reconciliation is:

```text
ProvenNotDispatched
```

`OutcomeUnknown` is never automatically retried.

The crate exposes a separate reconciliation operation:

```text
operation id
    ↓
backend.reconcile
    ↓
Applied | ProvenNotApplied | OutcomeUnknown
```

An Applied reconciliation is subjected to the same exact-target checks and external verification as the original dispatch.

## Crash/recovery interpretation

A deterministic operation ID helps reconstruct one exact target across retries, but V0.1 does not claim a separate durable local anchor-attempt log.

For a monotonic external source such as the intended Xenia ledger adapter, crash recovery should first retrieve and verify the source's fresh current frontier. Then #452/#456 determine whether local state is:

- already anchored;
- a verified descendant requiring a newer anchor;
- rolled back/divergent and contained.

This allows recovery to converge even if an earlier target was ambiguously applied before a newer local frontier later existed.

A concrete backend may additionally use operation-specific reconciliation when available.

## Source-side contract

The generic runtime cannot prove source-specific append semantics.

A concrete backend MUST own:

- source identity and epoch;
- bounded transport timeouts;
- operation-ID idempotency;
- source-side CAS/monotonic preconditions;
- durable append semantics;
- currentness/freshness verification;
- operation reconciliation after ambiguous outcomes.

For Xenia, the intended source precondition is a fresh signed ledger checkpoint plus the current witness-anchor state. An old signature or caller-supplied source counter is insufficient.

## Guard lifetime

Dispatch runs while `GuardedAnchorPermitV1` borrows #456's SQLite writer guard. This prevents another #449 writer from advancing local witness chronology during the external write critical section.

Backends therefore MUST impose hard external deadlines. A timeout after dispatch begins is `OutcomeUnknown`, after which the caller should release the guard and perform source reconciliation/current-state verification before another append.

## Tests authored

The source suite covers:

- exact Applied claim becomes a verified external frontier;
- deterministic operation ID is stable for the same source/frontier and changes across source epoch;
- mismatched Applied frontier becomes retry-unsafe unknown;
- freshness/source-verifier rejection after Applied becomes retry-unsafe unknown;
- explicit backend OutcomeUnknown requires separate reconciliation;
- later reconciliation can establish the exact Applied claim;
- ProvenNotDispatched is the only retry-safe dispatch failure.

## Non-authority statement

External witness anchoring is evidence chronology only. Neither a deterministic operation ID, an Applied source claim, a verified external frontier, nor successful reconciliation creates or amplifies Symthaea execution authority.
