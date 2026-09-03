# Systemd Durable Attempt Evidence v0.1

Status: **draft / unqualified**

## Purpose

Close the evidence gap exposed by the Xenia-authorized systemd profile without duplicating or weakening `SystemdRecoveryBroker`.

The key invariant is:

> The real restart backend cannot be called unless durable evidence already says that dispatch may occur.

This is stronger and more crash-useful than attempting to manufacture a final receipt only after the effect.

## Composition rather than reimplementation

This tranche wraps the two public #305 boundaries:

- `CheckpointStore` — publishes the exact durable Agency Kernel checkpoint head produced by the reservation write;
- `ServiceBackend` — writes attempt evidence before and after delegating to the real restart backend.

The underlying #305 recovery algorithm remains the only source of authority validation, reservation accounting, TOCTOU re-observation, dispatch accounting, reconciliation, and final health verification.

## Ordering

For an accepted recovery attempt:

```text
#305 reserve one use
  -> durable Agency Kernel checkpoint
  -> EvidencedCheckpointStore publishes exact head
  -> pre-dispatch world re-observation
  -> SQLite attempt journal append: DispatchArmed
  -> ONLY NOW call real ServiceBackend::restart
  -> append dispatch classification
  -> #305 persist accounting successor
  -> #305 independent health observation / reconciliation
  -> optional RecoveryCompleted evidence append
```

## Why `DispatchArmed` is conservative

`DispatchArmed` does **not** claim that an effect occurred. It means:

> The system crossed the final durable evidence boundary after which the external dispatch may occur.

Therefore:

- crash before calling the real backend may leave a conservative false-positive "may have occurred" record;
- crash after calling the backend but before terminal evidence leaves exactly the same conservative state;
- either case must be reconciled rather than automatically retried.

This intentionally prefers temporarily withholding one unit of authority over multiplying a possibly applied effect.

## Fail-closed evidence behavior

### Armed evidence cannot be persisted

The wrapper does **not** call the real backend and returns `NotDispatched` to #305. The reservation may therefore be released through the ordinary broker path.

### Real backend returns an unclassified error

The wrapper hashes the diagnostic and materializes it as `OutcomeUnknown`; raw backend errors are not treated as proof of non-dispatch.

### Terminal evidence cannot be persisted

The durable `DispatchArmed` record remains. The wrapper returns `OutcomeUnknown` even if the inner backend had returned `Applied` or `NotDispatched`, preventing definitive accounting from outrunning durable attempt evidence.

### Evidence state publication fails after a durable journal append

The wrapper again returns conservatively. Durable journal truth dominates an in-process convenience cache.

## Evidence schema

The attempt context stores commitments rather than raw operational text:

- execution-id digest;
- reservation-id digest;
- grant digest;
- plan digest;
- before-world digest;
- optional external authority-evidence digest (for example Xenia provenance).

Each record additionally binds:

- monotonically increasing evidence sequence;
- previous evidence digest;
- exact Agency Kernel checkpoint head;
- state;
- optional diagnostic commitment;
- optional after-world digest;
- optional final broker recovery outcome + verification.

No journal text, stderr, credentials, secrets, or arbitrary command payload is retained.

## SQLite evidence journal

The concrete v0.1 journal is append-only per attempt key.

It uses:

- WAL;
- `synchronous=FULL`;
- `BEGIN IMMEDIATE` before reading the attempt's latest evidence head;
- primary key `(attempt_key, sequence)`;
- exact previous-digest linkage;
- post-commit head readback;
- full chain re-hashing when loaded.

A later record never overwrites an earlier attempt record.

## Crash examples

### Crash after armed evidence, before backend call

Durable evidence:

`DispatchArmed`

Actual effect: none.

Safe interpretation: outcome unknown until recovery proves non-dispatch.

### Crash after backend call, before terminal evidence

Durable evidence:

`DispatchArmed`

Actual effect: possibly applied.

Safe interpretation: outcome unknown; no automatic retry.

### Backend applied, terminal evidence durable, accounting persistence fails

Durable evidence records that dispatch was applied; the Agency Kernel checkpoint may remain behind/contained. Recovery must reconcile the two evidence sources rather than silently refund authority.

## Relationship to Xenia

`AttemptEvidenceContext.authority_evidence_digest` provides a fixed-size seam for the Xenia capability/provenance evidence from #315/#317.

A small follow-up should make #317 construct this digest automatically and append `RecoveryCompleted` after a successful Xenia-authorized recovery receipt.

## Non-claims

v0.1 does not establish:

- that every attempt reaches a finalized `RecoveryCompleted` record;
- independent truth of the external effect;
- instant revocation under Xenia freshness suppression;
- trusted wall-clock integrity;
- TPM/IMA workload measurement;
- rollback immunity if both SQLite evidence and external trusted heads are reverted together;
- Byzantine storage resistance;
- production readiness.

The claim is narrower and important: **every invocation of the real restart backend is preceded by durable crash-conservative effect-entry evidence, or the backend is not invoked.**

## Qualification gate

Before promotion:

1. exact-head format/check/Clippy/tests;
2. pre-arm journal failure => zero inner restart calls;
3. terminal journal failure => one inner call, durable armed record, outward `OutcomeUnknown`;
4. successful effect => ordered `DispatchArmed -> Applied/NotDispatched/OutcomeUnknown` chain;
5. crash/fault injection around every write/call boundary;
6. #317 integration must bind Xenia provenance and final recovery receipt into the same attempt chain.
