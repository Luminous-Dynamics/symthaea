# MATH-RET-RUNTIME-002E — Transaction-Scoped Evidence Session

Status: draft production-hardening implementation

Authority: `MeasurementOnly`

002E closes the evidence-session hygiene gap identified after MATH-RET-RUNTIME-002C.

The frozen retrieval executor already calls `abort()` when trace staging, payload
audit staging, or evidence commit fails. But 002C performs candidate-universe and
materializer-identity preflight **before** entering that executor. If a caller
reused an `AtomicEvidenceSink` containing stale staged data, an early preflight
failure would not necessarily give the executor an opportunity to clear it.

002E makes clean staged state an ownership invariant instead of a caller
convention.

## Ownership boundary

```text
AtomicEvidenceSink S
        ↓ move
ProductionEvidenceSession<S>
        ↓
no public mutable sink access
        ↓
transaction guard
        ↓
production retrieval / evidence operation
        ↓
abort staged state on guard drop
```

`ProductionEvidenceSession` owns the sink by value.

It exposes read-only `inner()` inspection, but intentionally exposes no
`inner_mut()` method. Mutable sink access exists only inside `transact`.

## Cleanup semantics

The session calls `AtomicEvidenceSink::abort()`:

1. when the session is created;
2. before each transaction callback runs;
3. when the transaction guard drops;
4. when the session itself drops;
5. before `finish()` returns ownership of the sink.

The transaction guard is RAII-based, so its `Drop` runs for:

- successful callback return;
- ordinary `Err` return;
- Rust panic unwinding when the process uses `panic=unwind`.

The underlying sink contract defines `abort()` as clearing **staged** state.
Already committed evidence is not erased.

Therefore a successful executor commit survives transaction cleanup, while any
uncommitted staged trace/payload data is cleared before the next request.

## Production convenience path

002E exposes:

```text
ProductionEvidenceSession::execute(
    &mut ProductionRetrievalRuntime,
    QualifiedRetrievalRequest,
)
```

which is equivalent to running the frozen 002C runtime inside one evidence
transaction.

The resulting path is:

```text
clean evidence session
       ↓
002C candidate/materializer preflight
       ↓
guarded backend
       ↓
frozen RetrievalExecutor
       ↓
atomic evidence commit
       ↓
transaction guard drop
       ↓
no staged residue
```

## Mechanical canaries

The tests require:

1. constructor clears pre-existing staged state;
2. ordinary transaction error clears staged state;
3. successful committed transaction preserves committed evidence and clears staged state;
4. successful callback that stages but does not commit cannot leak staged state;
5. panic unwinding clears staged state;
6. `finish()` returns a clean sink without erasing prior commits;
7. a legal 002C production execution commits evidence through the session and finishes clean;
8. an early production candidate-preflight failure leaves no committed or staged evidence.

The integration fixture reuses the exact candidate artifact from 002C rather
than creating a second production test universe.

## Crash boundary

RAII is not a substitute for durable crash consistency.

002E does **not** guarantee cleanup after:

- `panic=abort`;
- `SIGKILL`;
- process termination before destructors run;
- kernel/power failure;
- a sink implementation that violates the frozen `AtomicEvidenceSink` contract.

A persistent production sink must still implement its own durable atomic commit
and recovery semantics. 002E establishes the in-process request/session
boundary so stale staged state cannot cross normal request boundaries or panic
unwinding.

## Relationship to 002D

002E is stacked after 002D only to preserve the ordered production-hardening
lineage. It does not interpret or modify the materializer implementation
receipt. Implementation provenance and evidence-session ownership remain
separate mechanical theorems.

## Nonclaims

002E proves only evidence-session lifecycle properties. It does not establish:

- that queued predecessor workflows passed;
- durable crash consistency for an arbitrary sink;
- materializer implementation provenance beyond 002D;
- retrieval relevance;
- HDC advantage;
- mathematical equivalence;
- theorem truth;
- proof success;
- evidence score;
- formal authority.
