# Muse Execution Authority

A dependency-free Rust transition contract for one-shot, evidence-bound mutation authorization and fail-closed reconciliation.

## Why it exists

Muse already separates edit planning, execution operators, independent observers and multi-turn constraint history. Those components still need one authoritative boundary between a proposal and a mutating renderer/tool invocation.

This crate defines that boundary as a replayable state machine:

prepare → authorize → persist → dispatch → observe → commit/reject/reconcile

An authorized grant binds the exact plan, registry, session ledger, parent output, action, operator artifact, edit scope, required capabilities, preconditions, observer obligations and resource cost.

## State transitions

- **Authorized:** a one-shot grant exists, but the operator has not been invoked.
- **Dispatched:** the dispatch event has been persisted; the host may now call the operator.
- **Committed:** the exact output digest and all required independent observer receipts pass. Only this transition advances the current session head.
- **Rejected:** the output is known but does not meet the acceptance policy. The artifact remains recorded; the current head does not advance.
- **Failed without mutation:** a definite failure receipt proves no output mutation occurred.
- **Outcome unknown:** the host cannot determine whether the operator wrote/committed output. Every subsequent mutation is blocked.
- **Reconciled:** an externally verified receipt resolves the unknown attempt as commit, reject, or no mutation.

## Host integration contract

The crate is deliberately a pure transition validator. It does not perform I/O, persist state, invoke renderers, verify digital signatures, or provide exactly-once execution.

A production host must:

1. Persist the AuthorizationIssued event atomically before dispatch can be prepared.
2. Persist DispatchStarted atomically before invoking the operator.
3. Use compare-and-swap or a serializable transaction over the ledger revision/current output so concurrent requests cannot dispatch the same grant twice.
4. Persist every terminal result. A crash after dispatch but before a terminal event is replayed as unresolved and blocks new mutations.
5. Authenticate the principal approval and reconciliation verifier outside this crate; the crate only checks their digest syntax and exact binding fields.
6. Persist events durably before performing side effects and replay the full event sequence after restart.
7. Treat operator idempotency, process isolation, artifact custody and recovery as deployment-level requirements.

Do not claim exactly-once side effects across process crashes. This contract provides one-shot transition validation and an explicit unknown-outcome interlock; it cannot determine whether an external process performed an action before crashing.

## Hard rejection conditions

- Stale parent, plan, registry, session ledger or ledger revision.
- Reused attempt/grant/reconciliation identities.
- Unsupported requested scope or missing capability.
- Missing, duplicate, failed or indeterminate precondition evidence.
- Missing observer coverage, observer/operator artifact identity collision, unapproved observer artifacts, or observation detached from the exact output digest.
- Insufficient resource budget.
- Grant reuse, wrong-node dispatch, invalid state transition or mutation while an attempt is unresolved.
- Reconciliation mismatch against the original attempt/plan/action/parent/output.

## Relationship to adjacent Muse programmes

- CQEP-001 / #11525 / PR #11528 compiles a capability-qualified edit DAG. This crate is the sole pre-mutation authority.
- MMAE-TRACE-001 / #11508 / PR #11517 maintains active multi-turn intent and preservation constraints. Its ledger digest is bound into every mutation.
- VMEA-001 / #11503 / PR #11506 and the independent acoustic/pitch observers provide evidence obligations.
- MEL-MR-EVAL-016/#9090 and MEL-MR-EVAL-017/#9128 provide renderer receipts and measured acoustic lineage. Their results are evidence inputs, not authorization.

## Test coverage

The authored synthetic tests exercise valid commit, stale parent/plan/registry/session, scope mismatch, approval mismatch, missing capabilities/preconditions, observer non-independence, missing/failed/detached observations, budget exhaustion, dispatched/unknown interlocks, reconciliation commit/reject/no-mutation, wrong attempt/output, safe retry after definite failure, grant reuse and malformed evidence.

Passing these tests would qualify only the pure state-transition contract. It would not demonstrate durable host persistence, concurrency safety, cryptographic approval verification, operator idempotency or safe live audio execution.
