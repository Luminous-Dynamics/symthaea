# Muse runtime authorization gate (MUSE-RUNTIME-001)

A dependency-free Rust transition validator for one-shot, content-bound mutation authorization and unknown-outcome reconciliation.

## Transaction

~~~text
Prepared plan node
   -> Authorize (persist state + one-use grant)
   -> Dispatch (persist DISPATCHED receipt before external call)
   -> Commit with exact output/execution receipt
      OR definite no-mutation failure with proof
      OR OUTCOME_UNKNOWN
           -> block every new mutation
           -> reconcile exact original grant/action/parent
              -> committed output advances head
              OR no-mutation proof permits safe retry
~~~

## Authorization bindings

A grant binds the exact CQEP plan digest and node, registry snapshot, parent audio, active session ledger, action, registered operator ID/version/artifact, and user-authorization receipt.

Before issuance the gate checks:
- the operator is in the exact current registry and has the required capabilities;
- requested locality does not exceed registered operator locality;
- the authorization receipt explicitly binds the same plan/node/parent/action and exact least-privilege scope;
- precondition receipt IDs exactly match the declared required set;
- observer evidence exactly matches the required metrics and the frozen authority registry, including evaluator artifact and calibration receipt;
- each observer is qualified and uses an artifact distinct from the operator;
- compute, latency and risk budgets are sufficient.

The estimated resource cost is reserved once at authorization. The same state is not charged again by later transitions.

## One-shot and unknown-outcome rules

- Grant IDs are unique in the retained attempt ledger.
- Only AUTHORIZED can dispatch.
- Only DISPATCHED can commit, prove a definite no-mutation failure or become OUTCOME_UNKNOWN.
- An undispatched authorization may be cancelled with a durable cancellation receipt.
- A failure is retry-safe only when an external receipt proves no mutation occurred.
- Any ambiguous dispatch becomes OUTCOME_UNKNOWN.
- OUTCOME_UNKNOWN blocks new mutation grants.
- Reconciliation must match the original grant, action, parent, registry and session; it can advance to one exact committed output or prove no mutation and make a new attempt possible.
- Current audio head advances only on COMMITTED.
- Reuse, changed payload or repeated transition is rejected.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/muse-runtime-authorization-gate/Cargo.toml
cargo +1.96.0 test --manifest-path tools/muse-runtime-authorization-gate/Cargo.toml
~~~

## Deployment boundary

This library returns new state values; the production host must persist each transition durably before continuing. In particular, persist DISPATCHED before invoking an external operator. On restart, replay the durable ledger; never infer no mutation from a timeout.

The crate does not provide a database, durable storage, cross-process lock, cryptographic signature verification, sandbox, operator idempotency or exactly-once semantics. A content digest's shape and binding are validated, but signature trust is external. If storage/host concurrency is not linearizable, two processes could race using the same old state; the host must use durable compare-and-swap or a serializable transaction.

## Claim boundary

A successful gate transition means only that the local contract's prepared identities, receipts, budgets and state transition were valid. It does not prove that operator output is correct, that observer declarations are empirically independent, or that the audio meets user intent. Post-render independent observation, VCCR and VMEA gates remain mandatory.
