# connectome-edit-transaction

A dependency-free Rust state machine for auditable, replayable connectome proofreading edit transactions.

## Paradigm shift

A proofreading model should not mutate a canonical connectome as a side effect of a prediction. It should submit a transaction against a frozen graph snapshot. The transaction advances the canonical head only after:

1. its base snapshot and parent transaction match the current ledger head;
2. every precondition assertion matches;
3. a human (or explicitly authorized review authority) approves it;
4. the patch adapter supplies a resulting snapshot digest;
5. an independent verifier passes;
6. every postcondition assertion matches;
7. annotation propagation is independently attested.

The proposer, reviewer and verifier must be three distinct identities. Failed or incomplete applied edits are quarantined and do not advance the canonical head.

Rollback is a new event. It must target the current committed head, use the inverse patch recorded by that edit, start from the edit's resulting snapshot, restore its base snapshot, and pass fresh independent verification. Both the original edit and rollback remain in the event history.

## API

- `EditTransactionLedger::new(initial_snapshot_sha256)`: create a ledger at a known graph snapshot.
- `submit_edit(EditSubmission)`: record a rejected, quarantined or committed edit receipt.
- `submit_rollback(RollbackSubmission)`: append a verified rollback event for the latest committed edit only.
- `replay_events(initial_snapshot_sha256, events)`: re-execute the append-only event stream; event sequence numbers and decisions must match exactly.
- `snapshot()`: obtain the current head, next event sequence, committed transaction stack and event count.

Rejected and quarantined proposals are retained in the append-only audit event stream and consume their idempotency key. They never advance the graph head. A retry must use a new event ID and idempotency key after the underlying evidence/base has changed.

## State semantics

- **Rejected:** stale base, wrong parent, failed precondition, explicit reviewer rejection, missing apply receipt or no-op snapshot. No graph head change.
- **Quarantined:** verifier failure/timeout/inconclusive outcome, failed postcondition, missing annotation propagation, or annotation-state mismatch. The patch may have been applied by the external adapter, but it is not committed to the canonical head.
- **Committed:** all preconditions, review, independent verification, postconditions and annotation propagation pass.
- **RolledBack:** a separately reviewed and verified inverse transaction restored the previous snapshot and popped the active committed stack. Original history is preserved.

The ledger uses a strict LIFO rollback policy in v0.1. Supporting arbitrary compensating edits or concurrent branch merges requires a separate, explicit conflict/merge protocol.

## Reproducible event fixture

- Schema: `research/connectomics/connectome-edit-transaction-ledger-v0.1.schema.json`
- Known-answer stream: `research/connectomics/fixtures/connectome-edit-transaction-ledger-known-answer-v0.1.json`

The fixture contains:
1. a stale-base edit that is rejected;
2. a valid merge that advances snapshot A -> B;
3. an inconclusive-verifier edit that is quarantined without advancing the head;
4. a separately verified rollback B -> A.

Expected final state: snapshot A, empty committed stack, four immutable events, next sequence 5. All digest strings in the fixture are synthetic sentinels.

## Critical security boundary

The core validates SHA-256 formatting and equality of supplied expected/observed digest assertions. **It does not hash external bytes.** A separately pinned ingestion/verification adapter must hash the actual patch, pre-edit and post-edit graph snapshots, verifier binary/configuration and annotation state, then independently inspect topology and annotation lineage before submitting receipts.

The ledger is not a distributed consensus service, transactional database, cryptographic signing system or graph correctness oracle. Production persistence must use an atomic append-and-head-update transaction, durable event storage, authenticated actor identities and independently verifiable signatures/attestations. Concurrent submissions require serialized compare-and-swap on the canonical head.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-edit-transaction/Cargo.toml
```

Twenty-two unit tests are authored for valid commit, stale base, failed pre/postconditions, verifier failure/inconclusive, annotation propagation, review rejection, actor separation, duplicate event/idempotency IDs, inverse-patch requirements, rollback success/failure, parent lineage, monotonic event sequence, deterministic replay, replay tampering, sequence overflow and malformed digests.

They have **not** been compiled or executed in the current authoring environment (no `cargo`, `rustc` or `rustfmt`).

## Industry benchmark targets

CAVE's published MICrONS65 medians are approximately 4.116 s for merge operations and 5.813 s for split operations, with over four million historical edits across more than 500 users. This crate cannot claim parity: those are server-side operational measurements for a mature graph platform, while this crate currently provides only an in-memory receipt state machine.

Use these as reference targets for later end-to-end benchmarking:
- local ledger validation/replay overhead separately from graph patch application;
- apply, independent verify, annotation propagation and full transaction latency at p50/p95/p99;
- throughput and head contention under concurrent clients;
- zero false commits in stale, duplicate, failed-verifier, topology-mismatch and annotation-mismatch controls;
- deterministic event replay and rollback success;
- human minutes per independently verified correction.

See [SCI-CONNECTOMICS-057](https://github.com/Luminous-Dynamics/symthaea/issues/10018).
