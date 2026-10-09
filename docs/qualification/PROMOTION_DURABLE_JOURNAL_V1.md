# Durable Promotion Journal v1 — Reference Boundary

## Purpose

This tranche is a disk-backed reference transaction model for the reservation/dispatch boundary described by PR #7210. It is deliberately separate from the in-memory Python ledger and the Rust transaction reference. It demonstrates the minimum journal contract a later production adapter must satisfy; it does not claim that a production dispatcher now uses this journal.

## Transaction boundary

The reference implementation is in **scripts/qualify_promotion_durable_journal_v1.py**. It uses Python's standard-library SQLite binding and a file-backed database.

- Reservation updates the materialized ledger state, inserts the reservation row, and appends its event in one BEGIN IMMEDIATE transaction.
- Dispatch preparation conditionally advances the ledger revision, records the positive attempt sequence and wall/monotonic readings, changes the reservation state, and appends the prepared-dispatch event in one transaction.
- A failed event append rolls back the ledger revision and reservation state as well as the journal write.
- A silent reservation compare-and-swap miss after the ledger revision update is treated as an invariant violation and raises, forcing the surrounding transaction to roll back rather than accidentally committing a revision-only update.
- A prepared attempt identity is reconstructed from the committed reservation row and its matching journal event. Caller-supplied fields alone cannot mint an identity through this API.
- The storage trigger checks the current head, lease, fencing token, trust-root generation, governance generation, and expected revision during the reserved-to-prepared transition. The application also uses a compare-and-swap update under BEGIN IMMEDIATE.
- An authority-fence rotation supersedes an unprepared reservation. A prepared reservation is not silently revoked by this reference model; it blocks rotation until a separate reconciliation/terminal-state protocol is specified.

## Journal integrity

Each event has a unique ID, monotonically allocated sequence, canonical payload, previous-event digest, and SHA-256 event digest. Verification checks the hash chain, sequence continuity, SQLite integrity and foreign keys, required trigger presence and critical SQL fragments, event replay, materialized ledger state, and reservation rows. This catches a same-name trigger replacement that drops the expected guard clauses, not just a missing trigger. SQLite triggers reject ordinary journal-row update/delete operations. A separate ledger-state trigger rejects fence regression, revision jumps, and head/lease/generation mutation without advancing the fence.

These are tamper-evidence and consistency checks, not protection against a fully privileged database owner. An owner able to alter the schema and rewrite every event can rebuild the hash chain. More importantly, SQLite does not provide per-connection database roles: any process with unrestricted write access to the file can attempt arbitrary SQL, including inserting fabricated rows/events or changing the schema. Therefore this model only demonstrates state-machine guards under a trusted database-writer boundary; it does **not** provide identity-aware fencing against a malicious or fully privileged writer. A production deployment needs a narrowly scoped storage service/API or database authorization boundary that does not give stale workers arbitrary mutation rights, plus an independently controlled signed/checkpointed digest or append-only evidence authority.

## Time semantics

Reservation and dispatch wall/monotonic readings are explicit inputs. The journal proves that the recorded readings were committed together and satisfy the model's ordering checks. It does not prove they came from a trusted clock, that monotonic clocks remain reliable across reboot/VM migration, or that the stated readings reflect physical time. The monotonic clock identifier is an identity label, not hardware attestation.

## Qualification corpus

The dedicated workflow executes on pushes to the reference branch and on pull-request events. It explicitly checks out and compares against the PR head SHA (rather than silently testing only a trusted default-branch checkout), uses read-only repository permissions, disables persisted checkout credentials, and records the tested SHA in the workflow summary.

The adversarial corpus covers durable reopen/reconstruction, single-use positive attempt sequences, rejected clock/order inputs, commit rollback after injected journal-write failure, rollback after a silently ignored compare-and-swap update, SQL-trigger fencing, stale writers after authority rotation, concurrent dispatch races across two connections, append-only triggers, rejection of same-name weakened trigger definitions, event-chain tampering, and materialized-state drift.

A workflow PASS would qualify only this reference model on the exact recorded Git head.

## Nonclaims and next integration boundary

This is not yet connected to a production dispatch service, lease issuer, authority-generation store, durable deployment journal, or external provider. It does not establish production authority, cross-host coordination, provider-side compare-and-swap, causality, merge success, hardware/OS clock guarantees, crash durability under every filesystem/storage stack, or production atomicity. SQLite WAL mode and synchronous FULL configure the reference's local persistence behavior; the actual deployment must qualify its database, filesystem, backup/restore, writer-fencing, and recovery configuration independently.

The next production tranche must identify the real authoritative storage owner and transaction API, then derive PromotionTemporalAttemptIdentityV1 only from a committed, independently verifiable reservation/dispatch record. It must exercise stale-writer fencing and crash/restart recovery against that real boundary. Until then, this remains source and local-model evidence only.
