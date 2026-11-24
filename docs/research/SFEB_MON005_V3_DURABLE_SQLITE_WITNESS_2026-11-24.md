# SFEB MON-005 v3 — durable witness state and restart/reconciliation preflight

**Date:** 2026-11-24  
**Status:** local Node.js/SQLite development reference model; 49/49 deterministic cases passed  
**PR:** https://github.com/Luminous-Dynamics/symthaea/pull/13518  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/13515

## Why this increment

MON-005 v1 added tenant-bound Merkle checkpoints and a 3-of-4 Ed25519 test quorum. V2 added RFC 9162 consistency proofs so a witness can verify an append-only extension without receiving the old record prefix. But the witness head and challenge receipts were still only in memory. A process restart would erase anti-equivocation state, defeating the central safety property.

V3 adds a file-backed SQLite state adapter for the witness's accepted per-tenant checkpoint head and the exact signature associated with each (tenant_id, challenge_nonce). This is a development preflight: it moves a critical invariant from an in-memory object into a transactionally updated local database, without pretending to qualify multiple independent operators.

## Executed evidence

Exact source: `benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-v1.mjs.gz`  
Report: `benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-report-v1.json`  
README/run instructions: `benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-README-v1.md`

Node.js v22.16.0:

- **49/49 deterministic cases passed.**
- **496 Merkle inclusion checks**; **2,016 consistency-proof pairs** across all 1 <= m < n for every new size n = 2..64.
- 3-of-4 Ed25519 test quorum signed tenant-scoped checkpoints of 12 and 18 records.
- Four new SQLite-specific cases passed:
  1. All four witness DB handles close and reopen; persisted checkpoint heads and challenge signatures remain; same challenge/checkpoint returns exact stored signature; changed checkpoint with same challenge is rejected.
  2. A simulated lost acknowledgement after successful COMMIT is reconciled after reopening; the exact signature is recovered from durable storage.
  3. An injected error before COMMIT rolls back both the checkpoint head and challenge receipt; a later retry succeeds.
  4. A second SQLite handle attempting BEGIN IMMEDIATE while a first handle holds the writer lock fails closed; after lock release the write succeeds.
- Raw and deterministic-gzip stdin execution are byte-identical; both raw and compressed syntax checks pass.

The source uses `node:sqlite`; Node v22.16.0 emits an ExperimentalWarning. The raw source is 57,698 bytes (SHA-256 `5573ef08bc19a512f8c8c2f5d07c8b60d092b433cebff1bab129b416f5b269e3`). The deterministic gzip is 12,433 bytes (SHA-256 `96beea3ee90281cef9f3eb945820117ef31a676b257e387c32a9dc4074da2e62`). Raw and compressed execution output SHA-256: `daa87d1f6f08786021e98bdb8d0187f35947f54e5e31e3f1f85bab641975e5a8`.

## Transactional design

Each witness instance has an independent SQLite file in the test harness. It persists:

- `witness_state`: one tenant's latest accepted epoch, tree size, Merkle root, checkpoint digest and last evidence-record digest;
- `witness_challenges`: (tenant_id, challenge_nonce), checkpoint digest and exact Ed25519 signature.

The accept path uses `BEGIN IMMEDIATE`, reads prior challenge and tenant state under the write transaction, verifies tenant identity, epoch/linkage, the RFC 9162 consistency proof, inclusion proof for each appended suffix record and the record hash chain, then inserts both the challenge receipt and new head in the same transaction. `PRAGMA journal_mode=DELETE`, `PRAGMA synchronous=FULL`, and `busy_timeout=0` are configured. A challenge retry for the exact checkpoint returns the signature already committed; a changed checkpoint attempting to reuse the challenge is rejected.

The transaction closes a local crash window: a witness cannot persist a new head without the corresponding challenge/signature receipt, or vice versa, under the exercised rollback case. It also provides a deterministic recovery behavior after a committed write whose response was lost.

## What this does not prove

- Four witness instances and separate SQLite files are exercised in **one process on one host**. This is not independent administration, network fault isolation, or independent durable witnesses.
- Close/reopen is same-process database-handle restart simulation, not an independently spawned process, abrupt process-kill matrix, power-loss, filesystem-corruption, storage-device or host-loss test.
- `synchronous=FULL` and `journal_mode=DELETE` are configuration settings, not proof of physical power-loss durability on a particular filesystem/device.
- `node:sqlite` is experimental in Node v22.16.0.
- Keys are deterministic test fixtures, not HSM/KMS-managed production identities. No key rotation, revocation or trusted clock is present.
- Canonicalization is not a full RFC 8785 implementation.
- The log proof generator still receives the full local event array, even though witness verification receives only the appended suffix plus compact consistency and inclusion proofs.
- No PostgreSQL adapter, cross-host CAS, external transparency-log integration, real external mutation sink, sink-side fence, or independent postcondition readback is implemented.

Therefore: **this is a local transactional-state preflight, not a distributed BFT or production durability result.**

## Industry fit and research direction

RFC 9162 supplies interoperable Merkle Tree Hash inclusion and consistency proofs. Rekor/Certificate Transparency-style systems provide public append-only commitments and monitoring. SFEB should publish or cross-anchor checkpoint digests to an established transparency ecosystem instead of replacing it. RFC 8785 JCS remains the canonical serialization target before cross-language signatures are trusted.

DBOS Go's RunAsTransaction is the fair same-database atomic workflow baseline; DBOS Rust steps are a separate at-least-once baseline; Temporal supplies durable event history/replay. SFEB must show incremental value only at the action evidence boundary: current tenant authority, objective fencing at the external sink, same operation key through lost-ack retry, independent readback, and externally witnessed proof. No throughput or SOTA superiority result is claimed by this local SQLite run.

## Next qualification gates

1. Replace the Node SQLite backend with four independently deployed witness services, each with durable monotonic state and an atomic per-tenant/per-epoch uniqueness/CAS guard.
2. Persist each witness's key identity and rotation/revocation metadata; provision keys via independent KMS/HSM boundaries.
3. Run independent process and host tests: kill/restart, DB restore, WAL/DELETE journal recovery, simulated power-cut where possible, filesystem errors, clock skew, network partitions and conflicting checkpoint races.
4. Differential-test RFC 9162 inclusion/consistency proofs against a mature independent implementation, and complete RFC 8785 JCS cross-language vectors.
5. Add independent witness gossip/split-view checks; anchor the checkpoint digest to Rekor/Tessera or a CT-compatible service.
6. Integrate one real idempotent/conditional-write sink, require sink-side objective/policy fencing, and verify outcome through a separately authorized readback.
7. Benchmark matched PostgreSQL outbox, DBOS Go transaction, DBOS Rust step and Temporal native/augmented arms, reporting synchronous action latency separately from asynchronous witness/anchor latency.

## Proposed performance and safety targets (not achieved)

- Zero accepted unauthorized or cross-tenant effects, conflicting witness heads, rollback checkpoints, or rewritten witnessed prefixes over at least 1,000,000 independently denominated adversarial attempts per release candidate under the stated at-most-one-Byzantine-witness assumption.
- 100% of results labelled externally_anchored independently verify from source, authority, execution and sink postcondition receipts, inclusion and consistency proof, quorum signatures and trusted checkpoint chain.
- Unknown sink outcomes remain unknown until authoritative readback; never mint a new operation key to escape reconciliation.
- <=10% p95/p99 incremental overhead on matched native external-action workloads. Report action execution separately from time-to-proof and time-to-external-anchor.
- Report checkpoint interval, proof bytes, signature CPU, witness egress, SQLite/PostgreSQL write lock wait, recovery time, storage growth and total cost per independently verified external action.

## References

- RFC 9162: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 8785: https://www.rfc-editor.org/rfc/rfc8785.html
- Sigstore Rekor security model: https://docs.sigstore.dev/about/security/
- Trillian/Tessera: https://google.github.io/trillian/
- DBOS Go transaction tutorial: https://docs.dbos.dev/golang/tutorials/transaction-tutorial
- DBOS Rust step tutorial: https://docs.dbos.dev/rust/tutorials/step-tutorial
- Temporal event history: https://docs.temporal.io/encyclopedia/event-history
