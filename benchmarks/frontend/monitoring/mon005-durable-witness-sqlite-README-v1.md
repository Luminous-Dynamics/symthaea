# SFEB MON-005 — durable SQLite witness reference model v1

## Executed locally

- Node.js v22.16.0.
- **49/49 deterministic cases passed.**
- **496 Merkle inclusion proofs** verified across tree sizes 1–31.
- **2,016 RFC 9162-style consistency proofs** verified across all prior sizes m < n for every new tree size n = 2..64.
- Four Ed25519 test witnesses; **3-of-4 quorum** signed tenant-scoped checkpoints containing 12 then 18 records.
- Four new SQLite cases pass: close/reopen persistence, recovery after commit/lost acknowledgement, transaction rollback before commit, and fail-closed competing-writer contention.
- Raw and gzip-stdin executions produce byte-identical test output; both syntax checks pass.

Node prints an `ExperimentalWarning` because `node:sqlite` is experimental in v22.16.0. This is intentional for a rapid local preflight, not a claim that the SQLite Node API is production-stable.

## Run

```sh
node --check benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-v1.mjs
node benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-v1.mjs

gzip -dc benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-v1.mjs.gz | node --input-type=module --check -
gzip -dc benchmarks/frontend/monitoring/mon005-durable-witness-sqlite-v1.mjs.gz | node --input-type=module
```

Only Node.js built-in modules are used, including `node:crypto`, `node:sqlite`, `node:fs`, `node:os` and `node:path`.

## What changed

The previous witness implementation kept accepted checkpoint heads and challenge/signature receipts in memory. This version adds a file-backed SQLite backend for a witness's monotonic per-tenant head and challenge response ledger.

The SQLite write protocol uses:

- `BEGIN IMMEDIATE` before checking the challenge and latest tenant checkpoint;
- one transaction for the new witness checkpoint head and its challenge/signature receipt;
- `PRAGMA journal_mode=DELETE` and `PRAGMA synchronous=FULL`;
- a unique primary key on `(tenant_id, challenge_nonce)` to prevent challenge reuse for conflicting checkpoint content;
- exact retry returns the persisted signature rather than generating a new response.

The new failure tests exercise close/reopen of database handles, post-COMMIT lost acknowledgement, injected pre-COMMIT failure, and a competing connection attempting to write while the first holds the writer lock.

## Important limits

This is a same-process, same-host SQLite preflight. It does **not** establish independent witness operators, independent failure domains, durable cross-host CAS, distributed Byzantine fault tolerance, process-kill/power-loss behavior, filesystem or hardware durability, or production-grade key custody. SQLite's `FULL` and `DELETE` settings are configured but no power-cut or storage-device qualification was performed. The Node SQLite API emits an experimental warning.

The witnesses still use deterministic test keys. Canonicalization remains a strict local JSON subset rather than full RFC 8785 JCS. The proof generator still reads the full local event array; each witness verifier receives only the appended suffix and consistency/inclusion proofs. There is no PostgreSQL witness adapter, network service, key rotation/revocation, HSM/KMS, trusted timestamp authority, external transparency-log publisher, real external mutation sink, or independent postcondition readback.

A signed checkpoint certifies that the configured test keys signed the checkpoint. It does not prove the truth of the underlying source receipt, authorization, remote mutation or postcondition. Those must be verified independently at their own boundaries.
