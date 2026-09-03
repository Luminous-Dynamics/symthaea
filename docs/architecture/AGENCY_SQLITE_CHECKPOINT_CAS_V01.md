# Agency SQLite Checkpoint CAS v0.1

Status: **draft / unqualified**

## Purpose

Provide the first concrete persistent implementation of `CheckpointCasStore` for the Symthaea Agency Kernel.

The target property is not merely "store checkpoints in SQLite". It is:

> Two processes that start from the same trusted checkpoint head cannot both durably advance the accepted authority frontier.

## Transaction profile

`SqliteCheckpointCasStore` opens SQLite with:

- WAL journaling;
- `synchronous=FULL`;
- foreign keys enabled;
- bounded busy timeout.

Every CAS uses `BEGIN IMMEDIATE` **before reading the current frontier**. The exact compare and successor installation therefore occur under one serialized SQLite write transaction.

For an expected head `H`, the store performs the semantic transition:

```text
BEGIN IMMEDIATE
  read durable frontier
  require frontier == H
  update row WHERE sequence == H.sequence AND digest == H.digest
COMMIT
```

The SQL predicate is retained even though `BEGIN IMMEDIATE` already serializes writers; the store therefore has both transaction-level exclusion and an exact expected-head update predicate.

## Generation zero

For `expected_previous = None`, the transaction requires the frontier table to be empty and inserts the first checkpoint as the singleton row.

A second process attempting `None -> checkpoint-0` after bootstrap observes an existing frontier and fails with `Conflict`.

## Stored object

The row contains:

- checkpoint sequence;
- exact 32-byte checkpoint digest;
- complete bincode-v1 `GrantAccountCheckpoint` bytes.

On every read the implementation deserializes the checkpoint, recomputes `checkpoint.head()`, and requires it to equal the stored sequence/digest. Corrupt or mismatched bytes are never returned as a trusted frontier.

After a successful transaction commit, the store re-reads the row through SQLite and requires the durable head to equal the requested next head before returning success.

## Cross-process test

The focused test opens two independent SQLite connections to the same file.

Both begin with the same genesis head. Writer A advances the row. Writer B then attempts the same expected predecessor and must receive `Conflict`. The durable row remains Writer A's successor.

This tests the real provider boundary rather than an in-memory mutex/CAS mock.

## Durability boundary

`PRAGMA synchronous=FULL` instructs SQLite to use its strongest normal sync discipline for this journal profile. This is necessary but not magical hardware proof.

The production claim still depends on:

- the filesystem honoring sync requests;
- the block device/controller honoring flush ordering;
- the database path being on persistent storage;
- no rollback of the entire filesystem/VM snapshot beneath SQLite;
- correct deployment permissions and backup/restore procedures.

For stronger anti-rollback deployments, the latest `CheckpointHead` should additionally be retained outside the SQLite file (Xenia witness, TPM/supervisor monotonic state, remote append-only witness, etc.). SQLite provides the local single-writer/durable transition; it does not make itself an independent anti-rollback root of trust.

## Crash model

SQLite transaction recovery is responsible for incomplete database commits. The Agency Kernel remains conservative above it:

- reservation state is checkpointed before effect dispatch;
- ambiguous post-dispatch outcomes remain charged;
- a failed/uncertain checkpoint write causes containment in the CAS adapter/broker.

This crate does not weaken those semantics.

## Non-claims

v0.1 does not establish:

- immunity to filesystem/VM snapshot rollback;
- TPM measured-state binding;
- Byzantine database-host resistance;
- distributed multi-node consensus;
- remote-datacenter linearizability;
- secure file ownership/permissions for every deployment environment;
- hardware power-loss correctness beyond SQLite/filesystem sync guarantees;
- Xenia/Symthaea distributed transaction atomicity;
- production qualification.

## Qualification gate

Before promotion:

1. exact-head Rust format/check/Clippy/tests must pass;
2. the two-connection stale-writer regression must pass repeatedly;
3. a crash/power-cut fault-injection campaign should be added for the selected production filesystem/storage profile;
4. deployment must define ownership/mode/backup/restore rules;
5. external latest-head retention must be selected for rollback-sensitive deployments.
