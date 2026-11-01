# BASALT SQLite WAL/FULL durable trusted-state backend — 2026-11-01

## Scope

This is the first durable local backend prototype for the trusted-state reference model in `xtask/src/basalt_trusted_state.rs`.

It narrows the implementation claim to a single-host local database:
- one normalized state-meta row with an explicit schema version;
- one row per transparency-log cursor;
- one row per stable source cursor;
- immutable accepted-receipt rows, each assigned a unique monotonic acceptance revision and retaining the transition-evidence digests used for profile, boot, log-origin and policy changes;
- one SQLite write transaction for every accepted state advancement.

It does **not** provide a remote authoritative anchor, malicious snapshot rollback protection, cryptographic verification, witness consensus, transparency freshness, observer qualification or close-coordinator qualification.

## Pinned local configuration

Every connection sets and verifies:
- `journal_mode=WAL`;
- `synchronous=FULL` (numeric pragma value 2);
- `foreign_keys=ON`;
- a 5-second busy timeout;
- bounded state cardinalities: 128 log cursors, 4,096 source cursors and 100,000 accepted receipts;
- open with rusqlite's SQLITE_OPEN_NOFOLLOW (without URI parsing), reject an existing symlink/non-regular file, and set the database file mode to 0600 on Unix. The containing directory must still be trusted against concurrent path replacement.

The backend starts each state mutation with `BEGIN IMMEDIATE`. It reads the complete current trusted state after obtaining the SQLite writer transaction, checks both the revision and exact expected state snapshot, then writes the source cursor, log cursor, accepted receipt and policy/revision record before committing.

The expected-state comparison is important. A revision-only compare is insufficient if two divergent local snapshots happen to carry the same revision number. The SQL write is additionally protected by primary/unique keys:
- one current source cursor per (host, source, logical partition lineage);
- one log cursor per log ID;
- one accepted envelope digest per receipt;
- one accepted receipt per (host, source, partition, generation).

The source key does not contain observer profile or boot ID. Those remain attributes in the source cursor and accepted receipt, so a profile update or reboot cannot create a new cursor and reset generation history.

Unsigned 64-bit revision/generation/tree-size values are stored as canonical decimal text because SQLite INTEGER is signed 64-bit. They are parsed with checked Rust conversion on read.

## Transaction and recovery behavior

The transaction writes in this order:
1. source cursor;
2. log checkpoint cursor;
3. immutable accepted-receipt row;
4. state revision, policy versions and deterministic state digest.

All four writes execute inside one `BEGIN IMMEDIATE` transaction. Any failure before commit returns while the transaction is live; dropping the rusqlite transaction rolls it back. Tests inject failures after each statement boundary and reopen the database to check that the complete pre-state remains.

An accepted-receipt retry is resolved by exact envelope digest and source/generation/predecessor/seal/profile/boot identity. It returns the prior acceptance decision without advancing state a second time. This remains valid if the current transparency checkpoint has since moved forward; the original checkpoint and policy are retained as the receipt's historical acceptance context.

A commit error is returned as `CommitOutcomeIndeterminate`, not guessed as a failure. The caller must reopen/re-read and query the exact receipt digest:
- exact accepted receipt present and bound to the expected source/generation/seal => already committed;
- exact receipt absent and old state intact => re-verify and retry the full transition;
- inconsistent or unavailable state => indeterminate / fail closed.

The current code's reopen + exact-retry test covers recovery after a committed receipt is reopened. The reader also checks that every accepted receipt revision from 1 through the current revision exists exactly once, every source history begins at generation 1 and advances without gaps with exact predecessor links, and the latest source cursor points to the newest accepted record. It checks that each profile/boot/log-origin transition retains its evidence reference and that global policy versions never roll back. A local cursor-only rollback is rejected even when the unkeyed local state digest is recomputed. It does not inject a transport-level lost acknowledgement at the SQLite API boundary or simulate kernel/power failure.

## Local state digest and trust boundary

The database stores a domain-separated SHA-256 digest over the normalized state. It is recomputed on every read and detects accidental or inconsistent database edits, partial out-of-band edits, and serialization/row mismatches. Schema version 1 is explicitly stored and checked. Each accepted receipt also stores its monotonic acceptance revision and all transition-evidence digest references, so reopening can verify the complete immutable history rather than only the latest cursor.

This digest is **unkeyed**. An attacker who can restore an older database or rewrite every row and recompute the digest can still create an internally consistent stale state. Therefore:
- local snapshot digest consistency is not anti-rollback;
- SQLite commit durability is not external freshness;
- local transaction success is not signer authorization;
- SQL uniqueness is not witness non-equivocation across independent trust domains.

The report hard-codes `anti_rollback_qualified=false`, `signature_verification_performed=false` and `external_freshness_established=false`.

The next production trust layer remains a remote authoritative compare-and-advance record or separately qualified hardware/witness anchor outside the local backup/restore domain. The anchor must commit the complete source cursor, log cursor, receipt digest, policy versions and monotonic revision as one authoritative record. The SQLite database is then a recoverable materialized cache.

## Filesystem and deployment constraints

SQLite WAL is a same-host multi-process backend; its shared-memory wal-index is not designed for clients on different machines using a network filesystem. The backend caps state at 128 log cursors, 4,096 source cursors and 100,000 accepted receipts so each read can safely materialize the complete state for exact-snapshot comparison. The database must live on a tested local filesystem with correct locking and durable sync semantics. Never place this WAL database on NFS or an unqualified network share.

The `FULL` pragma asks the SQLite VFS to synchronize the WAL at transaction commit. Actual durability still depends on the VFS, filesystem, storage controller, device write-cache and hardware honoring flush/barrier semantics. A pinned reference machine and fault-tested storage stack are required before making a deployment durability claim.

WAL allows readers to proceed alongside a writer, but SQLite still serializes writers. This backend is a local single-writer authority, not the multi-host remote anchor.

## Commands

Inspect or initialize a local database:

```sh
cargo run -p xtask -- basalt-trusted-state-sqlite-inspect --database ./basalt-trusted-state.sqlite
```

The report includes WAL mode, synchronous level, current revision, cursor/receipt counts and state digest. The three non-claims remain false.

Reference-model command:

```sh
cargo run -p xtask -- basalt-trusted-state-model-check
```

Unit tests:

```sh
cargo test -p xtask basalt_trusted_state
```

## Authored tests

The module now contains 24 Rust unit tests, including:
- database creation and state persistence across close/reopen;
- WAL and FULL pragma inspection;
- exact accepted-receipt retry after restart;
- injected transactional abort after source, log, receipt and metadata writes;
- two independent connections racing the same prepared plan;
- same-revision but divergent database snapshots;
- eight independent SQLite connections racing the same state plan;
- source-generation uniqueness and cross-table foreign-key constraints exercised by the SQLite schema;
- unkeyed state-digest mismatch rejection.

These tests are authored, not executed in this environment. The deterministic write-boundary injection checks SQLite transaction rollback, not real power-loss behavior, filesystem locking under every platform, VFS durability, or hardware flush integrity.

## Benchmark protocol

The accompanying benchmark defines:
- atomicity/reopen and exact-retry gates;
- transaction-abort injection at every SQL write boundary;
- independent-connection CAS contention;
- state snapshot fork rejection;
- schema uniqueness and foreign-key enforcement;
- state-digest mismatch rejection;
- explicit anti-rollback unknown behavior.

Efficiency measurements must report p50/p95/p99 commit latency, throughput, lock wait, busy/conflict rates, bytes written per accepted epoch, WAL growth/checkpoint behavior, restart/recovery time and database size over increasing receipt history. Compare WAL/FULL against WAL/NORMAL only as an explicitly unsafe durability control; do not use NORMAL results to qualify the FULL production profile.

Initial research targets, not results:
- zero partial state advances in all injected failures;
- zero duplicate accepted generations;
- zero digest-mismatch acceptance;
- exact receipt replay resolves without a second state advance;
- p95 local durable commit <= 5 ms at up to 1,000 accepted receipts on a pinned NVMe reference host;
- p95 <= 25 ms with 16 competing same-host writers at up to 1,000 receipts, with busy timeout and conflict counts reported separately;
- publish unthresholded scaling curves at 10k and 100k receipts because the v1 reference store intentionally materializes the bounded state history for full transition validation.

## Industry positioning

SQLite is the local transactional baseline, not the external anti-rollback baseline. A remote candidate such as etcd offers atomic transactions guarded by key/value and revision comparisons; linearizable reads/writes have a different latency and availability profile because they coordinate through Raft. A serializable/stale read must not be used for an admission freshness check unless followed by a linearizable compare-and-advance.

Benchmark the local SQLite adapter and a remote authority adapter separately. Do not equate an etcd cluster's ordinary linearizability with protection against a malicious whole-cluster snapshot restore; the cluster backup, provisioning and witness trust domain must be included in the threat model.

## References

- SQLite WAL: https://www.sqlite.org/wal.html
- SQLite synchronous pragma: https://www.sqlite.org/pragma.html
- SQLite file locking and corruption caveats: https://www.sqlite.org/howtocorrupt.html
- SQLite over network: https://www.sqlite.org/useovernet.html
- rusqlite transaction API: https://docs.rs/rusqlite/0.37.0/rusqlite/struct.Connection.html
- etcd v3.7 transaction API: https://etcd.io/docs/v3.7/learning/api/
- etcd API guarantees: https://etcd.io/docs/v3.3/learning/api_guarantees/
- BASALT remote trusted-state contract: https://github.com/Luminous-Dynamics/symthaea/issues/9612

## Validation status

No Rust compile, rustfmt, test execution, real process-kill test, kernel crash test, power-loss test, filesystem/VFS qualification, performance benchmark, remote-anchor test or external anti-rollback qualification is claimed. The SQLite DDL was independently executed against a local Python SQLite engine: table creation, foreign-key check and uniqueness constraint behaved as intended. This is schema validation only, not validation of the Rust adapter.
