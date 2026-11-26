# BASALT Storage Engine Decision: Do Not Collapse the Three Storage Guarantees

**Date:** 2026-11-01  
**Status:** Research ADR v0.1; no backend migration approved  
**Tracking:** [BASALT-STORE-QUAL-001 #9764](https://github.com/Luminous-Dynamics/symthaea/issues/9764)

## Decision summary

Do not choose one database for all of BASALT or Symthaea. Keep three roles separate:

1. **Local durable state:** retain SQLite via rusqlite with WAL + synchronous=FULL as the current BASALT reference implementation and compatibility baseline.
2. **Remote coordination/authority:** keep etcd as the primary implementation candidate for scoped, linearizable compare-and-advance. Add PostgreSQL 18 as a serious competing remote adapter, not as an assumed replacement.
3. **Independent anti-rollback:** neither SQLite, Turso, PostgreSQL nor etcd can by itself prove that an administrator did not restore an old internally consistent whole-store snapshot. Keep an independently administered witness/monitor holding an authenticated authority-incarnation and accepted-state commitment.

Add Turso Database as a Rust-native local SQL qualification arm. It is promising, but do not put it in the trusted path until it passes the same exact-byte, crash, power-loss, restore, and concurrency suite.

There is no blanket migration from SQLite to PostgreSQL or Turso in this ADR.

## 1. Resolve the terminology first

Three things are often conflated:

### PostgreSQL

PostgreSQL is a server database. Its engine is not Rust. It is a mature relational database with transaction isolation, write-ahead logging, replication, backup and recovery facilities.

### SQLx

SQLx is a Rust SQL client/toolkit, not a database. It supports PostgreSQL, MySQL/MariaDB and SQLite. Its PostgreSQL driver is pure Rust; its SQLite driver uses the native libsqlite3 library. Thus SQLx + PostgreSQL gives a Rust application a pure-Rust PostgreSQL client, not a Rust database engine. rusqlite is also a Rust API binding over SQLite, with this workspace enabling its bundled SQLite library.

References:
- https://docs.rs/sqlx/latest/sqlx/
- https://docs.rs/crate/sqlx/latest
- https://docs.rs/rusqlite/latest/rusqlite/

### Turso Database

Turso Database is an in-process SQL database engine implemented in Rust and designed for SQLite compatibility. It is a rewrite, distinct from Turso's earlier libSQL fork. It supports an experimental PostgreSQL frontend, but the PostgreSQL wire/dialect frontend is not a substitute for a mature PostgreSQL server deployment.

The current project README says the project is not yet 1.0, some compatibility gaps remain, and some features are experimental. Its changelog identifies release 0.8.0 (2026-09-28). It reports production use and deterministic simulation/Antithesis testing, which makes it a serious candidate for evaluation, but not automatically qualified for BASALT's durability boundary.

References:
- https://github.com/tursodatabase/turso
- https://github.com/tursodatabase/turso/blob/main/COMPAT.md
- https://github.com/tursodatabase/turso/blob/main/CHANGELOG.md

## 2. What Symthaea already does

The current storage architecture explicitly separates workloads:

| Role | Current approach | Decision |
|---|---|---|
| Hot cognitive state | In-memory structures | Keep out of synchronous DB reads |
| Durable local metadata and memory | SQLite / rusqlite | Retain |
| Embedding cache | redb | Retain as a cache/KV role |
| Persistent vectors/HDC | LanceDB and HDC store behind optional backends | Retain role separation |
| Analytics and telemetry | DuckDB / Parquet | Retain off the hot path |
| Distributed governance/history | Mycelix/Holochain bridge | Keep its own semantics |
| Remote BASALT authority | Proposed etcd; Postgres alternative | Benchmark both |

docs/architecture/STORAGE_ARCHITECTURE.md says new cognitive code should depend on narrow storage traits, not concrete engines, and that Postgres/pgvector should be added when a concrete query shape requires them. This is consistent with the BASALT decision: storage is a set of capability-specific components, not a contest to find one universal database.

The local BASALT prototype is a different durability class from a casual cache or genealogy ledger. Its SQLite WAL/FULL, immediate transaction and history-validation requirements should not be imposed on every SQLite user, but neither should a NORMAL-configured cache be substituted for it.

## 3. Local durable state: SQLite versus Turso

### SQLite / rusqlite baseline

Keep the present local reference backend:

- WAL journal mode;
- synchronous=FULL;
- explicit BEGIN IMMEDIATE;
- bounded busy timeout;
- one transaction for source cursor, log cursor, immutable receipt and policy/revision state;
- reopen/recovery and concurrency tests;
- independent external anti-rollback remains false/unknown.

SQLite documents that in WAL mode, synchronous=FULL adds a WAL sync at each transaction commit. WAL is same-host only; it is not a multi-host shared database. NORMAL may lose a recently committed transaction after power loss even though the database remains consistent.

References:
- https://www.sqlite.org/wal.html
- https://www.sqlite.org/pragma.html

### Turso qualification opportunity

Turso deserves a first-class benchmark arm because it is a Rust engine with SQLite-oriented SQL/file compatibility, native bindings and concurrent-write work. It is exactly the kind of alternative worth testing under BASALT rather than adopting based on language preference.

An important example: Turso issue #8194 documented an acknowledged commit that could miss its WAL fsync under synchronous=FULL on a fresh WAL generation / first commit shape. The issue is closed. Fix commit fa8aeccfcf50feb715801a76478336dca72935fa changed the predicate to include prepared frames as well as the dirty-WAL flag and added power-loss regression tests for a first fresh-database commit and a first commit after wal_checkpoint(TRUNCATE). The issue's verification comment reports the witness passed at the fix and current main.

This is not a reason to dismiss Turso. It is a strong example of why this qualification programme matters: the bug was subtle, the process page cache could hide it under process-kill testing, and the actual durability property needed a power-loss model.

Required Turso tests:
1. Exact pinned release and commit; no floating dependency.
2. Explicit PRAGMA synchronous=FULL, not just an assumed default.
3. First commit on a fresh DB.
4. First commit on every newly opened connection.
5. First commit after WAL checkpoint/truncate.
6. Small non-spilling and large/spilling transactions.
7. Commit after response acknowledgement followed by a power-loss model.
8. Reopen from the durable disk image, not the live page cache.
9. Concurrent writers and exact-retry recovery.
10. Differential SQL/schema behavior versus the exact SQLite version used as baseline.

Turso's own #8194 regression test is an input to our corpus, not the entirety of our corpus. We need an independent harness and an exact-version pin.

## 4. Remote authority: etcd versus PostgreSQL

These are both credible remote arms, but their operating models differ.

### etcd v3

Strengths:
- Coordination-oriented key/value data model.
- Native atomic compare/then transactions.
- Linearizable range operations by default.
- Small, narrow source/log/policy/receipt state fits the model.
- Scoped compares avoid an unnecessary application-level global counter key.

Costs/limits:
- Raft still orders cluster commits; scoped keys do not remove cluster-level consensus latency.
- Operationally another cluster to deploy, monitor, back up and restore.
- Whole-cluster snapshot restoration remains an anti-rollback problem.
- The numeric revision alone is not an independent witness. --bump-revision and --mark-compacted can make a restored cluster appear monotonic to watch consumers.

### PostgreSQL 18 + SQLx

Strengths:
- Relational constraints, indexes, joins and durable audit queries in the same service.
- Strong transaction isolation. PostgreSQL SERIALIZABLE emulates serial execution for committed transactions through Serializable Snapshot Isolation.
- Mature tooling, migration ecosystem, backups and operational expertise.
- Rust application integration via SQLx.
- Natural fit if Luminous/Mycelix ultimately needs multi-tenant relational workflows, policy queries, analytics metadata, operator tools and application state.

Costs/limits:
- A general-purpose relational server is heavier than a small coordinator.
- A SERIALIZABLE transaction may fail with SQLSTATE 40001; the complete transaction and all logic based on its prior reads must be retried. Do not retry only the final failed statement.
- WAL durability and synchronous replication must be explicitly configured and observed. PostgreSQL's default synchronous_standby_names is empty, so commits do not wait for a synchronous standby unless configured.
- A remote SQL database is not magically a consensus/anti-rollback witness; failover and restore topology still matter.
- PITR recovery creates timeline branches. A timeline/LSN is useful evidence, but it is not proof that the current authority descends from the latest independently witnessed application receipt.

References:
- https://www.postgresql.org/docs/18/transaction-iso.html
- https://www.postgresql.org/docs/18/mvcc-serialization-failure-handling.html
- https://www.postgresql.org/docs/18/runtime-config-wal.html
- https://www.postgresql.org/docs/18/runtime-config-replication.html
- https://www.postgresql.org/docs/18/continuous-archiving.html

### Recommendation

Keep etcd as the leading narrow coordination candidate because the remote BASALT contract is fundamentally scoped CAS and linearizable currentness. Build a PostgreSQL 18 reference arm before finalizing the decision. PostgreSQL may win operationally if the same service is also the application's relational system of record and we can prove synchronous durability/failover behavior under the intended threat model.

Do not choose PostgreSQL solely because it is mature, and do not choose etcd solely because it is called a consensus store. The selection must be workload- and threat-model-specific.

For the PostgreSQL arm, pin a configuration:
- fsync=on;
- synchronous_commit=on;
- full_page_writes=on;
- explicit synchronous_standby_names and commit wait policy for any replicated-durability arm;
- primary/linearizable authoritative reads only;
- SERIALIZABLE with complete retries, or a reviewed conditional-update/CAS transaction;
- a restore monitor that checks external accepted-state commitments, not only LSN or timeline numbers.

## 5. One state-machine contract across all engines

Keep the transition model pure and backend-neutral. Every adapter receives a plan bound to the full expected snapshot and must either:

- atomically advance the required cursor/receipt/policy state; or
- advance nothing and return a typed conflict/failure/indeterminate state.

Do not let database-specific retry behavior alter the admission algebra.

The remote retry API must distinguish:

- current_authority_revision: the revision of the current authority read/transaction response;
- receipt_acceptance_revision: the historical revision attached to the immutable receipt.

This preserves the current reference-model meaning of AlreadyCommitted.revision as the current state revision without losing the receipt's historical provenance.

The database under test must never be the only source of truth for the benchmark result about its own durability. Preserve raw fault schedule, durable disk image, expected/observed rows, exact engine version, configuration, and verifier receipt outside the tested store.

## 6. Common benchmark protocol

### Arms

A. SQLite WAL/FULL via rusqlite, local single host.  
B. Turso FULL via Rust binding, local single host.  
C. PostgreSQL 18 primary, local durable commit, via SQLx.  
D. PostgreSQL 18 with explicitly configured synchronous standbys.  
E. etcd v3, three and five members, linearizable reads and transactions.  
F. redb as a narrow local KV comparison only; not treated as SQL parity.

Keep local and remote operation classes separate. Publish network/quorum boundaries; do not compare unlike operations as if they were the same measurement.

### Common transaction

Persist together:
1. expected source generation/predecessor cursor;
2. expected log checkpoint cursor;
3. immutable receipt identity and original acceptance context;
4. relevant policy/witness pointer version;
5. current store/authority revision metadata.

The state-machine test fixture and receipt bytes must be identical across all arms.

### Fault matrix

- process termination before begin, during writes, after commit before response;
- OS crash and power loss immediately after acknowledged commit;
- failure during fsync or WAL flush;
- fresh connection/first commit and post-checkpoint commit;
- simultaneous writers from same and distinct source/log scopes;
- policy change between read and commit;
- exact retry after response loss and after later unrelated commits;
- local DB/backup/PITR snapshot restoration;
- etcd restore with revision bump and mark-compacted;
- PostgreSQL PITR/timeline branch;
- witness unavailable, stale, conflicting or independently restored;
- disk full, short write and storage I/O error.

SIGKILL is a process-crash control, not a power-loss model: the kernel page cache survives process termination. The test must model the durability boundary being claimed.

### Metrics

Safety gates (all mandatory):
- zero acknowledged commits missing from the modeled durable image;
- zero partial logical advancement;
- zero generation-equivocation acceptance;
- zero stale-policy commits;
- zero stale-read freshness promotions;
- zero duplicate semantic advancement;
- zero anti-rollback false-freshness after normal or revision-bumped restore;
- 100% seeded-fault detection within the declared supported model.

Efficiency and operations:
- p50/p95/p99 commit latency;
- accepted commits per second by concurrency and key skew;
- fsync/flush count and time;
- bytes written per accepted receipt;
- transaction conflict and retry rate by scope;
- CPU, memory, disk and network bytes;
- reopen/recovery and cache-rebuild time;
- operational/configuration/backup/restore complexity.

Do not build a weighted score that lets a durability failure be offset by throughput. Report raw vectors, configuration, confidence intervals, and failure schedules.

## 7. Relationship to coding-agent and wider engineering work

BASALT is the evidence/continuity substrate; the coding-agent benchmark is one producer and consumer of evidence, not a competing project.

- #8311's mutation-tested oracles establish whether the task suite can distinguish correct from plausible incorrect implementations.
- The coding-agent run receipt establishes which commit, task fixtures, compiler, oracle and execution actually ran.
- BASALT can bind and qualify those receipts, protect the accepted sequence, and prevent stale results from becoming current after rollback or fixture drift.
- The engine qualification suite itself becomes a coding-agent task family: agents must implement a backend adapter that passes the same mutation and crash corpus.
- Do not store the sole authoritative benchmark receipt inside the database engine being scored. The tested DB may index and serve records, but the raw signed result and external acceptance commitment must be independently retained.

This supports a shared evidence model across Symthaea, Mycelix and other engineering domains without forcing their workloads into one database.

## 8. Contributor onboarding boundary

A contributor with PostgreSQL/Prisma integration experience can make a useful first contribution without being assigned ownership of BASALT's trust semantics:

- provide a disposable PostgreSQL 18 service fixture;
- implement a small four-record compare-and-advance prototype;
- add migration, conflict, SQLSTATE 40001 whole-transaction retry, and lost-response exact-receipt tests;
- report actual configuration and test receipts;
- do not touch production credentials, signer keys, trust roots, merge gates, or the authoritative main-branch transition contract.

A separate maintainer must review the Rust adapter and security semantics. Demonstrated PostgreSQL application experience is valuable, but does not by itself qualify Rust systems, cryptographic or anti-rollback work.

## 9. Current recommendation

- **Keep:** rusqlite/SQLite WAL-FULL as the local BASALT reference baseline.
- **Benchmark:** Turso as a Rust-native local SQL candidate; it has encouraging testing machinery and a concrete fixed durability defect that our suite should independently reproduce.
- **Build:** PostgreSQL 18 + SQLx as a remote authority comparison arm.
- **Lead with:** etcd scoped CAS for the first narrow remote-authority implementation, pending the matched comparison.
- **Always separate:** the independent witness/anti-rollback plane.
- **Do not do:** a workspace-wide database migration or a one-engine mandate.

This recommendation is provisional until the common fault and concurrency corpus is executed on exact pinned versions and configurations.

## 10. Explicit non-claims

No benchmark results in this ADR are achieved results. No adapter, crash campaign, physical power-loss qualification, replicated Postgres HA qualification, or Turso qualification is claimed by this document. The Turso #8194 issue is cited as an upstream report and upstream fix/regression, not as our independent test result.
