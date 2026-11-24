# SFEB MON-004: PostgreSQL outbox, external fencing and fair industry comparison

**Date:** 2026-11-24  
**Status:** SQL reference contract committed; PostgreSQL execution pending  
**Tracking issue:** [#13497](https://github.com/Luminous-Dynamics/symthaea/issues/13497)  
**Predecessor:** [MON-003 SQLite persistence preflight](SFEB_MON003_SQLITE_PERSISTENCE_AND_EVIDENCE_JOURNAL_2026-11-24.md)

## Why this is the next frontier

MON-003 now has a local persistence preflight: 38/38 deterministic cases pass on Node.js 22.16.0 and SQLite 3.49.1. Five fresh-process sequential runs gave a median of 482.3 cycles/s (range 422.4–520.4), with median p95 local acceptance of 1.023 ms and p95 dispatch/readback of 1.980 ms.

Those are single-host SQLite smoke measurements, not SOTA comparisons. They do not test PostgreSQL MVCC under contention, multi-process worker claims, broker delivery, multi-host failure, or a real external sink.

The next challenge is to demonstrate that the same evidence/authority contract survives concurrent workers without confusing these three independent identities:

1. **Objective fence:** current authority for the actual external mutation, scoped by tenant and objective.
2. **Outbox claim lease:** temporary ownership of one delivery row, with its own epoch and claim token.
3. **Operation idempotency identity:** stable tenant/action key and request digest across every retry and worker replacement.

A claim token is not proof of authority to mutate the external system. A database row lock is not an external-system fence. A lost acknowledgement does not justify generating a new operation key.

## SQL reference artifact

`benchmarks/frontend/monitoring/mon004-postgres-outbox-contract-v1.sql`

The SQL artifact defines:
- tenant policy revision and objective fencing rows;
- immutable source receipts plus separately recorded revocations;
- operation and resource state;
- outbox event identity, aggregate key/sequence, claim epoch/token/worker lease and separate remote-outcome state;
- tenant-scoped evidence-chain heads and append-only journal rows;
- short `FOR UPDATE SKIP LOCKED` claim transactions;
- pre-network persistence of `remote_outcome='unknown'`;
- worker completion guarded by the exact worker identity, claim epoch, claim token and unexpired claim lease;
- per-aggregate ordering so later events cannot overtake an earlier undelivered event.

The SQL deliberately treats a delivery claim lease as separate from the objective fence checked by the remote mutation sink. It is a reference contract, not a complete migration, stored procedure library or tested adapter.

**Execution status:** not executed. The environment had no `psql` or PostgreSQL server. SQL syntax, locking behavior, isolation anomalies, performance and recovery are unqualified. The next implementation must run against a pinned PostgreSQL release and include multi-process tests.

## Industry comparison: use the strongest native baseline

### DBOS Go datasource transaction

DBOS Go's `RunAsTransaction` executes application writes and the DBOS durability/completion record in one datasource transaction. Its docs specify that this is exactly-once for that same-database transaction even across a crash/recovery; sharing the same engine handle as the system database collapses it into one transaction. A plain outbox implementation must not be compared against a weaker alternative that omits this native baseline.

Sources: [DBOS Go transaction tutorial](https://docs.dbos.dev/golang/tutorials/transaction-tutorial), [datasource reference](https://docs.dbos.dev/golang/reference/datasources).

### DBOS Rust steps

DBOS Rust steps execute at least once: if a side effect happens and the process crashes before checkpoint completion, the step can be retried. The Rust step tutorial currently states there is no transactional step yet. This is a different baseline from Go's datasource transaction, so measure it separately and use the same idempotent external sink.

Sources: [DBOS Rust steps](https://docs.dbos.dev/rust/tutorials/step-tutorial), [DBOS Rust workflow/step reference](https://docs.dbos.dev/rust/reference/workflows-steps).

### Temporal

Temporal's durable event history records commands and results, then replays workflow code after worker failure to reconstruct state. SFEB should integrate with this machinery, not rebuild it. Its incremental value must be measured at the source-provenance, current external-authority, unknown-outcome reconciliation and independent postcondition boundary.

Sources: [Temporal event history](https://docs.temporal.io/encyclopedia/event-history), [Temporal durable execution](https://docs.temporal.io/durable-execution).

### PostgreSQL SKIP LOCKED

PostgreSQL documents that `SKIP LOCKED` deliberately produces an inconsistent view and is suitable for queue-like consumers, not general-purpose consistent reads. This is why the SQL uses it only to claim outbox rows and keeps authoritative state/policy checks elsewhere. Aggregate ordering requires a separate head-of-stream predicate; simply ordering the overall queue does not guarantee per-aggregate order after workers skip locked rows.

Source: [PostgreSQL SELECT and row-locking clauses](https://www.postgresql.org/docs/current/sql-select.html).

### DBOS throughput numbers

DBOS's production checklist reports a provider benchmark of a single 16-vCPU PostgreSQL server sustaining 20K workflows or steps per second. This is a provider-reported sizing reference, not an independently reproduced universal throughput figure. MON-003's local SQLite number is not directly comparable; hardware, workload, concurrency, write count, durability settings and network boundaries differ.

Source: [DBOS production checklist](https://docs.dbos.dev/production/checklist).


### Tenant-scope schema audit follow-up

A static review found the source-receipt relation needed tenant scope in its referential integrity, not just in application checks. The SQL contract now declares `UNIQUE (tenant_id, receipt_id)` and uses composite `(tenant_id, receipt_id)` foreign keys for both operation receipt references and receipt-revocation records. This prevents a correctly formed receipt ID from being attached to another tenant's operation or revocation row. This is a source-level correction only; it remains pending execution against PostgreSQL.

## Executed pre-PostgreSQL claim/fence model

Before a real PostgreSQL runtime was available, a bounded reference model was added to test the separate claim lease and objective fence contract:

- Source: `benchmarks/frontend/monitoring/mon004-postgres-claim-model-v1.mjs.gz`
- Report: `benchmarks/frontend/monitoring/mon004-postgres-claim-model-report-v1.json`
- Run instructions: `benchmarks/frontend/monitoring/mon004-postgres-claim-model-README.md`
- **21/21 deterministic cases passed** under Node.js v22.16.0.
- **5,040 sequential operation orderings explored** (all permutations of seven lifecycle steps).
- Observed across the finite model: 378 mock remote effects; 1,062 stale-claim rejection observations; 2,772 schedules in which read/reconciliation evidence was unavailable at the modeled point; 0 duplicate effects; 0 invariant violations.

The model separately exercises outbox claim epoch/token, tenant/objective action fence, persisted uncertainty before remote mutation, same-key remote idempotency, separate reader grants, per-aggregate ordering, cross-tenant isolation and independent readback before delivery completion.

This increases confidence in the **contract ordering**, not in PostgreSQL itself. All schedules are sequential permutations; the model assumes an atomic remote sink and deterministic clocks. It is not PostgreSQL MVCC, a concurrent memory model, network-partition testing or a reliability probability. The SQL contract still must be applied to a pinned PostgreSQL version, run with multiple real processes, and tested with crash/failover injection.

Reproduce:
```sh
gzip -dc benchmarks/frontend/monitoring/mon004-postgres-claim-model-v1.mjs.gz | node --check -
gzip -dc benchmarks/frontend/monitoring/mon004-postgres-claim-model-v1.mjs.gz | node -
```

## Matched benchmark matrix

All arms receive identical event traces, tenant policies, external sink behavior, idempotency keys, payload sizes and injected failure schedules.

| Arm | Native baseline | SFEB comparison |
|---|---|---|
| A | PostgreSQL transactional outbox + worker | Baseline local atomicity and multi-worker queueing |
| B | DBOS Go `RunAsTransaction` on the same datasource | Native atomic app writes + DBOS completion row |
| C | DBOS Go transaction + SFEB external boundary | Additional source evidence, live sink authority, reconciliation and readback |
| D | DBOS Rust step + idempotent external sink | At-least-once step semantics |
| E | DBOS Rust step + SFEB | Additional external-action assurance |
| F | Native Temporal workflow/activity + idempotent sink | Durable event history/recovery baseline |
| G | Temporal + SFEB | Additional external-action assurance |
| H | Kafka transactional consume/produce + external sink | Broker-contained transaction vs external side-effect boundary |
| I | Kafka + SFEB sink/evidence adapter | Additional cross-system assurance |

Measure both steady state and recovery:
- 1, 8, 32, 128 and 512 workers; separate process counts and connection pool sizes;
- throughput and p50/p95/p99 local transaction, claim, remote action and independent readback latency;
- lock wait, skipped claims, per-aggregate lag, oldest pending age and fairness;
- retries per successful action, remote duplicate-effect count and outcome-unknown duration;
- storage/IOPS, evidence size, signature-verification cost and cost per independently verified external action;
- time to converge after crash, permission revocation, sink timeout and connector replay gap.

Failure strata:
- concurrent claim contention and worker death while holding a claim;
- stale claim completion after lease replacement;
- objective fence changes after claim but before remote commit;
- policy or receipt revocation between acceptance and sink execution;
- remote commit + lost ACK + DB restart;
- outbox duplicate, out-of-order delivery, aggregate sequence fork;
- incorrect or stale independent readback;
- Postgres failover, replica lag, serialization retry and clock skew.

## Proposed targets—not achieved results

- 0 unauthorized or cross-tenant mutations/disclosures over at least 1M adversarial attempts per release candidate, with exact per-family denominators.
- 0 duplicate external effects for one stable operation identity over at least 1M duplicate/retry schedules against an idempotent test sink.
- 100% of claimed successes linked to source, authority, execution and independent outcome receipts.
- 100% of effects with durable sink evidence reconcile to the original operation identity; absent proof remains `outcome_unknown`.
- Added SFEB p95/p99 overhead <=10% on matched native external-action workloads, with absolute latency and total cost published.
- No throughput superiority claim until DBOS/Temporal baselines are reproduced with the same workload, hardware and durability configuration.

## Promotion gates

1. **Static SQL review:** schema, lock order, constraint and migration audit.
2. **Real PostgreSQL:** apply schema and execute all local atomicity cases; capture server version and settings.
3. **Multi-process:** concurrent claims, lease expiry, stale completion and aggregate ordering under process kills.
4. **Failure injection:** crash at each local/remote boundary, acknowledgement loss, DB restart/failover and readback disagreement.
5. **Native baseline:** reproduce DBOS Go datasource transaction, DBOS Rust step and Temporal activity arms.
6. **External system:** qualify one real idempotent or conditional-write sink and independent readback.
7. **Evidence anchoring:** sign periodic per-tenant evidence-chain heads and store checkpoints outside the database administrator boundary.

**Decision:** MON-003 establishes a local SQLite preflight. MON-004 is the next proof obligation: PostgreSQL concurrency plus fair, version-aware DBOS/Temporal comparison. SFEB should compete on evidence and authority assurance at the external mutation boundary, not claim ownership of durable workflow execution or transactional outbox patterns that existing systems already provide.
