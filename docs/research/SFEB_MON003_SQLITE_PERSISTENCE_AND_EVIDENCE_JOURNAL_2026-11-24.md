# SFEB MON-003: persistent operation ledger, transactional outbox and evidence journal

**Date:** 2026-11-24  
**Status:** Local executable SQLite preflight; 38/38 deterministic cases passed  
**Related research PR:** [#13398](https://github.com/Luminous-Dynamics/symthaea/pull/13398)

## Executive result

This increment moves MON-003 from in-memory state-machine references to a file-backed persistence model. The objective acceptance transaction writes the operation record, resource reservation intent, outbox event and causally ordered evidence-journal entries in one local SQLite transaction. A separate SQLite database models the remote sink, with its own live permission revision, tenant/objective fencing epoch, idempotency key and resource state. A third database handle performs independent readback.

The critical failure model is not “did the worker return success?” It is:

1. Local intent commits atomically.
2. Before crossing the remote boundary, the local operation is marked `outcome_unknown`.
3. The remote sink may commit and the acknowledgement may be lost.
4. Retry uses the exact same tenant/action key and request digest.
5. The sink returns the pre-existing operation identity instead of repeating the side effect.
6. A separate read handle verifies the remote operation and resource postcondition.
7. Only then does the local ledger move to `remote_committed`.

A missing or mismatched readback leaves the local state `outcome_unknown`. It is not reclassified as success, and the action key is not replaced.

## Executed artifacts

- Source: [`mon003-sqlite-transactional-outbox-v1.mjs`](../../benchmarks/frontend/monitoring/mon003-sqlite-transactional-outbox-v1.mjs)
- Machine-readable report: [`mon003-sqlite-transactional-outbox-report-v1.json`](../../benchmarks/frontend/monitoring/mon003-sqlite-transactional-outbox-report-v1.json)

**Local result:** Node.js v22.16.0, SQLite 3.49.1, **38/38 deterministic cases passed**. Node's `node:sqlite` API reports an experimental warning in Node 22; this is a toolchain limitation as well as a qualification caveat.

### Test families

- One local transaction commits or rolls back the operation, resource intent and outbox together. Failure injection after each of the three writes leaves no partial operation.
- Tenant/objective fence scope, live action policy revision, source receipt validity and resource compare-and-set are checked before local acceptance.
- Same-key/same-digest retries require read-only reconciliation; same-key/changed-digest attempts are rejected.
- Read grants bind tenant, objective, principal, operation, permission revision and validity interval. Reader revocation, expiry, principal substitution and cross-tenant access fail closed.
- The remote sink independently checks its live policy revision and tenant/objective fence before a new mutation. Its own unique operation key and digest prevent repeat effects.
- Pre-commit remote failure, timeout after remote commit, acknowledgement loss, close/reopen recovery, altered request digest and independent postcondition mismatch are exercised.
- Source receipts are immutable. Revocation is a separate append-only record carrying timestamp, reason and authority.
- A hash-chained evidence journal records accepted objectives, reservation intent, outbox enqueue, revocation, dispatch attempts, postcondition verification and remote commit. The verifier detects modified event content; triggers reject normal row updates/deletes.

### Local benchmark

Five fresh-process repetitions, each with 200 sequential end-to-end cycles, yielded:

| Metric | Observed local result |
|---|---:|
| End-to-end cycles/s, median | 482.3 |
| End-to-end cycles/s, range | 422.4–520.4 |
| Local-accept p95, median across runs | 1.023 ms |
| Remote dispatch + independent readback p95, median across runs | 1.980 ms |

Environment: one container host, single-process sequential test workload, two file-backed SQLite databases, WAL mode and `synchronous=FULL`, no network and no broker. These values are **not SOTA comparisons or production SLOs**. They must not be compared directly to multi-vCPU PostgreSQL/DBOS claims without matching workload, durability, concurrency and hardware.

## Paradigm shift: state becomes a projection over evidence

An ordinary workflow engine stores the current status and execution history. SFEB should add a stronger, explicit contract:

- **Evidence first:** immutable source receipt, event identity, revocation history and source-capability lease.
- **Authority at use time:** action authority, tenant scope, policy revision and fencing epoch are rechecked at the sink.
- **Operation identity survives uncertainty:** one stable idempotency identity is preserved through retries and recovery.
- **Read and act are separate capabilities:** a read-only reconciliation grant does not confer permission to mutate.
- **Outcome is independent:** the executor's return value is not the postcondition; a separate system-of-record readback is required.
- **Every transition carries evidence:** state changes are linked to append-only records with hashes, not just mutable status columns.

The hash chain is tamper-evident, not tamper-proof. An actor able to rewrite the entire SQLite file and recompute the complete chain can create a new internally consistent history. The next hardening step is to periodically anchor journal heads to an independent trust domain—e.g. signed checkpoints stored in a separate account or an append-only transparency service—and test rollback/fork detection.

## Industry baseline: where not to compete

### DBOS is a stronger local-transaction comparator than the earlier benchmark plan assumed

DBOS documents transactional datasource operations that co-commit application writes with the durability/checkpoint record in the application's database. Its Go datasource docs explain that application writes and the DBOS durability record commit atomically in the same transaction; if the datasource is the same database handle as the DBOS system database, the transaction can collapse to one shared transaction. DBOS also provides workflow ownership tokens checked in the same transaction as checkpoint writes, preventing a zombie executor from recording stale progress. (Official: [DBOS datasources](https://docs.dbos.dev/golang/tutorials/transaction-tutorial), [concurrent executions](https://docs.dbos.dev/explanations/concurrent-executions), [architecture](https://docs.dbos.dev/architecture).)

DBOS Rust steps still have at-least-once execution around the side-effect/checkpoint crash window, and its Rust steps tutorial says there is no transactional step yet. This makes the correct benchmark language version-specific: test native DBOS transactional datasource support where available, DBOS Rust step retry separately, and DBOS ownership-token behavior. Do not benchmark against a weaker nontransactional path and call that “DBOS”. (Official: [DBOS Rust steps](https://docs.dbos.dev/rust/tutorials/step-tutorial), [DBOS Go transactions](https://docs.dbos.dev/golang/tutorials/transaction-tutorial).)

DBOS's production checklist reports that a single 16-vCPU PostgreSQL server sustained 20K workflows or steps per second in its benchmarks. Treat that as a **provider-reported sizing reference**, not an independently reproduced universal throughput guarantee. The local SQLite number above is a different workload and is not directly comparable. (Official: [DBOS production checklist](https://docs.dbos.dev/production/checklist).)

### Temporal already solves durable workflow progress

Temporal stores an ordered event history and rebuilds workflow state by replaying it after worker failure. Completed activities replay their recorded outcomes rather than being blindly re-executed as workflow logic. SFEB should integrate with Temporal's event history, timers and activity execution—not recreate the durable workflow engine. The testable delta is source provenance, current external-action authority, durable sink idempotency, scoped reconciliation and independent postcondition proof. (Official: [Temporal durable execution](https://docs.temporal.io/durable-execution), [understanding Temporal](https://docs.temporal.io/evaluate/understanding-temporal).)

### Transactional outbox and queues are established patterns

AWS's transactional-outbox guidance explicitly covers the dual-write failure: application state and outbox event are committed together, while message delivery can be duplicated and therefore consumers must be idempotent. It also warns about preserving event order and using saga-style coordination when multiple service databases are involved. SFEB is not inventing the outbox; its differentiation must be the evidence/authority/postcondition contract layered over it. ([AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html).)

PostgreSQL's `FOR UPDATE SKIP LOCKED` semantics are specifically useful for multiple consumers of a queue-like table. The documentation warns that SKIP LOCKED provides an inconsistent view and is intended for queue-consumer patterns rather than general querying. This is the natural production adapter path for multi-worker outbox claiming. ([PostgreSQL SELECT locking clauses](https://www.postgresql.org/docs/current/sql-select.html), [explicit row locks](https://www.postgresql.org/docs/current/explicit-locking.html).)

SQLite WAL allows readers and a writer to proceed concurrently but requires processes to share the same host; WAL does not operate over a network filesystem. `synchronous=FULL` syncs the WAL on each commit and is the documented higher-durability configuration in WAL mode. This preflight is therefore a local persistence test, not a clustered database qualification. ([SQLite WAL](https://www.sqlite.org/wal.html), [SQLite PRAGMA synchronous](https://www.sqlite.org/pragma.html).)

## Revised benchmark design: fair, adversarial and version-aware

We should compare matched end-to-end traces, not generic “agent success rates”. Run all systems on identical source-event seeds, operation payloads, tenant policies, sink adapters, timing distributions and failure injection points.

| Arm | Baseline | What is measured |
|---|---|---|
| A | Direct DB transaction + naïve outbox consumer | Local atomicity, duplicate effects, loss window |
| B | PostgreSQL outbox + `SKIP LOCKED` consumers | Multi-worker claim fairness, ordered delivery, p95/p99 contention |
| C | Native Temporal workflow + idempotent external sink | Durable workflow recovery and external sink correctness |
| D | Temporal + SFEB | Incremental evidence and authorization assurance cost |
| E | Native DBOS datasource transaction (shared database) | Strong local transaction/checkpoint baseline |
| F | DBOS ownership-token workflow + external idempotent sink | Zombie worker prevention and external retry behavior |
| G | DBOS + SFEB external-boundary assurance | Incremental provenance, sink authority, reconciliation and readback |
| H | DBOS Rust step retry path | At-least-once side-effect/checkpoint crash window, with idempotent consumer |
| I | Kafka transactional consume/produce + external sink | Broker transaction boundary vs arbitrary external side effect |
| J | SQLite local preflight | Embedded/local durability, contention and restart behavior |

Failure strata must include local crash before commit; process termination after commit; remote pre-commit rejection; remote commit + response loss; duplicate and out-of-order delivery; stale worker; permission revocation after grant; tenant/objective ID collision; source receipt revocation; read-grant revocation; wrong postcondition; replay retention expiry; connector capability drift; database lock contention; clock skew; and recovery after both local and remote database handles are closed/reopened.

### Proposed qualification targets (not achieved)

**Safety gates**
- 0 unauthorized/cross-tenant mutations or reconciliation disclosures over at least 1,000,000 generated adversarial operation attempts per release candidate, with per-family denominators.
- 0 duplicate external effects for stable operation identity across at least 1,000,000 retry/duplicate-delivery schedules on an idempotent test sink.
- 100% of success claims contain a linked source, authority, execution and independent outcome receipt.
- 100% of remotely committed effects with durable sink evidence reconcile to the same operation identity; where evidence is unavailable, remain `outcome_unknown`.
- 0 accepted tampered evidence chains; externally anchored checkpoint rollback/fork detection must be tested before claiming independent tamper evidence.

**Performance gates**
- First reproduce DBOS's published 16-vCPU PostgreSQL workload or obtain a directly comparable baseline with explicit step mix, transaction count, storage durability, connection pool, payload size and concurrency.
- Target no more than 10% p95/p99 overhead for SFEB's added checks against the matched native system on local DB operations; publish absolute latencies too.
- Target throughput within 10% of native DBOS/Temporal on equivalent durable-workflow workloads before claiming a competitive execution layer.
- Report recovery time after injected failures, not only steady-state throughput.
- Measure cost per independently verified external action, including DB writes, evidence storage, signature verification, readback, replay, retry and external anchoring.

These are proposed qualification bars, not measured outcomes. The immediate priority is correctness and apples-to-apples measurement, not advertising a throughput number.

## Immediate engineering sequence

1. **PostgreSQL adapter:** replace SQLite's single-writer claim with a row-leased queue using `FOR UPDATE SKIP LOCKED`; add claim leases, fencing tokens, partition ordering and concurrency stress.
2. **DBOS native benchmark:** implement a matched native datasource transaction and separate Rust step retry arm; verify exactly where the local transaction and durability record co-commit.
3. **Temporal paired adapter:** use native workflow history for orchestration, SFEB only at source/authority/sink/outcome boundaries.
4. **External trust anchoring:** sign periodic evidence-chain heads and persist them outside the ledger's administrator boundary; add key rotation and revocation tests.
5. **Real sink:** choose one supported conditional/idempotent API and implement an independent readback adapter; no generic “exactly once” claim.
6. **Chaos and concurrency:** multi-process kill/restart schedules, database restore/rollback, network partitions, duplicate outbox dispatch and linearizability checking.
7. **Public benchmark harness:** pin versions, publish raw traces, exact source commits, run environment, per-arm failure denominators and confidence intervals.

**Decision:** MON-003 now demonstrates local SQLite transaction atomicity and a separate idempotent sink model with independent readback. It does not yet demonstrate Postgres multi-worker semantics, real external-sink correctness, production cryptographic provenance or superiority to DBOS/Temporal. The next competitive step is a matched PostgreSQL + DBOS/Temporal comparison, with SFEB's incremental value isolated to evidence, authority and outcome assurance.
