# MON-025 — Live Recovery Challenge Contract v1

**Profile:** `mon025-live-recovery-challenge-v1`  
**Status:** reproducible lab source committed; live execution and independent qualification pending.

## 1. Objective

Exercise the actual source, CDC, durable message, backup, restore and application-readiness boundaries under controlled fault injection. MON-025 is intended to generate independently observed facts for MON-019 through MON-024, not to replace PostgreSQL, Kafka, Debezium or pgBackRest.

Primary question:

> Under a pinned workload and fault schedule, can we reconstruct the exact acknowledged source history, account for declared database surfaces, recover to the requested point, and demonstrate that the restored service is safe before enabling traffic?

## 2. System topology

- PostgreSQL 18.7 or PostgreSQL 19.0 source selected via Compose profile.
- Kafka broker 4.3.1 and Kafka Connect/Debezium 3.7.1.Final.
- A source SQL truth reader independent of Debezium's connector status.
- A Kafka event reader independent of the producer path.
- Physical base-backup and WAL archive volumes.
- A separate restore target with its own data volume and loopback-only port.
- A future MON-024 receipt adapter; this initial increment writes raw run artifacts and does not mint production-qualified receipts.

The topology is deliberately small and reproducible. It is not a high-availability Kafka or PostgreSQL topology and must not be used to infer production HA behavior.

## 3. Version control and compatibility lanes

### Compatibility baseline

- PostgreSQL 18.7 / 19.0, official image tags `postgres:18.7-bookworm` and `postgres:19.0-bookworm`.
- Kafka `apache/kafka:4.3.1`.
- Debezium `quay.io/debezium/connect:3.7.1.Final`.

Debezium 3.7.1.Final states that it was built against Kafka Connect 4.3.1 and tested with Kafka brokers 4.3.1. This is the reproducible declared compatibility baseline.

### Extension lanes

- Kafka 4.3.2: test separately as a same-series maintenance upgrade.
- Kafka 4.4.0: test separately as a newer minor/feature line after checking the exact Debezium test matrix.
- pgBackRest 2.60.1: test as a separate backup/PITR implementation, matching the local and remote versions exactly.
- No extension lane may replace or overwrite the baseline run artifacts.

## 4. Workload and state surfaces

The fixture intentionally includes state that row-based CDC should not be assumed to cover:

- ordinary transactional rows and stable event IDs;
- a stored generated column;
- a separately advanced sequence;
- a large object with content digest;
- a materialized view;
- a partitioned table and partition topology;
- an unlogged table;
- schema/catalog metadata and an explicit publication;
- role/ACL state.

The fixture is an adversarial test database, not a complete inventory for all PostgreSQL installations. MON-022 remains responsible for per-database inventory discovery, including extensions, foreign/external state, routines, triggers, policies, constraints, indexes and tablespaces.

## 5. Required evidence per run

A run directory must contain:

1. Immutable version lock and Compose configuration digest.
2. Run ID, selected PostgreSQL major, Kafka/Connect versions and image digests.
3. Source system identifier, database, source epoch, timeline, publication, slot and snapshot metadata.
4. Workload manifest: expected event IDs, transaction groups, payload digests and generation parameters.
5. Connector configuration, task state, observed source offsets, restart count and error log digest; transaction BEGIN/END receipts, `event_count`, transaction ID joins and per-event order checks; byte-identical replay is counted and deduplicated, while conflicting BEGIN/END payloads for the same transaction ID block.
6. Independent Kafka readback: observed event IDs, exact payload digests, duplicate/replay groups, conflicting identities and missing events.
7. Source SQL state manifest at the intended recovery point.
8. Base-backup manifest, required archived WAL interval, requested recovery LSN/timeline and achieved recovery state.
9. Restored target identity, timeline and SQL state manifest.
10. Separate database-ready and service-ready timestamps, clock domain and uncertainty.
11. Dependency/application probe outcomes, write-probe rollback, old-primary fencing and stale-endpoint fencing results.
12. Raw output hashes and a manifest index. Never normalize away the raw source evidence.

The dedicated Debezium role has REPLICATION, CONNECT, schema USAGE and SELECT only on the published fixture table; it is not a superuser. The independent truth reader must not consume Debezium's reported event count as its expected answer. Expected IDs are derived directly from source fixture SQL and stored before the fault injection.

## 6. Execution protocol

1. Create a fresh uniquely named Compose project and isolated volumes.
2. Run preflight and record all resolved image IDs and repository digests before starting the lab.
3. Start the selected PostgreSQL source, Kafka and Connect services.
4. Verify PostgreSQL settings, system identifier, slot/publication and source boundary.
5. Capture pre-workload SQL manifest; provision a dedicated least-privilege replication role, register the connector, and independently verify initial snapshot completion.
6. Create a deterministic transaction batch and capture expected IDs directly from PostgreSQL.
7. Execute one fault at a time from the fault matrix. Record injection time and exact command outcome.
8. Recover connector/broker/database as required. Read Kafka independently and reconcile exact IDs and payload digests; independently reconcile Debezium transaction boundary metadata and per-event ordering.
9. Capture source state manifest at a known recovery boundary.
10. Create physical base backup and ensure archive coverage for the required WAL interval.
11. Capture a repeatable-read state manifest with its LSN, timeline, system identifier and wall-clock observation; restore into a separate volume at that LSN and pinned timeline; keep the restored service isolated.
12. Verify actual target position, source identity, timeline, catalogs and per-surface digests.
13. Run read probe, transactional write/readback and rollback; test dependencies and application invariants.
14. Verify old primary and stale endpoints are fenced. Only a separately authorized controller may activate traffic after proof is independently accepted.
15. Export run artifacts and map normalized facts into MON-024 receipts.

## 7. Fault injection requirements

A fault must be tied to an explicit control point, with preconditions and postconditions. “Restart the connector” is not evidence that a crash occurred between a specific database commit and offset commit. When a desired boundary cannot be controlled deterministically through the public API, report the experiment as a randomized crash-window test rather than claiming an exact cut.

At minimum test:
- connector stopped/restarted after source commit;
- connector killed while processing a transaction;
- broker killed and restarted after event production;
- source PostgreSQL immediate shutdown and restart;
- WAL archive gap / missing required segment;
- restore point before and after a committed transaction;
- publication/filter/schema changes;
- sequence advance, generated-column change, large object write, materialized-view refresh, unlogged-table write and partition change;
- source identity/timeline mismatch;
- dependency unavailable;
- former primary or stale endpoint remains reachable;
- unknown command outcome requiring reconciliation before another mutation.

Never inject faults into a production system or a shared developer database.

## 8. Restore semantics

The native restore lane uses `pg_basebackup` plus PostgreSQL WAL archive and point-in-time recovery. The restore target receives a distinct volume and loopback-only endpoint. Recovery uses a caller-specified LSN and pinned numeric timeline and pauses at the target. `pg_verifybackup` checks the base-backup manifest and embedded WAL; an archive-seal observation is recorded separately, while the restore itself must prove the complete required WAL range was available. The operator records requested and observed replay LSN/timeline; a server reaching a paused recovery state is not by itself service-ready.

pgBackRest is a separately version-pinned comparator. Its archive-check, backup manifest, verify, restore and recovery target evidence should be normalized into the same run model. The local and remote pgBackRest versions must match exactly.

## 9. Safety invariants

- Test-only credentials and loopback-bound service ports.
- No host networking and no public ingress.
- No production credentials or production database data.
- Destructive volume deletion requires explicit confirmation.
- Source and restore target data directories never share a writable volume.
- Recovery target remains isolated and read-only at the infrastructure boundary until acceptance.
- A write probe must be rolled back and its rollback observed.
- Former-primary and stale-endpoint fencing must precede traffic activation.
- Unknown mutation outcome ⇒ no new mutation until reconciliation.
- CI skipped/queued is not a pass; source-level tests do not qualify runtime behavior.

## 10. Out-of-scope claims

This version does not prove distributed HA, multi-broker durability, synchronous replica guarantees, storage failure resilience, cloud-object-store semantics, multi-region failover, production security hardening or end-to-end business correctness. Those require additional topologies and independent observers.

It does not claim that PostgreSQL 19 sequence synchronization means every sequence allocation is emitted as an ordered CDC event. The fixture separately observes sequence state. PostgreSQL 19's sequence replication support is tied to publication/subscription sequence synchronization operations.

## 11. Qualification criteria

A challenge run is valid only when:
- all artifacts and images are pinned and digest-recorded;
- expected fixture events are generated and retained by an independent oracle;
- fault timing and outcome are recorded;
- no missing event or conflicting event identity is concealed as a duplicate;
- backup/WAL coverage reaches the requested target;
- the target's actual recovered state matches the expected state at that same target;
- database-ready and service-ready are measured separately;
- all required database surfaces are covered or explicitly proven N/A under the chosen claim;
- traffic activation is withheld on every failed or unknown prerequisite.

No single scalar score can compensate for a false-qualified recovery.
