# MON-025 — Live Recovery Challenge Lab

**Status:** reproducible lab definition, Compose stack, database fixtures, fault matrix and operator scripts added; container execution, live CDC tests, PITR drills and independent runtime qualification pending.

MON-025 moves MON-019–024 from semantic contracts into a repeatable live-system challenge. The lab exercises an actual PostgreSQL server, Debezium PostgreSQL connector, Kafka broker, independent SQL truth reads, physical base backup and WAL-archive recovery. It exports a test run manifest suitable for later MON-024 receipt adaptation.

## Pinned baseline

| Component | Baseline |
|---|---|
| PostgreSQL source | 18.7 or 19.0; selectable Compose profiles |
| Kafka broker | Apache Kafka 4.3.1, the compatibility version named by Debezium 3.7.1 |
| Kafka Connect / Debezium | `quay.io/debezium/connect:3.7.1.Final` |
| Backup / PITR baseline | PostgreSQL `pg_basebackup` + WAL archive |
| External backup comparator | pgBackRest 2.60.1; adapter contract only, not embedded in Compose |

Kafka 4.3.2 is a same-series patch extension; Kafka 4.4.0 is a newer-series extension. Neither silently replaces the declared Debezium-tested 4.3.1 baseline. See `VERSIONS-v1.json`.

## Quick start

Copy `.env.example` to `.env`, set both local-only passwords, and assign a fresh `COMPOSE_PROJECT_NAME` for each independent run. Never reuse production credentials.

```sh
cp .env.example .env
# Edit .env: set MON025_DB_PASSWORD, MON025_CONNECTOR_DB_PASSWORD and a unique COMPOSE_PROJECT_NAME.
bash scripts/static-preflight.sh
bash scripts/labctl.sh preflight
bash scripts/labctl.sh up 18
bash scripts/capture-run-manifest.sh 18 mon025-run-001
bash scripts/provision-connector-user.sh 18
bash scripts/register-connector.sh 18
bash scripts/verify-cdc-events.sh 18 baseline 2
bash scripts/capture-source-boundary.sh 18
bash scripts/seed-batch.sh 18 mon025-run-001 100
bash scripts/verify-cdc-events.sh 18 mon025-run-001 100
bash scripts/verify-transaction-metadata.sh 18 mon025-run-001 100
```

To run the PostgreSQL 19 lane, use `bash scripts/labctl.sh up 19`, provision the connector role for major 19, register the connector, then verify the `baseline` snapshot before seeding. Do not run the two source majors under the same connector identity/slot simultaneously.

The database fixture is initialized on first start from `compose/initdb/001-mon025-schema.sql`. The scripts are invoked with `bash` because repository file mode may not yet mark them executable.

The lab binds exposed ports to loopback only. It is single-node and non-HA. Kafka replication factors are 1. It is not a production deployment template.

## Recovery drill

The native backup lane creates a physical base backup and verifies its embedded manifest/WAL with `pg_verifybackup`. It separately seals at least one later archived WAL segment; the restore itself must consume the complete required archive interval.

A minimal drill after the CDC test batch:

```sh
bash scripts/backup.sh 18 mon025-backup-001

# Quiesce all source writers before recording the target snapshot.
bash scripts/capture-state.sh 18 postgres18 out/source-state.json

# Deliberately create post-target state so PITR must exclude it.
bash scripts/seed-batch.sh 18 mon025-post-target-001 10
bash scripts/seal-wal-archive.sh 18

TARGET_LSN="$(jq -r '.metadata.capture_lsn' out/source-state.json)"
TARGET_TIMELINE="$(jq -r '.metadata.timeline_id' out/source-state.json)"
bash scripts/labctl.sh restore 18 mon025-backup-001 "$TARGET_LSN" "$TARGET_TIMELINE"
bash scripts/wait-for-restore.sh 18 restore18 "$TARGET_LSN" "$TARGET_TIMELINE"
bash scripts/capture-state.sh 18 restore18 out/restore-state.json
bash scripts/compare-state.sh out/source-state.json out/restore-state.json
```

The source capture is a repeatable-read snapshot tied to `capture_lsn`, `timeline_id`, system identifier and capture time. The connector uses Debezium's string decimal mode so the independent SQL oracle can compare exact numeric values, not only event IDs and counts. Fixture writers must remain quiesced during capture. The target is restored to that LSN on that timeline and paused; the wait script also verifies the WAL filename's timeline prefix. RPO uses the recorded capture time and bounded clock uncertainty, not a comparison of unlike clocks.

The comparison report checks source/target database identity, system identifier, timeline and recovery state, then reports every state surface separately. A difference (for example, unlogged data not restored by WAL replay) is an explicit coverage failure, not something to average away.

Fault injection is opt-in and scoped to the Compose project named in `.env`: export `MON025_ALLOW_FAULT_INJECTION=YES` and use `bash scripts/fault-inject.sh 18 connect-kill`, `broker-kill`, or `postgres-kill`. It sends SIGKILL only to the selected lab service. Never use the injector against any non-lab service.

The Compose restore service remains isolated on loopback-only port 56418/56419. It does not configure a load balancer or activate production traffic.

## Files

- `LIVE-RECOVERY-CHALLENGE-CONTRACT-v1.md` — lab claims, controls and execution protocol.
- `VERSIONS-v1.json` / schema — version/compatibility lock profile.
- `FAULT-MATRIX-v1.json` / schema — positive controls and fault injection seeds.
- `compose/compose.yaml` — PostgreSQL 18/19, Kafka and Debezium Connect.
- `compose/postgres/001-mon025-schema.sql` — database surfaces and seed fixture.
- `compose/postgres/state-manifest.sql` — independent SQL manifest query for rows, schema, indexes, constraints, triggers, routines, extensions, sequences, large objects, views, partitions, unlogged data and authorization state.
- `compose/restore/restore-entrypoint.sh` — isolated physical restore and PITR.
- `scripts/` — static preflight, lifecycle, least-privilege connector provisioning, deterministic workload, independent CDC value readback, Debezium transaction-boundary verification, physical backup verification, WAL archive sealing, LSN/timeline recovery, wait/restore verification, per-surface comparison and opt-in fault injection.
- `scripts/transaction-metadata-report.jq` and `scripts/test-transaction-metadata-order.sh` — independently runnable transaction-order adjudication filter and regression fixtures.
- `BENCHMARKS-v0.1.md` — matched comparison protocol and targets.

## Execution status

No Docker/Podman runtime is available in the authoring environment. Therefore no image pull, Compose startup, live connector registration, PostgreSQL transaction, Kafka event capture, pgBackRest invocation, physical backup or restore was performed here. Run `bash scripts/static-preflight.sh` locally to validate the JSON structure and shell syntax; optional JSON Schema validation, ShellCheck and Compose parsing run when those tools are installed. Live integration remains pending.
