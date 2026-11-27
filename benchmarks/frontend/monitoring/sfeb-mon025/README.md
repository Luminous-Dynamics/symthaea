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

Copy `.env.example` to `.env` and set a unique local-only password before running the lab.

```sh
cp .env.example .env
# Edit .env: set MON025_DB_PASSWORD and confirm the local-only settings.
bash scripts/labctl.sh preflight
bash scripts/labctl.sh up 18
bash scripts/capture-run-manifest.sh 18 mon025-run-001
bash scripts/capture-source-boundary.sh 18
bash scripts/register-connector.sh 18
bash scripts/seed-batch.sh 18 mon025-run-001 100
bash scripts/verify-cdc-events.sh 18 mon025-run-001 100
```

To run the PostgreSQL 19 lane, use `./scripts/labctl.sh up 19` and register the connector against that service. Do not run the two source majors under the same connector identity/slot simultaneously.

The database fixture is initialized on first start from `compose/initdb/001-mon025-schema.sql`. The scripts are invoked with `bash` because repository file mode may not yet mark them executable.

The lab binds exposed ports to loopback only. It is single-node and non-HA. Kafka replication factors are 1. It is not a production deployment template.

## Recovery lane

`scripts/backup.sh` creates a physical base backup on the named backup volume. The source archives WAL under its data volume. The restore profile copies the selected backup into an isolated target volume and replays archived WAL to a caller-supplied recovery target time, pausing at the target.

A drill must capture an independent source-state manifest at the exact intended recovery point, then compare it to the restored target manifest. Do not compare the restored target only with the source's later state: legitimate post-target writes are supposed to be absent.

The Compose restore service starts with traffic inaccessible on loopback-only port 56418/56419. It does not configure a load balancer or activate production traffic.

## Files

- `LIVE-RECOVERY-CHALLENGE-CONTRACT-v1.md` — lab claims, controls and execution protocol.
- `VERSIONS-v1.json` / schema — version/compatibility lock profile.
- `FAULT-MATRIX-v1.json` / schema — positive controls and fault injection seeds.
- `compose/compose.yaml` — PostgreSQL 18/19, Kafka and Debezium Connect.
- `compose/postgres/001-mon025-schema.sql` — database surfaces and seed fixture.
- `compose/postgres/state-manifest.sql` — independent SQL manifest query.
- `compose/restore/restore-entrypoint.sh` — isolated physical restore and PITR.
- `scripts/` — preflight, lifecycle, connector registration, batch seeding, CDC verification, state capture and backup.
- `BENCHMARKS-v0.1.md` — matched comparison protocol and targets.

## Execution status

No Docker/Podman runtime is available in the authoring environment. Therefore no image pull, Compose startup, live connector registration, PostgreSQL transaction, Kafka event capture, pgBackRest invocation, physical backup or restore was performed here. Static JSON and schema structure are intended to be checked before use. Exact-head shell syntax and Compose parsing have not yet been run; live integration remains pending.
