# MON-031 — Version-Aware Backup Lineage Qualification

MON-031 converts version-specific recovery hazards into explicit evidence policy. Its first executable guard addresses PostgreSQL 18.7's warning that existing incremental backup files should not be trusted after upgrading; a new full backup should become the base for subsequent incrementals.

## Executable policy

The standard-library Go module `lineage/` evaluates a candidate physical backup against an explicit version and lineage policy:

- exact server major-version agreement;
- expected PostgreSQL system identifier;
- a complete, acyclic parent chain terminating at one full backup;
- strict parent-before-child timestamps;
- verified integrity for every backup in the chain;
- verified WAL archive coverage for the selected candidate;
- a configured minimum version for the full base and every incremental node.

For the PostgreSQL 18.7 transition, set `minimum_full_backup_version` to `18.7`. A chain rooted in an 18.6 full backup is blocked even if a later incremental was created by an 18.7 server. A missing parent, unknown version, absent integrity proof or unknown WAL coverage is indeterminate; it is never interpreted as a pass.

The evaluator does not run `pgBackRest`, inspect a physical backup, verify WAL bytes or prove application recovery. It consumes independently collected backup metadata and evidence statuses. The backup adapter and full restore remain separate obligations.

## Use

@@BT@@sh
cd benchmarks/frontend/monitoring/sfeb-mon031/lineage
go run ./cmd/qualify --input INPUT-example.json
@@BT@@

Exit codes: `0` qualified, `2` blocked, `3` indeterminate, `64` CLI usage error, `65` invalid input, `66` input unavailable. The input JSON is strict: unknown fields and trailing documents are rejected.

## Upstream regression matrix

`REGRESSION-MATRIX-v1.json` captures eight version-specific recovery regression targets:

- PostgreSQL 18.7 incremental-base invalidation;
- Debezium DBZ-2889 (`lsn_events_processed=1` and transactional logical-message offset restart);
- Debezium DBZ-2894 (decoderbufs BEGIN sharing stored LSN);
- Kafka KAFKA-20732 and KAFKA-20979 (tiered retention and time-index restart data loss);
- pgBackRest 2.60.1 multi-repository annotate and SFTP block-incremental restore fixes.

The matrix distinguishes the default profile from optional extension lanes. It is a test programme, not evidence that the fixes have been independently reproduced. Only the PostgreSQL incremental-lineage policy is implemented here; other scenarios remain explicitly pending their runtime adapters and experiments.

## Validation

Run `bash CHECKS.sh` for JSON structure, Go formatting, vet, unit tests and race tests. These checks do not start a database, Kafka cluster, SFTP server, pgBackRest repository or Debezium connector.
