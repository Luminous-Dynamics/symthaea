# MON-023 — Restore-to-Serve Recovery Assurance

**Status:** research contract, 38 scenario seeds, Go semantic evaluator and benchmark harness; exact-head execution and runtime adapters pending.

MON-023 separates three claims: artifact validated, database recovered, and service recovered. A successful backup or restore command does not prove application recovery.

## Files
- RESTORE-TO-SERVE-CONTRACT-v1.md — claim levels, evidence, RPO/RTO, dependencies, app validation and threats.
- SCENARIOS-v1.json / SCENARIOS-v1.schema.json — synthetic fault scenarios.
- restore-assurance/ — planned standard-library Go evaluator, tests and benchmarks.
- BENCHMARKS-v0.1.md — matched recovery measurement programme.

## Industry baseline
- [pgBackRest command reference](https://pgbackrest.org/command.html): backup consistency WAL ranges, archive checks, repository verification and PITR restore.
- [pgBackRest user guide](https://pgbackrest.org/user-guide.html): backup WAL start/stop and archive status.
- [AWS Backup Restore Testing](https://docs.aws.amazon.com/aws-backup/latest/devguide/restore-testing.html): periodic restore jobs and restore duration, with optional validation.
- [PostgreSQL 19 release notes](https://www.postgresql.org/docs/19/release-19.html): recovery/standby read-your-writes improvements.

MON-023 composes with those systems; it is not a replacement backup engine.

## Claim ceiling
No exact-head Go test/race/vet/fuzz or numeric benchmark is claimed. No live restore, RPO/RTO result, independent restore observer, production readiness or SOTA superiority is claimed. A successful artifact or database restore is not a service-recovered claim.
