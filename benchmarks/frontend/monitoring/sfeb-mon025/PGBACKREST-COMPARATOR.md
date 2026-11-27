# MON-025 pgBackRest Comparator Profile

**Comparator version:** pgBackRest 2.60.1 (stable on 2026-11-27).  
**Execution status:** specified, not integrated into the Compose stack or run in this authoring environment.

## Why it is a separate lane

The default MON-025 recovery lane uses PostgreSQL `pg_basebackup` plus an archive directory on the source volume. This provides a zero-extra-product physical backup/PITR baseline.

pgBackRest is compared as a distinct backup implementation. It must not share state or artifacts with the native lane, and every recovery should be evaluated with the same independent target-state observer.

## Version discipline

pgBackRest local and remote binaries must match exactly. Record:
- exact `pgbackrest version` output from the database host and repository host;
- binary SHA-256 for each host;
- stanza/config digest with secrets redacted;
- repository type, path/bucket, encryption, retention and TLS settings;
- PostgreSQL system identifier and WAL archive interval;
- backup set/label, manifest digest, archive-check result and restore target.

## Required execution path

1. Build/install pgBackRest 2.60.1 on both the PostgreSQL host and repository host from the immutable upstream release distribution, or use packages pinned to that exact version.
2. Configure a dedicated stanza and repository for the lab; never point it at a production repository.
3. Configure PostgreSQL `archive_command` to run the local pgBackRest binary, and validate the WAL archive with pgBackRest's archive check before the backup is considered complete.
4. Create a full backup; retain command output, backup label and manifest.
5. Generate the same deterministic post-backup transaction schedule used by the native lane.
6. Restore into a new isolated data directory at the same target timestamp used for the native lane.
7. Verify source system identity, target recovery point, timeline lineage, catalog inventory and MON-022 state surface digests.
8. Measure database-ready and service-ready RTO separately and export normalized recovery evidence for MON-024.

The official pgBackRest build guide provides a source-build route and release distribution tarball; installation packages are preferred where available. The official release notes describe the 2.60.1 fixes. The exact upstream source URL is:

`https://github.com/pgbackrest/pgbackrest/releases/download/release%2F2.60.1/pgbackrest-2.60.1.tar.gz`

## Acceptance

A pgBackRest lane is admissible only after:
- exact local/remote version equality;
- repository and backup integrity checks;
- full required WAL interval coverage;
- successful isolated restore to the declared target;
- no state-surface digest divergence for the declared claim;
- independent RPO/RTO and application readiness checks.

Do not use a third-party Docker image with a different patch version as a substitute. If the image does not provide 2.60.1, record the exact version and treat it as a separate exploratory comparator.
