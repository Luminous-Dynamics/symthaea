# MON-021 — PostgreSQL CDC Continuity Assurance

**Status:** research contract, 44-case synthetic seed corpus and standard-library Go semantic prototype; live PostgreSQL 18/19 integration and independent qualification pending.

MON-021 connects MON-016/017 bounded source capture with MON-013 destination readback by qualifying PostgreSQL logical-decoding continuity from an exported snapshot through WAL decoding, durable evidence persistence, acknowledgements and failover.

## Research question

Can an independent verifier establish that a database snapshot and every logical-decoding message within a declared slot/plugin/database/source boundary were preserved through a terminal LSN, that the consumer never acknowledged beyond durable evidence, and that failover did not silently fork, skip or misidentify the source history?

## Files

- [CDC-CONTINUITY-CONTRACT-v1.md](./CDC-CONTINUITY-CONTRACT-v1.md) — source identity, snapshot, WAL/slot state, durability, failover and benchmark semantics.
- [SCENARIOS-v1.json](./SCENARIOS-v1.json) and [SCENARIOS-v1.schema.json](./SCENARIOS-v1.schema.json) — synthetic adversarial seed corpus.
- [cdc-continuity/](./cdc-continuity/) — standard-library Go semantic reference, 35 adversarial tests, fuzz target and scaling benchmarks.
- [BENCHMARKS-v0.1.md](./BENCHMARKS-v0.1.md) — local reference-package measurements and prospective targets.
- [BENCHMARKS-v0.1.md](./BENCHMARKS-v0.1.md) — local preflight and exploratory measurements.

## PostgreSQL 18/19 research basis

- [Logical decoding concepts and replication slots](https://www.postgresql.org/docs/18/logicaldecoding-explanation.html).
- [Replication-slot state](https://www.postgresql.org/docs/18/view-pg-replication-slots.html).
- [Logical replication failover, PostgreSQL 18](https://www.postgresql.org/docs/18/logical-replication-failover.html).
- [PostgreSQL 19 release notes (2026-11-19)](https://www.postgresql.org/docs/19/release-19.html).
- [PostgreSQL 19 sequence replication](https://www.postgresql.org/docs/19/logical-replication-sequences.html).
- [PostgreSQL 19 WAIT FOR LSN](https://www.postgresql.org/docs/19/sql-wait.html).
- [WAL/replication configuration](https://www.postgresql.org/docs/18/runtime-config-replication.html).
- [Timeline history](https://www.postgresql.org/docs/18/continuous-archiving.html).

## Local semantic reference

The Go package parses PostgreSQL LSN values and validates source/slot/plugin identity, exported-snapshot/consistent-point binding, bounded capture intervals, message monotonicity, exact replay identity, durable-evidence/acknowledgement ordering, retention state and failover readiness. It deliberately does not require numeric LSN adjacency because LSNs are WAL byte positions, not a consecutive logical-event counter.

The package consumes normalized state. It is not a PostgreSQL replication-protocol client, SQL adapter, timeline-history parser, cryptographic verifier or independent witness. Its `Witnessed` field represents a pre-qualified upstream custody/witness condition and is not authenticated by this package.

## Claim ceiling

The Go reference validates supplied normalized state and manifests. It is not a PostgreSQL decoder, database integration, WAL parser, signature verifier, or independent witness. It does not prove changes outside the plugin/filter scope, source truth, or external effects. No PostgreSQL runtime test or production qualification is claimed.
