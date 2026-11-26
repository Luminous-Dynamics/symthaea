# MON-021 — PostgreSQL CDC Continuity Assurance

**Status:** research contract, scenario seed corpus and standard-library Go semantic prototype; live PostgreSQL 18 integration pending.

MON-021 connects MON-016/017 bounded source capture with MON-013 destination readback by qualifying PostgreSQL logical-decoding continuity from an exported snapshot through WAL decoding, durable evidence persistence, acknowledgements and failover.

## Research question

Can an independent verifier establish that a database snapshot and every logical-decoding message within a declared slot/plugin/database/source boundary were preserved through a terminal LSN, that the consumer never acknowledged beyond durable evidence, and that failover did not silently fork, skip or misidentify the source history?

## Files

- [CDC-CONTINUITY-CONTRACT-v1.md](./CDC-CONTINUITY-CONTRACT-v1.md) — source identity, snapshot, WAL/slot state, durability, failover and benchmark semantics.
- [SCENARIOS-v1.json](./SCENARIOS-v1.json) and [SCENARIOS-v1.schema.json](./SCENARIOS-v1.schema.json) — synthetic adversarial seed corpus.
- [cdc-continuity/](./cdc-continuity/) — standard-library Go semantic reference, tests and benchmarks.
- [BENCHMARKS-v0.1.md](./BENCHMARKS-v0.1.md) — local preflight and exploratory measurements.

## PostgreSQL 18 research basis

- [Logical decoding concepts and replication slots](https://www.postgresql.org/docs/18/logicaldecoding-explanation.html).
- [Replication-slot state](https://www.postgresql.org/docs/18/view-pg-replication-slots.html).
- [Logical replication failover](https://www.postgresql.org/docs/18/logical-replication-failover.html).
- [WAL/replication configuration](https://www.postgresql.org/docs/18/runtime-config-replication.html).
- [Timeline history](https://www.postgresql.org/docs/18/continuous-archiving.html).

## Claim ceiling

The Go reference validates supplied normalized state and manifests. It is not a PostgreSQL decoder, database integration, WAL parser, signature verifier, or independent witness. It does not prove changes outside the plugin/filter scope, source truth, or external effects. No PostgreSQL runtime test or production qualification is claimed.
