# MON-022 — Database Surface Coverage Assurance

**Status:** research contract, 36 scenario seeds and standard-library Go semantic reference; PostgreSQL/driver runtime adapters and independent qualification pending.

MON-022 answers a question ordinary row CDC cannot answer by itself:

> Which parts of database state are covered by the declared capture method, how was each surface evidenced, and what remains outside the claim?

The profile distinguishes a scoped row-stream continuity claim from a full recoverable-database-state claim. It inventories sequences, schema/DDL, generated columns, large objects, materialized views, unlogged data, foreign tables, partition topology, publication filters, extensions, constraints, triggers, routines and access-control state.

## Files

- DATABASE-SURFACE-CONTRACT-v1.md — inventory ontology, coverage modes, claim scopes, threats and benchmark targets.
- SCENARIOS-v1.json and SCENARIOS-v1.schema.json — synthetic adversarial scenario seeds.
- surface-coverage/ — standard-library Go coverage evaluator, tests and scale benchmarks.
- BENCHMARKS-v0.1.md — workload contract and prospective targets.

## Current research basis

PostgreSQL 19 adds sequence synchronization, but cached sequence values mean a sequence page LSN alone cannot prove exact nextval state. Logical decoding does not generally replicate DDL; pgoutput does not include generated-column values; and large-object contents are outside ordinary logical row replication. See:

- https://www.postgresql.org/docs/19/release-19.html
- https://www.postgresql.org/docs/19/logical-replication-sequences.html
- https://www.postgresql.org/docs/19/functions-sequence.html
- https://debezium.io/documentation/reference/stable/connectors/postgresql.html

## Claim ceiling

The Go evaluator checks normalized coverage attestations, exact inventory/surface scope, a pinned attestation profile, freshness, state digests and independently evidenced observer identity/control-root membership. It does not collect PostgreSQL catalogs, sequence state, large-object contents or generated-column values; it does not authenticate attestations or prove the underlying state truthful. The runtime must supply pinned, independently verified evidence. No exact-head test or PostgreSQL integration/production qualification is claimed.
