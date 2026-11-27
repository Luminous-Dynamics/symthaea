# MON-021 PostgreSQL CDC continuity reference

A standard-library Go semantic prototype for PostgreSQL 18/19 logical decoding, including version/effective-WAL-level pinning, initial snapshot/slot consistent-point binding, capture range, complete transaction-envelope joins, decoded-event ordering, exact duplicate replay, durable-evidence watermark, consumer acknowledgement bound, slot retention health and failover readiness.

The reference deliberately does not require consecutive numeric LSNs: WAL positions can be sparse relative to logical messages. It checks monotonic message positions and exact event identity/fields/digests. Exact duplicate replay is idempotent; reuse of an event ID with different payload or fields is divergence.

The prototype consumes normalized slot state and manifests. It requires system identifier, server major and timeline binding in the slot/manifest; validates transaction begin/commit/end LSN and exact XID/member-event/ordinal binding; checks requested and actual sent acknowledgements; and reconciles complete transaction envelopes across replay. It requires a witnessed direct-parent timeline/fork point on failover, and returns indeterminate for two-phase decoding. It is not a PostgreSQL decoder, replication-protocol client, SQL runtime integration, timeline history parser, cryptographic verifier or independent witness. The Witnessed and timeline-history witness fields are semantic preconditions supplied by the upstream MON custody/witness boundary; this package does not authenticate them. A live adapter must provide pinned verifier reports for source/slot identity, state snapshots and manifest/witness claims.

Run `go test -race -count=5 ./...`, `go vet ./...`, `go test -cover ./...`, `go test -run=^$ -fuzz=FuzzLSNParse -fuzztime=2s`, and `go test -run=^$ -bench='BenchmarkValidate(Capture|Replay)' -benchtime=3x`.


PostgreSQL 19 can have configured `wal_level=replica` with `effective_wal_level=logical` after logical slot creation; both fields are pinned in the profile. `WAIT FOR LSN` is not a timeline proof. Sequences, DDL/schema, generated columns and large objects remain outside this row-stream coverage profile.


No exact-head Go test, vet, fuzz or numeric benchmark is claimed for the current evolving branch because a fresh remote checkout could not be obtained in this environment.
