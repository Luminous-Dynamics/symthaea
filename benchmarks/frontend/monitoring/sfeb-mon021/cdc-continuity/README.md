# MON-021 PostgreSQL CDC continuity reference

A standard-library Go semantic prototype for validating a declared PostgreSQL logical-decoding source profile, initial snapshot/slot consistent-point binding, capture range, decoded-event ordering, exact duplicate replay, durable-evidence watermark, consumer acknowledgement bound, slot retention health and failover readiness.

The reference deliberately does not require consecutive numeric LSNs: WAL positions can be sparse relative to logical messages. It checks monotonic message positions and exact event identity/fields/digests. Exact duplicate replay is idempotent; reuse of an event ID with different payload or fields is divergence.

The prototype consumes normalized slot state and manifests. It is not a PostgreSQL decoder, replication-protocol client, SQL runtime integration, timeline history parser, cryptographic verifier or independent witness. The `Witnessed` field is a semantic precondition supplied by the upstream MON custody/witness boundary; this package does not authenticate it. A live adapter must provide pinned verifier reports for source/slot identity, state snapshots and manifest/witness claims.

Run `go test -race -count=5 ./...`, `go vet ./...`, `go test -cover ./...`, `go test -run=^$ -fuzz=FuzzLSNParse -fuzztime=2s`, and `go test -run=^$ -bench='BenchmarkValidate(Capture|Replay)' -benchtime=3x`.
