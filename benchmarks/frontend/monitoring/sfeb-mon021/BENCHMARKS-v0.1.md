# MON-021 PostgreSQL CDC continuity benchmark v0.1

**Status:** workload contract and reference benchmark harness; no numeric result is frozen in this version.

## Reproducible commands

From `benchmarks/frontend/monitoring/sfeb-mon021/cdc-continuity`:

- `go test -race -count=5 ./...`
- `go vet ./...`
- `go test -count=1 -cover ./...`
- `go test -run=^$ -fuzz=FuzzLSNParse -fuzztime=2s`
- `go test -run=^$ -bench='BenchmarkValidate(Capture|Replay)' -benchtime=3x`

The benchmark harness uses 10, 100, 1,000, 10,000 and 100,000 decoded-event fixtures for capture and replay reconciliation. Fixture setup occurs outside the timed loop.

## Correctness-first workload matrix

| Workload | What it measures | Mandatory invariant |
|---|---|---|
| LSN parsing | LSN string normalization and validation | malformed LSNs never normalize to zero |
| Initial snapshot | source/slot identity and exported snapshot anchor | snapshot and consistent point bind to the exact slot profile |
| Capture validation | slot health, bounded range, event ordering and watermark | no false-complete capture; no ack beyond durable evidence |
| Replay reconciliation | restart overlap and event identity | exact duplicates idempotent; changed event fields/digest diverge |
| Retention health | restart LSN, WAL status and invalidation state | lost history cannot qualify |
| Failover readiness | standby slot persistence/sync, replay position and old-primary fencing | no failover success with unsynced/invalidated/stale lineage |

## Matched industry comparison

Do not compare the semantic checker to raw Debezium/Kafka Connect rows per second. A fair comparison must pin PostgreSQL version, output plugin, database schema, publication/filter, transaction shape, slot settings, WAL retention, snapshot mechanism, failover topology, sink durability and recovery behavior.

Measure separately:
- decoder/connector throughput;
- time from source commit to decoded message;
- time to durable MON evidence;
- consumer acknowledgement lag;
- retained WAL bytes and slot age;
- duplicate replay rate and conflict detection;
- failover recovery time;
- evidence bytes per logical message;
- independent witness and cryptographic verification cost;
- qualified source intervals per total assurance cost.

## Prospective qualification targets

These are targets, not achieved results:

- zero acknowledgements beyond a durably persisted and independently reconcilable manifest watermark across every finite crash schedule;
- 100% exact replay idempotency and 100% conflicting-event detection in the finite corpus;
- zero false failover-continuity qualifications;
- 100% expected classifications for frozen scenarios, including positive controls;
- at least 500 independently authored and blinded crash/failover/omission holdouts;
- for the in-memory reference, p95 under 200 ms for 100,000 decoded event records after optimization on a pinned reference host; memory and allocations must also be reported.

The end-to-end target is not maximum decoded rows per second. It is maximum **qualified source intervals per total assurance cost**, including database/WAL retention, compute, memory, storage, network, cryptography, sealing/witnesses, recovery and operator intervention.

## Measurement limitations

The current Go package is a semantic envelope checker over normalized slot state/manifests. It is not a live PostgreSQL decoder or replication client. Its booleans and witness markers represent upstream facts that a future pinned verifier/adapter must independently establish. No PostgreSQL server, failover topology, cryptographic verifier, Debezium/Kafka Connect comparator or independent witness is qualified by this file.

A numeric benchmark report should be frozen only after the exact committed source is checked out, test/benchmark commands are recorded with hardware/runtime metadata, and result artifacts are attached to the exact commit.
