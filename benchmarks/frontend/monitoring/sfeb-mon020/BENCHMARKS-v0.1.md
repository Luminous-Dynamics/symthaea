# MON-020 event-order benchmark v0.2

**Status:** local exploratory measurement of the current semantic reference profile. Not a vendor comparison or production service result.

## Environment

- Go 1.23.2
- Linux amd64
- Intel Xeon Platinum 8573C
- Command: `go test -run=^$ -bench='BenchmarkEvaluate(SameSource|CrossSourceWitness)' -benchtime=3x`
- Race gate: `go test -race -count=5 ./...`
- Static analysis: `go vet ./...`
- Coverage: `go test -count=1 -cover ./...`
- Fuzz: `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`

## Current preflight

- `go test -race -count=5 ./...`: PASS.
- `go vet ./...`: PASS.
- `go test -count=1 -cover ./...`: PASS, 88.0% statement coverage.
- `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`: PASS, 70,950 executions in the recorded run.
- 22 named tests plus one fuzz target.
- 23 scenario seeds, including 6 positive controls; descriptive only, not yet fully parameterized executable fixtures.

## Results

Go 1.23.2, Linux amd64, Intel Xeon Platinum 8573C, three measured iterations per point:

| Workload | Events / witnesses | ns/op | MB allocated/op | allocs/op |
|---|---:|---:|---:|---:|
| Same-source ordering | 24 events | 119,278 | 0.082 | 221 |
| Same-source ordering | 204 events | 1,096,538 | 0.712 | 1,534 |
| Same-source ordering | 2,004 events | 12,007,756 | 7.976 | 16,121 |
| Same-source ordering | 20,004 events | 234,743,413 | 88.709 | 162,243 |
| Cross-source handoffs | 10 witnesses | 100,166 | 0.093 | 271 |
| Cross-source handoffs | 100 witnesses | 1,502,315 | 0.798 | 1,869 |
| Cross-source handoffs | 1,000 witnesses | 21,323,172 | 9.117 | 18,963 |
| Cross-source handoffs | 5,000 witnesses | 78,169,076 | 49.630 | 95,862 |

Each point used three measured iterations after fixture setup. These are directional microbenchmarks, not stable latency distributions. The workload uses synthetic source/event/witness objects and normalized verifier reports in memory. It excludes cryptographic operations, actual transport, external source collection, capture completeness, remote witness services, storage durability, PostgreSQL/Holochain, dispatch, destination readback and physical instrumentation.

## Prospective targets

Not achieved:
- 100,000 same-source events: p95 under 500 ms after optimization on a pinned host.
- 10,000 cross-source handoff proofs: p95 under 250 ms after optimization.
- Incremental re-verification for a changed 1,000-event neighborhood: p95 under 50 ms.
- Zero false-qualified results on mandatory finite order tests plus at least 500 independently authored blind holdouts.

Do not compare these numbers to telemetry ingestion or raw trace export throughput unless comparators include the same source-sequence proof, normalized claim binding, verifier costs, cross-source witness and PROV constraint checks.

## Qualification boundary

This is a local semantic oracle, not a source collector, cryptographic implementation or event-order authority. It does not prove source truth or completeness. The local source was reconstructed and formatted because a fresh GitHub clone was unavailable; these are not exact remote-head replay, hosted CI, external independent reproduction or production qualification.
