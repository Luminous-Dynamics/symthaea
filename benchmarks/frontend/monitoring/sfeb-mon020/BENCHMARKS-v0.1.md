# MON-020 event-order benchmark v0.1

**Status:** local exploratory measurement of the semantic reference package. Not a vendor comparison or production service result.

## Environment

- Go 1.23.2
- Linux amd64
- Intel Xeon Platinum 8573C
- Command: `go test -run=^$ -bench='BenchmarkEvaluate(SameSource|CrossSourceWitness)' -benchtime=3x`
- Race gate: `go test -race -count=5 ./...`
- Static analysis: `go vet ./...`
- Coverage: `go test -count=1 -cover ./...`
- Fuzz: `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`

## Results

| Workload | Events / witnesses | ns/op | MB allocated/op | allocs/op |
|---|---:|---:|---:|---:|
| Same-source ordering | 24 events | 109,263 | 0.076 | 221 |
| Same-source ordering | 204 events | 947,060 | 0.671 | 1,539 |
| Same-source ordering | 2,004 events | 13,638,079 | 7.555 | 16,111 |
| Same-source ordering | 20,004 events | 153,026,250 | 84.555 | 162,273 |
| Cross-source handoffs | 10 witnesses | 99,876 | 0.084 | 268 |
| Cross-source handoffs | 100 witnesses | 901,064 | 0.734 | 1,870 |
| Cross-source handoffs | 1,000 witnesses | 12,866,927 | 8.498 | 18,987 |
| Cross-source handoffs | 5,000 witnesses | 66,858,275 | 46.499 | 95,828 |

Each point used three measured iterations after fixture setup. These are directional microbenchmarks, not stable latency distributions. The workload constructs synthetic source/event/witness objects, validates verifier reports in memory, and checks event-order constraints. It excludes cryptographic operations, real transport, source completeness collection, remote witness lookups, storage durability, database/WAL adapters, Holochain, dispatch, destination readback and physical instrumentation.

## Correctness preflight

- `go test -race -count=5 ./...`: PASS.
- `go vet ./...`: PASS.
- `go test -count=1 -cover ./...`: PASS, 89.5% statement coverage.
- `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`: PASS, 67,732 executions in the recorded run.
- 22 named tests plus one fuzz target.

The test seed corpus contains 23 descriptive scenarios including 6 positive controls; it has not yet been converted into a fully parameterized executable corpus or run with a separate oracle.

## Prospective targets

Not achieved:
- 100,000 same-source events: p95 under 500 ms after optimization on a pinned reference host.
- 10,000 cross-source handoff proofs: p95 under 250 ms after optimization.
- Incremental re-verification for a changed 1,000-event neighborhood: p95 under 50 ms, with correct dependency invalidation.
- Zero false-qualified outcomes on mandatory finite event-order cases and at least 500 independently authored blind holdouts.

Do not compare these numbers to telemetry ingestion or raw trace export throughput unless the comparators include the same sequence proof, cryptographic verifier, cross-source witness and constraint checking costs.

## Qualification boundary

This is a local semantic oracle, not a source collector, cryptographic implementation or event-order authority. It does not prove sources are complete or truthful and does not prove action effects. The source used for local preflight was reconstructed and formatted locally because a fresh GitHub clone was unavailable; these results are not exact remote-head checkout replay, hosted CI, external independent reproduction or production readiness.
