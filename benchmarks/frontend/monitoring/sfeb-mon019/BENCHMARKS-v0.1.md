# MON-019 causal-graph benchmark v0.1

**Status:** local exploratory measurement of a semantic reference package. Not a vendor comparison or production service result.

## Environment

- Go: 1.23.2
- OS/architecture: Linux amd64
- CPU reported by Go: Intel Xeon Platinum 8573C
- Benchmark command: `go test -run=^$ -bench=BenchmarkEvaluateValidGraph -benchtime=3x`
- Race gate: `go test -race -count=5 ./...`
- Static analysis: `go vet ./...`
- Coverage: `go test -count=1 -cover ./...`
- Fuzz smoke: `go test -run=^$ -fuzz=FuzzDigestProfileIsolation -fuzztime=2s`

## Workload

The benchmark constructs one Activity with N authenticated external input entities, one `used` edge and one `attributed_to` edge per input, exact equality between declared/observed/graph-linked input sets, verifier reports for each input, and a bounded capture manifest. The resulting graph is approximately N+4 nodes and 2N+3 edges.

The benchmark measures in-memory semantic graph validation and pinned normalized verifier-report joins. It does not include cryptographic verification, transport authentication, source collection, PostgreSQL, Holochain, network calls, independent witness infrastructure, storage durability, action dispatch, or physical instrumentation.

## Results

| Inputs | Approx. nodes | Approx. edges | ns/op | MB allocated/op | allocs/op |
|---:|---:|---:|---:|---:|---:|
| 10 | 14 | 23 | 56,226 | 0.021 | 52 |
| 100 | 104 | 203 | 292,206 | 0.209 | 290 |
| 1,000 | 1,004 | 2,003 | 4,004,410 | 2.465 | 2,280 |
| 10,000 | 10,004 | 20,003 | 32,433,831 | 20.468 | 21,949 |
| 50,000 | 50,004 | 100,003 | 197,689,933 | 89.784 | 115,630 |
| 100,000 | 100,004 | 200,003 | 458,734,669 | 179.695 | 231,734 |

The table reports Go allocation bytes as decimal MB for readability. Each point used only three measured iterations after setup and therefore is a directional baseline, not a statistically stable latency distribution. The 100,000-input point is approximately 459 ms for this in-memory graph-validation profile on this host.

## Correctness preflight

- `go test -race -count=5 ./...`: PASS.
- `go vet ./...`: PASS.
- `go test -count=1 -cover ./...`: PASS, 92.0% statement coverage.
- `go test -run=^$ -fuzz=FuzzDigestProfileIsolation -fuzztime=2s`: PASS, 67,017 fuzz executions in the recorded run.
- 35 named tests plus one fuzz target.

## Interpretation and next targets

The verifier reports are pre-indexed by subject, avoiding a repeated full report scan for every node. The semantic validator is broadly close to linear for this workload, but allocation volume remains material: about 180 MB allocated at 100,000 inputs. This is a clear optimization opportunity.

Prospective targets for a pinned, reproducible reference host—not achieved results:

- 100,000 inputs / roughly 200,000 edges: p95 under 250 ms after allocation and traversal optimization.
- 500,000 nodes / 1,000,000 edges: under 2 seconds for in-memory structural validation.
- Incremental re-verification of a 1,000-edge changed subgraph: p95 under 50 ms, with dependency invalidation correct.
- Preserve zero false-qualified outcomes on the mandatory finite adversarial corpus while optimizing.

These targets must be validated using repeated runs, fixed hardware, graph distributions (chain, star, high fan-in, high fan-out, sparse and dense), memory profiles, adversarial graphs, and an independent implementation/oracle. They do not imply a SOTA win. Throughput is not comparable to telemetry ingestion, trace export or workflow no-op throughput unless the tested semantics include the same graph verification and evidence costs.

## Qualification boundary

The source used for local preflight was reconstructed from the public branch content and formatted/tested locally because a fresh GitHub clone was unavailable in the execution environment. These results are not an exact-head checkout replay, hosted CI pass, cryptographic conformance run, external independent reproduction or production readiness claim.
