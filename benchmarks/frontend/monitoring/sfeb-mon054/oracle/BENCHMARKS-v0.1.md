# MON-054 benchmark protocol v0.1

## Primary outputs

- Witness-set exactness and witness trust/completeness.
- Pairwise shared/missing failure-domain dimensions, not a scalar independence score.
- Exact witness inventory member agreement.
- Covered / missing / excluded / unsupported / unknown counts.
- Captured extras and contradictions.
- Coverage numerator/denominator, with exclusions visible and never counted as covered.

## One-iteration smoke observations

Platform: linux/amd64, Go 1.23.2, Intel Xeon Platinum 8573C. Invocation: `go test -run '^$' -bench . -benchmem -benchtime=1x`.

| Witnesses × members | Time/op | Bytes/op | Allocs/op |
|---:|---:|---:|---:|
| 2 × 100 | 131.6 µs | 34,152 | 26 |
| 10 × 100 | 109.0 µs | 66,104 | 48 |
| 50 × 100 | 1.50 ms | 526,168 | 138 |
| 10 × 1,000 | 461.6 µs | 501,016 | 50 |

These are smoke observations only. Do not interpret the non-monotonic 2×100 vs 10×100 timing as a performance trend.

## Repeated benchmark protocol

Pin exact commit, Go toolchain, OS/kernel, CPU, memory and power state. Run at least 30 independent benchmark processes per workload. Publish raw results and p50/p95/p99 across process runs. Workloads should include large sparse inventories, high-fanout domains, shared identity/control planes, missing dimensions, conflicting members and many witness pairs.

## Proposed initial gates (not yet achieved)

- 0 qualified results with fewer than two preregistered witnesses.
- 0 independence claims with any shared or missing declared dimension.
- 100% detection of seeded missing required resources.
- 0 excluded resources counted as covered.
- 0 undocumented captured extras silently accepted.
- 100% exact known-answer scenario verdicts.
