# MON-050 benchmark protocol v0.1

## Workload

The microbenchmark creates one evidence node and N claim nodes, each claim having one mandatory dependency on the shared evidence node. One evidence-revocation event therefore affects all N claims. The benchmark includes graph validation and transitive impact evaluation; Go's benchmark harness reports allocations and bytes/op.

## One-iteration smoke observation

Observed in this execution container on linux/amd64, Go 1.23.2, Intel Xeon Platinum 8573C. Invocation: `go test -run '^$' -bench . -benchmem -benchtime=1x ./...`.

| Claims | Time/op | Bytes/op | Allocs/op |
|---:|---:|---:|---:|
| 1,000 | 2.39 ms | 1,047,648 | 2,229 |
| 10,000 | 21.51 ms | 9,345,024 | 21,293 |
| 100,000 | 257.20 ms | 88,758,336 | 223,615 |

These observations verify that the workload shapes execute. A single iteration is not a stable latency estimate, cross-host comparison, or accepted performance target.

## Next measurement protocol

- Pin toolchain, OS/kernel, CPU model, memory, power governor and exact commit.
- Run at least 30 independent benchmark processes per size; retain raw output and environment receipt.
- Report p50/p95/p99 process-level latency and confidence intervals; report allocation and GC cost separately.
- Add sparse, deep-chain, high-fan-out, high-overlap, mixed required/context, and concurrent-event graphs.
- Compare full recomputation with incremental reverse-index maintenance; do not trade exact invalidation for speed.
- Measure graph-validation cost separately from event-impact traversal.
- Require exact set equality and all anti-gaming gates for every performance run.

## Proposed gates (not yet achieved)

- 100% exact affected/unaffected set agreement on frozen deterministic fixtures.
- Zero missed mandatory invalidations and zero unrelated invalidations.
- Invalidate-all fails precision; invalidate-none fails recall.
- Event replay idempotence and commuting-event order invariance.
- Performance targets are set only after a repeated, pinned baseline is available.
