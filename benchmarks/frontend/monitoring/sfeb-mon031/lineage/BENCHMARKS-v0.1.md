# MON-031 Backup-Lineage Benchmark Plan v0.1

## Current local reference sample

Environment: Go 1.23.2, Linux/amd64, virtual AMD EPYC 9V74 host. A 20-node full-plus-incremental chain produced three exploratory samples:

| Sample | Time | Bytes/op | Allocs/op |
|---|---:|---:|---:|
| 1 | 10.942 µs/op | 14,240 | 87 |
| 2 | 11.236 µs/op | 14,240 | 87 |
| 3 | 10.118 µs/op | 14,239 | 87 |

These are local microbenchmark observations, not a portable performance claim, and not a comparison against pgBackRest.

## Correctness targets (prospective)

- 100% of valid post-upgrade full and incremental controls qualify.
- 100% of chains rooted in pre-18.7 full backups block under an 18.7 cutoff.
- 100% of missing-parent, unknown-integrity and unknown-WAL-coverage cases remain indeterminate.
- 100% of duplicate IDs, cycles, cross-major lineages, identity mismatches and non-causal parent timestamps block.
- No candidate is qualified solely from a tool's summary success flag.

## Performance study

Measure evaluator cost separately from physical backup/restore:
- chain length 1, 2, 10, 100, 1,000 and 10,000;
- p50/p95/p99 wall time and allocations;
- malformed-chain early-exit time;
- memory versus input inventory size;
- JSON decode cost separately from lineage evaluation.

A prospective target is p95 below 1 ms for a 100-node chain on a documented 2-vCPU reference machine. Measure with repeated pinned runs and report environment/variance before treating the target as achieved.

## Industry comparison

pgBackRest already performs backup integrity, manifests, repository, WAL and restore operations. MON-031 does not replace those functions. Its proposed contribution is a portable, explicit version-transition and parent-lineage eligibility decision that prevents an otherwise valid-looking incremental from being considered after an upstream advisory invalidates its old full base.

A fair comparison must use identical backup artifacts and independent restore truth. No performance or superiority claim is made.
