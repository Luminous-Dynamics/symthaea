# Precision-aware REA Evidence admission and claim replay benchmark

This development benchmark is not a production decision path and adds no external dependencies.

## Workload matrix

The harness constructs semantically valid Evidence records from the checked-in public fixture. It varies:
- 1 / 4 / 16 / 64 Evidence records;
- 0 B / 64 KiB / 1 MiB irrelevant raw_result payload per record;
- precision mode off vs two unsafe integer fields per Evidence record, plus the same two fields in the asserted normalized value.

That produces 24 independent cells. Every receipt reports input bytes, result bytes, maximum record size and precision-warning count. All dimensions are separate; there is no aggregate leaderboard score.

## Four timed operations

1. **Evidence admission:** raw JSON parsing, duplicate-key rejection, schema validation and semantic Evidence identity.
2. **Precision-warning scan:** recursive JCS integer-warning discovery over already admitted claim/Evidence Values. This is isolated from parsing and from producer/replay overhead.
3. **Producer evaluation:** full typed claim evaluation, including warning discovery and derivation commitment, on already admitted Evidence values.
4. **Independent replay:** raw request/result/Evidence parsing, Evidence revalidation, independent warning/scope/disposition reconstruction and full result comparison.

The producer runs once outside timing to construct the valid result consumed by replay. Percentile distributions are not additive. A fast replay mismatch is not performance success.

## Protocol

- Three warmups and 50 measured samples by default.
- Smoke mode runs 24 cells with one measured sample and no warmups; this validates wiring/schema/correctness only.
- Nearest-rank p50/p95/p99/min/max are reported per operation and cell.
- 50-sample p99 is effectively the maximum; use at least 1,000 samples for exploratory tails and substantially more plus independent process-level repeats for defensible tail estimates.
- Record exact source SHA, Cargo.lock SHA-256, Rust compiler and profile, runner CPU/memory/OS/kernel, governor, concurrency and container limits.
- Compare paired revisions on the same machine, same lockfile/toolchain, same generated evidence, same workload cell and same concurrency.

## Interpretation

This benchmark does not define a performance target until stable repeated baselines exist. A provisional engineering goal is that the incremental precision-warning scan and claim-evaluation overhead should remain measurable and bounded; do not impose a percentage budget until representative payload distributions and run-to-run variation have been established.

Correctness gates dominate: all replay samples must return ReplayMatch; warning counts must match expected synthetic construction; the benchmark may not weaken duplicate-key rejection, schema/identity checks, scope binding, claim ceilings or resource limits.

Results qualify only the exact tree, lockfile, compiler, build profile, runner and workload matrix.
