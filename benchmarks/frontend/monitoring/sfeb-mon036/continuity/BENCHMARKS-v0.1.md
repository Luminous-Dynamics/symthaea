# MON-036 Benchmark Plan v0.1

## What to measure

MON-036 is a selective claim-disposition evaluator, not a CDC engine or recovery orchestrator. Measure whether it revalidates only claims whose support dependencies changed, while never retaining evidence across unknown, revoked or contradictory events.

## Local semantic preflight

Environment: Linux/amd64, Go 1.23.2, Intel Xeon Platinum 8573C.

- \`go vet ./...\`: pass.
- \`go test ./...\`: pass.
- \`go test -race ./...\`: pass.
- Five repetitions, 100 claims x 1,000 unrelated events: approximately 3.01–3.66 ms/op, 1.318 MB/op, 8,428–8,429 allocations/op.

This is a local algorithm microbenchmark, not an end-to-end product comparison or SOTA claim. The workload intentionally tests event-index selectivity: 1,000 events are unrelated to all 100 claims.

## Prospective targets (not achieved)

| Dimension | Target | Test |
|---|---:|---|
| Selective correctness | 100% correct dispositions on blinded known-answer controls | Exact event/claim oracle |
| Stale evidence | 0 stale claims retained | Mutate config, timeline, authority and verifier inputs |
| Unaffected reuse | 100% valid unaffected controls retained | Mix unrelated changes into the same ledger |
| Unknown handling | 0 unresolved cases promoted to retain | Missing events/states, sequence gaps, global unknown changes |
| Revocation | 100% of affected claims invalidated | Revoke signer/authority for unexpired claims |
| Completeness | 100% event sequence and dependency accounting | Drop, duplicate, reorder, splice and tamper entries |
| Scale | p95 < 10 ms for 1,000 claims / 100,000 events on pinned 2-vCPU host | Fixed-input benchmark and heap profile |
| Selectivity | Unrelated events do not increase affected-claim count | Sweep unrelated event count with fixed support graph |

Do not use a weighted average that trades a false retain against speed. Report false-retain, false-invalidate, false-revalidate, missed-revalidation and unresolved rates separately.

At 0 false-retain outcomes in 10,000 independent unsafe trials, the one-sided 95% upper bound remains about 0.03%; zero observed failures is not proof of zero risk.

## Matched comparison

Use identical fault schedules and independent source-of-truth oracles:

1. ordinary freshness/TTL only;
2. rerun-all-on-any-change;
3. path/dependency heuristic;
4. MON-036 selective event-ledger disposition;
5. expert adjudication.

Measure exact claim dispositions, stale-evidence use, false invalidation, unnecessary revalidation burden, evidence acquisition cost, recovery RPO/RTO and operator effort. All arms receive the same event stream, dependency inventory, initial claims, budgets and source/target observations.

Do not equate MON-036 microseconds per claim with Debezium throughput or pgBackRest restore rate.
