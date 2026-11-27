# MON-028 benchmark scorecard v0.1

## Safety scorecard

| Measure | Prospective target |
|---|---:|
| False-qualified outcomes | 0 across ≥10,000 independently sampled unsafe trials |
| One-sided 95% Wilson upper bound | Report; target ≤0.03% for 0/10,000 |
| Positive-control acceptance | 100% |
| Assigned replicate accounting | 100% |
| Missing/skipped/timed-out observations hidden | 0 |
| Exact per-family fault attribution | 100% |
| Suite digest agreement across independent operators | 100% |
| Candidate result expected-answer fields | 0 |

## Usefulness and operational scorecard

- Exact classification agreement and safe-outcome mismatches by fault family.
- False-rejected positive controls and unresolved positive controls.
- p50/p95/p99 scorer overhead, separate from database/service RTO.
- Domain RPO/RTO, missed event IDs, transaction closure, state surface divergences.
- CPU, memory, disk I/O, network and retained evidence bytes.
- Manual interventions, experiment setup/cleanup time and incomplete experiment count.

## Matched comparison

Every candidate/baseline pair uses the same signed suite, oracle, fault matrix, workload, environment profile, resource budget and case/replicate set. Implementation identity can differ. Report the full Pareto vector; do not create a weighted scalar that trades away false qualification.

## Sample size and independence

A 10,000-trial zero-error result has a nonzero one-sided Wilson upper bound of roughly 0.027%. Repeated runs sharing a host, database reset, failure-injection mechanism or operator are not automatically independent. Confirmatory experiments should randomize order, use independent reset domains and report cluster-aware intervals.

## What a win means

A system can claim a safety improvement only when it reduces false-qualified outcomes on the same ground-truth cases without increasing false-rejected positives, while retaining complete accounting. A latency improvement is a separate claim and must use matched resource budgets. A candidate that is merely more conservative may improve the safety metric while worsening usefulness; the scorecard must show both.

No MON system is currently measured against this scorecard. Targets are not results.
