# MON-037 Benchmark Plan v0.1

## Hypothesis

Composing subsystem outputs with independently captured SQL truth, Kafka capture manifests, transaction metadata, surface inventory, event identity and transaction closure detects cross-layer faults that isolated component-green checks miss, without turning absent evidence into pass.

## Prospective targets (not achieved)

| Dimension | Target |
|---|---:|
| Unsafe promotion | 0 across 10,000 deterministic adversarial mutations |
| Known-answer controls | 100% exact expected component disposition |
| Source/capture completeness | 100% complete and custody-verifiable manifests |
| CDC reconciliation | 100% event identity, data_collection, payload and transaction accounting |
| Duplicate handling | 100% conflicting duplicate detection; exact replay requires a digest-bound idempotency proof that covers the exact duplicate event-ID set | Replay identity mutation corpus |
| Transaction closure | 100% detection of BEGIN/END/count/data_collections/total_order gaps |
| Timeline | 100% ancestry/fork-boundary classification |
| Observer independence | 100% detection of shared domains in the v1 mutation matrix |
| Evidence continuity | 0 revalidate/invalidate/unresolved claims treated as retain |
| State inventory | 0 required full-state surfaces omitted, excluded or unknown |
| Evaluation overhead | p95 < 10 ms for 1,000 events and 100 transactions on a pinned 2-vCPU host |

The 10,000 mutations rotate across deterministic fault families. They are not independent live trials and cannot justify a rare-event confidence bound.

## Industry comparison

Use identical transaction scripts, PostgreSQL/Debezium/Kafka/Connect versions, WAL/backup snapshot, recovery target, fault schedule and independent SQL oracle. Compare component-only success flags, row-count-only matching, event-ID-only matching, rerun-all evidence policy, MON-037 and independent maintainer adjudication.

Report false-qualified, false-blocked and indeterminate rates, missing/duplicate/conflicting events, transaction closure, surface completeness, RPO, RTO, operator effort and all-in cost separately. Do not compare verifier microseconds with Debezium rows/s or pgBackRest restore throughput.

All live comparisons remain pending.
