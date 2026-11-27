# MON-038 Benchmark Plan v0.1

## Hypothesis

A stable per-partition capture vector can establish transactional visibility and event-set closure for multi-partition Kafka without a global ordering assumption. It should preserve legal offset gaps while detecting truncation, partition drift, topic recreation, missing capture ranges and event/capture disagreement.

## Prospective targets (not achieved)

| Dimension | Target |
|---|---:|
| Partition inventory | 100% of expected partitions have exactly one bounded range |
| Cluster/topic identity | 100% cluster, topic ID and partition mismatch detection |
| Visibility | 100% of non-read_committed captures rejected |
| Range closure | 100% detection of LSO/HW instability, truncation and out-of-range records |
| Business-event join | 100% exact event/transaction/payload/partition-offset linkage |
| Legal sparse offsets | 100% of known aborted/control-offset gaps accepted without inventing missing business events |
| Unsafe promotion | 0 across 10,000 deterministic adversarial mutations |
| Runtime overhead | p95 < 50 ms for 10,000 records over 256 partitions on a pinned 2-vCPU host |

The 10,000 mutation sweep is a software regression control, not 10,000 independent live Kafka failures or a rare-event confidence bound.

## Matched industry comparison

Compare: (1) Kafka read_committed consumer capture with offsets only; (2) consumer capture plus topic/partition inventory; (3) MON-038 with LSO/HW/leader-epoch and producer-fence evidence; (4) MON-038 plus MON-037 SQL/CDC truth reconciliation; and (5) independent operator adjudication.

Hold Kafka broker/client versions, topic configs, producer transaction schedule, transactional IDs, partition count, retention, consumer isolation and machine budget constant. Include aborted and open transactions, producer fencing, leader changes, retention movement, topic delete/recreate, compaction, partition expansion and cross-partition transactions.

Report false-qualified, false-blocked and indeterminate outcomes; visible business-event recall/precision; legal gap acceptance; capture time; RPO/RTO; CPU/memory; operator intervention; and total restore cost. Do not compare verifier microseconds directly to Kafka throughput benchmarks.

## Version matrix

Preserve Debezium 3.7.1.Final / Kafka Connect and broker 4.3.1 as the reproduction baseline. Run Kafka 4.3.2 as a separate extension rather than silently changing a control. Exercise the fixed 3.7.1.Final path against a pre-fix reproduction control for the PostgreSQL LSN/transactional-message restart boundary. Include Kafka transactional producer/consumer behavior, not just broker offset sequences.

All live runs and industry comparisons remain pending.
