# MON-037 — Unified Recovery Fault Campaign

MON-037 composes normalized outputs from MON-033, MON-034, MON-035, MON-036 and MON-024 with MON-025's independent source/CDC oracle. It does not duplicate the domain-specific evaluators.

## Offline use

    cd benchmarks/frontend/monitoring/sfeb-mon037/campaign
    go run ./cmd/qualify --input INPUT-example.json
    bash CHECKS.sh
    bash RUN-OFFLINE-CAMPAIGN.sh

The offline runner executes MON-025 static preflight and parser fixtures, MON-024 Go tests, and the MON-034/035/036/037 check scripts. It never starts containers, invokes live fault injection, restarts a service or mutates Kafka offsets. A clean worktree and exact Git HEAD are recorded.

## Integrated invariants

- Source SQL truth, observed Kafka capture, transaction metadata and required database-surface inventory each carry an explicit completeness assertion and SHA-256 manifest reference.
- Each event includes both Kafka topic and Debezium data_collection label. END.data_collections is reconciled against collection labels, not incorrectly inferred from topic names.
- Missing, extra, conflicting or mismatched event IDs block.
- Exact replays are recorded; without a digest-bound idempotency proof they remain indeterminate.
- Transaction BEGIN/END, expected event counts, END.event_count, collection-label counts and transaction total_order are checked.
- Kafka offsets are checked monotonically within topic/partition. Row collection_order is checked for contiguity within each transaction/collection. Logical-message collection labels are not assumed to be globally stable or contiguous across transactions.
- V1 refuses to infer one global order across multiple Kafka partitions.
- Full-state claims cannot exclude required surfaces; row-stream exclusions require a digest-bound exclusion manifest.
- Timeline, observer independence and evidence continuity results must use their exact versioned profiles.
- MON-024 cannot promote a blocked or unknown lower layer. Synthetic fixtures never qualify production traffic.
- This tool never enables traffic.

SCENARIOS-v1.json contains 27 public scenario definitions. They are a development corpus, not a sealed holdout or 27 live recovery trials.
