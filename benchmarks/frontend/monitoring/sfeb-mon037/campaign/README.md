# MON-037 — Unified Recovery Fault Campaign

MON-037 composes normalized outputs from MON-033, MON-034, MON-035, MON-036 and MON-024 with MON-025's independent source/CDC oracle. It does not duplicate the domain-specific evaluators.

## Offline fixture use

The public example is deliberately unbound to its canonical record arrays until the fixture-authoring command computes their digests:

    cd benchmarks/frontend/monitoring/sfeb-mon037/campaign
    go run ./cmd/qualify --input INPUT-example.json --derive-fixture-record-digests > /tmp/mon037-bound-input.json
    go run ./cmd/qualify --input /tmp/mon037-bound-input.json
    bash CHECKS.sh
    bash RUN-OFFLINE-CAMPAIGN.sh

Digest derivation is accepted only when mode is `offline_fixture`. Runtime candidates must provide digests from independently captured source material; the evaluator verifies them but never derives them on behalf of a runtime claimant.

The offline runner executes MON-025 static preflight and parser fixtures, MON-024 Go tests, and MON-034/035/036/037 check scripts. It never starts containers, invokes live fault injection, restarts a service or mutates Kafka offsets. It requires a clean worktree, records the exact Git HEAD and hashes each step log.

## Integrated invariants

- SQL source truth, observed Kafka capture, transaction metadata, topic/partition inventory, required database surfaces and surface dispositions have canonical record-set digests plus completeness manifest references.
- Each event includes both Kafka topic and Debezium data_collection label. END.data_collections is reconciled against collection labels, not inferred from topic names.
- Missing, extra, conflicting or mismatched event IDs block.
- Exact replays are recorded; without a digest-bound idempotency proof they remain indeterminate.
- Transaction BEGIN/END, expected event counts, END.event_count, row counts, logical-message counts and transaction total_order are checked.
- Kafka offsets are checked monotonically within topic/partition. Row collection_order is checked within each transaction and collection. Logical-message labels are not assumed globally stable across transactions.
- V1 refuses to infer a global order across multiple Kafka partitions.
- Full-state claims cannot exclude required surfaces; row-stream exclusions require a digest-bound exclusion manifest.
- Timeline, observer independence and evidence continuity results must use their exact versioned profiles.
- MON-024 cannot promote a blocked or unknown lower layer. Synthetic fixtures never qualify production traffic.
- This tool never enables traffic.

SCENARIOS-v1.json contains 27 public scenario definitions. They are development cases, not a sealed holdout or 27 live recovery trials.
