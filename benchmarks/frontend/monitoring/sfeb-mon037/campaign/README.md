# MON-037 — Unified Recovery Fault Campaign

MON-037 composes normalized outputs from MON-033, MON-034, MON-035, MON-036 and MON-024 with MON-025's independent source/CDC oracle. It does not duplicate the domain-specific evaluators.

## Offline use

    cd benchmarks/frontend/monitoring/sfeb-mon037/campaign
    go run ./cmd/qualify --input INPUT-example.json
    bash CHECKS.sh
    bash RUN-OFFLINE-CAMPAIGN.sh

The offline runner executes MON-025 static preflight and parser fixtures, MON-024 Go tests, and the MON-034/035/036/037 check scripts. It never starts containers, invokes live fault injection, restarts a service or mutates Kafka offsets. A clean worktree and exact Git HEAD are recorded.

## Integrated invariants

- Exact source-event identity and payload digests are reconciled against observed topic records.
- Missing, extra, conflicting or mismatched event IDs block.
- Exact replays are recorded; without independent idempotency proof they remain indeterminate.
- Transaction BEGIN/END, expected event counts, END.event_count, data_collections, transaction total_order and per-topic collection_order are checked.
- Ordering is local to each topic/partition. V1 refuses to infer one global order across multiple partitions.
- Full-state claims cannot exclude required surfaces; row-stream scope may use explicit exclusions.
- Timeline, observer independence and evidence continuity results must use their exact versioned profiles.
- MON-024 cannot promote a blocked or unknown lower layer. Synthetic fixtures never qualify production traffic.
- This tool never enables traffic.

The 27 positive/negative controls are public synthetic development cases, not a sealed holdout or live recovery results.
