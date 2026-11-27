# MON-037 — Unified Recovery Fault Campaign Contract v1

Profile: mon037-unified-recovery-campaign-input-v1. Status: offline normalized-output composer and deterministic adversarial corpus implemented; live adapter pending.

## Research question

Can one independent campaign oracle detect cross-component recovery failures that are invisible when PostgreSQL backup/PITR, Debezium offsets, Kafka topics, timeline ancestry, observer domains and evidence continuity are each tested in isolation?

MON-037 composes MON-024 through MON-036; it does not replace MON-025's PostgreSQL/Kafka lab, MON-032 transaction accounting or domain-specific evaluators.

## Inputs

Each trial carries a stable identity, mode and evaluation time; optional MON-033 offset proof; MON-034 timeline result; MON-035 observer result; MON-036 event-ledger/claim disposition; MON-024 proof-graph result; source SQL truth and observed CDC envelopes; transaction closure summaries; source/restore surfaces; and restore-to-service checks.

Each component result retains its own profile and version. Missing evidence is never synthesized into a pass.

## Reconciliation

Expected and observed records are joined by event ID. The comparison tuple is transaction ID, topic, operation, payload SHA-256, transaction total_order and collection_order.

- Missing acknowledged events, unexpected records, conflicting identity reuse and payload mismatch block.
- Exact duplicate replay is counted separately. Without independent idempotency evidence it is indeterminate.
- Each transaction requires BEGIN and END, exact member count, matching END.event_count, data_collections, contiguous total_order and per-topic collection_order.
- V1 requires one partition per topic for order adjudication. Multi-partition ordering is indeterminate; no cross-partition global order is guessed.
- Full-state claims cannot exclude required surfaces. Row-stream exclusions must be explicit. Unknown surface state remains indeterminate.

## Component composition

- MON-033: exact offset target, raw-offset digest and predicate are checked. An unobserved target is indeterminate.
- MON-034: blocked ancestry blocks; indeterminate holds.
- MON-035: a shared domain blocks; incomplete observer inventory is indeterminate.
- MON-036: invalidated claims block; revalidate/unresolved claims hold; retain is necessary but not sufficient.
- MON-024: blocked/indeterminate results propagate. A qualified graph from an offline synthetic fixture is blocked as an unsafe promotion.
- All 12 restore-to-service assertions must be explicitly passed.

## Verdicts and limits

The component verdict reports lower-layer checks. The campaign verdict additionally incorporates MON-024 and execution mode. V1 never sets traffic_activation_permitted=true. Both offline_fixture and runtime_candidate remain indeterminate absent a separately authenticated runtime verifier and custody adapter.

The evaluator consumes normalized outputs. It does not authenticate signatures, fetch source truth, inspect live PostgreSQL/Kafka, validate WAL bytes or perform recovery. No live lab trial or SOTA claim follows from a passing offline case.
