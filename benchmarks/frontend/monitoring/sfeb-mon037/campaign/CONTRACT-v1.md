# MON-037 — Unified Recovery Fault Campaign Contract v1

Profile: mon037-unified-recovery-campaign-input-v1. Status: offline normalized-output composer and deterministic adversarial corpus implemented; live adapter pending.

## Research question

Can one independent campaign oracle detect cross-component recovery failures that are invisible when PostgreSQL backup/PITR, Debezium offsets, Kafka topics, timeline ancestry, observer domains and evidence continuity are each tested in isolation?

MON-037 composes MON-024 through MON-036; it does not replace MON-025's PostgreSQL/Kafka lab, MON-032 transaction accounting or domain-specific evaluators.

## Evidence inputs and completeness

A trial binds:
- complete SQL/source truth with source-truth manifest SHA-256;
- complete observed topic capture with capture-manifest SHA-256;
- complete transaction metadata export with report SHA-256;
- complete required database surface inventory with inventory SHA-256;
- optional MON-033 exact offset-boundary proof and raw offsets SHA-256;
- MON-034 timeline ancestry result;
- MON-035 observer-independence result;
- MON-036 event-ledger and claim-disposition result;
- MON-024 proof-graph result;
- source and observed event envelopes, partition inventory, transaction proofs, and restore-to-service checks.

Manifest hashes bind named artifacts but do not authenticate them. A trusted capture adapter and MON-026 custody verifier must authenticate the bytes, origin, authority and completeness before a runtime claim is admissible. Incomplete or unbound manifests remain indeterminate.

Each component result retains its own profile/version. Missing evidence is never synthesized into a pass.

## Event reconciliation

Expected and observed records are joined by event ID. The comparison tuple is transaction ID, topic, Debezium data_collection label, operation, payload SHA-256, transaction total_order and collection_order.

- Missing acknowledged events, unexpected records, conflicting identity reuse and payload mismatch block.
- Exact duplicate replay is counted separately. It remains indeterminate unless a digest-bound consumer idempotency proof is supplied. The proof digest is not authenticated by this semantic evaluator.
- Kafka offsets must increase within each single-partition topic.
- V1 requires one partition per topic for ordering adjudication. Multi-partition ordering is indeterminate; no cross-partition global order is guessed.

## Transaction closure

For each transaction require BEGIN and END observations, the exact number of row plus logical-message records, END.event_count equality, exact Debezium data_collections label counts, and contiguous transaction total_order.

Row collection_order must be contiguous within each transaction and collection. Message collection_order must be present and positive, but v1 deliberately does not assume logical-message pseudo-collection labels or counters are stable across transactions. Kafka offset order is independent of transaction total_order.

## Database surface coverage

The required surface IDs come from an explicitly complete inventory bound by a SHA-256 manifest. Every required ID must appear exactly once.

For full_state, each surface must be digest-matched or proven not_applicable with a justification digest. Exclusions block. For row_stream, explicit exclusions require a digest-bound exclusion manifest. Unknown, omitted or unbound surfaces remain indeterminate.

## Component composition

- MON-033: exact offset target, raw-offset digest and predicate are checked. An unobserved target is indeterminate.
- MON-034: blocked ancestry blocks; indeterminate holds.
- MON-035: a shared failure domain blocks; incomplete observer inventory is indeterminate.
- MON-036: invalidated claims block; revalidate/unresolved claims hold; retain is necessary but not sufficient.
- MON-024: blocked/indeterminate results propagate. A qualified graph from an offline synthetic fixture is blocked as an unsafe promotion.
- All 12 restore-to-service assertions must be explicitly passed.

## Verdicts and limits

The component verdict reports lower-layer checks. The campaign verdict additionally incorporates MON-024 and execution mode. V1 never sets traffic_activation_permitted=true. Both offline_fixture and runtime_candidate remain indeterminate absent a separately authenticated runtime verifier and custody adapter.

The evaluator consumes normalized outputs. It does not authenticate signatures, fetch source truth, inspect live PostgreSQL/Kafka, validate WAL bytes or perform recovery. No live lab trial or SOTA claim follows from a passing offline case.
