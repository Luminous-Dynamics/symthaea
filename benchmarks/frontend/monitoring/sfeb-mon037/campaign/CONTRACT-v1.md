# MON-037 — Unified Recovery Fault Campaign Contract v1

Profile: mon037-unified-recovery-campaign-input-v1. Status: offline normalized-output composer and deterministic adversarial corpus implemented; live adapter pending.

## Research question

Can one independent campaign oracle detect cross-component recovery failures that are invisible when PostgreSQL backup/PITR, Debezium offsets, Kafka topics, timeline ancestry, observer domains and evidence continuity are each tested in isolation?

MON-037 composes MON-024 through MON-036; it does not replace MON-025's PostgreSQL/Kafka lab, MON-032 transaction accounting or domain-specific evaluators.

## Evidence inputs and completeness

A trial binds:
- complete SQL/source truth with source-truth artifact SHA-256 and canonical event-set SHA-256;
- complete observed Kafka capture with capture-manifest SHA-256 and canonical observed-event SHA-256;
- topic/partition inventory with its own record-set digest;
- complete transaction metadata export with report SHA-256 and canonical transaction-proof digest;
- complete required database-surface inventory with inventory SHA-256 and required-ID digest;
- surface dispositions and their source/restored/N/A/exclusion digests, bound by a surface-assertion digest;
- optional MON-033 exact offset-boundary proof and raw-offset SHA-256;
- MON-034 timeline ancestry result;
- MON-035 observer-independence result;
- MON-036 event-ledger and claim-disposition result;
- MON-024 proof-graph result.

Artifact digests bind external files; record-set digests bind the actual arrays presented to this evaluator. Neither digest authenticates its producer. A trusted capture adapter and MON-026 custody verifier must authenticate bytes, origin, authority and completeness before a runtime claim is admissible.

A helper can derive canonical record-set digests only for `offline_fixture` input. It refuses `runtime_candidate`; production evidence must arrive already bound. Missing, mismatched or incomplete manifests never qualify.

## Event reconciliation

Expected and observed records are joined by event ID. The comparison tuple is transaction ID, topic, Debezium data_collection label, operation, payload SHA-256, transaction total_order and collection_order.

- Missing acknowledged events, unexpected records, conflicting identity reuse and payload mismatch block.
- Exact duplicate replay is counted separately. It remains indeterminate unless a digest-bound consumer idempotency proof is supplied. The proof digest is not authenticated by this semantic evaluator.
- Kafka offsets are monotonically checked within each topic/partition.
- V1 requires one partition per topic for ordered qualification. Multi-partition ordering is indeterminate; no cross-partition global order is guessed.

## Transaction closure

For each transaction require BEGIN and END observations, exact member count, matching END.event_count, and contiguous transaction total_order.

For the scoped `public.ledger` lane, row collection_order must be contiguous within each transaction and collection. Debezium END.data_collections must report the row collection count and the aggregate logical-message count. Message pseudo-collection labels are retained for audit but are not assumed stable or contiguous across transactions. Kafka offset order is independent of transaction total_order.

## Database surface coverage

The required surface IDs come from an explicitly complete inventory bound by a SHA-256 manifest and a canonical required-ID digest. Every required ID must appear exactly once; extra surfaces block.

For `full_state`, every surface must be digest-matched or proven `not_applicable` with a justification digest. Exclusions block. For `row_stream`, explicit exclusions require a digest-bound exclusion manifest. Unknown, omitted or unbound surfaces remain indeterminate.

## Component composition

- MON-033: exact offset target, raw-offset digest and predicate are checked. An unobserved target is indeterminate.
- MON-034: blocked ancestry blocks; indeterminate holds.
- MON-035: a shared failure domain blocks; incomplete observer inventory is indeterminate.
- MON-036: invalidated claims block; revalidate/unresolved claims hold; retain is necessary but not sufficient.
- MON-024: blocked/indeterminate results propagate. A qualified graph from an offline synthetic fixture is blocked as an unsafe promotion.
- All 12 restore-to-service assertions must be explicitly passed.

## Verdicts and limits

The component verdict reports lower-layer checks. The campaign verdict additionally incorporates MON-024 and execution mode. V1 never sets `traffic_activation_permitted=true`. Both `offline_fixture` and `runtime_candidate` remain indeterminate absent a separately authenticated runtime verifier and custody adapter.

The evaluator consumes normalized outputs. It does not authenticate signatures, fetch source truth, inspect live PostgreSQL/Kafka, validate WAL bytes or perform recovery. No live lab trial or SOTA claim follows from a passing offline case.
