# MON-032 — Composite-ID and Multi-Topic Transaction Closure Contract v1

**Profile:** mon025-multitopic-transaction-closure-v3  
**Status:** Offline jq verifier, fixture workload and adversarial checks implemented; live Kafka/PostgreSQL capture pending.

## What changed

MON-030 incorrectly grouped Debezium PostgreSQL events by the entire transaction ID string. Debezium's PostgreSQL integration test constructs the event ID from the numeric PostgreSQL transaction ID plus the LSN of the individual operation. The BEGIN, row events, logical messages and END can therefore have different suffixes. MON-032 correlates on the numeric XID prefix and preserves the full operation IDs for audit.

MON-030 also inspected only the ledger row topic. Debezium can emit transactional PostgreSQL logical decoding messages on a separate topic.prefix.message topic. Its integration test exercises BEGIN, row, logical message, row, END, with END.event_count counting all three data events.

## Lab workload

Each seed batch now emits one transactional logical message with prefix mon025_transaction in the same transaction as the ledger inserts. The connector is configured to include only this message prefix, the public.ledger table and transaction metadata.

## Verification rules

- Correlate row events, transactional logical messages and BEGIN/END using the numeric XID prefix before the first colon.
- Preserve the full operation-scoped IDs.
- Deduplicate exact row replays by event ID; reject conflicting reuse.
- Deduplicate logical-message replays by XID plus total_order; reject content or metadata conflicts at the same ordinal.
- Validate the union of row and logical-message total_order values as exactly 1 through END.event_count, with no gaps or collisions.
- Validate row data_collection_order in the observed single-partition ledger topic sequence.
- Preserve logical-message data_collection_order but require positive values. Do not assume the connector's internal logical-message pseudo-collection label is a stable table identifier.
- Reconcile END.data_collections: ledger count must equal observed rows; summed non-ledger counts must equal observed logical messages.
- Require exactly one unique BEGIN and END per numeric transaction identity. Count exact replays; block conflicting boundaries.
- Require one partition per topic for this profile. Multi-partition ordering needs a different partition/offset-aware benchmark.

## Evidence basis

Debezium upstream PostgreSQL transaction integration test:
https://github.com/debezium/debezium/blob/f7d50ae29417d160020ff52d477bd938e122d256/debezium-connector-postgres/src/test/java/io/debezium/connector/postgresql/TransactionMetadataIT.java

Debezium upstream logical decoding message integration test:
https://github.com/debezium/debezium/blob/f7d50ae29417d160020ff52d477bd938e122d256/debezium-connector-postgres/src/test/java/io/debezium/connector/postgresql/LogicalDecodingMessageIT.java

## Offline fixture matrix

Ten local jq fixture classes cover:
1. Row/message/row transaction with different operation LSN suffixes.
2. Missing logical-message observation.
3. Unrelated transaction XID at boundaries.
4. Reversed observed row-topic order.
5. Exact row replay.
6. Exact logical-message replay.
7. Conflicting logical-message content at the same XID/order.
8. Row/message total-order collision.
9. Multiple logical messages with non-assumed pseudo-collection labels.
10. Inconsistent END data-collection counts.

These are offline fixture results, not live connector results.

## Limits

This verifier is scoped to one captured table (public.ledger) and one configured logical-message prefix (mon025_transaction). It is not a generic multi-table Debezium auditor. Generalization requires a separately signed collection inventory, explicit capture scope and all configured data topics.

No Docker, Kafka, PostgreSQL or Debezium service was started during authoring. Live event capture, connector restart drills and the DBZ-2889 offset experiment remain pending.
