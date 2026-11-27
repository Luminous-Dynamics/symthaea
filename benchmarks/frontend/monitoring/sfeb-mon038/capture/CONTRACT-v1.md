# MON-038 — Kafka Partition-Vector Capture Closure Contract v1

Profile: mon038-kafka-partition-capture-input-v1. Status: offline partition-vector evaluator and adversarial fixtures; live Kafka adapter pending.

## Research question

Can a recovery verifier establish complete, transactionally visible Kafka capture for every relevant partition without pretending that independent partition offsets define a total order?

## Partition vector

Each trial binds one recovery boundary: incident, claim, PostgreSQL source cluster, database, source epoch, target timeline, backup snapshot, terminal source position, scope digest and inventory digest.

The Kafka side binds one cluster ID and a topic inventory, including topic ID, partition count, cleanup policy and configuration digest. V1 requires exactly one Kafka cluster per vector. Every declared partition must have one range receipt.

Each range is half-open: [start_offset_inclusive, end_offset_exclusive). The receipt includes log-start offsets, LSO and high-watermark observations before and after capture, leader epochs, isolation level, capture times, completeness assertion, record count, record-set digest and range-manifest digest.

## Transactional visibility and range closure

V1 requires read_committed. It uses the Last Stable Offset as the exclusive boundary for committed visibility. The capture must end at that LSO. A closed, producer-fenced range additionally requires stable LSO and high watermark with LSO equal to high watermark. An LSO that has not caught up to the high watermark remains indeterminate; LSO beyond the high watermark blocks.

The producer-fence time must precede the capture window. The input binds a fence proof digest. The semantic evaluator does not authenticate that proof; the runtime adapter/custody verifier must.

## Offset holes and ordering

Offsets need only be strictly increasing within the partition, not contiguous. The reported hole count for a range is its width (end-exclusive minus start-inclusive) minus the number of visible records. This includes leading, interior and trailing offset positions not represented by visible records. Such holes can arise from aborted transactions or control records hidden from a read_committed consumer; the evaluator does not infer a cause from the count.

Ordering is local to (cluster, topic, partition). No global order is inferred between partitions. A transaction may span multiple partitions; its relationship is joined by transaction/event identity and verified separately by MON-037.

## Record closure

Every captured row-change or logical-message record must appear exactly once in the business-event set at the exact topic ID, partition and offset. Every business event must resolve to the matching captured record and match event ID, transaction ID, record class, collection label and payload SHA-256.

Transaction metadata, heartbeat, tombstone and opaque records remain separate record classes. Tombstones require a key digest. V1 requires cleanup policy delete; compacted topics are blocked because compaction can remove prior keyed values and does not support exact historical record completeness on its own.

## Digests

The input binds canonical topic inventory, per-partition visible-record digest, partition manifest, partition-range vector, business-event set and common recovery-boundary digests.

The offline fixture helper derives these digests only in offline_fixture mode. Runtime candidates must arrive pre-bound and independently authenticated. Hashes are byte binding, not source authenticity.

## Verdicts and non-goals

- blocked: identity collisions, invalid ranges, wrong topic IDs, non-read_committed isolation, compaction, digest mismatch or business-event/captured-record mismatch.
- indeterminate: incomplete inventory or capture, unstable LSO/high watermark/epoch, unfenced producer, missing receipt or unresolved open transaction.
- qualified: the submitted vector is internally consistent for the component's defined capture contract only.

Both offline_fixture and runtime_candidate remain campaign-indeterminate because live runtime attestation is not implemented. The verifier does not fetch Kafka, authenticate a broker, restore PostgreSQL or activate traffic. It does not infer global order.
