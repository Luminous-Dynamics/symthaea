# MON-038 — Kafka Partition-Vector Capture Closure

MON-038 is the multi-partition capture complement to MON-037. It qualifies a bounded set of Kafka log ranges without pretending that independent partition offsets define one global order.

## Offline fixture use

    cd benchmarks/frontend/monitoring/sfeb-mon038/capture
    go run ./cmd/qualify --input INPUT-example.json --derive-fixture-proof-digests > /tmp/mon038-bound.json
    go run ./cmd/qualify --input /tmp/mon038-bound.json
    bash CHECKS.sh

The public example has two ledger partitions and one logical-message partition. It is deliberately gap-free for the positive control. The fixture helper derives proof digests only for offline_fixture mode; it refuses runtime_candidate.

## Core model

Each partition contributes its own half-open range [start_offset_inclusive, end_offset_exclusive), keyed by Kafka cluster ID, topic name, immutable topic ID and partition number.

A valid v1 vector requires a complete topic/partition inventory; exactly one range for every declared partition; producer fencing before capture; read_committed isolation; stable log-start offset, Last Stable Offset (LSO), high watermark and leader epoch; end_offset_exclusive equal to LSO; stable LSO equal to high watermark for a closed capture; topic cleanup policy delete; record, partition-manifest, inventory and vector digests; and exact linkage between captured business records and MON-037 business-event envelopes.

V1 reports offset slots not represented by visible records as `offset_gap_slots_unclassified`. Any positive count makes the capture verdict indeterminate. V1 does not infer that a gap is legal merely because Kafka `read_committed` hides aborted transactional records and control markers. The current input does not contain raw broker-log/control-marker evidence sufficient to classify every exact interval.

## Offset-hole safety rule\n\nThe negative-control suite removes the same in-range business record from both the captured-record array and business-event array, then recomputes fixture digests. The verifier must still return indeterminate because the range has an unexplained offset slot. A matching pair of incomplete arrays cannot qualify a capture.\n\nA future version may accept a gap only when independently authenticated, version-pinned broker-log/record-batch evidence accounts for the exact offsets as aborted data or control markers. Until then, no offset gap is silently accepted.\n\n## Ordering semantics

Ordering is checked only inside a topic partition. Transactions may span partitions. MON-038 never infers a global order from partition offsets, capture timestamps or topic names.

V1 explicitly requires one Kafka cluster per vector so (topic, partition, offset) cannot collide across clusters. Topic IDs detect delete/recreate drift.

## Trust boundary

A qualified component result means the submitted vector is internally consistent with its normalized metadata and digests. It does not authenticate the broker, prove producer fencing, fetch SQL source truth or authorize production recovery. Runtime use requires independently authenticated broker metadata, capture receipts, producer-fence evidence and MON-026 custody verification.

Offline and runtime-candidate inputs remain campaign-indeterminate. production_use_permitted and global_order_inferred are always false.

The scenarios are synthetic development evidence, not live Kafka validation or a SOTA comparison.
