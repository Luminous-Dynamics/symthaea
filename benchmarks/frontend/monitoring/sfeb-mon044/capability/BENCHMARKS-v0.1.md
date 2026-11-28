# MON-044 Benchmark Plan v0.1

## Hypothesis

A version/configuration-bound capability contract plus explicit negative-space inventory reduces false-complete recovery claims compared with reconciling only the records a connector happened to emit.

## Proposed targets (not achieved)

| Dimension | Gate |
|---|---:|
| Required inventory omissions | 100% held indeterminate |
| Duplicate/ambiguous declarations | 100% blocked |
| Unsupported required capability | 100% blocked |
| Excluded required operation without approved policy adjustment | 100% blocked |
| Conditional capability missing prerequisite | 100% indeterminate |
| Not-applicable without justification | 100% indeterminate |
| Capability identity/config drift | 100% blocked |
| Conformance report from older identity | 100% blocked |
| Transaction metadata before deployment epoch | 100% not claimed |
| DDL/schema state without independent capture lane | 100% not claimed |
| Generated/TOAST/truncate limitations | 100% exact expected disposition |
| Offline campaign authorization | 0 permitted |

## Negative-control matrix

- PostgreSQL DDL changes required but only logical row CDC exists.
- Transactions before Debezium transaction metadata activation.
- Generated columns required from pgoutput row events without recomputation/extraction lane.
- TOAST/unavailable-value placeholder mistaken for original value.
- TRUNCATE routed to multiple partitions and treated as globally ordered.
- skipped.operations or include/exclude filter removes required events.
- snapshot-to-stream boundary changed between capture receipts.
- schema history is missing or mismatched.
- a conformance report is reused after connector configuration changes.
- capability manifest claims a newer connector version than the tested binary.
- missing scope source digest or untrusted source inventory.
- optional unknown capability with all mandatory scope items covered.

## Matched industry experiment

Compare native connector config and event reconciliation, row-count reconciliation, event-ID reconciliation, MON-037 without capability closure, MON-044 plus MON-037–041, and independent catalog/application adjudication.

Use identical connector/server/plugin versions, source schema and workload, configuration, publication, snapshot boundary, partitions, event truth and fault schedule. Report false-complete, false-blocked and indeterminate, missing required surfaces, schema correctness, transaction metadata epoch, generated/TOAST completeness, truncate ordering, capture throughput, overhead, evidence bytes and operator effort.

Do not compare the assurance evaluator's microbenchmark to connector throughput. Publish the two separately.

## Non-claims

The module is an offline evaluator, not a runtime capability registry, vendor certification, live connector adapter, signed conformance service, or production recovery qualifier. All benchmark thresholds are proposed. No superiority or SOTA claim is made.
