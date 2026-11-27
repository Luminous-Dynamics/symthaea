# MON-033 — PostgreSQL Offset-Boundary Restart Challenge Contract v1

**Profile:** mon033-postgres-offset-restart-challenge-v1  
**Status:** read-only offset observer, target matcher and guarded restart helper implemented; live connector experiment pending.

## 1. Upstream regression target

Debezium 3.7.1.Final (2026-11-06) lists DBZ-2889: PostgreSQL changes may be lost on restart when lsn_events_processed is 1 or when the stored offset is at a transactional logical message. DBZ-2894 is a separate decoderbufs case where a BEGIN sharing the stored LSN can consume one row-event skip and re-emit a processed row.

At the pinned Debezium source:
- lsn_events_processed counts INSERT, UPDATE, DELETE and TRUNCATE.
- BEGIN, COMMIT, logical MESSAGE, ORIGIN and NOOP do not consume that counter.
- A stored offset at MESSAGE is treated separately because the counter may describe earlier row events at a different LSN.
- The Kafka Connect source offset keys include lsn, lsn_proc, lsn_commit, messageType, lsn_events_processed and txId.

## 2. Two exact target states

1. Same-LSN row boundary: lsn_events_processed == 1 and messageType is one of INSERT/UPDATE/DELETE/TRUNCATE.
2. Transactional logical-message boundary: messageType == MESSAGE with txId present; preserve the complete lsn_events_processed value because it can refer to rows preceding the message.

The target is reached only when the exact source offset returned by Kafka Connect GET /connectors/{name}/offsets satisfies the predicate. The tools do not patch or delete offsets.

## 3. Experiment sequence

- Poll connector status and read current source offsets.
- Capture raw offset JSON, status JSON, UTC capture time and SHA-256.
- Stop polling only after the exact target state is observed; save the target proof.
- Use the guarded restart helper to restart only the MON-025 Compose connect service.
- Wait for the connector/task to return to RUNNING and save the post-restart source offset.
- Reconcile SQL source truth against row, transactional logical-message and transaction topics through MON-025/MON-032.

## 4. Fail-closed semantics

- API unavailable, malformed response or multiple source partitions: no target qualification.
- Exact target not observed before timeout: indeterminate, not pass.
- Connector/task not RUNNING: no qualified target observation.
- Missing LSN fields, wrong message type or wrong event count: reject target.
- Restart not completed or connector not RUNNING afterwards: indeterminate.
- A restart without an apparent duplicate is not by itself proof of capture completeness; the independent SQL/CDC reconciliation remains mandatory.

## 5. Safety boundary

The observer uses GET requests only. It does not call Kafka Connect PATCH/DELETE offset endpoints. The only mutation helper requires MON025_ALLOW_OFFSET_RESTART=YES, a valid target proof, a disposable Compose project name and targets only the Compose connect service. Never use this helper against a non-lab service.

## 6. Known-answer fixtures

- lsn_events_processed=1 with INSERT qualifies as a target observation.
- messageType=MESSAGE with a nonzero preceding counter qualifies as a separate target observation.
- lsn_events_processed=0 does not satisfy the first target.
- MESSAGE does not satisfy the first target even if its event counter is one.

These test the target parser, not whether the running connector can reach the target.

## 7. Qualification metrics

Every trial records connector/worker and image digests, PostgreSQL version, broker version, Connect version, connector configuration digest, slot identity, exact before/after source offsets, raw topic evidence, independent SQL truth, transaction closure report, lost/duplicate/conflicting event IDs, RPO/RTO and operator interventions.

A trial qualifies only if the intended offset boundary was reached, restart completed, all assigned source events are independently accounted for, and no transaction or identity conflict remains.

## 8. Limits

Kafka Connect GET offsets exposes the connector's stored source offset; it does not prove that all corresponding output-topic records are durable or visible to an independent consumer. Topic reconciliation is separate. The lsn_events_processed=1 state may not be reached by every workload or plugin; absence is indeterminate. The decoderbufs DBZ-2894 COPY case is an optional compatibility lane, not the default pgoutput baseline.
