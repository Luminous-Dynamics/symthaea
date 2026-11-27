# MON-033 — PostgreSQL Offset-Boundary Restart Challenge Contract v1

**Profile:** `mon033-postgres-offset-restart-challenge-v1`  
**Status:** read-only offset observer, target matcher and guarded restart helper implemented; live connector experiment pending.

## 1. Upstream regression and version baseline

The regression IDs are now mapped to release versions rather than treating the currently pinned post-fix source as an affected build.

- Debezium 3.7.1.Final (November 6, 2026) lists **DBZ-2889** (PostgreSQL changes lost on restart at `lsn_events_processed == 1` or a stored transactional logical-message offset) and **DBZ-2894** (decoderbufs BEGIN sharing the stored LSN can consume a row-event skip and re-emit a processed row) among its fixes.
- 3.7.1.Final was built against Kafka Connect 4.3.1 and tested with Kafka brokers 4.3.1. Use this as the stable fixed-version regression-control arm, not as the presumed affected version.
- Debezium 3.8.0.Alpha1 (development) also lists DBZ-2889 among fixes. 3.8.0.Alpha2 (development) lists DBZ-2894 and DBZ-2901 (PostgreSQL DML lost after restart from a transactional logical message following COPY). The 3.8 alpha line is an exploratory compatibility arm, not a production baseline.
- 3.7.0.Final is the proposed pre-fix reproduction candidate for DBZ-2889/2894. Its exact source/package and image digest must be pinned before interpreting any run as a reproduction.

See [VERSION-MATRIX-v1.md](VERSION-MATRIX-v1.md) for the trial matrix, upstream references and controls. Presence of an issue in release notes is not itself a reproduced data-loss result.

## 2. Offset semantics used by the observer

At the pinned Debezium PostgreSQL implementation, the source offset keys include `lsn`, `lsn_proc`, `lsn_commit`, `messageType`, `lsn_events_processed` and `txId`.

- `lsn_events_processed` counts INSERT, UPDATE, DELETE and TRUNCATE.
- BEGIN, COMMIT, logical MESSAGE, ORIGIN and NOOP do not consume that counter.
- A stored offset at MESSAGE is handled separately because the counter may describe preceding row events at another LSN.

The implementation details are used to construct exact target predicates; they do not imply that the fixed release still exhibits the defect.

## 3. Two exact target states

1. Same-LSN row boundary: `lsn_events_processed == 1` and `messageType` is one of INSERT/UPDATE/DELETE/TRUNCATE.
2. Transactional logical-message boundary: `messageType == MESSAGE` with `txId` present; preserve the complete `lsn_events_processed` value because it can refer to rows preceding the message.

The target is reached only when the exact source offset returned by Kafka Connect GET `/connectors/{name}/offsets` satisfies the predicate. The tools do not patch or delete offsets.

## 4. Experiment sequence

- Run a pre-fix candidate and fixed stable control with identical PostgreSQL, Kafka Connect/broker, connector config, plugin, schema, workload and storage settings.
- Poll connector status and read current source offsets.
- Capture raw offset JSON, status JSON, UTC capture time and SHA-256.
- Stop polling only after the exact target state is observed; save the target proof.
- Use the guarded restart helper to restart only the MON-025 Compose `connect` service.
- Wait for the connector/task to return to RUNNING and save the post-restart source offset.
- Reconcile SQL source truth against row, transactional logical-message and transaction topics through MON-025/MON-032.
- Run decoderbufs and COPY/logical-message cases as separately labelled sub-lanes; never pool their results with the default pgoutput case.

## 5. Fail-closed semantics

- API unavailable, malformed response or multiple source partitions: no target qualification.
- Exact target not observed before timeout: indeterminate, not pass.
- Connector/task not RUNNING: no qualified target observation.
- Missing LSN fields, wrong message type or wrong event count: reject target.
- Restart not completed or connector not RUNNING afterwards: indeterminate.
- A restart without an apparent duplicate is not by itself proof of capture completeness; independent SQL/CDC reconciliation remains mandatory.
- A post-fix release that does not reproduce is a passing regression-control observation only if the target was reached and all independent event-set checks were performed.

## 6. Known-answer fixtures

- `lsn_events_processed=1` with INSERT qualifies as a target observation.
- `messageType=MESSAGE` with a nonzero preceding counter qualifies as a separate target observation.
- `lsn_events_processed=0` does not satisfy the first target.
- MESSAGE does not satisfy the first target even if its event counter is one.

These test the target parser, not whether the running connector can reach the target.

## 7. Qualification metrics

Every trial records connector/worker and image digests, PostgreSQL version, broker version, Connect version, connector configuration digest, slot identity, exact before/after source offsets, raw topic evidence, independent SQL truth, transaction closure report, lost/duplicate/conflicting event IDs, RPO/RTO and operator interventions.

A trial qualifies only if the intended offset boundary was reached, restart completed, all assigned source events are independently accounted for, and no transaction or identity conflict remains.

## 8. Limits

Kafka Connect GET offsets exposes the connector's stored source offset; it does not prove that all corresponding output-topic records are durable or visible to an independent consumer. Topic reconciliation is separate. The `lsn_events_processed=1` state may not be reached by every workload or plugin; absence is indeterminate. This PR contains no live offset observation, restart or PostgreSQL/Kafka runtime result.
