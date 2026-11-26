# MON-016 industry boundaries and integration strategy v1

## Why this is a separate research problem

A transparency log can show that submitted statements were appended and were not later altered or removed from the witnessed tree. It cannot, by that fact alone, prove that every event that should have been submitted actually was submitted. SCITT's model permits issuers to choose whether to register statements; that makes selective submission a first-class boundary, not a verifier defect.

Likewise, a trace ID or log correlation field improves observability but is not a completeness proof. OpenTelemetry describes trace/log correlation as fragile in the presence of context propagation gaps, sampling and different telemetry sources.

PostgreSQL logical decoding has separate slot, LSN, restart, replay and failover semantics. A consumer must track progress and deal with redelivery/replay without duplicating downstream effects. A sequence of received records alone does not establish source completeness unless the adapter verifies the snapshot, slot state, cursor/high-watermark and terminal condition.

## Existing capabilities to respect

| Technology | Strong capability | Boundary MON must not overclaim | Integration plan |
|---|---|---|---|
| RFC 9162 / transparency logs | Inclusion proofs and append-only consistency between witnessed tree heads | Does not prove all external actions were submitted or observed | Use inclusion/consistency as manifest custody; add source-side enumeration evidence |
| SCITT architecture (RFC 9943) | Signed statement registration and receipt verification | Issuer can selectively register; receipt covers the registered statement, not all source activity | Register query-bound capture manifests; define a source-completeness profile separately |
| OpenTelemetry | Trace/span/log correlation, context propagation and diagnostic search | Context may be missing, sampled or lost; trace correlation is not a complete event ledger | Use trace IDs as optional diagnostics, not as proof identity; capture at receiver/source boundary |
| PostgreSQL logical decoding | Ordered change stream and WAL cursor, slot, snapshot and replay model | Slot replay, retention gaps, lag and source admin paths need their own qualification | Build a PostgreSQL adapter that binds consistent snapshot, slot identity, start/end LSNs, terminal flush LSN and WAL retention evidence |
| Transactional outbox | Couples business transaction and event row in one database transaction | Does not prove consumer processing or source honesty beyond DB/control root | Qualify outbox enumeration and consumer acknowledgement independently |
| Temporal / DBOS | Durable workflow execution and retries | Workflow completion is not a completeness proof over actions outside the workflow boundary | Capture at actual destination, reconcile operation ledger against workflow IDs |

## MON-016 integration profiles to build next

### PostgreSQL logical decoding

Record database system identifier, timeline, slot name and plugin, consistent snapshot/LSN, start LSN, end/flush LSN, slot restart LSN, WAL retention state, publication/table filters, and decoding transaction boundary. A decoder must detect lost slot, invalidated slot, retention overflow, timeline switch, duplicate replay, and incomplete transaction. Do not treat numeric LSN arithmetic as a dense ordinal stream. Map decoded transactions to an ordered source result ordinal inside a declared snapshot/range, while retaining the real LSN as native source identity.

### Transactional outbox

Bind table/schema version, tenant filter, exact query, snapshot isolation point, primary key range, commit LSN or monotonically increasing outbox sequence, retention policy, and consumer checkpoint. Reconcile outbox enumeration against source table state and consumer acknowledgements. The outbox's own operator/database is still a trust root.

### Telemetry and trace export

Declare sampling policy, dropped-span counters, exporter queue drops, collector restart markers, source IDs, time window, and clock-quality bounds. A query returning zero spans in a sampled trace backend cannot establish that no action occurred.

### Cross-source corroboration

Require two separately administered capture sources, each with its own query digest, snapshot, terminal cursor, manifest and evidence root. Define deterministic event identity mapping and classify one-sided records as divergence/indeterminate—not silently union them into a complete corpus. Shared data stores, operators, KMS, build pipelines or cloud admins remain common-cause dependencies.

## Research questions and benchmark targets

- Can source-side watermark + stable snapshot + gap detection distinguish genuine no-event windows from truncated exports with 100% accuracy on an independently labelled fault corpus?
- How quickly does each adapter detect capture gaps after source restart, retention pressure, failover, slot invalidation or collector drop?
- What is the assurance overhead in bytes/event, CPU, storage, WAL, network and wall-clock latency?
- What fraction of negative statements can be independently anchored without confusing an anchored manifest with global absence?
- What common-cause dependencies remain between source, witness, and second enumerator?

Proposed targets (not measured): zero false-complete results under injected truncation, zero global-absence assertions from a single source, 100% detection of all labelled gaps, and explicit indeterminate outcomes whenever a watermark, snapshot, cursor or verification proof is missing.
