# MON-021 — PostgreSQL CDC Continuity Assurance Contract v1

**Status:** research contract and semantic prototype; PostgreSQL 18 runtime adapter pending.  
**Relationship:** connects MON-016 bounded capture and MON-017 source reconciliation to MON-013 destination-side commit/readback.

## 1. Research objective

Can an independent verifier establish that a database snapshot and every logical-decoding message within a declared slot/plugin/database/source boundary were preserved through a terminal WAL position, that the consumer never acknowledged beyond durable evidence, and that failover did not silently fork, skip or misidentify the source history?

MON-021 is a bounded-source assurance profile. It does not assert that the database emitted every possible business event, that the output plugin decoded every semantic state change, or that the data is truthful. It qualifies continuity only within the explicitly identified logical-decoding source profile and observed interval.

## 2. PostgreSQL facts that shape the assurance model

PostgreSQL 18 documentation establishes the following relevant semantics:

1. A logical replication slot represents an ordered stream of changes from a single database. Logical slots normally emit a change once, but after a crash the persisted slot position may move back to an earlier checkpoint and recent changes may be sent again. Consumers must avoid harmful duplicate processing.
2. The slot's `confirmed_flush_lsn` represents the LSN up to which the consumer confirmed receipt. It does not independently attest that the consumer durably stored and witnessed the corresponding MON evidence.
3. The slot's `restart_lsn` is the oldest WAL address that may still be required. `wal_status`, `safe_wal_size` and `invalidation_reason` expose important retention/invalidation state.
4. Logical slots can be synchronized to a hot standby for failover, but slot synchronization is asynchronous. A failover-capable slot must be synchronized and persistent on the standby, and the standby must have received sufficient WAL before promotion.
5. An exported logical-slot snapshot and `consistent_point` provide a defined starting state and WAL point for a snapshot-plus-stream capture. The snapshot must be imported under PostgreSQL's own lifecycle constraints.
6. PostgreSQL timelines distinguish histories after recovery/promotion. Timeline history records show the parent timeline and fork point; numeric LSNs on divergent branches must not be blindly compared as if they were one linear history.

See official PostgreSQL 18 documentation linked in the README.

## 3. Source identity: a CDC stream is not just a slot name

The source profile must bind:

- cluster `system_identifier`;
- database OID and database name;
- logical slot name and slot type;
- output plugin identity and version/configuration digest;
- publication/filter/query profile and schema/decoder profile;
- server major/minor version and relevant logical-decoding feature flags;
- source primary incarnation and timeline ID;
- verified timeline-history chain and fork points where failover/recovery occurred;
- initial slot creation record, consistent point and exported snapshot ID/digest;
- source administrative control root and its independently verified membership evidence.

Slot names are cluster-wide identifiers, but a slot record is still scoped to its source cluster, database and plugin. A name alone is not a stable source identity. Recreating a slot with the same name is a new source epoch unless an authoritative continuity record proves otherwise.

## 4. Snapshot-to-stream continuity

The initial capture binds an exported snapshot to the slot creation response:

- exact slot name and output plugin;
- exported snapshot name and source snapshot transaction context;
- `consistent_point` LSN;
- exact database/source identity and timeline;
- schema/publication/filter profile;
- snapshot enumeration manifest, terminal marker and digest;
- the first decoded change-stream boundary.

A snapshot is not complete merely because an export name was returned. The snapshot worker must complete an independently verifiable enumeration inside the correct snapshot transaction, and the stream consumer must retain changes from the matching consistent point.

The source adapter must reject:
- a snapshot created from another slot or database;
- a mismatched consistent point;
- a snapshot that expired before being imported;
- incomplete table/keyspace enumeration;
- schema or publication profile changes without a new profile epoch;
- an unexplained gap between snapshot completion and first decoded stream coverage.

## 5. LSNs and event identity

LSNs are positions in WAL, not consecutive application event numbers. Valid logical-decoding output may have gaps in LSN values because WAL contains records that the output plugin does not emit as row/change events. MON must not require numeric LSN adjacency.

Each decoded event should carry a stable identity built from the exact source incarnation, database, slot/profile, transaction identity, commit/end LSN and deterministic per-transaction message ordinal, where the output plugin can provide those fields. The exact identity profile must be versioned and must bind the full decoded message digest.

Requirements:
- LSNs are parsed and compared only within the same timeline/source lineage.
- Within one captured stream, decoded event positions must be monotonic under the profile's declared ordering.
- Multiple messages may share transaction/commit LSNs; message ordinal or another source-defined stable discriminator is required.
- Exact duplicate replay of the same event identity and digest is idempotent.
- Same event identity with different payload digest is divergence.
- A crash-induced replay may overlap previously captured positions; overlap must reconcile to exact prior event IDs/digests.
- A numeric LSN gap is not by itself evidence of missing logical changes. Completeness is established by the source's bounded decoded-stream terminal/watermark and collector health, not by adjacency of integers.

## 6. The critical invariant: durable evidence before acknowledgement

The consumer must obey:

`acknowledged_lsn <= durable_evidence_through_lsn`

The durable evidence watermark is advanced only after:
1. all decoded messages through the candidate boundary have been normalized and persisted;
2. the exact range manifest and event digests are committed;
3. the manifest is independently sealed or witnessed under its declared custody profile;
4. replay/reconciliation state is durable enough to safely handle a crash;
5. the MON evidence store confirms the transaction committed.

Only then may the consumer send a replication feedback/flush acknowledgement that can move `confirmed_flush_lsn` forward.

A client receipt, in-memory buffer, local function return or merely decoded maximum LSN is not a durable evidence watermark.

If PostgreSQL's confirmed flush position advances beyond the last durable/witnessed evidence boundary, the missing evidence may no longer be recoverable from the slot. The result is a capture-integrity incident and MON must not claim completeness.

## 7. Slot retention and health qualification

At each declared checkpoint, preserve the exact `pg_replication_slots` state, including:

- `restart_lsn`;
- `confirmed_flush_lsn`;
- `wal_status`;
- `safe_wal_size`;
- `xmin` and `catalog_xmin`;
- `temporary`, `active`, `failover`, `synced`;
- `invalidation_reason`;
- `plugin`, `database`, `slot_type`.

Interpretation:
- `wal_status = lost` or non-null invalidation reason: continuity unavailable; deny/hold.
- `wal_status = unreserved`: WAL is at risk of removal; do not promote the capture to complete without a proven retained range.
- non-logical, wrong-database, wrong-plugin or temporary slot: deny the declared profile.
- a `NULL` restart or confirmed-flush position is not a numeric zero; treat it as an explicit unknown/uninitialized state.
- `catalog_xmin` and `xmin` are retention constraints and must be preserved as source-health evidence; their presence is not proof that all required WAL was retained.
- `safe_wal_size` is a risk indicator, not a completeness proof.

## 8. Failover and timeline continuity

For a failover to a standby, qualification requires evidence that:
- the logical slot was explicitly failover-enabled;
- standby slot synchronization was configured and ran;
- the slot exists as persistent and `synced = true` on the intended standby before promotion;
- its invalidation reason is null;
- required WAL/catalog history was retained and the standby was sufficiently advanced relative to the consumer;
- primary/standby identities and timeline history are authenticated;
- no old-primary writer/decoder can continue to be treated as the current source after promotion;
- the new primary incarnation/timeline epoch and handoff point are sealed in the next capture manifest.

The `synced` flag has different meaning on primary vs hot standby and must be interpreted only on the standby. It is not a generic "stream complete" bit.

If timeline histories branch, LSNs after the fork on different timelines are incomparable unless a verified ancestry/handoff profile establishes their relation. v0.1 refuses to compare divergent timeline positions numerically. A future version may qualify an exact timeline-history chain and fork-point continuity relation.

## 9. Capture manifest

Each bounded manifest contains:
- source identity and CDC profile digest;
- slot/database/output plugin/filter/schema profile;
- timeline ID and timeline-history digest;
- initial snapshot/consistent-point reference (for initial capture);
- start and terminal LSN, with exact inclusivity semantics;
- ordered decoded event identities and payload digests;
- replay/duplicate classification and prior manifest reference;
- slot health/retention observations at start and end;
- collector identity and independent control root;
- observed dropped/overflow/retry/error counters;
- durable evidence watermark and external witness/seal reference;
- PostgreSQL acknowledged/confirmed-flush LSN observed afterward;
- capture start/end observation metadata and clock uncertainty.

The manifest describes completeness only for the declared slot/plugin/database/source interval. It does not establish global absence or business-level completeness beyond that source profile.

## 10. Decision model

| Result | Meaning |
|---|---|
| qualified | Snapshot/stream/slot identity, range, replay, durable evidence and bounded terminal state all reconcile |
| denied | A mandatory identity, range or durability predicate is proven false |
| indeterminate | Evidence is missing/stale/unbounded, retention is at risk, or failover continuity is not established |
| divergent | Same event identity has conflicting payloads, source lineage forks, timeline/slot identity conflicts, or durable records equivocate |

No scalar trust score may compensate for an unqualified interval or an acknowledgement beyond durable evidence.

## 11. Adversarial scenarios

The scenario corpus includes:
- exact slot name reused by another cluster/database;
- output plugin or filter profile changed mid-capture;
- snapshot consistent point mismatched with the first WAL boundary;
- exported snapshot expired or imported in the wrong transaction;
- duplicate WAL decoding after crash with exact same event bytes;
- duplicate event ID with altered payload;
- consumer ack advances before MON evidence store commit;
- evidence store commit acknowledgement is lost and reconciled;
- `confirmed_flush_lsn` ahead of durable evidence watermark;
- `restart_lsn` / WAL retention becomes unavailable;
- `wal_status = lost`, `unreserved`, or invalidation reason set;
- failover slot configured but not synced;
- slot temporary or nonpersistent on standby;
- standby timeline history absent or contradictory;
- old primary continues emitting after promotion;
- cross-timeline LSN values compared as if same linear history;
- missing event/capture manifest terminal watermark;
- exact replay overlap across consumer restart.

## 12. Industry position and fair comparison

| Existing capability | Strength | MON-021 extension |
|---|---|---|
| PostgreSQL logical decoding | WAL-derived ordered logical-change stream | Independent evidence manifests, durable-before-ack and source-scope claims |
| PostgreSQL replication slots | Retention and consumer progress state | Qualify restart/confirmed-flush, retention and acknowledgement semantics |
| Debezium | Mature CDC connectors, offsets and schema handling | Independently witnessed source continuity, timeline/slot identity and durable evidence watermark |
| Kafka Connect | Durable offset-based source/sink integration | Prove offsets are not advanced beyond sealed evidence and preserve exact source range |
| OpenTelemetry | Operational visibility of capture services | Capture-health observability is evidence, not completeness proof |
| MON-016/017 | Bounded capture and cross-source reconciliation | PostgreSQL-specific snapshot/slot/WAL/timeline profile |

Fair benchmarking must match source database version, data shape, output plugin, filter/publication, slot settings, checkpoint/wal retention, message volume, transaction sizes, failover model, downstream persistence and end-to-end assurance costs. CDC throughput alone is not source-completeness assurance.

## 13. Benchmark programme

### Mandatory finite-corpus goals

- zero false-qualified ack-ahead-of-durable-evidence cases;
- zero false-complete snapshot/stream range cases;
- zero silent conflicting event-ID replays;
- zero stale/old-primary or unverified timeline-continuity acceptance;
- 100% expected classifications for the frozen finite corpus, including positive controls.

### Matched workload dimensions

- initial snapshot sizes: 1 GB, 100 GB, 1 TB (controlled synthetic environments);
- CDC rates: 100, 1k, 10k, 100k decoded messages/s;
- transaction sizes: single-row, 100-row, 10k-row, multi-MB payload transactions;
- crash/restart point before/after decode, evidence persistence, witness seal, client feedback and checkpoint;
- failover with synchronized slot, delayed slot sync, catalog xmin gap, missing WAL, unsynced slot, old-primary split-brain;
- exact replay, partial replay and conflicting message substitution.

Record p50/p95/p99 capture latency, time-to-durable evidence, ack lag, retained WAL bytes, slot age, CPU, memory, network, storage, evidence bytes/message, witness cost, recovery time, duplicate rate and operator interventions.

### Prospective targets

- no acknowledgement beyond durable evidence watermark in every tested crash schedule;
- 100% exact replay idempotency and 100% conflict detection in finite tests;
- zero false failover-continuity acceptance;
- 500+ independently authored/blinded crash/failover mutations;
- report capture throughput and qualified evidence throughput separately;
- maximize qualified source intervals per total assurance cost, not raw decoded rows/s.

These are research targets, not results. Finite corpus success is not a universal guarantee.

## 14. Implementation sequence

1. Implement a Go/Rust semantic reference for LSN parsing, source/slot profile identity, snapshot-to-stream handoff, event ID replay, durable watermark and failover state.
2. Define a versioned normalized output-plugin event envelope and per-transaction event ordinal; do not assume a universal row ID.
3. Add PostgreSQL 18 SQL probes for server/DB/slot state and a reproducible disposable-server integration harness.
4. Implement snapshot/exported-snapshot acquisition and logical replication protocol capture, binding the slot consistent point to snapshot enumeration.
5. Persist event manifests before sending flush feedback. Exercise commit-ack loss and restart/replay.
6. Implement timeline-history / failover slot sync adapter with explicit control-root and old-primary fencing evidence.
7. Connect manifest output into MON-016/017 bounded capture and cross-source comparison; connect resulting state observation to MON-013/014.
8. Run PostgreSQL version-pinned integration, failure schedules, independent oracle and matched Debezium/Kafka Connect baselines.

## 15. Executable semantic reference

A standard-library Go package is provided in `cdc-continuity/`. It implements:
- PostgreSQL LSN parsing/formatting;
- exact source/cluster/database/slot/output-plugin/profile binding;
- exported snapshot and slot `consistent_point` binding;
- slot type, temporary/invalidation, WAL retention and `restart_lsn` checks;
- bounded capture interval and decoded-message monotonicity;
- per-transaction ordinal ordering for messages sharing an LSN;
- exact duplicate replay acceptance and event identity/field/payload conflict detection;
- equality of durable evidence watermark and terminal LSN;
- consumer acknowledgement and observed `confirmed_flush_lsn` bounds;
- replay overlap/gap reconciliation;
- failover slot identity, `failover`, standby `synced`, standby replay position, retained range, timeline epoch and old-primary fencing checks.

The reference intentionally does not require consecutive numeric LSNs. It also does not implement WAL decoding, replication protocol, SQL observation, actual slot synchronization, timeline history file parsing, cryptography or verifier-report transport. The `Witnessed` marker is a semantic precondition produced by an upstream MON custody adapter, not proof generated by this Go package.

Local source-level preflight was run on a semantically equivalent local prototype:
- `go test -race -count=5 ./...`: PASS.
- `go vet ./...`: PASS.
- `go test -count=1 -cover ./...`: PASS.
- `go test -run=^$ -fuzz=FuzzParseLSN -fuzztime=2s`: PASS.

The local source and public GitHub branch were not freshly cloned and replayed as an exact checkout. Treat these as local prototype results, not remote exact-head or hosted-CI qualification. The scenario corpus is 34 descriptive seeds / 11 positive controls, not yet a fully parameterized independent oracle corpus.

## 16. Claim ceiling

This contract and semantic prototype do not prove PostgreSQL SQL/runtime correctness, actual WAL retention, successful logical slot failover, source completeness, plugin semantic completeness, event truth, independent witness operation, database commit effect or physical outcome. A live PostgreSQL 18 integration and independently controlled source/collector/witness qualification remain mandatory.
