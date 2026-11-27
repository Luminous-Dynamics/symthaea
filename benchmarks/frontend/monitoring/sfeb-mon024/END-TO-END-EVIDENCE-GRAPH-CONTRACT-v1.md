# MON-024 — End-to-End Evidence Graph Qualification Contract v1

**Profile:** `mon024-end-to-end-evidence-graph-v1`  
**Status:** semantic contract and local Go reference implemented; adapters, signed receipt verification and production qualification pending.

## 1. Research question

Can an independent verifier compose causal provenance, temporal ordering, database change-stream continuity, state-surface coverage and restore-to-service evidence into one bounded claim—and determine whether service activation is justified without trusting the recovery orchestrator's success flag?

MON-024 is a *composition layer*. MON-019 through MON-023 remain responsible for their domain-specific semantics. MON-024 consumes normalized receipts and checks that the claims refer to one coherent recovery boundary, that the proof graph is intact, and that each stage's required assertion is qualified.

## 2. Why composition is a distinct problem

A collection of individually valid subsystem results can still describe different databases, source epochs, snapshots, incompatible inventories or unrelated terminal positions. A successful restore can be based on incomplete capture; a complete image can be restored while dependencies remain unavailable; and a healthy-looking service can still be unsafe to activate while an old primary or stale endpoint remains writable.

The integration problem is not “did every tool return success?” It is “do independently checked facts form one connected proof of the exact state that the service will expose?”

## 3. Required proof graph

| Stage | Required parent(s) | Key claim |
|---|---|---|
| `causal_provenance` | none | all relevant inputs and effects attributed; authority bound |
| `temporal_order` | causal provenance | generation-before-use and independently witnessed cross-source handoffs |
| `cdc_continuity` | temporal order | snapshot bound, transaction set closed, ACK no further than durable evidence, timeline continuity |
| `surface_coverage` | CDC continuity | complete inventory; each surface represented under a declared claim scope |
| `restore_to_service` | CDC continuity + surface coverage | artifact/database recovery, dependencies, application probes/invariants, rollback, fencing and activation ordering |

Every parent reference carries the exact parent stage, evidence ID and output digest. Unknown stages, extra/missing edges, duplicate edges, missing parent receipts and digest substitutions are rejected.

The restore stage has two direct parents because recovery must bind both the source event history and the declared database-state inventory. It must not infer full-state coverage from a row-stream continuity claim.

## 4. Shared recovery boundary

All five receipts must match this identity tuple exactly:

- incident ID and recovery claim ID;
- source cluster identity and database;
- source epoch and source timeline identity;
- exported snapshot identity;
- terminal source position, interpreted by MON-021's PostgreSQL-specific rules;
- claim scope digest and database inventory digest.

The composition evaluator compares the terminal-position token for equality only. It does not parse PostgreSQL LSNs or compare positions across timelines; those semantics remain with MON-021. An adapter must not normalize away source epoch, timeline, snapshot or inventory identity.

## 5. Receipt quality and freshness

Each stage receipt includes a stable evidence ID, subject/output digests, exact parent references, stage status, evidence grade, producer and observer control roots, observation/expiry times, clock domain/uncertainty and required typed checks.

A production-eligible grade is `runtime_independently_verified`. `structurally_checked` and `synthetic` evidence cannot qualify production traffic. The composition oracle does not authenticate grade labels or verify signatures; a future receipt-verification adapter must do so.

Every receipt must be fresh at evaluation time. All receipt timestamps must be comparable against the declared evaluation clock domain. Missing clock metadata, a mismatched domain, or uncertainty above the configured bound yields `indeterminate`; a future observation or invalid validity interval blocks.

Every observer must identify a control root distinct from the producer's control root. The graph must include at least two observer control roots. This is a minimum diversity check only: actual administrative separation, tenancy, identity-provider independence and common-cause risk require external attestation; comparing strings does not establish real independence.

## 6. Typed stage assertions

### 6.1 Causal provenance

Required checks: `inputs_attributed`, `effects_attributed`, `authority_bound`. These summarize MON-019's causal input-to-effect graph and bounded authority/evidence lineage. They do not assert that an input was truthful merely because provenance is complete.

### 6.2 Temporal ordering

Required checks: `generation_before_use`, `cross_source_handoffs_witnessed`. These summarize MON-020's source-local sequence and cross-source order witnesses. Wall-clock order alone is not an acceptable handoff proof.

### 6.3 CDC continuity

Required checks: `snapshot_bound`, `transaction_set_closed`, `ack_within_durable_watermark`, `timeline_continuous`. These summarize MON-021's PostgreSQL-specific exported snapshot, complete transaction envelopes, durable-before-ack and timeline/failover checks. This package does not duplicate slot, WAL, LSN or transaction parsing.

### 6.4 Database surface coverage

Required check: `inventory_complete`.

Counts are separate and cannot be collapsed:

- `Covered`: directly represented by an accepted evidence method;
- `NotApplicable`: excluded only through an evidenced N/A determination;
- `Excluded`: outside the explicit claim scope;
- `Unknown`: unresolved.

For `full_recoverable_database_state`, every required surface must be `Covered` or proven `NotApplicable`; no required surface may be excluded or unknown. NotApplicable is not counted as Covered. For `row_stream_continuity`, explicit exclusions are permissible only when their manifest digest is present; unknown surfaces are not permissible.

### 6.5 Restore-to-service

Required checks: `artifact_validated`, `database_recovered`, `dependencies_ready`, `application_read_probe`, `application_write_readback`, `write_probe_rollback`, `application_invariants`, `old_primary_fenced`, `stale_endpoints_fenced`, `independent_service_acceptance`, `activation_order_verified`, `traffic_disabled_until_ready`.

A successful backup or database restore does not promote itself to a service-recovered claim. Application write probes must be rolled back. Old-primary and stale-endpoint fencing, independent service acceptance and activation ordering are explicit checks. The evaluator only emits a gate decision; it does not enable traffic itself.

## 7. Verdict semantics

- `qualified`: every required stage exists exactly once; identities and graph edges match; all required checks pass; evidence is runtime-independently graded and fresh; observer roots meet the minimum diversity; surface counts satisfy the selected claim; and there are no unresolved mutation outcomes.
- `blocked`: an explicit failed check, contradictory boundary, malformed graph edge, unsupported stage, duplicate stage/evidence ID, producer/observer control-root alias, impossible freshness interval or full-state exclusion exists.
- `indeterminate`: a required receipt/field is missing; a check is unknown; evidence is stale, non-runtime or not clock-qualified; a parent cannot be resolved; required surface coverage is unknown; observer-domain diversity is insufficient; or a prior mutation outcome is unresolved.

If both blocked and indeterminate findings exist, `blocked` takes precedence. `may_enable_traffic` is true only for `qualified`.

An indeterminate result is not permission to retry a mutating operation. The command identity and side effect must first be reconciled. Invariant: **unknown outcome ⇒ no new mutation until reconciliation**.

## 8. Threat model and limits

The composition layer is intended to expose cross-stage source/database/snapshot/timeline substitution; missing, duplicated, reordered or digest-substituted evidence edges; incomplete causal/temporal proof; uncommitted or prematurely acknowledged CDC events; database surfaces silently omitted from a full-state claim; NotApplicable incorrectly presented as copied coverage; stale, synthetic or structural receipts presented as production evidence; producer-controlled observers; clock mismatch/uncertainty; restore success promoted despite dependency, application, rollback or fencing failures; and unresolved command outcomes followed by unsafe mutation.

Out of scope for this semantic evaluator: cryptographic signature verification, HSM/key custody, observer identity federation, actual source/target state comparison, PostgreSQL WAL decoding, snapshot export, pgBackRest, Kafka transactions, network isolation, cloud control-plane operations and actual service traffic control. Those require runtime adapters and external evidence.

## 9. Benchmark hypothesis

The hypothesis is that a shared, typed proof graph can reduce false recovery claims across heterogeneous tools without requiring a single trusted orchestrator. This is not yet demonstrated against industry baselines. MON-024 will be evaluated by false-qualified rate under adversarial faults, positive-control qualification, coverage accounting, detection latency, evaluator overhead and operator reconciliation burden.

## 10. Integration sequence

1. Freeze the five normalized receipt schemas and profile versions.
2. Build adapters that translate MON-019–023 output into receipts without dropping source identity, scope, time or uncertainty fields.
3. Verify receipt signatures and observer control domains independently of the recovery orchestrator.
4. Run the same fault-injection corpus against Debezium/Kafka Connect, PostgreSQL logical replication and backup/PITR stacks.
5. Perform restore drills against PostgreSQL 18 and 19 with independent source and target observers.
6. Gate a canary service first; validate rollback and fencing behavior; only then widen traffic.

No production activation or SOTA claim follows from this contract alone.
