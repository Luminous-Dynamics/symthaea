# MON-044 — Capture Capability and Negative-Space Qualification v0.1

Status: executable offline evaluator specification and development plan. No live connector inventory, vendor-signed capability registry, or production qualification is implemented by this document.

## Objective

Prevent a recovery campaign from treating a CDC stream, backup manifest, broker topic, or adapter output as complete merely because the records it did capture reconcile successfully.

A capture mechanism has a versioned observation boundary. It may omit, transform, delay, coalesce, classify as unavailable, or intentionally exclude parts of source state. A recovery claim must qualify that boundary against an independently enumerated required-state inventory.

MON-044 evaluates **positive coverage and negative space**:
- what the capture mechanism is capable of observing;
- what is conditionally observable under a particular configuration;
- what is unsupported by the connector/server combination;
- what is intentionally excluded;
- what is not applicable, with justification;
- what remains unknown.

An unsupported required surface or an explicit exclusion cannot be converted into complete recovery by matching the events that happened to be emitted.

## Research premise

The relevant contract is not "CDC is on." It is:

`RequiredRecoveryScope ⊆ VerifiedCaptureCapability(scope, version, configuration, evidence)`

The subset relation is only established when each required scope item has exactly one explicit disposition and the disposition is supported by version/configuration-bound conformance evidence or an independently approved exception.

## Capability identity

A capability manifest MUST bind:
- connector and plugin identity;
- exact connector build/version and distribution digest;
- database engine/version and logical-decoding plugin;
- broker/client version and protocol compatibility profile;
- connector configuration digest, including include/exclude filters and topic routing;
- publication identity/digest and replication-slot identity where applicable;
- schema-history identity and schema epoch;
- snapshot mode and snapshot completion boundary;
- transaction metadata settings and their activation epoch;
- operation filtering/skipping settings;
- column handling policy, including generated/toasted/unavailable values;
- topic partitioning/routing behavior;
- test-profile digest and conformance-suite version;
- manifest profile/version and canonical digest.

A connector name or version string without the complete configuration and source-plugin identity is not a capability identity.

## Coverage model

Every required recovery-scope item is described as a stable tuple:
- scope item ID;
- state domain (row data, schema/DDL, transaction boundary, generated value, large/toasted value, truncate, logical message, sequence, extension state, publication/filter state, etc.);
- object identity (database/schema/table/column/sequence/topic/partition or explicit global scope);
- operation or observation class;
- recovery requirement (required, optional, or explicitly permitted exclusion);
- source scope digest and inventory provenance reference.

Every declared capability item has exactly one status:
- `supported`: the pinned connector/server/configuration combination is expected to capture this class;
- `conditional`: capture is supported only when an explicit prerequisite is satisfied and evidence proves it;
- `unsupported`: the pinned combination does not capture this class;
- `excluded`: configuration intentionally filters or skips this class;
- `not_applicable`: the requirement genuinely does not apply to this source/scope and requires justification;
- `unknown`: the evidence does not establish any of the above.

A `supported` or `conditional` claim MUST bind a conformance-test profile and passing test report, plus the exact connector/source/configuration identity. A vendor documentation reference is useful provenance but does not replace runtime conformance evidence.

An `unsupported` or `excluded` item MUST remain visible in the output. It blocks a complete-recovery claim whenever the item is required. It can be accepted only when an independently authorized recovery policy explicitly permits the exclusion and the scope of that exception is digest-bound. A silent filter is never an implicit exception.

A `not_applicable` item needs a reason and a justification artifact digest. Missing evidence is never equivalent to not applicable.

## Required Debezium/PostgreSQL negative-space cases

The public development corpus MUST include these version-aware cases:

1. **DDL/schema evolution:** PostgreSQL logical decoding does not emit DDL changes. The recovery scope therefore needs independent schema/catalog history, migration ledger, or an explicit verified schema snapshot.
2. **Transaction metadata activation epoch:** Debezium transaction metadata is available only for transactions occurring after the connector is deployed with transaction metadata enabled. Historical transactions before the activation epoch cannot be retroactively certified from those metadata records.
3. **Generated columns:** the PostgreSQL `pgoutput` plugin does not capture generated-column values as ordinary row-change payload. The recovery policy must require independent recomputation, separate extraction, or explicit non-requirement.
4. **TOAST/unavailable values:** placeholder/unavailable-value semantics must not be mistaken for the actual source value. The capability statement must bind the column handling policy and any source-side fetch mechanism.
5. **TRUNCATE and partition ordering:** truncate events affect whole tables and have no message key; when routed to multiple Kafka partitions, per-table ordering is not automatically guaranteed across partitions.
6. **Skipped operations and filters:** configured skipped operations, publication filters, table/column filters and topic routing are part of the recovery boundary.
7. **Snapshot-to-stream handoff:** snapshot completion boundary and subsequent WAL position must be bound so that rows changed during the snapshot are neither omitted nor double-counted without idempotency evidence.
8. **Schema-history loss or drift:** data events without the exact schema history required to interpret their payload are not full recovery evidence.
9. **Sequences and non-row state:** sequence values, extensions, roles, grants, publications and other non-row objects require separate inventory/capture lanes where they are part of the recovery policy.

These cases are not assumptions about every future connector release. Each capability is version-pinned and must be re-evaluated when connector, database, plugin, config or schema changes.

## Evaluator rules

For each required scope item:
- exactly one declared capability disposition must match its stable item ID and scope digest;
- duplicate declarations block;
- missing declarations are indeterminate;
- mismatched object/scope/configuration identity blocks;
- supported requires a passing conformance report bound to the same capability manifest;
- conditional requires every named prerequisite to have explicit passing evidence;
- unsupported/excluded required items block unless an approved, scope-bound exception exists;
- not_applicable requires a valid justification digest;
- unknown is indeterminate.

The evaluator must distinguish:
- `capability_verdict`: what the pinned mechanism can observe under the declared capability contract;
- `coverage_verdict`: whether the required recovery scope is covered;
- `campaign_verdict`: whether authenticated runtime evidence and all other MON obligations have been satisfied.

For offline fixtures, a coverage-qualified result remains campaign-indeterminate. MON-044 does not authenticate the manifest, documentation, conformance report, exception authority, or source inventory. MON-041 custody is required for runtime admission.

## Evidence and trust boundaries

MON-044 consumes:
- independently sourced required-scope inventory;
- capability manifest bound to exact connector/server/plugin/configuration digests;
- conformance test report for every supported/conditional class;
- configuration and publication/filter evidence;
- schema/migration/catalog evidence for non-CDC state;
- exception authority and scope proof where exclusions are permitted;
- MON-041 authenticated receipts for runtime use.

Hashes bind bytes; they do not authenticate the issuer. A passing conformance test proves only the tested behavior over its declared fixture and version/configuration, not universal connector correctness. A vendor statement is not a substitute for the exact conformance test.

## MON-044 outputs

- capability manifest digest and identity summary;
- required-scope inventory digest;
- per-item disposition and reason codes;
- supported/conditional conformance evidence links;
- unsupported/excluded/not-applicable/unknown lists;
- unfulfilled required-scope items;
- permitted exception list and policy digest;
- capability verdict;
- coverage verdict;
- campaign verdict (always indeterminate in offline v1);
- explicit limitations and revalidation triggers.

## Benchmark programme

All targets below are proposed, not achieved.

| Dimension | Proposed gate |
|---|---:|
| Required-scope inventory omissions | 100% held indeterminate |
| Duplicate/ambiguous capability dispositions | 100% blocked |
| Unsupported required capability | 100% blocked |
| Configured exclusion of required operation | 100% blocked absent signed exception |
| Conditional capability without every prerequisite | 100% indeterminate |
| Not-applicable without justification | 100% indeterminate |
| Connector/version/configuration drift | 100% invalidated or held |
| DDL/transaction-metadata activation/generated-column/TOAST/TRUNCATE negative controls | 100% exact expected disposition |
| Unsupported and excluded item visibility | 100% retained in reports |
| Offline fixture production authorization | 0 permitted |

The mutation corpus is a deterministic evaluator regression set, not an operational failure probability.

## Matched industry comparison

Use the same source schema, data, transaction workload, connector version, configuration, snapshot boundary and independent source truth.

Compare:
1. Native Debezium + Kafka configuration and emitted records only.
2. Native CDC plus row-count reconciliation.
3. Native CDC plus event-ID reconciliation.
4. MON-037 event reconciliation without MON-044 capability closure.
5. MON-044 capability/negative-space closure plus MON-037–041.
6. Independent schema/catalog and application-level adjudication.

Measure:
- required-state coverage and unobserved surface count;
- false-complete recovery claims;
- unsupported/excluded operation detection;
- schema/DDL correctness;
- transaction metadata activation-boundary correctness;
- generated/TOAST value completeness;
- truncate and multi-partition ordering correctness;
- snapshot-to-stream handoff correctness;
- runtime overhead, capture throughput, evidence volume and operator effort.

The benchmark must distinguish connector-native capability from assurance added by MON-044. No claim of replacing or outperforming Debezium is warranted; the research question is whether capability-aware scope closure prevents false-complete recovery claims.

## Revalidation triggers

A capability statement is invalidated or requires revalidation when:
- connector, plugin, database, broker or client version changes;
- connector configuration, publication, include/exclude filters, topic routing, or operation skipping changes;
- schema/migration state changes;
- transaction metadata is enabled/disabled or activation epoch changes;
- the source database identity, snapshot epoch, replication slot or publication changes;
- conformance suite or test report is superseded;
- a capability exception is revoked or expires.

Revalidation is scoped to dependency edges; unrelated claims need not be discarded if their exact inputs and evidence remain unchanged.

## Execution sequence

1. Implement the shared capability/required-scope schema and canonicalization vectors.
2. Implement a deterministic offline evaluator with fail-closed negative controls.
3. Add Debezium/PostgreSQL public cases for DDL, activation epoch, generated columns, TOAST, TRUNCATE, filters and snapshot handoff.
4. Bind capability manifests and conformance receipts through MON-041 custody.
5. Add PostgreSQL/Kafka lab adapters and capture source/catalog truth independently.
6. Integrate MON-044 into MON-037 composition without altering lower-layer verdicts.
7. Add benchmark holdouts and independent SRE adjudication.
8. Publish measured matched results only after exact-head execution and independent replication.

## Explicit non-claims

This document does not implement a connector capability registry, live adapter, vendor attestation, source catalog capture, signed exception authority, test runner, or production qualification. Offline capability/coverage results are not evidence of runtime truth. Unsupported capabilities must not be hidden by successful reconciliation of the records that were emitted.
