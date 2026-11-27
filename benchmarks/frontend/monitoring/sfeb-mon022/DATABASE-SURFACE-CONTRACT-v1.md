# MON-022 — Database Surface Coverage Assurance Contract v1

**Status:** research contract and semantic Go reference; live PostgreSQL adapters and independent qualification pending.  
**Relationship:** extends MON-021 row-stream continuity by qualifying database state surfaces that are not necessarily represented by logical row changes.

## 1. Research question

Can an independent verifier state exactly which database surfaces a backup/CDC/replication process covers, how each surface is evidenced, and what remains out of scope—without promoting row-stream success into a false claim of complete database recoverability?

This is not a trust score. It is an inventory with typed evidence for each surface.

## 2. Two different claims; never collapse them

### 2.1 Scoped row-stream continuity

This claim is deliberately narrow. It may cover rows and transactions emitted within a named publication, table/column/row filter, plugin version, source epoch and LSN interval. It does not claim DDL, sequences, generated values, large objects or external table contents unless separate evidence is attached.

A row-stream claim can qualify while explicitly marking other surfaces excluded, because the claim name and output must say it is a row-stream claim. It may not be advertised as the database being fully replicated.

### 2.2 Full recoverable database state

This claim requires every required database surface in the declared inventory to be represented by one of:
- **Native replication:** a pinned native replication mechanism covers the exact source/target surface and profile.
- **Independent comparison:** an observer under a separate administrative control root reads source and target state and compares canonical state digests.
- **Recomputation and comparison:** a deterministic, version-pinned computation recomputes a derived surface and compares it with target state, with dependency provenance and function/version identity.

Excluded by contract, Unknown, missing or stale coverage cannot satisfy a full-state claim. Required Not applicable is permitted only when a separately qualified applicability proof exists.

## 3. Surface inventory ontology

Every surface must have a stable surface ID, class, source scope digest and claim applicability. The default inventory is:

| Surface class | Why row CDC alone is insufficient | Candidate evidence |
|---|---|---|
| Table row state | Publication/table/row/column filters may omit data | WAL/CDC range + independent table snapshot |
| Schema and DDL | Logical decoding does not provide general DDL event replication | Catalog/schema fingerprint + migration attestation |
| Sequence state | nextval is not a normal row event; WAL/page LSN can lag cached values | last_value, is_called, definition, cache and sequence synchronization evidence |
| Generated columns | pgoutput may omit generated-column values | Pinned generation expression + independent readback or deterministic recomputation |
| Large objects | OID metadata and LO byte content are separate from table rows | OID inventory + chunk/content digest + target verification |
| Materialized views | Rows may require refresh/rebuild rather than logical replay | Definition digest + refresh epoch + source/target contents digest |
| Partition topology | Attach/detach/routing changes affect data placement and boundaries | Catalog topology fingerprint + migration event |
| Publication/filter scope | Row filters and column lists can silently narrow the stream | Publication membership + actual filter/profile digest |
| Foreign tables/external data | External source state is outside the local WAL history | Upstream source identity and its own MON capture profile |
| Unlogged tables | Their contents are not WAL-replicated like logged table rows | Explicit contract exclusion or independent snapshot/restore |
| Extension-owned state | Extension tables/files/shared state may have separate lifecycle | Extension version, owned object inventory and adapter-specific evidence |
| Constraints, indexes, triggers and routines | Catalog semantics and execution behavior are not ordinary row changes | Catalog fingerprint + migration/deployment attestations |
| Roles, grants and security policies | Access-control state is part of recoverability | Roles/ACL/policy fingerprint with secret-safe representation |
| Large configuration and tablespace state | Database-level/runtime state can sit outside logical changes | System configuration/tablespace manifest and restore-test evidence |

The inventory is intentionally extensible. New surface classes require schema versioning and known-answer tests. An omitted class is not implicitly covered.

## 4. Coverage modes and evidence semantics

Each surface attestation selects exactly one mode:

- native_replication: supported by the pinned native replication method and exact profile.
- independent_comparison: source and target state are compared by a separate observer/control domain.
- recomputed_comparison: a deterministic, version-pinned computation is compared with target state.
- excluded_by_contract: explicitly outside a narrow claim; requires a non-empty reason and is forbidden as a way to qualify full database state.
- not_applicable: a separate applicability proof shows the surface does not apply to this source/claim.
- unknown: no trustworthy evidence; always prevents a complete claim for a required surface.

Every evidence record binds:
- the exact inventory digest and surface evidence subject ID under the pinned mon022-surface-coverage-attestation-v1 profile;
- stable surface ID and class;
- source and target cluster/system identities;
- source and target administrative control roots;
- exact inventory/scope digest;
- source and target snapshot/epoch/timeline references;
- source and target canonical state digests;
- method ID/version/profile digest;
- evidence digest and verifier identity/build/trust-store reference;
- observation time, freshness deadline and completeness result;
- exclusion reason or applicability proof as required.

A source and target digest mismatch is divergence. A missing/ambiguous/stale/unverified attestation is indeterminate. An excluded required surface prevents full-state qualification. An independent comparison is only independent if its observer identity and observer control-root membership are separately verified, its observer ID is not an endpoint alias, and its control root differs from both source and target roots.

## 5. Scope and decision rules

The request carries the complete inventory and the claim scope. The evaluator must:
1. reject duplicate surface IDs or conflicting evidence records;
2. require evidence for every surface required by the selected claim;
3. check exact inventory digest, source/target identity and per-surface scope digest;
4. check freshness, completeness and non-empty evidence/method digests;
5. check source/target state digests agree for replicated/recomputed/compared modes;
6. check independent observer identity and control-root membership are separately evidenced and separate from both endpoints;
7. require explicit reason for excluded surfaces;
8. reject exclusions for required full-database-state surfaces;
9. report excluded surfaces in a scoped row-stream result.

A row-stream claim is qualified only for its explicitly listed row-stream-required surfaces. It reports every other surface as excluded/not covered, and is not a full-state claim. The output separately counts covered surfaces, not-applicable surfaces, and excluded surfaces; not-applicable is never presented as an observed/copy-verified surface.

A full recoverable-database-state claim is qualified only when every full-state-required surface has valid coverage evidence or a valid not-applicable proof. No scalar confidence score can compensate for a missing surface.

## 6. High-risk surfaces

### 6.1 Sequences

For PostgreSQL 19, sequence synchronization is available, but it is not equivalent to a continuously emitted nextval event stream. Sequence cache means a page LSN may not advance with every value handed to a client. Compare at least:
- last_value and is_called;
- start, increment, min, max, cache, cycle;
- owner, schema/name and ACL/ownership;
- source/target sequence snapshot epoch and database identity;
- subscription/publication sequence refresh state where applicable.

Do not treat page_lsn equality alone as exact sequence-value equality. last_value may be ahead of the last value handed out because of caching. Capture and compare sequence state at a controlled boundary; concurrent nextval requires a defined quiescence or serialization protocol.

### 6.2 Generated columns

For stored generated columns, qualify the exact expression and dependency schema version, and compare target readback. Recompute only if the expression is deterministic under a declared function/extension version. Volatile functions, time, collation differences, extension upgrades or external dependencies make unqualified recomputation indeterminate.

### 6.3 DDL, catalogs and executable semantics

Logical replication does not mean schema replication. DDL/migration changes need their own ordered attestation. Catalog fingerprints should include object identity and normalized definitions, not OIDs alone when restore or migration can change them. Include constraints, indexes, triggers, policies and routines where they affect the claimed behavior.

### 6.4 Large objects and external state

Large-object OIDs are not the content. Inventory OIDs, metadata, byte length and canonical chunk/content digests. External tables and extension-owned state require separately identified source/collector profiles. An unlogged table may need an independent snapshot/restore path or explicit exclusion.

## 7. Decision model

| Result | Meaning |
|---|---|
| qualified | Every required surface has valid, fresh, complete evidence within the claim scope |
| denied | A mandatory scope/mode predicate is proven false (for example excluded surface in full-state claim) |
| indeterminate | Required evidence is missing, stale, unverified, not comparable or incomplete |
| divergent | Admissible evidence for the same surface conflicts or source/target state digests differ |

No aggregate confidence score can compensate for a missing required surface.

## 8. Threat model

Assume a source, target, CDC connector, migration runner, extension, observer or operator may be faulty or malicious. Adversarial cases include:
- publication row filter silently narrows scope;
- newly added table or partition omitted from inventory;
- sequence cache advances without page LSN change;
- last_value equal but is_called differs;
- generated column expression changes while materialized result appears unchanged on one fixture;
- generated column absent from pgoutput event;
- DDL applied only to one side;
- materialized view stale after base-table changes;
- large object OID replicated but bytes missing/altered;
- unlogged table present on primary but absent after failover;
- extension-owned state omitted;
- two named observers share a cloud admin/key custodian;
- evidence is stale or attached to wrong database epoch;
- an exclusion is misrepresented as full-state coverage;
- source/target canonicalization algorithm/profile differs.

## 9. Industry comparison and fair differentiation

| Existing product/standard | What it does well | MON-022's additional assurance question |
|---|---|---|
| PostgreSQL logical replication | Ships row changes and, in PostgreSQL 19, supports sequence synchronization | Which non-row database surfaces are still outside the claimed recovery state? |
| Debezium | Snapshots and streams supported row changes with offsets and connector-specific semantics | What exact database behavior and state did its stream not represent? |
| Kafka Connect source EOS | Couples source records and source offsets in Kafka transactions when the connector supports correct resume semantics | Does source EOS cover sequences, DDL, LOs or other state surfaces? |
| pg_dump / physical backup | Captures broad database/cluster state under defined modes | Which items were included, consistent, restored and independently compared? |
| Schema migration tooling | Orders and applies DDL changes | Did source and target end at the same schema and executable behavior epoch? |
| MON-021 | Qualifies bounded PostgreSQL logical-decoding continuity and transaction/replay/failover | MON-022 composes row-stream evidence with explicit state-surface coverage |

Fair comparisons must use matched source/target versions, schema/object count, extension versions, publication filters, sequence cache sizes, concurrent writers, snapshot/quiescence protocol, backup mode and restore/failover procedure. Raw CDC rows/s is not a database-state coverage metric.

## 10. Benchmark programme

### Coverage correctness

- 100% expected classifications on the frozen finite known-answer corpus, including positive controls.
- Zero false full-state-qualified outcomes when any required surface is missing, unknown, stale or excluded.
- Zero false independent-comparison results when the observer shares a control root with either endpoint.
- Zero sequence-equality claims based only on page LSN.
- Zero generated-column coverage claims without verified expression/version and output readback/recomputation.
- 500+ independently authored blind surface-omission and substitution holdouts.

### Scaling

Test inventory sizes of 12, 100, 1,000, 10,000, 100,000 and 1,000,000 surface objects, with class distributions and sparse/large evidence records. Measure p50/p95/p99 evaluation time, memory/allocations, evidence bytes/object, independent observer cost, false-inclusion rate, missing-surface detection recall and operator effort.

### Prospective targets

- 100% detection of every missing required surface in the finite corpus.
- Zero full-state false qualifications in 500+ independently blinded holdouts.
- p95 under 250 ms for 100,000 pre-normalized surface attestations on a pinned reference host after optimization.
- Incremental invalidation of only affected surfaces when schema/extension/publication profile changes, while preserving correct dependency invalidation.
- Maximize independently qualified recoverable state per total assurance cost, not raw row throughput.

Targets are not results. They must be measured under exact, reproducible environments and compared to real backup/restore and CDC systems.

## 11. Implementation sequence

1. Freeze the database-surface ontology and explicit row-stream/full-state claim distinction.
2. Implement the normalized Go semantic coverage evaluator and finite known-answer tests.
3. Add PostgreSQL 18/19 adapters for catalogs, publications/filters, sequences and generated-column metadata.
4. Add large-object inventory and digest streaming.
5. Add materialized view, partition, extension, unlogged, role/ACL, trigger/routine and tablespace surface adapters.
6. Add restore/failover comparison harness and independent control-root witness.
7. Join MON-021 row-stream evidence with MON-022 non-row surface coverage.
8. Run blind omissions, matched backups/restore tests and total assurance-cost benchmarks.

## 12. Claim ceiling

The Go package consumes normalized surface inventory and evidence. It does not query PostgreSQL or compute canonical digests itself. No live source/target snapshot, sequence reconciliation, large-object verification, DDL attestation, independent observer, restore test or production coverage claim is made. A row-stream result never means full database state is recoverable.


## 13. Semantic reference execution status

The Go reference in surface-coverage/ consumes normalized attestations and enforces the two claim scopes, explicit row-stream exclusions, full-state required-surface completeness, inventory/scope/evidence subject binding, freshness, state-digest equality and separate observer identity/control-root evidence. It does not execute catalog queries, canonicalization, signature verification, observer deployment or restore operations.

The code includes 29+ named adversarial tests, one fuzz target and scaling benchmarks. No exact-head Go test, race test, vet, fuzz run or numeric benchmark is claimed for the current evolving branch. A fresh GitHub clone could not be obtained in this environment. The 40 scenario rows are descriptive seed cases, not a frozen independently authored executable oracle corpus.
