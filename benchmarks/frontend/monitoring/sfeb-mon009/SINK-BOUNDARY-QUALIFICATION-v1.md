# MON-009 — Sink-Boundary Effect Qualification

**Status:** research protocol and reference transaction design; not yet implemented against PostgreSQL.  
**Parent:** MON-008 authenticated checkpoints, PR #13620.  
**Tracking:** issue #13638.  
**Version:** 1.0, 2026-11-25.

## 1. Research thesis

A consequential action should not be considered effect-qualified merely because an agent, harness, workflow engine, serving party, or transparency service recorded it.

The qualification target is a three-way causal reconciliation:

1. **Intent/admission:** what operation was requested, by which principal, under which authority revision and policy?
2. **Destination commit:** what exact state transition did the destination atomically accept, under which operation identity, scope and fencing token?
3. **Independent observation:** what resulting state did a separately authorized observer read from a declared observation path?

Transparency then anchors the evidence and its history. It does not convert an issuer's statement into physical or business truth.

The differentiating question becomes:

> Can a fresh auditor reconcile caller intent, destination commit and independent observation after the original operator, database credentials and orchestration history are unavailable—and identify exactly which parts of the action space were inside the observation boundary?

This is not a claim of global exactly-once execution. It is a sink-specific qualification protocol for conditional state transitions.

## 2. Relationship to MON-007 and MON-008

MON-007 already contains a semantic reference model for source identity/digest, tenant/resource scope, authority-at-commit, sink fencing, stable operation identity, sink receipt, separate-reader readback and causal ordering. Its Go model consumes adapter-produced verified facts; it is not a live sink.

MON-008 adds custom-profile authenticated checkpoints, trust-root rotation, monotonic state and witness continuity. Its custom Ed25519 envelope is explicitly not COSE and does not implement RFC 9942/9943.

MON-009 must not duplicate either layer. It adds the missing boundary adapter and integration evidence:

- a real PostgreSQL instance;
- destination-side atomic mutation and deduplication;
- a durable operation ledger;
- destination-issued evidence tied to committed state;
- a separately authorized readback;
- a bounded observation/coverage statement;
- an offline auditor that consumes only portable evidence.

The initial implementation should use PostgreSQL as a reversible test sink, not claim that a generic model or in-memory map proves destination enforcement.

## 3. Claims and verdicts

Keep these claims independent:

| Claim | Minimum evidence | What it does not establish |
|---|---|---|
| intent_bound | authenticated principal, operation ID, canonical request digest | authority to mutate |
| authority_valid | policy/capability digest, resource scope, authority revision, validity/freshness evidence | that the sink enforced it |
| sink_committed | destination operation ledger + destination-issued receipt | independent observation of the result |
| readback_verified | separate-principal observation of exact resource revision and digest | organizational independence if control planes are shared |
| transparency_anchored | SCITT Signed Statement/Receipt and applicable VDS proof | factual truth of the statement |
| checkpoint_current | authenticated checkpoint chain, monotonic state and witness comparison | absence of unobserved actions |
| coverage_bounded | boundary manifest, event watermark, independent reconciliation and bypass tests | universal completeness outside the manifest |
| offline_audited | fresh verifier, portable bundle and independent trust material | present-day state beyond the observation time |

Each claim is one of: **verified, missing, contradicted, stale or indeterminate**. Preserve the evidence pointer, verifier version, trust context, observation time and reason for every result. Do not collapse this into a single green boolean.

Freshness is orthogonal to signature validity: a correctly signed observation can be stale. Historical verification is also distinct from current-state verification.

## 4. Destination operation contract

### 4.1 Immutable operation identity

Every mutation is bound to:

- tenant ID;
- resource ID;
- stable operation ID;
- canonical request digest;
- authority revision or capability digest;
- fencing token / lease epoch;
- expected prior resource revision;
- operation profile version.

The operation ID is unique within the destination idempotency namespace, not merely per tenant. The same operation ID with the same immutable binding is a retry of the same operation. The same operation ID with any changed tenant, principal, resource, digest, authority or fence is a conflict and must not mutate state.

### 4.2 Atomic destination semantics

The destination must enforce, in one transaction or an equivalent linearizable primitive:

1. look up the operation ID under a unique constraint;
2. if an identical committed operation exists, return its original outcome without a second mutation;
3. if the ID exists with different scope, digest, authority or fence, reject it as an operation-identity conflict;
4. lock or conditionally update the target resource;
5. compare the supplied expected revision to the current revision;
6. compare authority revision to the destination's current accepted authority epoch and enforce the documented freshness/revocation rule;
7. reject a fence lower than the resource's accepted fencing high-water mark;
8. apply the resource mutation and update its revision/digest;
9. append the operation record and destination event sequence in the same transaction;
10. commit the state and evidence record together.

A unique database constraint alone is not enough if the side effect occurs outside the transaction. For an external downstream effect, the first sink must either own the effect or expose a transactional outbox whose receiver also performs stable-ID deduplication.

### 4.3 Fencing and authority freshness

The prototype must state how fencing tokens are issued, scoped and advanced. A token from an old lease must not become valid because the operator restored an old database snapshot.

Authority revocation is not instantaneous merely because a policy service changed state. The sink must document the authority epoch it has installed, its update channel, freshness bound and fail-closed behavior during loss of contact. Test the revocation propagation window as a measured safety property.

When the sink cannot establish a sufficiently fresh authority epoch, it must refuse before mutation or return an explicit indeterminate result. Do not infer current authority from a caller-supplied timestamp.

### 4.4 Unknown outcomes

A timeout or connection loss after a possible commit is not a failed action and not a safe retry with a new ID.

The only recovery sequence is:

1. retain the original operation ID and immutable request binding;
2. query/reconcile that same operation ID at the destination;
3. if a committed ledger entry exists, recover the original outcome and receipt;
4. if the destination can prove no commit occurred, retry the same operation ID;
5. if the outcome cannot be determined, remain indeterminate and prohibit a new mutation that could duplicate the effect.

The benchmark must inject process termination immediately before commit, immediately after commit, after durable commit but before acknowledgement, and after acknowledgement but before receipt delivery.

## 5. PostgreSQL reference design

This section specifies a design to implement and integration-test; it is not an assertion that a production SQL implementation already exists.

### 5.1 Logical tables

A minimal schema should have:

- resource_state: tenant, resource, current revision, canonical state digest, accepted authority epoch, fencing high-water mark and per-resource event sequence;
- operation_ledger: tenant, operation ID, resource, request digest, authority revision, fence, prior revision/digest, committed revision/digest, outcome and destination sequence;
- mutation_events: immutable per-resource sequence, operation ID, exact state transition digest and commit metadata;
- authority_state: the currently installed authority revision/status and freshness metadata for each relevant scope;
- boundary_manifest: enrolled resources, permitted mutation procedures, role/grant digest, observer identity, capture source, interval and policy version.

The operation ledger has a unique key on (tenant_id, operation_id). The mutation event sequence should be transactionally advanced with the resource row, not allocated from a non-transactional sequence if the benchmark expects gap-free committed per-resource ordering. Any gap-free property must be tested under rollback, concurrent retry and crash.

### 5.2 Privilege boundary

The ordinary application role must not have direct INSERT, UPDATE or DELETE privileges on protected state, operation-ledger or event tables. It may call only the constrained mutation procedure. Use a dedicated non-login function owner and narrowly granted execution rights; separate read-only observer credentials from mutation credentials.

The prototype must record who can alter functions, grants, triggers, extensions, replication settings and audit configuration. PostgreSQL superuser, host root, storage administrator, backup/restore operator and cloud control-plane bypasses are explicit privileged exclusions unless separate controls and external evidence cover them.

A procedure that is atomic for the application role is not proof against a database superuser. Do not overstate the boundary.

### 5.3 Receipt construction

The destination receipt is generated only from a committed operation-ledger record. It should bind at least:

- receipt profile and schema version;
- destination issuer identity and key identifier;
- operation ID and canonical request digest;
- tenant/resource identifiers or privacy-preserving scoped identifiers;
- authority revision and fence;
- prior and committed resource revisions/digests;
- destination event sequence;
- committed/rejected outcome;
- destination identity and relevant software/profile measurement;
- commit time plus monotonic sequence evidence.

The signer must not accept caller-supplied result fields as fact. A signing failure after commit leaves the mutation committed but evidence delivery pending; the receipt must be recoverable from the immutable committed ledger entry under the same operation ID.

The preferred interoperability path is a SCITT Signed Statement registered under RFC 9943 with an RFC 9942 COSE Receipt. This document does not define a competing COSE or transparency format. Registration proves what was registered under the TS policy; it does not prove that the destination's underlying claim was true.

## 6. Independent observation

### 6.1 Readback contract

A separate observer reads the exact resulting resource revision and digest after the destination commit. Its record binds:

- observer principal and credential/key identifier;
- authorization policy/revision used for the read;
- independently selected read endpoint/source;
- tenant/resource and operation ID;
- observed revision and state digest;
- destination sequence / replay position where available;
- observation freshness and timestamp;
- verifier and adapter versions.

The readback must match the committed revision/digest, not merely return HTTP success or a resource that happens to contain the same high-level value.

### 6.2 Independence is multidimensional

A different credential is necessary but not sufficient. Report the observer's independence across:

| Dimension | Question |
|---|---|
| identity | Is it a distinct principal with separately governed credentials? |
| privilege | Can the writer alter, impersonate or authorize the reader? |
| process | Is readback performed by an independent process/service? |
| administrator | Do the same humans or automation control both paths? |
| key custody | Are signing and observer keys independently governed? |
| storage | Is the observation source independent from the write path or explicitly shared? |
| network | Can the operator reroute or suppress the observation path? |
| operator | Is the observer institutionally independent where the assurance tier claims it? |

If both paths share a PostgreSQL instance, owner role, administrator or host, disclose that common cause. The first prototype may qualify privilege separation and exact-state readback while explicitly not qualifying organizational or infrastructure independence.

### 6.3 Staleness

A read replica is not authoritative merely because it is read-only. Bind readback to a replay position, LSN or resource sequence and prove that it has observed at least the commit being checked. If the read source cannot establish that lower bound, return stale or indeterminate, never verified.

## 7. Bounded observation completeness

### 7.1 The boundary manifest

Completeness is a scoped claim, not a universal property. The manifest must declare:

- enrolled tenants/resources and the covered time interval;
- every permitted mutation path and API/procedure version;
- database role/grant and schema/function digest;
- expected event-sequence semantics and observed high-water mark;
- observer identity, endpoint and polling/reconciliation policy;
- privileged actors and bypass routes excluded from scope;
- monitoring gaps, outages and unresolved intervals;
- the version of the capture and enforcement configuration.

The declared boundary should initially be narrow: mutations to enrolled resources through the constrained PostgreSQL procedure. Direct table owners, superusers, infrastructure administrators, restore processes and any unmediated API are not silently assumed covered.

### 7.2 Sequence and watermark reconciliation

Use transactionally committed per-resource event sequence numbers to detect missing interior events. A third-party observer independently records the highest sequence it has reconciled and anchors periodic watermarks outside the operator's control.

A contiguous event sequence without an independently retained watermark cannot detect deletion of the entire tail. A watermark without an independently observed source cannot establish that the source reported every mutation. Test both limits.

For a claimed interval [s, w], the auditor checks:
- the manifest identifies the exact enrolled scope and interval;
- every committed event in the independently anchored range is present and ordered;
- every caller/serving-party action expected at the sink has a matching sink outcome or explicit refusal;
- every sink commit has a matching receipt or an explicit evidence-delivery gap;
- every qualified commit has a separate readback;
- all configuration or privilege changes that alter the boundary are represented or cause the coverage claim to become indeterminate.

### 7.3 Third-party measurement

Evaluate an adapter to the active SCITT Declared-versus-Observed Measurement Capsule draft. Use an independent measuring party to record the subject's declared action/coverage and the observer's measured state, including the "could not be checked" case.

Do not clone its capsule schema or imply that its draft has been adopted as an RFC. The SFEB contribution is the sink-specific transaction/readback evidence and adversarial qualification of the declared boundary.

### 7.4 Coverage verdicts

- coverage_verified: only the declared boundary and interval have complete, independently reconciled evidence under the stated trust assumptions;
- coverage_missing: required evidence or watermark is absent;
- coverage_contradicted: observed state or sequence conflicts with the declaration;
- coverage_stale: the required watermark or observation exceeds its freshness bound;
- coverage_indeterminate: source, control plane or observer failure prevents a conclusion.

Absence of a receipt is not proof that an action did not happen. A path outside the declared boundary must remain an explicit residual risk.

## 8. Portable audit bundle

The bundle should contain the exact bytes needed to reproduce each supported claim:

1. intent and caller/serving-party evidence;
2. identity, authority revision, policy digest and historical trust collateral;
3. canonical request binding;
4. destination-issued operation receipt and committed ledger evidence;
5. independent readback and observer trust material;
6. boundary manifest, sequence range, external watermark and reconciliation report;
7. SCITT Signed Statement, RFC 9942 Receipt, inclusion/consistency proofs and authenticated checkpoint/witness collateral where available;
8. verifier binaries/source versions, profile identifiers and machine-readable verdicts.

The auditor is run in a fresh environment with the original orchestrator database, credentials and network services unavailable. Trust roots are supplied independently, not learned from the bundle being evaluated. The auditor must reject substituted trust roots, non-canonical encodings, wrong issuer/subject binding, stale authority, mismatched operation digests, stale readback and unanchored coverage ranges.

An audit of an historical readback proves what the observer recorded at that observation time. It does not prove the resource remains unchanged today.

## 9. Adversarial test matrix

| Attack | Expected sink behavior | Expected audit behavior |
|---|---|---|
| same operation ID, same binding | return original outcome; no second mutation | verify one committed effect |
| same ID, changed digest/scope/authority/fence | reject identity conflict | contradicted / reject |
| cross-tenant or cross-resource substitution | reject before mutation | reject |
| stale authority revision | reject or indeterminate under freshness policy | not effect-verified |
| revocation races with commit | enforce documented sink epoch ordering | record exact race boundary |
| stale/lower fencing token | reject | contradicted / reject |
| same fence with concurrent valid operations | serialize with explicit revision precondition | verify causal order |
| lost acknowledgement after commit | reconcile original operation ID | recover one original effect |
| crash before commit | no committed effect | missing/indeterminate until reconciled |
| crash after commit, before receipt | committed ledger remains; receipt recoverable | evidence delivery pending, not failed |
| receipt signed for fabricated result | no sink source record | reject effect claim |
| stale or colluding reader | no qualified readback | stale/indeterminate |
| out-of-band SQL mutation | denied for application role | detect or mark privileged bypass outside scope |
| event deletion / interior sequence gap | immutable/privilege control prevents or exposes | coverage contradicted/missing |
| tail deletion without external watermark | not necessarily detectable locally | coverage indeterminate |
| operator DB snapshot rollback | sink fence/independent watermark prevents silent acceptance | detect rollback or fail closed |
| observer outage | mutation may proceed only under declared policy | coverage/readback indeterminate |
| transparency registration delayed | effect result remains distinct from anchor status | anchor_pending, not anchored |
| key/root/policy rollback | no trust-state rollback | reject historical verification |

Every attack family must include a negative control that demonstrates the test fails when its intended control is removed.

## 10. Assurance Gap Challenge extension

Run identical workload semantics, destination, payloads, durability, concurrency, retries and deterministic fault schedules across:

- native PostgreSQL conditional mutation;
- PostgreSQL transactional outbox + idempotent receiver;
- DBOS durable transaction/step with the documented external-effect pattern;
- Temporal Activity with stable destination idempotency key;
- SCITT RFC 9942/9943 transparency path;
- Rekor with available independent monitor/witness path;
- applicable AEE/WIMSE/Vaara profiles;
- a third-party measurement-capsule adapter;
- SFEB overlays on the applicable systems.

Only compare claims a given arm implements. Record adapter-derived functionality separately from native guarantees.

| Tier | Capability | Primary output |
|---|---|---|
| W0 | no-op durable workflow | orchestration ceiling only |
| W1 | durable database mutation/outbox | DB durability and dispatch |
| W2 | destination-enforced conditional mutation and dedupe | sink-accepted effects/s |
| W3 | caller/serving-party linkage to exact request | two-sided linked actions/s |
| W4 | destination receipt + separate-principal readback | readback-verified effects/s |
| W5 | SCITT/RFC 9942 receipt, historical policy, authenticated checkpoint and independent witnesses | externally anchored effects/s |
| W6 | original operator DB/credentials unavailable or malicious; portable audit | offline-audited effects/s |

Keep W0 separate from W2-W6. Do not compare no-op workflow throughput to readback-verified or offline-audited external effects.

### Required metrics

Performance:
- sink-accepted, readback-verified, externally anchored and offline-audited effects/s;
- p50/p95/p99 per tier and stage;
- reconciliation time and age of unresolved outcomes;
- event-to-watermark and action-to-anchor delay;
- CPU-seconds, memory, storage, IOPS, network bytes and cost per qualified effect;
- receipt, proof and bundle sizes.

Assurance:
- duplicate committed mutations;
- unauthorized, stale-authority, stale-fence and cross-tenant accepts;
- caller/sink/readback mismatches;
- false accepts and false rejects;
- missing receipts and indeterminate outcomes;
- readback staleness/collusion detection;
- omission-detection precision/recall within the declared boundary;
- fraction of W6 claims audited with original operator services unavailable;
- time to detect watermark gaps and checkpoint equivocation.

Publish raw attempt ledgers, exact versions, hardware, database durability/isolation settings, concurrency, payload distributions, fault seeds, analysis scripts and confidence intervals. Use at least five independent performance runs per point. A million negative attempts is a release-candidate campaign target, not proof of zero risk.

### Proposed gates (targets, not results)

- zero duplicate destination mutations under repeated same-ID delivery;
- zero accepted stale-fence, cross-tenant or unauthorized effects in at least 1,000,000 seeded negative attempts per release candidate;
- zero effect_verified outputs without a destination receipt and separately authorized exact-revision readback;
- 100% of W6-qualified bundles independently audited offline;
- all coverage claims restricted to explicit, versioned boundaries and demonstrated negative controls;
- no unknown outcome retried under a fresh operation ID;
- two independent verifier implementations agree on the supported receipt/proof corpus before interoperability is claimed.

No performance target should be set until the real PostgreSQL baseline has been measured. After the baseline is reproducible, set a separate overhead budget for W2, W4, W5 and W6 rather than one blended target.

## 11. Execution sequence and stop/go gates

### Phase 0 — freeze semantics
- Freeze operation ID, digest, authority, fence, expected revision and unknown-outcome rules.
- Freeze the observer independence matrix and declared boundary manifest.
- No production claim.

### Phase 1 — PostgreSQL sink
- Implement schema, mutation procedure, restricted roles and transactionally committed operation/event ledger.
- Use real concurrent transactions and fault injection.
- Demonstrate same-ID replay, changed-binding conflict, stale-fence rejection and lost-ACK reconciliation.
- Run integration tests against a real PostgreSQL service; do not substitute the semantic model.

### Phase 2 — receipt and readback
- Derive sink statements from committed ledger rows.
- Sign using destination-governed key custody.
- Read back exact state through a separate identity and disclose common control planes.
- Test stale replica and reader-compromise controls.

### Phase 3 — coverage and measurement
- Publish the boundary manifest and independently anchored event watermarks.
- Inject bypasses, privileged changes, tail truncation and observer outage.
- Add a version-pinned Measurement Capsule adapter if its declared/observed model maps without semantic loss.

### Phase 4 — offline auditor
- Build the portable bundle and fresh-process verifier.
- Remove original operator DB and credentials during audit.
- Verify historical policy, trust roots, sink receipt, readback and coverage verdicts independently.

### Phase 5 — matched benchmark
- Run native, outbox, DBOS, Temporal, transparency-only and SFEB-overlay arms with matched semantics.
- Publish all failures, unknowns, costs and residual coverage limits.

A skipped or unavailable CI job is never a pass. Local tests, PostgreSQL integration, GitHub CI, independent review and production qualification are distinct evidence classes.

## 12. Related standards and research

Published foundations:
- [RFC 9942 — COSE Receipts](https://www.rfc-editor.org/rfc/rfc9942.html)
- [RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html)
- [RFC 9162 — Certificate Transparency Version 2.0](https://www.rfc-editor.org/rfc/rfc9162.html)
- [RFC 9052 — CBOR Object Signing and Encryption](https://www.rfc-editor.org/rfc/rfc9052.html)

Version-pinned work in progress:
- [Agent Execution Evidence -02](https://datatracker.ietf.org/doc/html/draft-thaha-scitt-agent-execution-evidence-02)
- [WIMSE Evidence Binding -03](https://datatracker.ietf.org/doc/html/draft-thaha-wimse-agent-evidence-binding-03)
- [Vaara Receipt -15](https://datatracker.ietf.org/doc/html/draft-sirkkavaara-vaara-receipt-15)
- [SCITT AI Agent Execution -01](https://datatracker.ietf.org/doc/html/draft-emirdag-scitt-ai-agent-execution-01)
- [Declared-versus-Observed Measurement Capsules -01](https://datatracker.ietf.org/doc/html/draft-templeman-scitt-measurement-capsule-01)
- [Physical-Site Engagement Receipts -06](https://datatracker.ietf.org/doc/html/draft-wilder-scitt-physical-site-engage-receipt-06)

Use these as interoperability and benchmark inputs. Do not claim their draft profiles are published RFCs or that SFEB implements them until independently tested.

## 13. Open questions to resolve during the prototype

1. Can PostgreSQL's transaction boundary own the actual external effect, or is a downstream receiver required?
2. Which authority epoch source is sufficiently fresh at the exact sink commit boundary?
3. Can the observer obtain an authoritative commit position without sharing the writer's administrative control plane?
4. Which privileged bypasses are technically closed, independently monitored, or explicitly excluded?
5. Should transparency anchoring be synchronous or asynchronous for each risk tier? If asynchronous, expose anchor_pending.
6. Which measurement-capsule fields and correction semantics map cleanly to the sink/readback evidence without importing unsupported authority claims?

These questions are part of the experiment; they are not permission to overclaim completeness.
