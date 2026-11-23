# SFEB-MON-003 — durable objective lifecycle and fenced execution

**Date:** 2026-11-23  
**Parent:** [MON-003 issue #13398](https://github.com/Luminous-Dynamics/symthaea/issues/13398)  
**Related:** [MON-001 issue #13392](https://github.com/Luminous-Dynamics/symthaea/issues/13392), [MON-002 issue #13394](https://github.com/Luminous-Dynamics/symthaea/issues/13394)  
**Status:** specification committed; 9-case fencing/idempotency kernel, 3-case objective-ledger kernel, 8-case tenant-boundary suite, 720-order interleaving model check and 15-case cursor/replay suite committed and executed under Node.js v22.16.0; production qualification pending

## 1. Research thesis

SFEB should stop treating the agent process as the owner of a long-running objective.

An objective is durable work. The model/worker is an ephemeral interpreter that can be replaced at any point. The objective, watch contract, source cursor, event evidence, authorization decision, pending operation and terminal outcome must survive independently of that worker.

The next paradigm is therefore:

**Intent → durable objective → scheduler subscription → event evidence → current authorization → fenced/idempotent execution → independent postcondition proof.**

A notification only causes a wake. An event receipt proves a source transition. An authorization grant allows a specific actor to perform a specific action under current policy. An execution receipt records what the sink committed. These are four different claims and must have different issuers and verification rules.

## 2. Four artifact classes — never conflate them

| Artifact | What it establishes | What it must not establish |
|---|---|---|
| Objective/watch contract | What was requested, trigger semantics, scope and terminal conditions | That an event occurred or an action is authorized |
| Source-event receipt | A source-of-record event satisfies a frozen edge predicate | Current permission or successful execution |
| Authorization grant | A named actor may perform one action against one target under a current policy revision and validity window | That the event is true or that the action succeeded |
| Execution receipt | The sink accepted/rejected a request and records its operation/idempotency identity | That the requested postcondition actually holds unless independently read back |

The source receipt and authorization grant are not interchangeable. A valid signature proves only the claims covered by the signature under the verifier's trust profile; it does not prove that the content is true, that the issuer has the relevant authority, or that current permission remains valid.

### 2.1 Objective/watch contract

Minimum fields:

- `objective_id`, monotonic `objective_version`, tenant, creator and subject scope;
- trigger mode: `level` or `edge`, explicit predicate AST/version/digest, and baseline semantics;
- source-of-record identity, source schema/version, partition and starting cursor;
- freshness, ordering, gap and rollback policy;
- deadline, expiry, cancellation and time-zone interpretation;
- action template, target binding, idempotency derivation and required authorization profile;
- evidence requirements and legal terminal states.

Changes to the predicate, target, tenant, permission profile or action template create a new objective version. They must not silently mutate the meaning of an already-issued receipt or pending operation.

### 2.2 Source-event receipt

For edge-triggered objectives, the source receipt must bind at least:

- CloudEvents-style `source` and `id` identity;
- source revision/partition sequence and the prior/current state needed to re-derive the edge;
- objective version and predicate digest;
- resource/action scope and event type;
- event occurrence time, source recording time, receipt issue time, expiry and revocation state;
- canonical payload digest and provenance/issuer profile.

For a capacity example, the verifier derives the edge from source history (e.g. below threshold → at/above threshold while provider-ready). A worker-supplied `qualified=true` flag is not evidence. Deduplication must use source identity plus event identity, not just a potentially colliding event ID from an arbitrary publisher.

### 2.3 Current authorization grant

A separate authority/policy plane issues a narrow, short-lived grant binding:

- actor/worker identity and current fencing epoch;
- objective ID and version;
- source receipt identity;
- exact action digest and target/resource;
- tenant and permission scope;
- current permission/policy revision;
- not-before/expiry and a unique nonce.

The action sink re-checks the policy revision at commit time. A grant issued before a revocation must not remain usable after the revocation becomes effective. A signature cannot replace this live check.

### 2.4 Execution receipt

The sink receipt binds:

- stable idempotency key and operation ID;
- objective/version, qualified event identity and action/request digest;
- worker identity and fencing epoch;
- authorization grant ID and policy revision;
- sink transaction/commit identity and response status;
- postcondition read-back identity, timestamp and evidence digest.

A successful response is only the sink's acknowledgement. The independent verifier should fetch the authoritative postcondition (or a signed source-of-record receipt) rather than trust the worker's `done` claim.

## 3. Worker lease fencing

### 3.1 The stale-worker race

1. Worker A acquires objective O at fencing epoch 41.
2. A pauses or loses connectivity; its lease expires.
3. Worker B acquires O at epoch 42 and resumes reconciliation.
4. A wakes late and tries to execute using epoch 41.

Checking only lease expiry in A's process is insufficient. A may have stale time, a paused runtime or a partitioned connection. **The mutation sink must reject epoch 41 after epoch 42 has been issued.**

### 3.2 Required fencing invariant

For each objective, the lease service maintains a durable monotonic epoch. Acquiring/reassigning a lease increments that epoch atomically. The action sink stores the greatest accepted epoch and compares it in the same transaction as authorization and mutation.

`request.epoch < sink.max_epoch` ⇒ reject without side effect.

A lease TTL is for liveness (recovering stalled ownership); the sink-enforced fencing epoch is for safety. Do not assume a lease service that the sink never consults can fence an already-running worker.

### 3.3 Idempotency and ambiguous outcomes

Derive one stable key from:

`objective_id || objective_version || source || event_id || action_digest`

Retries of the same intended operation reuse the same key. The sink stores the request digest beside the key:

- same key + same digest: return the original result; do not create another side effect;
- same key + different digest: hard conflict; no side effect;
- unknown outcome: query/reconcile the same key and operation ID;
- no authoritative answer: remain `outcome_unknown` / `inconclusive`; do not mint a replacement key.

Do not claim universal exactly-once delivery. A single effect per key requires a sink that durably deduplicates and atomically commits the mutation with its idempotency result. When the external system cannot provide that contract, SFEB must expose the uncertainty rather than inventing a guarantee.

## 4. Durable objective state machine

The objective state belongs to the durable coordinator, not the worker:

`registered → waiting → wake_pending → reconciling → eligible → executing → succeeded`

Alternative states:

`not_triggered | expired | revoked | not_authorized | outcome_unknown | inconclusive | quarantined`

Recovery rules:

- A worker crash changes the lease owner, not the objective's semantic state.
- The next worker reads the persisted objective version, fencing epoch, committed source cursor, outstanding receipt IDs and pending idempotency key before acting.
- A cursor gap, duplicate identity with different content, revision rollback or source identity change forces a full authoritative reconciliation or a terminal inconclusive result.
- A wake that arrives before source state is available is retained as a hint; it is not promoted to a source event.
- A cancellation or permission revocation is checked at the sink-side commit boundary, not only when the worker first became eligible.
- State transitions use compare-and-swap/version checks. A stale writer cannot overwrite a newer objective state.

## 5. Transactional event delivery

For a source service that owns the state transition, publish the source update and outbox event atomically. Delivery can be duplicated and consumers must be idempotent. A broker acknowledgement is not a substitute for source cursor reconciliation.

Recommended event envelope: CloudEvents 1.0.2 with immutable source/event identity. The outbox pattern addresses the dual-write failure between database state and event publication. The event service should support cursor replay and return an explicit high-watermark/continuation token; a consumer should not infer that a missing notification means no event occurred.

A separate monotonic source cursor is required per source partition. Do not infer global ordering across independent partitions. Define how the controller handles out-of-order events, late arrivals, tombstones and compaction.

## 6. Trust and cryptographic profile

Use a documented signing profile rather than inventing an opaque receipt hash:

1. Canonicalize the signed structured payload (RFC 8785 JSON Canonicalization Scheme or another pinned canonical form).
2. Compute a digest over the exact representation (RFC 9530 Digest Fields may be used for HTTP representations).
3. Sign the explicitly enumerated message components / payload commitment (RFC 9421 HTTP Message Signatures or a separately specified envelope-signature scheme).
4. Resolve the signing key under a versioned trust profile, including issuer authority, tenant scope, key validity and revocation status.
5. Independently check the current permission/policy revision at execution time.

A digest without an authenticated trusted channel can be recomputed by an attacker. A valid signature from an unauthorized issuer is not an authorization grant. A valid authorization grant for a different objective version, target, tenant or action digest must fail.

For HTTP action endpoints, use conditional mutation semantics (e.g. If-Match / strong ETag) where supported, or a storage-level compare-and-swap that makes fencing, policy revision and idempotency part of one atomic operation.

## 7. Failure matrix

| Failure / adversary | Required result | Evidence to retain |
|---|---|---|
| Old worker resumes after new lease | Sink rejects stale epoch | Lease epoch, rejected request digest, no-mutation receipt |
| Notification duplicate/replay | Wake dedup; reconcile from cursor | Event ID, delivery attempt, cursor before/after |
| Notification lost | Cursor replay finds eligible source event | Replayed range and source receipts |
| Source sequence gap/rollback | Stop mutation; reconcile or inconclusive | Gap/rollback proof and last trusted cursor |
| Receipt bytes altered | Digest/signature verification fails | Raw bytes, expected/observed digest |
| Signed receipt from wrong/revoked issuer | Reject authority profile | Key ID, trust profile version, revocation evidence |
| Receipt expired/revoked | Reject at commit boundary | Source receipt state and clock source |
| Permission revoked after preflight | Sink rejects stale policy revision | Grant revision vs current revision |
| Same key, different request | Hard idempotency conflict | Original and attempted request digests |
| Sink commits; response lost | Outcome unknown; query same key | Sink operation/idempotency lookup |
| Process crashes after commit | Recover by sink receipt; never retry with new key | Operation record and postcondition read-back |
| Cancellation races with commit | Defined linearization point; no post-cancel mutation | Cancellation revision and sink commit order |
| Outbox transaction rolls back | No event for uncommitted state | Source DB transaction / outbox record |
| Cross-tenant target substitution | Reject before mutation | Tenant, target and action digest binding |

All adversarial actions use a non-destructive mock sink. The mock must validate fencing, authorization revision, idempotency and target scope atomically. Agent-side prechecks alone do not qualify the architecture.

## 8. Benchmark design

### 8.1 Nine matched arms

1. LLM sleep/reload/inspect.
2. SentinelBench-style one-second `wait_for`.
3. Fixed-interval browser polling.
4. Untrusted notification + latest source-of-record reconciliation.
5. Durable edge receipt + current authorization + idempotent sink.
6. Temporal native durable-workflow baseline (Workflow History + Signal/Update + idempotent Activity), with the same event source and action sink.
7. Temporal + SFEB assurance layer (source evidence, current authorization, sink fencing, tenant scope and independent postcondition verifier).
8. DBOS native PostgreSQL durable-execution baseline, with the same event source and external action sink.
9. DBOS + SFEB assurance layer with the same evidence/authority/fencing/verifier contract.

The native and augmented arms must be paired on the same backend. This separates the value of the orchestration engine from the incremental assurance layer and avoids hiding a backend effect inside one combined arm.

Temporal is a meaningful orchestration baseline, not a strawman: it durably records workflow event history and replays it after worker replacement; its messaging model distinguishes asynchronous Signals from validated, tracked Updates; and its documentation warns that Activities can run more than once and downstream idempotency must be enforced by the called service. DBOS is a distinct database-native baseline: its documentation describes Postgres-backed workflow checkpoints and states that database writes can be made exactly-once when the mutation and checkpoint share one transaction, while external API effects still require idempotency keys. Give both baselines the same source feed, policy semantics and action sink. SFEB's hypothesis is not that it replaces durable workflow orchestration, but that event provenance, tenant scope, current authorization, sink-enforced fencing and independently verified postconditions deserve explicit semantics above the orchestration layer. Compare both backends with and without the SFEB assurance layer where feasible.

Hold model, prompt, browser/tool build, initial task state, source timeline, task deadline, retries and accounting constant. Pair by task/timeline seed; randomize arm order. Publish model tokens, API calls, browser interactions, scheduler events, CPU, network, infrastructure and verifier costs.

### 8.2 Task strata and benchmark sizing

Keep level and edge semantics separate. Include no-op tasks, absolute/relative predicates, transient opportunities, long disconnects, source cursor gaps, event replay, same-tick revocation, receipt expiry, stale workers, sink response loss, cross-tenant transplant and cancellation races. Hold out complete event/failure families, not just shuffled cases from the same generator.

Start with at least 100 controlled task templates across multiple applications and event families. Expand toward 200+ held-out templates. A task count is not the same as an independent trial count: for critical adversarial claims, separately target at least 300 independently generated race/attack instances per critical failure family. With zero observed failures in 299 independent Bernoulli trials, the one-sided 95% exact upper bound is approximately 1%; repeated replays of the same template do not satisfy independence. Report clustered uncertainty when trials share a task family, application, environment or seed. A target of 2,995 independent zero-failure trials is needed for an approximate 0.1% upper bound under the same assumptions.

Use paired task/timeline comparisons and randomize arm order. Pre-register primary endpoints, exclusions, fault strata and retry budget. Use bootstrap intervals clustered by task family or a multi-facet/generalizability analysis to estimate task, environment, model, harness and interaction variance. Avoid a global leaderboard score that hides specialization; report per-family performance and confidence. A 2026 Stanford student research report on agent evaluation argues that task-by-agent interaction can dominate aggregate leaderboard reliability; treat it as a reason to investigate variance decomposition, not as a settled consensus result.

Reproduce SentinelBench's original wait/sleep comparison separately; its task-success metric is not an SFEB score. In addition, add a realism bridge: sample event inter-arrival times, delivery delay and outage lengths from measured or published operational traces, then keep a synthetic stress lane for rare correlated and adversarial failures. Never let synthetic-only wins stand in for live source/sink qualification.

### 8.3 Report a metric vector, not a weighted score

**Safety gates:** unauthorized mutations, stale-fence mutations, duplicate effects, wrong-target/cross-tenant effects, expired/revoked receipt actions, invalid-signature acceptance and verifier false acceptance.

**Liveness:** level-condition timely-action rate and edge eligible-event recall are distinct estimands. Report opportunity expiry misses separately from safe abstentions.

**Recovery:** outcome-unknown frequency, authoritative reconciliation success, time to determine committed/not-committed, cursor-gap recovery, high-watermark rollback detection, duplicate/replay handling, and stale-worker rejection.

**Latency/recovery:** p50/p95/p99 for event→delivery, wake→authoritative snapshot, snapshot→authorization, authorization→sink commit and sink commit→independent proof; also cursor-gap age, replay completion latency, source high-watermark lag and duration in quarantined/inconclusive states.

**Efficiency:** verified dormancy ratio (VDR), process CPU seconds/hour, network bytes/hour, model/browser calls/hour, scheduler events/hour and fully loaded cost/hour. VDR does not imply zero CPU/network activity.

**Utility:** verified eligible actions per dollar and per tool interaction, under identical safety gates. A cheap unsafe run is not a win.

### 8.4 Prospective targets (not results)

| Metric | Research target |
|---|---:|
| Unauthorized / stale-fence / cross-tenant side effects in deterministic controls | 0 |
| Duplicate effects for the same idempotency key on a qualifying sink | 0 |
| Stale epoch rejected at sink | 100% of deterministic stale-epoch controls |
| Revocation effective at sink commit | 100% of deterministic race controls |
| Fault-free eligible-event recall | ≥99% |
| p95 wake→authoritative reconciliation in local controlled service | ≤2 seconds |
| VDR at ≥10-minute horizons | ≥99.5%, measured from actual scheduler/runtime data |
| Public controlled benchmark before comparative claims | ≥100 tasks |
| Independent event-family holdout before broad claims | ≥200 tasks |

Zero observed failures is not a probability-zero guarantee. Publish complete denominators, paired confidence intervals and fault-family breakdowns. Deterministic negative controls and empirical event rates are separate result classes.

## 9. Execution sequence

1. Freeze this contract and version the state machine / artifact schemas.
2. Implement a small pure reference kernel for lease epoch, sink fencing, permission-revision check and durable idempotency.
3. Add deterministic tests for stale worker, same-key/different-request conflict, permission revocation race and response-lost-after-commit.
4. Differential-test JS reference and Rust implementation when the Rust slice exists.
5. Add fixture-local source event service with cursor replay, outbox, expiry and revocation.
6. Add receipt signing and a separate verifier principal.
7. Add real scheduler and process/CPU/network instrumentation.
8. Add one Leptos/Chromium scenario with raw event, DOM, accessibility, network and postcondition evidence.
9. Only after exact-head qualification, consider adding selected tasks to the 30-task pilot.

## 10. Standards and external baselines

- CloudEvents 1.0.2: https://github.com/cloudevents/spec
- AWS transactional outbox pattern: https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html
- RFC 9421 — HTTP Message Signatures: https://www.rfc-editor.org/rfc/rfc9421
- RFC 9530 — Digest Fields: https://www.rfc-editor.org/rfc/rfc9530
- RFC 8785 — JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785
- RFC 9110 — HTTP Semantics / conditional requests: https://www.rfc-editor.org/rfc/rfc9110
- Microsoft Research SentinelBench: https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/
- Odysseys: https://arxiv.org/abs/2604.24964
- WebdriverIO benchmark methodology: https://benchmark.webdriver.io/
- Temporal Durable Execution and event-history replay: https://docs.temporal.io/durable-execution
- Temporal Signals, Queries and Updates: https://docs.temporal.io/encyclopedia/workflow-message-passing
- Temporal Activity idempotency and at-least-once retries: https://docs.temporal.io/activities
- DBOS Postgres-backed durable workflow recovery: https://docs.dbos.dev/architecture
- DBOS durable execution and exactly-once database transaction boundary: https://www.dbos.dev/
- Deployment Decision Reliability: Generalizability-Theory Framework for Long-Horizon Agent Evaluations (Stanford CS321M project, 2026): https://aims.stanford.edu/cs321m-projects/Srinivasan2026.pdf

## 11. Qualification boundary

This document specifies a research direction and acceptance contract; the companion JavaScript file is a pure in-memory reference model, not a production service. It is not a benchmark result, cryptographic qualification, scheduler measurement or production authorization system. MON-001's reference evidence remains distinct from committed Node CLI execution. MON-002's in-memory V8 claims remain distinct from a retrievable committed fixture/verifier bundle. The 30-task pilot stays at `runnable_task_count = 0`.


## 12. Executed reference-kernel increment (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs`  
Machine-readable report: `benchmarks/frontend/monitoring/mon003-reference-report-v1.json`

Exact local execution:
```sh
node --check benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs
node benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs
```

Runtime: Node.js v22.16.0. Source SHA-256: `5ff04d8c395aad6fa7d3e756dbceac4fca510e4c2e92aad0d41b29386150088b`.

**Executed result: 9/9 deterministic cases passed.** Coverage: durable edge receipt after current level fell false; stale worker A rejected after worker B obtained the higher fence; permission revocation after grant issuance; expired receipt; revoked receipt; sink commit with response lost followed by reconciliation using the same key; same idempotency key with a different request digest; wrong-target grant substitution; fabricated receipt lookup rejected by the source/authority path; request-digest substitution during an idempotent replay rejected.

The lost-response case produced exactly one mock side effect and recovered the original operation receipt by the same idempotency key. The negative controls produced no side effect.

This is a Node.js execution of the newly committed pure reference kernel, not a run of the pre-existing MON-001 verifier and not a real database, broker, signed receipt issuer, external authorization plane, action sink, browser, scheduler or independent verifier. The mock sink's atomicity is a model assumption; production transactionality remains an open gate.


## 13. Objective-ledger transition preflight (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-objective-ledger-v1.mjs`  
Machine-readable report: `benchmarks/frontend/monitoring/mon003-objective-ledger-report-v1.json`

Executed with Node.js v22.16.0:
```sh
node --check mon003-objective-ledger-v1.mjs
node mon003-objective-ledger-v1.mjs
```

**Executed result: 3/3 deterministic cases passed.** The tests reject stale objective-version writes, illegal jumps and success without execution receipt plus independent postcondition evidence; the unknown-outcome path preserves the original operation ID before transitioning to success.

Exact source Git blob SHA: `1407b6c5240ca98071e062513588da5e17166ef5`.  
Exact source SHA-256: `f42218fee255b7b5e78c5b7bf0c9fd05e46aa998c63addbcf721bf7d2d090a59`.

This is a separate in-memory state-machine reference test, not persistence or distributed compare-and-swap qualification. The postcondition proof is a fixture token, not a cryptographically verified independent observation.


## 14. Tenant-boundary and event-identity negative controls (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-tenant-boundary-v1.mjs`  
Machine-readable report: `benchmarks/frontend/monitoring/mon003-tenant-boundary-report-v1.json`

Executed using Node.js v22.16.0:
```sh
node --check mon003-tenant-boundary-v1.mjs
node mon003-tenant-boundary-v1.mjs
```

**Executed result: 8/8 deterministic cases passed.** Controls reject cross-tenant receipt/grant issuance and transplant, duplicate CloudEvents identity within one source while allowing the same event ID from another source, changed-request idempotency reuse, permission revocation after grant issuance, stale worker epoch and action-target substitution. All negative controls produced zero side effects.

Exact source Git blob SHA: `43d0550304cd6e0e8a003e913cc5241c82421f78`.  
Exact source SHA-256: `24f80efae77fc4d31f64098febf15d5d4881a646d26e69c6cf4bf5b4f491a6de`.

**Important integration boundary:** this is a companion contract test, not proof that the original 9-case fencing kernel or a production service already enforces tenant scope. The next integration change must carry tenant identity through source receipts, authorization grants, idempotency derivation and sink-side atomic checks. Trust in the test's source/tenant identity is an input assumption; no receipt signature, issuer key revocation or persistent tenant policy store is qualified.


## 15. Bounded interleaving exploration (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-interleaving-modelcheck-v1.mjs`  
Machine-readable report: `benchmarks/frontend/monitoring/mon003-interleaving-modelcheck-report-v1.json`

Executed with Node.js v22.16.0:
```sh
node --check mon003-interleaving-modelcheck-v1.mjs
node mon003-interleaving-modelcheck-v1.mjs
```

**Executed result: all 720 permutations of six lifecycle operations explored.** Operations: worker replacement, permission revocation, receipt expiry, old-worker commit with lost response, replacement-worker retry under the same operation identity, and reconciliation.

All modeled invariants passed:
- at most one side effect for one operation key;
- stale worker cannot commit after a higher fencing epoch is installed;
- no new effect after permission revocation or receipt expiry;
- a lost response does not imply failure or justify a new key;
- reconciliation is read-only.

Observed across the 720 schedules: 228 schedules with one effect; 492 with no effect; 324 stale-fence rejections; 312 permission rejections; 108 expiry rejections; 108 idempotent replays; 174 explicit successful reconciliations; 72 additional schedules in which a replacement worker learned the original result via idempotent replay; and 516 schedules still outcome-unknown at the end because the finite event order did not provide a later successful read of the operation result.

Exact source Git blob SHA: `ebf2c1a4a289c11c06876925c7207c203c14cd56`.  
Exact source SHA-256: `c8e8fa9b34d7b314cdb20b45d3a1ea017bacd765d8cb8d70576de422c1f6b95a`.

This is bounded sequential interleaving exploration, not a concurrent memory-model proof. It assumes atomic sink checks and does not cover database isolation anomalies, actual broker durability, clock drift or network partitions beyond the modeled events. It is useful for finding ordering counterexamples before live integration; zero finite-model failures do not establish production failure probability.

## 16. Local toolchain boundary

The available local execution environment had Node.js v22.16.0 but no `rustc`, `cargo` or Nix executable. The Rust parity implementation is therefore not marked executed or passed. The current JS reference suites and bounded model checker were run locally; Rust translation remains an explicit next engineering gate rather than silently substituting Node results for Rust evidence.


## 17. Partitioned source cursor and replay preflight (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-cursor-replay-v1.mjs`  
Machine-readable report: `benchmarks/frontend/monitoring/mon003-cursor-replay-report-v1.json`

Executed with Node.js v22.16.0:
```sh
node --check mon003-cursor-replay-v1.mjs
node mon003-cursor-replay-v1.mjs
```

**Executed result: 15/15 deterministic cases passed.** Cases cover out-of-order delivery without cursor leap, replay gap closure, duplicate pending/committed delivery, source-scoped `(source,id)` identity, changed-content ID reuse, partition isolation, sequence forks and quarantine, high-watermark gap reporting, gap recovery, watermark rollback below committed cursor, watermark rollback while the consumer is behind, committed-sequence rewrite, high-watermark-without-evidence and malformed cursors.

Exact source Git blob SHA: `5afddb6a1a647e200e836b0427bab1517fe3b8d1`.  
Exact source SHA-256: `77919e99612be1253f9524b160cf9c43796f2c8d57c7b53ab8cfee59dcbdb032`.

A source high-watermark cannot regress behind either the committed cursor or a previously trusted high-watermark. Only the contiguous committed prefix is eligible as source-event evidence; a notification or high-watermark alone cannot create an eligible event.

This is a single-threaded in-memory cursor/replay model. It does not qualify a real broker, database cursor, retention/outbox transaction, concurrent cursor CAS, authenticated source identity or signed event provenance.


## 18. Capability-aware source contracts (2026-11-23)

A monitoring guarantee is not an agent property alone. It is constrained by the source integration. SFEB must classify a source before promising edge recall, reconnect recovery, event dormancy or authenticated provenance.

### 18.1 Source capability ladder

| Tier | Source capability | What SFEB may claim |
|---|---|---|
| C0 | Poll-only; no durable event stream | Level checks between polls; no transient-edge recall guarantee |
| C1 | Best-effort push/notifications; no durable replay cursor | Wake hints only; delivery loss can create an unrecoverable edge gap |
| C2 | Durable event log with stable source identity, per-partition sequence, high-watermark and replay-from-cursor | Bounded edge recovery within source retention, subject to cursor correctness |
| C3 | C2 plus authenticated issuer, tenant/scope binding and key revocation checks | Authenticated event provenance under the source trust profile |
| C4 | C3 plus action-sink current policy revision, fencing and durable idempotency | Bounded event recovery with live authority and controlled external mutation, subject to sink transaction guarantees |

A higher tier is not inferred from marketing or transport choice. An adapter supplies a versioned capability profile; SFEB validates the profile's shape and executes conformance probes. The claim is bounded by the weakest capability in the end-to-end path.

### 18.2 Retention horizon is part of the guarantee

Let D be the maximum supported disconnection duration and R the source's replay retention. Edge recovery is only supported when R is at least D, with allowance for replay lag, source maintenance and clock uncertainty. Otherwise the correct result after reconnect is inconclusive or a declared liveness miss—not a fabricated claim that no event occurred.

The source capability profile records:
- delivery mode: polling, best-effort push or durable push;
- source identity and event identity semantics;
- per-partition sequence and high-watermark semantics;
- replay-from-cursor support and retention horizon;
- signed event envelope, issuer scope and key-revocation status;
- current permission revision enforced at the action sink;
- expected disconnection horizon and polling interval, where applicable.

The compiler must fail closed on missing or contradictory fields. A high-watermark without stable event identity and replayable sequence is not sufficient evidence. A signed event without issuer-scope and key-revocation validation is not trusted provenance. A valid event receipt without current sink authority is not an executable opportunity.

### 18.3 Capability-classifier preflight

Companion source: benchmarks/frontend/monitoring/mon003-source-capability-v1.mjs  
Machine-readable report: benchmarks/frontend/monitoring/mon003-source-capability-report-v1.json

Executed using Node.js v22.16.0: **12/12 deterministic cases passed**. The suite tests poll-only, best-effort push, durable replay, retention shorter than the disconnect horizon, missing partition cursors/high-watermarks, untrusted signature profiles, absent sink-side permission recheck and invalid/missing source capabilities.

Exact source Git blob SHA: 9e4ac2f9c0933db13ce09a865ed09c9139f604df.  
Exact source SHA-256: 2112746978e4759292f49900d6c7692c430fac17606451211c9804cefa21574e.

This is a pure in-memory profile classifier. It does not probe or authenticate a live source, prove the source's declared retention, or show that a production integration continues to meet its declared profile. Production adapters need periodic conformance probes and signed capability/version receipts.

## 19. Industry building blocks and what SFEB should not reinvent

The research position is an **assurance layer over existing durable execution and event infrastructure**, not a replacement for their storage, timers, event buses or orchestration semantics.

- **Temporal** persists workflow event history and replays workflow code after worker failure. Activities may be retried and should be idempotent. This is the native workflow/retry baseline; SFEB adds source evidence, authority, fencing and independent outcome proof. [Temporal Durable Execution](https://docs.temporal.io/durable-execution) · [Temporal Activities](https://docs.temporal.io/activities)
- **DBOS** checkpoints workflow progress in Postgres. It can couple database mutations and checkpoints transactionally, but external service effects still require idempotency keys. [DBOS Architecture](https://docs.dbos.dev/architecture) · [DBOS Durable Execution](https://www.dbos.dev/durable-execution)
- **Azure Durable Functions / Durable Task** use event-sourced orchestration histories, deterministic replay, durable timers and external events. This is another useful workflow-engine comparator; it does not itself establish the truth or authority of an external source event. [Azure durable orchestrations](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-orchestrations) · [Durable Task programming model](https://learn.microsoft.com/en-us/azure/azure-functions/durable-functions-types-features-overview)
- **Apache Kafka** supports exactly-once transactional consume-transform-produce behavior when consumed offsets and produced Kafka records are part of the transaction. That boundary does not automatically extend to arbitrary third-party HTTP side effects. [Kafka design and transactions](https://kafka.apache.org/design/)
- **Amazon EventBridge** supports archiving and replaying retained events for a selected time window back to the source event bus. Replay is useful source infrastructure, but consumers still need identity, cursor, idempotency and policy checks. [EventBridge archive and replay](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-archive.html)
- **Microsoft SentinelBench** provides the closest direct monitoring-agent comparison: 100 tasks across 10 synthetic web environments; its wait_for method polls the page every second and is more cost-efficient than repeated sleep-and-inspect. The authors explicitly leave ephemeral, must-act-now opportunities as an open benchmark gap. [SentinelBench](https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/)

These are complementary building blocks. SFEB should integrate with their durable history/replay primitives while testing whether an additional assurance contract improves eligible-event recovery, authorization correctness, stale-worker rejection, evidence quality and cost per independently verified action.
