# SFEB-MON-003 — durable objective lifecycle and fenced execution

**Date:** 2026-11-23  
**Parent:** [MON-003 issue #13398](https://github.com/Luminous-Dynamics/symthaea/issues/13398)  
**Related:** [MON-001 issue #13392](https://github.com/Luminous-Dynamics/symthaea/issues/13392), [MON-002 issue #13394](https://github.com/Luminous-Dynamics/symthaea/issues/13394)  
**Status:** specification committed; 8-case deterministic JavaScript reference kernel committed and executed under Node.js v22.16.0; production qualification pending

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

### 8.1 Six matched arms

1. LLM sleep/reload/inspect.
2. SentinelBench-style one-second `wait_for`.
3. Fixed-interval browser polling.
4. Untrusted notification + latest source-of-record reconciliation.
5. Durable edge receipt + current authorization + idempotent sink.
6. Full durable objective + sink-enforced fencing + signed receipts + independent postcondition verifier.

Hold model, prompt, browser/tool build, initial task state, source timeline, task deadline, retries and accounting constant. Pair by task/timeline seed; randomize arm order. Publish model tokens, API calls, browser interactions, scheduler events, CPU, network, infrastructure and verifier costs.

### 8.2 Task strata

Keep level and edge semantics separate. Include no-op tasks, absolute/relative predicates, transient opportunities, long disconnects, source cursor gaps, event replay, same-tick revocation, receipt expiry, stale workers, sink response loss and cancellation races. Hold out complete event/failure families, not just shuffled cases from the same generator.

Start with at least 100 controlled tasks and enough repeated event opportunities to estimate per-stratum rates. Expand toward 200 tasks and independent holdout families before broad claims. Reproduce SentinelBench's original wait/sleep comparison separately; its task-success metric is not an SFEB score.

### 8.3 Report a metric vector, not a weighted score

**Safety gates:** unauthorized mutations, stale-fence mutations, duplicate effects, wrong-target/cross-tenant effects, expired/revoked receipt actions, invalid-signature acceptance and verifier false acceptance.

**Liveness:** level-condition timely-action rate and edge eligible-event recall are distinct estimands. Report opportunity expiry misses separately from safe abstentions.

**Recovery:** outcome-unknown frequency, authoritative reconciliation success, time to determine committed/not-committed, cursor-gap recovery and stale-worker rejection.

**Latency:** p50/p95/p99 for event→delivery, wake→authoritative snapshot, snapshot→authorization, authorization→sink commit and sink commit→independent proof.

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

## 11. Qualification boundary

This document specifies a research direction and acceptance contract; the companion JavaScript file is a pure in-memory reference model, not a production service. It is not a benchmark result, cryptographic qualification, scheduler measurement or production authorization system. MON-001's reference evidence remains distinct from committed Node CLI execution. MON-002's in-memory V8 claims remain distinct from a retrievable committed fixture/verifier bundle. The 30-task pilot stays at `runnable_task_count = 0`.


## 12. Executed reference-kernel increment (2026-11-23)

Companion source: `benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs`

Exact local execution:
```sh
node --check benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs
node benchmarks/frontend/monitoring/mon003-objective-kernel-v1.mjs
```

Runtime: Node.js v22.16.0. Source SHA-256: `eee472accbff167138e71791448a782ec73e36c271e69f412c8a4fd5a5c5e182`.

**Executed result: 8/8 deterministic cases passed.** Coverage: durable edge receipt after current level fell false; stale worker A rejected after worker B obtained the higher fence; permission revocation after grant issuance; expired receipt; revoked receipt; sink commit with response lost followed by reconciliation using the same key; same idempotency key with a different request digest; wrong-target grant substitution; fabricated receipt lookup rejected by the source/authority path.

The lost-response case produced exactly one mock side effect and recovered the original operation receipt by the same idempotency key. The negative controls produced no side effect.

This is a Node.js execution of the newly committed pure reference kernel, not a run of the pre-existing MON-001 verifier and not a real database, broker, signed receipt issuer, external authorization plane, action sink, browser, scheduler or independent verifier. The mock sink's atomicity is a model assumption; production transactionality remains an open gate.
