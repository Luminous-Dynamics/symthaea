# CEB Review Causal-Topology Development Challenge Key V0 — 2026-11-07

**Status:** Public development answer key. Not secure, sealed, or eligible for later holdout use.  
**For:** Human coordinators and post-lock adjudicators using [Causal-Topology Development Challenge V0](CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md).  
**Do not expose before first-pass lock:** Reviewers must submit and lock their independent forms before this key is disclosed. Because this key is public, that is procedural concealment only, not secure blinding.

## 1. Key use and semantic-equivalence rule

The diagrams below are one compact representation of the expected causal structure, not the only acceptable node decomposition. Reviewers may use different intermediate nodes and still be correct if they preserve:

- target outcome and invariant identity;
- the same sufficient path(s) and AND/OR logic;
- required temporal ordering;
- the P/I/F/N/M/IRR counterfactual predictions;
- residual-path status after repair;
- source and authority boundaries.

Do not award full credit to a graph that contains the right file or node but has an incorrect gate, omits a necessary condition, asserts an unsupported edge, or predicts the wrong intervention effect. Do not penalize harmless node splitting/merging where semantics are equivalent.

All evidence in these synthetic cases is source-declared. No compiler, runtime or intervention observation is supplied.

## 2. CTG-S01 — alternate ingress authorization bypass

**Expected target mechanism:** A valid scheduler signature is treated as if it proved actor authority over the tenant/resource. The import-worker route reaches the mutation without the authorization invariant.

**Canonical topology:**

\`\`\`text
API request
  -> authorize(actor, tenant, resource)
  -> mutate
  -> outcome (permitted mutation)

Import job
  -> verify scheduler signature
  -> worker mutation (no actor/tenant/resource authorization)
  -> outcome (unauthorized cross-tenant write possible)
\`\`\`

Ingress is an OR choice. The worker path is unsafe; the API path remains guarded. The worker signature establishes scheduler authenticity only. A valid signature is not an authority edge from the original actor to the requested resource.

| State | Expected target | Reason |
|---|---|---|
| P | absent | The in-scope worker bypass path does not exist; API route is guarded. |
| I | present | Worker path reaches mutation after signature verification only. |
| F | absent | Worker now checks actor, tenant and resource before mutation. |
| N | target absent | Scoped capability validation is a distinct valid authority mechanism. |
| M | absent | The minimal missing authorization check is inserted on the worker path. |
| IRR | present | Logging does not alter the worker's path or authority. |

**Nearest wrong mechanisms:** “The API handler is missing authorization” (false; it calls the check); “scheduler signature verification is broken” (not the target); “all background jobs are unsafe” (overgeneralized).

**Residual-path rule:** Fixing the worker path does not justify a claim about callers outside the declared two-path source package. The API path is safe only under the stated contract.

## 3. CTG-S02 — conjunction of legacy mode and missing policy

**Expected target mechanism:** The fail-open return is reachable only when both \`legacy_mode = true\` AND \`policy_version = None\`. Either condition alone is insufficient to activate the target branch.

**Canonical topology:**

\`\`\`text
legacy_mode = true ----\
                        AND gate -> Decision::Allow fallback -> resource mutation
policy_version = None -/
\`\`\`

The gate output is the unsafe decision. Ordinary policy evaluation is a separate branch, not a causal parent of the fail-open outcome.

| State | Expected target | Reason |
|---|---|---|
| P | absent | No permissive missing-policy fallback exists. |
| I | present | Both declared conditions activate the Allow fallback. |
| F | absent | Missing policy now fails closed. |
| N | target absent | A signed, scoped override is a different authority-bearing exception. |
| M | absent | Replacing Allow with Deny removes the sufficient conjunction. |
| IRR | present | Logging/counter changes preserve the decision path. |

**Nearest wrong mechanisms:** “Legacy mode alone is insecure” (does not establish the gate); “missing policy alone always allows” (does not establish the gate); “the policy loader is defective” (not the described failure).

**Residual-path rule:** Other policy fallback behavior outside this function is out of scope and unknown.

## 4. CTG-S03 — retry plus attempt-scoped deduplication

**Expected target mechanism:** A committed transfer is followed by a missing acknowledgement; the queue redelivers the same logical operation with a new attempt ID; the deduplication key includes that attempt ID, so the retry is treated as a new operation and the effect can commit twice.

**Canonical topology:**

\`\`\`text
commit_transfer(operation_id)
          |
          v
ack not durably observed (lost response / worker interruption)
          |
          v
same logical operation redelivered with new attempt_id
          |
          v
dedup key (operation_id, attempt_id) is distinct
          |
          v
dedup miss -> second commit -> duplicate effect
\`\`\`

Necessary conditions in the declared scenario:
1. side effect commits before acknowledgement is durably established;
2. the acknowledgement is lost or not observed, causing redelivery;
3. deduplication identity changes across attempts.

Retry alone is not sufficient if the logical operation ID is durably deduplicated.

| State | Expected target | Reason |
|---|---|---|
| P | absent | The described post-commit retry path is absent. |
| I | present | New attempt identity defeats logical-operation deduplication after redelivery. |
| F | absent | Durable operation identity and effect are atomically recorded before acknowledgement. |
| N | absent | Stable operation ID plus retention through the redelivery horizon suppresses repeats. |
| M | absent | Using the stable operation ID as the dedup key removes the attempt-specific miss under the declared retention. |
| IRR | present | Jitter/logging changes do not change the identity or commit/ack sequence. |

**Nearest wrong mechanisms:** “The queue is at-least-once” (not independently a defect); “network loss alone duplicates transfers” (only if idempotency fails); “TTL alone is causal” (the supplied I key already differs on every attempt).

**Residual-path rule:** The key does not establish atomicity outside the described transfer/dedup store. If a reviewer claims end-to-end exactly-once semantics, that is an overclaim.

## 5. CTG-S04 — time-of-check/time-of-use stale authorization

**Expected target mechanism:** Authorization is evaluated at enqueue time; revocation occurs while the job waits; the worker relies on the serialized boolean and mutates without checking current authority.

**Canonical temporal topology:**

\`\`\`text
t0: authorize_now(actor, tenant, resource) = true
       |
       v
    enqueue(Job { allowed: true })
       |
t1: role/permission revoked
       |
t2: worker reads stale allowed=true
       |
       v
    mutate after authority revoked
\`\`\`

The relevant ordering is t0 < t1 < t2. The defect requires a delay window and use of the stale decision. Do not claim the race occurs when revocation is after the protected mutation.

| State | Expected target | Reason |
|---|---|---|
| P | absent | Synchronous authorization and mutation occur in the same declared transaction scope. |
| I | present | Asynchronous delay separates check from mutation; stale boolean is trusted. |
| F | absent | Current authorization is checked at the mutation boundary. |
| N | target absent | Capability epoch is checked against current revocation state. |
| M | absent | Reauthorization immediately before mutation closes the stale window. |
| IRR | present | Queue ordering/observability changes do not refresh the boolean. |

**Nearest wrong mechanisms:** “All async work is unsafe” (too broad); “signature validation is broken” (not the mechanism); “revocation always wins instantly” (not established unless worker checks it).

**Residual-path rule:** Other workers or mutation paths are not covered unless included in the source package.

## 6. CTG-S05 — malformed-input recovery bypasses size bound

**Expected target mechanism:** Invalid encoding selects the recovery path; attacker-controlled declared length reaches a proportional allocation before the bound is applied; resource exhaustion is therefore reachable through the fallback even though the normal path is guarded.

**Canonical topology:**

\`\`\`text
                    +-> valid encoding -> size check -> bounded allocation --+
input --------------|                                                        +-> consume
                    +-> invalid encoding -> recovery -> allocation by length -+
                                             (missing size check)
\`\`\`

The two paths are OR alternatives. The normal path is not defective under the supplied contract. On the recovery path, malformed encoding and oversized declared length jointly expose the allocation.

| State | Expected target | Reason |
|---|---|---|
| P | absent | Malformed input is rejected before proportional allocation. |
| I | present | Recovery path allocates by untrusted length without the bound. |
| F | absent | Common bound is applied before either allocation route. |
| N | target absent | Bounded streaming with backpressure avoids proportional allocation. |
| M | absent | The pre-allocation bound check is inserted into the recovery path. |
| IRR | present | Error text and telemetry do not alter allocation order. |

**Nearest wrong mechanisms:** “The normal parser lacks a size check” (false); “all malformed input causes OOM” (not shown); “the allocator is defective” (not the target).

**Residual-path rule:** A fix that guards the normal branch alone is partial and leaves the recovery path exposed.

## 7. CTG-S06 — interacting size and concurrency limits

**Expected target mechanism:** Per-item allocation is below its per-request cap, but the increased per-item size AND increased concurrency allow aggregate simultaneous allocations to exceed the shared memory budget.

**Canonical topology:**

\`\`\`text
request size = 24 MiB --------\
                               AND gate -> aggregate allocation = 288 MiB
12 concurrent jobs ----------/                 |
                                                v
                                       256 MiB budget exceeded
                                                |
                                                v
                                            worker OOM
\`\`\`

The configuration values are causally interacting parents of aggregate allocation. At 24 MiB × 8 jobs = 192 MiB, the outcome is not reached. At 16 MiB × 12 jobs = 192 MiB, the outcome is not reached. At 24 MiB × 12 jobs = 288 MiB, it is reachable. A count-only admission controller does not account for aggregate bytes.

| State | Expected target | Reason |
|---|---|---|
| P | absent | 16 × 8 = 128 MiB, within budget. |
| I | present | 24 × 12 = 288 MiB, exceeding 256 MiB. |
| F | absent | Weighted semaphore enforces the aggregate budget, including release/cancellation. |
| N | target absent | 24 × 8 = 192 MiB, within budget. |
| M | absent | Restoring concurrency to 8 yields 192 MiB maximum under the raised item limit. |
| IRR | present | Logging does not change either factor or admission semantics. |

**Nearest wrong mechanisms:** “Large request alone exceeds the limit” (24 MiB is under 32 MiB and 256 MiB); “concurrency alone is unsafe” (12 × 16 MiB would be 192 MiB); “the memory budget is wrong” (the invariant is declared and fixed for this case).

**Residual-path rule:** A weighted semaphore is only complete if all allocation paths acquire weight and release it on success, error, cancellation and timeout. The simplified source excerpt does not prove those lifecycle semantics; they must remain an explicit source question.

## 8. Cross-case equivalence and failure labels

Accept different graph decompositions when they preserve these semantics. Use the existing adjudication categories and record topology-specific disagreements as diagnostic annotations:

- missed alternative path;
- incorrect AND/OR gate;
- omitted necessary condition;
- false causal edge inferred from code adjacency;
- incorrect temporal order;
- intervention predicted to remove a path that remains;
- residual path missed;
- over-broad graph/scope;
- graph incomplete because the source package is insufficient.

Do not use a graph-size penalty as a substitute for semantic adjudication. A concise graph with one missing required path is wrong; a longer graph with unsupported edges is also wrong.

## 9. Claim boundary

This key is a synthetic expected-answer guide. It is not a gold-standard authority for real repositories, not an exhaustive graph representation, and not a benchmark result. Any ambiguity discovered by independent reviewers must be preserved, adjudicated and, when necessary, used to revise the case under a new version. No source-only answer may be promoted to a runtime/intervention claim.
