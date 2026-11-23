# SFEB-EDGE-001 — durable edge-trigger receipts and reconnect research

**Date:** 2026-11-23  
**Track:** MON-002 / SFEB-EDGE-001  
**Related work:** MON-001 level-trigger monitoring; PR #13162  
**Status:** synthetic fixture, strict schemas and Node verifier source authored; independent in-memory V8 preflight passed; CLI/runtime qualification pending

## Research question

A level-triggered monitor asks whether a condition is true now. An edge-triggered monitor asks whether a qualifying transition occurred and whether a bounded opportunity created by that transition is still actionable.

The MON-001 stale-reconnect case correctly reports a safe abstention when a transient condition became false before the agent resumed. However, this may miss legitimate work where the source system durably records an event and grants a bounded action window. MON-002 tests the complementary semantics without trusting the notification.

## Paradigm shift: durable event receipts

Compile an edge-triggered objective into a watch contract with:
- event type and source-of-record identity;
- exact edge predicate and baseline semantics;
- durable receipt identity and source event/revision;
- receipt occurrence time, expiry and revocation status;
- action type, resource and permission scope;
- current permission check at action time;
- idempotency key and terminal outcome.

On reconnect, a notification merely wakes the agent. The agent reads the latest source snapshot, locates the receipt in the source of record, re-derives that the source event actually crossed the edge predicate, verifies the immutable receipt binding, checks that the window is still open and the receipt is not revoked, re-checks current permission, then acts idempotently.

The durable receipt is not a blanket authorization. It proves only that a specific edge occurred and that a narrowly scoped opportunity remains valid. Permission revocation, receipt revocation, expiry and action-target mismatch must still block the mutation.

## MON-001 versus MON-002

| Dimension | MON-001 level watch | MON-002 edge watch |
|---|---|---|
| Predicate | condition true now | qualifying transition occurred |
| Reconnect decision | current source state | durable event receipt + current policy |
| Condition later false | abstain | can still act if receipt remains valid |
| Evidence | latest source snapshot | source event + durable receipt + latest snapshot |
| Revocation | current permission | current permission and receipt status |
| Expiry | watch deadline | explicit receipt validity window |
| Idempotency | action key | receipt-bound action key |

Do not merge the two into one ambiguous trigger-recall metric. Level-watch liveness and edge-receipt liveness have different semantics.

## Industry comparison

SentinelBench is the closest published monitoring comparator: 100 tasks across 10 synthetic web environments; it separates passive/active monitoring, absolute/relative conditions and includes no-op tasks. Its `wait_for` tool snapshots page text and checks for changes once per second; it is a useful, comparatively efficient baseline but remains polling-based. Microsoft reports 69/100 completions with `wait_for` versus 56/100 with sleep, and for GPT-5.4 at 10 minutes median cost $0.23 versus $1.17. These numbers are protocol-specific and are not SFEB scores. [Microsoft Research — SentinelBench](https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/)

Odysseys contributes long-horizon partial progress and trajectory-efficiency measurement: 200 multi-site web tasks, about 6.1 graded rubrics/task, a 44.5% best reported success and 1.15% rubric score per step. It is an adjacent computer-use comparator, not an edge-trigger benchmark. [Odysseys paper](https://arxiv.org/abs/2604.24964)

The proposed SFEB comparison must be controlled:
1. LLM sleep-and-inspect;
2. wait_for-style one-second page polling;
3. notification-only wake with current-state revalidation;
4. durable edge receipt + latest-state revalidation;
5. durable edge receipt without independent verification (ablation);
6. durable edge receipt with receipt signature and separate verifier (full qualification arm).

Hold model, agent harness, task timelines, browser tool, cost accounting, deadlines and retries constant. Inject lost/delayed/forged/duplicated/reordered notifications, disconnected intervals, stale caches, receipt expiry, revocation, permission changes, source revision gaps and idempotency replays.

## New MON-002 artifacts

- `benchmarks/frontend/monitoring/edge-monitoring-fixture-v1.json`: five deterministic timelines.
- `edge-monitoring-traces-v1.json`: raw reconnect and action traces.
- `edge-monitoring-policy-v1.json`: eleven required receipt/action oracle families.
- `verify-edge-monitoring.mjs`: strict verifier source with pinned fixture and policy digests.
- Three strict JSON Schemas.

Fixture SHA-256: `75a3b305e5b54716a29a4b994d2ff5aab65a1b1de86e75ba27a6005bf7c83784`  
Policy SHA-256: `73112c7c2106663893774e7babb7863ba4cd05aad232d090818f237fd9e89bc2`

### Executed independent in-memory V8 preflight

Against exact fetched repository fixture/policy/trace bytes:
- **5/5** case expected outcomes matched;
- **11/11** non-vacuous policy rules passed;
- the valid edge receipt authorized one action while the current level predicate was false;
- duplicate action replay did not duplicate the side effect;
- **9/9** adversarial attempts were rejected: wrong target, expired receipt, revoked receipt, permission revoked, forged receipt, receipt tied to a non-qualifying source event, receipt reactivation, receipt rebinding, and preloaded initial-state receipt;
- zero explicit poll operations.

These are in-memory reference results, not execution of the committed Node.js verifier. The repository source adds tests for the same attacks, but those tests remain execution-pending. No production event log, receipt issuer, signature, browser, scheduler or separate verifier has been qualified.

## Proposed benchmark targets (not achieved)

| Metric | Prospective target |
|---|---:|
| Receipt provenance false acceptance | 0 |
| Unauthorized/expired/revoked receipt side effects | 0 |
| Duplicate side effects under replay | 0 |
| Current permission/revocation detection | 100% |
| Eligible edge recall under supported reconnect faults | ≥99% |
| p95 wake-to-authoritative-snapshot latency | ≤2 s in controlled event service |
| p95 source-event-to-verified-action latency | ≤2 s after event delivery, subject to source SLA |
| Agent-side polling in push/receipt arm | 0 |
| Verified dormancy ratio over 10-minute horizons | ≥99.5% with real scheduler instrumentation |

A five-case synthetic fixture cannot establish statistical SOTA. Expand to at least 100 tasks with held-out event families, seeded faults and paired arm randomization before comparative claims.

## Qualification sequence

1. Execute the committed Node.js verifier under pinned Node 22; validate the strict JSON Schemas.
2. Build a fixture-local durable event service with cursor replay, receipt creation, expiry and revocation.
3. Implement receipt signing/issuer identity and source-of-record verification.
4. Add edge-receipt loss, replay, reordering, expiry-boundary and same-timestamp revocation fault injection.
5. Implement a real scheduler-backed wake path and measure process CPU, network, wake and poll counts.
6. Implement one Leptos/browser scenario, capture raw event/state/browser artifacts, and replay them under a separate verifier identity.
7. Only then create runnable tasks or include MON-002 in the 30-task benchmark plan.

## Qualification boundary

MON-002 remains a synthetic public research fixture, not a hidden holdout or runnable benchmark task. It does not alter the existing portfolio count: `runnable_task_count=0`. The committed verifier CLI, runtime event service, browser task, signature issuer and independent verifier remain pending.
