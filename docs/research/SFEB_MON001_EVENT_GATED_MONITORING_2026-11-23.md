# SFEB-MON-001 — event-gated monitoring research and execution

**Date:** 2026-11-23  
**Parent programme:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Track:** SFEB-MON-001 (separate from the 30-candidate visual/interactive/full-stack pilot plan)  
**Status:** six-case synthetic fixture and 11-rule policy authored; in-memory V8 preflight passed; committed CLI / production event bus / browser execution pending

## Thesis

A long-running agent should not spend hours in a self-induced action loop. It should compile a user's objective into a **watch contract**, hand waiting to a scheduler, and resume only when a wake hint justifies a fresh read of the source of record.

The safe control path is:

`Intent → Watch Contract → Scheduler Registration → Dormancy → Untrusted Wake Hint → Authoritative Reconciliation → Predicate + Permission Check → Idempotent Action / Abstention → Evidence Receipt`

The message bus is a wakeup mechanism, not an authority. A stale, forged, replayed or duplicated notification cannot establish that a condition is currently true. Every mutation must be based on an authoritative snapshot with a current revision and a matching permission check.

## Industry comparison

### SentinelBench — closest direct comparator

Microsoft Research's SentinelBench defines 100 monitoring tasks across 10 synthetic web environments, with time-evolving external state. It explicitly separates passive monitoring from active monitoring and absolute from relative conditions, and includes no-op scenarios where the condition never occurs. Its baseline compares ordinary `sleep` with a `wait_for(condition, timeout)` tool. The `wait_for` implementation takes a baseline textual snapshot and checks for changes once per second; it is substantially more efficient than repeated LLM-driven sleep-and-inspect loops, but it is still a polling implementation. Microsoft reports 69/100 tasks completed with `wait_for` versus 56/100 with `sleep`; for GPT-5.4 at the 10-minute setting, median cost was $0.23 versus $1.17 per task, a 5.1× difference. These are SentinelBench-specific figures, not SFEB scores. [Microsoft Research SentinelBench report](https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/)

The design space left to test is not “poll less” alone. It is **event-wake plus authoritative replay**: wake on a relevant message, fetch the latest revision, evaluate a frozen watch contract, check authorization and act idempotently. We should reproduce SentinelBench's task definitions and compare against its sleep and wait_for baselines before claiming superiority. The current MON-001 traces model this control path but do not exercise a real event bus or browser.

### Odysseys — long-horizon success is not enough

Odysseys evaluates 200 realistic multi-site web tasks with an average of 6.1 graded rubrics per task. Its paper reports a strongest model success rate of 44.5% and only 1.15% rubric score per step under its Trajectory Efficiency metric. Its central lesson is that binary completion hides partial progress and long trajectories hide waste. MON-001 therefore reports latency, observations, polling, duplicate work and cost separately from terminal outcome. [Odysseys paper](https://arxiv.org/abs/2604.24964)

### FrontierCode 2.0 — maintainer quality

FrontierCode 2.0 is a 132-task maintainer-oriented coding benchmark with unit tests, repository-specific rubrics and additional verifiers; blocking failures zero the task score. It is useful for the full-stack maintenance lane, not a direct monitoring score. The official site has a dynamically rendered leaderboard and publishes methodology/scoring changes; third-party snapshots can show different settings and values. The current public mirrors should therefore be treated as date- and configuration-specific, not as one universally fixed “SOTA” number. [Cognition methodology and leaderboard](https://cognition.com/frontiercode) · [Priors snapshot, 23 Nov 2026](https://mypriors.com/benchmarks/frontiercode-2.0-main) · [Frontierlog configuration table](https://frontierlog.ai/benchmark/frontiercode-2-0)

### WebdriverIO and interaction-grounded judging

WebdriverIO's browser-agent benchmark fixes the model/harness/prompt and varies browser tooling over a fixed random sample, reporting Wilson intervals and cost, token and tool-call telemetry. It is a good template for a controlled browser-tool arm, but not a frontend engineering score. Interaction-as-a-Judge shows that interaction-grounded evaluators can improve defect diagnosis and iterative repair in its own protocol; use such models as secondary diagnostic signals, never as authority for permissions, state transitions or deterministic acceptance. [WebdriverIO methodology](https://benchmark.webdriver.io/) · [Interaction-as-a-Judge](https://arena.ai/blog/interaction-as-a-judge)

## New MON-001 artifact set

- `benchmarks/frontend/monitoring/monitoring-fixture-v1.json`: six deterministic cases, external event timelines, source revisions and expected state/outcomes.
- `monitoring-reference-traces-v1.json`: raw registration, wait, wake, observe, act and finish traces, bound to exact fixture bytes by SHA-256.
- `monitoring-policy-v1.json`: eleven required oracle families, pinned by SHA-256 in the verifier.
- JSON Schemas for fixture, raw trace and policy.
- `verify-monitoring.mjs`: source-authored Node.js verifier, including adversarial self-tests for forged notifications, altered action targets and worker-claim invariance.

### Executed in-memory V8 preflight

The independent in-memory implementation used the exact fetched fixture, trace bundle and policy:
- 6/6 case expectations matched;
- 11/11 policy rules passed;
- worker-claim mutation left all case results unchanged;
- forged notification followed by an attempted mutation was rejected;
- action target substitution was rejected;
- duplicate action replay was accepted as an idempotent retry but produced only one side effect;
- 7 authoritative observations and zero explicit poll operations were recorded across the traces;
- the reconnect-stale case was correctly reported as safe abstention with a liveness miss.

This is an in-memory reference preflight, **not** execution of the committed Node.js CLI. The synthetic scheduler's `wait_until` is virtual time; no actual process dormancy, CPU usage, timer accuracy, network traffic or event-bus latency has been measured.

## Paradigm shifts

### 1. Replace agent loops with typed watch contracts

Compile the user request into a durable object containing:
- predicate type: level-triggered or edge-triggered;
- initial baseline and comparison semantics;
- authoritative source and revision cursor;
- freshness and ordering requirements;
- deadline, expiry and time-zone rules;
- required permissions and revocation behavior;
- action schema, target and idempotency key;
- terminal outcomes and evidence requirements.

This moves waiting and scheduling out of the LLM's tool-call loop.

### 2. Treat notifications as hints and authority as a separate plane

The notifier can say “something may have changed.” It cannot assert state, permission or successful action. On wake: deduplicate the notification; resume from a durable cursor; fetch source-of-record state; check monotonic revision; evaluate policy; then act with an idempotency key. If source reconciliation is unavailable, outcome is inconclusive and mutation is blocked.

### 3. Separate level-triggered and edge-triggered semantics

The current stale-reconnect case is deliberately **level-triggered**: the condition was true at revision 2, false at revision 3, and the delayed notification arrives after expiry. Correct behavior is abstention, even though liveness misses the transient opportunity.

An edge-triggered watch has a different contract. It requires a durable source event or signed event receipt proving that the qualifying edge occurred while disconnected. It must also define a valid action window and re-check current permission and cancellation state. A notification alone is never that receipt. The next fixture extension should test both:
- level predicate that expired before reconnect → no action;
- durable edge receipt still within its authorized action window → exactly one action;
- expired edge window or revoked permission → no action.

Do not collapse these into one ambiguous “trigger recall” score.

### 4. Measure useful dormancy and verified responsiveness

Add these metrics:
- **Trigger recall:** fraction of eligible source events resulting in a verified action before expiry.
- **False-action rate:** mutations without a valid source event and currently true policy.
- **No-op abstention:** fraction of never-triggering tasks ending without side effects.
- **Wake precision / recall:** relevance and coverage of notification deliveries.
- **Event-to-verified-action latency:** from authoritative event timestamp to independent receipt.
- **Reconciliation latency:** wake to current source-of-record observation.
- **Observation/poll ratio:** authoritative reads and explicit polls per eligible event.
- **Duplicate side-effect count:** must be zero.
- **Cost per monitored hour:** compute, API, browser, infrastructure and verifier cost.
- **Verified Dormancy Ratio (VDR):** fraction of elapsed monitoring time spent yielded to the scheduler with no model/browser tool calls. Measure process CPU separately; VDR does not imply zero CPU or infrastructure activity.

### 5. Use lexicographic safety, not a blended score

A cost reduction must never compensate for unauthorized action, false completion, missed revocation or fabricated evidence. Publish a vector of safety, liveness, latency, cost and efficiency. Any critical safety failure blocks the run regardless of its efficiency score.

## Proposed research targets (not achieved)

| Metric | Target |
|---|---:|
| Unauthorized or duplicate side effects | 0 |
| Premature action on forged/stale wake | 0 |
| No-op task abstention | 100% |
| Critical permission-revocation detection | 100% |
| Trigger recall on fault-free persistent-event tasks | ≥99% |
| p95 event-to-verified-action latency after delivered wake | ≤2 s in the controlled event-bus lane |
| Explicit agent-side polling in event-wake arm | 0 |
| VDR on ≥10-minute horizons | ≥99.5% (requires real scheduler/browser instrumentation) |
| Verifier false acceptance under seeded adversarial controls | 0 |

These are prospective goals. MON-001 has six synthetic cases, which is not enough for statistically meaningful estimates or a credible SOTA claim.

## Experimental programme to beat the field

1. Reproduce SentinelBench's 100 tasks and official `sleep` / `wait_for` baselines.
2. Add a third fixed-polling arm and a fourth event-wake + durable-cursor + source-revalidation arm.
3. Hold model, prompt, browser harness, task timeline, deadlines and retry budget constant; randomize task order and tool-arm assignment.
4. Add fault injection: duplicate, delayed, forged, reordered, lost and replayed notifications; same-timestamp revocations; disconnect/reconnect; stale source cache; event-bus outage; clock skew.
5. Measure task success, safety violation rate, trigger recall, false actions, no-op abstention, event-to-action latency, token/API/browser cost, CPU, network, wake count, authoritative reads and VDR.
6. Use paired task-level comparisons, task-clustered bootstrap intervals, pre-registered primary outcomes and complete denominators. Publish all timeouts, infra failures and inconclusive cases.
7. Keep public-pilot tasks out of hidden-holdout claims. Hold out entire event families and failure patterns, not shuffled copies.

## Next engineering gates

1. Execute the committed Node verifier under the pinned Node runtime and resolve any parity failures against the in-memory preflight.
2. Extend the fixture with a durable edge-trigger receipt and explicit action-window semantics.
3. Implement a fixture-local event service with durable cursor, replay, duplicate/loss/delay controls and monotonic virtual clock.
4. Add a scheduler-backed wait tool; prove that the agent process yields rather than polling.
5. Add raw execution identity, signed run statement, source snapshot receipts and idempotency receipt; verify them in a separate identity.
6. Implement one Leptos monitoring scenario and measure browser/CPU/network evidence before benchmark task promotion.

## Qualification boundary

MON-001 is a new synthetic research fixture; it is not one of the 30 pilot tasks and is not counted as runnable. No browser candidate, production event bus, signed verifier identity or real scheduler has been qualified. The main pilot ledger remains at `runnable_task_count = 0`; this work does not change that status.


## Complementary edge-triggered semantics

The stale-reconnect case in MON-001 is intentionally level-triggered. A separate MON-002 experiment now tests whether a transient event can remain actionable after reconnect through a durable source-of-record receipt, with immutable event/revision/time/scope binding, expiry, revocation and current permission checks. MON-002 is not a relaxation of MON-001; it has separate fixtures, policy and outcome semantics.

See [MON-002 README](../../benchmarks/frontend/monitoring/edge-monitoring-README.md), [research note](SFEB_EDGE001_DURABLE_EVENT_RECEIPTS_2026-11-23.md) and [tracking issue #13393](https://github.com/Luminous-Dynamics/symthaea/issues/13393). Its independent in-memory preflight passed 5/5 cases, 11/11 policy rules and 9/9 seeded adversarial rejections. The committed Node CLI and production event receipt service remain unexecuted.
