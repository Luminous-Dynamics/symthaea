# SFEB long-horizon state reconciliation and dynamic interface work

**Date:** 2026-11-23  
**Status:** research design; INT-004 remains planned/not authored and is not runnable  
**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Task-family issue:** [#13251](https://github.com/Luminous-Dynamics/symthaea/issues/13251)

## Why this is the next frontier

Frontend benchmarks often emphasize visual reconstruction, short deterministic interactions or a static final state. Real operational applications are dynamic: records change while a user works, server events arrive late or more than once, permission/ownership changes can invalidate an action, and the user may need to clarify an ambiguous request before an irreversible transition.

Two 2026 benchmarks expose complementary gaps:

- **FrontierCode 2.0** is a maintainer-oriented mergeability benchmark. Cognition reports 132 tasks and an official launch weighted score of 47.3% for Claude Opus 5.5 at its best reported effort. Any failed blocking criterion zeros a task. Its release analysis shows that thoroughness is a major remaining weakness: at maximum effort, pooled blocker failure rates are 15.3% for thoroughness, 12.3% for core behavior and 5.2% for backwards compatibility. This argues for tests that inspect the entire affected invariant family rather than merely the named example. [Official announcement and methodology](https://cognition.com/blog/frontier-code-2.0).
- **OSWorld 2.0** evaluates 108 long-horizon workflows across 31 self-hosted websites. Its task summary reports a median human execution time around 1.6 hours, more than 250 agent steps on average and 27.25 scoring checkpoints. The paper's primary 500-step result is 20.6% strict binary completion and 54.8% partial score for Claude Opus 4.8 with maximum thinking and batched tools. Its challenge tags include cross-source reasoning, implicit-state inference, multi-item state tracking, conflict disambiguation, dynamic environments, streaming interaction and proactive clarification. [Benchmark](https://osworld-v2.xlang.ai/) · [Paper](https://arxiv.org/abs/2606.29537).

These numbers are not comparable to Vision2Web or SFEB task completion rates. FrontierCode tests repository patch mergeability; OSWorld tests long-horizon computer use. SFEB should borrow their strongest *task-design lessons*, not their score scales.

## Proposed task: SFEB-PILOT-INT-004 — keyboard-accessible work board with dynamic state reconciliation

This is the expanded candidate already represented in `task-family-plan-v1.json`. It remains a public-pilot brief only. No fixture, service or oracle has been authored, and it is not counted as runnable.

### User-facing goal

A coordinator manages a work board containing assigned jobs, priorities, statuses and owners. They move a job through a permitted state transition using either pointer or keyboard controls. While the user is acting, the authoritative service may deliver:
- a late event that changes the job's status or owner;
- a duplicate event with the same event identity;
- a conflicting update that makes the pending action stale;
- a permission change that removes the user's authority;
- an ambiguous instruction that requires clarification before a protected mutation.

The interface must reconcile with authoritative state, not preserve an optimistic local fiction.

### Deterministic local event model

Use a fixture-local service and a frozen event schedule; no live third-party systems, network dependencies or real credentials. Each event has:
- stable event ID and idempotency key;
- entity ID and monotonically increasing entity version;
- authoritative actor/role and allowed transition;
- event timestamp/order controlled by a deterministic test clock;
- causal predecessor/version and expected state digest.

The browser receives events through a deterministic stream adapter. The evaluator injects late delivery, duplicate delivery, pauses/resumes and a state-changing message at frozen checkpoints. A separate expected-state reducer derives the authoritative final state. The candidate must not author that reducer or its verdict.

### Required behavior

1. Pointer drag/drop and keyboard alternatives produce equivalent valid state transitions.
2. Optimistic UI is visibly pending and is committed only after authoritative acknowledgement.
3. A late authoritative event invalidates a stale local action; the candidate must reject or rebase it according to the transition contract.
4. Duplicate event delivery is idempotent: no duplicate mutation, notification, audit entry or activity row.
5. A rejected optimistic mutation rolls back to authoritative state with an accessible explanation.
6. Permission changes are applied before the next protected mutation; stale authorization must fail closed.
7. If task instructions and authoritative evidence conflict or leave a required decision ambiguous, the system asks for clarification before committing an irreversible action.
8. Activity history is append-only and corresponds exactly to accepted authoritative transitions; rejected attempts are separately identifiable.
9. Reconnection/replay converges to the same state as uninterrupted delivery.

## Counterfactual and metamorphic campaign

### Negative controls

- Ignore a late authoritative event and accept a stale action.
- Apply a duplicate event twice.
- Show an optimistic state as committed before acknowledgement.
- Fail to roll back after service rejection.
- Accept a transition after permission revocation.
- Record an activity entry for a rejected transition.
- Commit an ambiguous protected action without a clarification gate.
- Remove keyboard access while pointer drag/drop still works.

### Positive and metamorphic controls

- Duplicate delivery of the same event ID does not change final state or audit history.
- Permuting non-conflicting events produces the same final state.
- Replaying the same frozen event schedule produces byte-identical final-state and audit digests.
- Pointer and keyboard operation produce equivalent authoritative outcomes.
- Pausing and resuming the event stream converges to the uninterrupted result.
- A semantics-preserving visual/style change does not alter the event-state result.

The current plan records these as candidate fault and metamorphic families only. They have not been instantiated or executed.

## Metrics to add to SFEB

Do not collapse these into an average quality score. Report each with task-level denominators and uncertainty:

| Metric | Proposed target |
|---|---:|
| Protected stale actions accepted | 0 |
| Duplicate protected effects | 0 |
| Duplicate event application | 0 |
| Authoritative final-state agreement | 100% |
| Append-only audit completeness | 100% |
| Permission revocation enforcement | 100% |
| Keyboard/pointer semantic equivalence | 100% |
| Ambiguous irreversible action without clarification | 0 |
| Unrelated invariant regressions | <5% |
| Reconnection/replay convergence | 100% |

For non-critical UX measures, record reconciliation latency p50/p95, time spent in optimistic/pending state, clarification precision/recall, time-to-recovery, number of retries, cost per accepted workflow and human intervention. Establish latency budgets from a pinned local environment before setting a universal threshold.

## Horizon design

Avoid manufacturing difficulty by repeating trivial clicks. Grow the dependency depth:

- 10 transitions: parser/state-machine and basic idempotency smoke level;
- 50 transitions: multi-entity changes, retries and keyboard/pointer parity;
- 100 transitions: several late/duplicate/conflicting events and permission changes;
- 250 transitions: long workflow with injected mid-task updates and final-state/audit reconciliation.

Use the same frozen task contract at every horizon, with a pre-registered event schedule and checkpoint set. Report partial completion checkpoints separately from strict final completion. Repeated runs on one task are not independent task samples.

## How SFEB can go beyond the current industry

Combine four complementary strengths:
1. Vision2Web's static, interactive and full-stack coverage.
2. FrontierCode's maintainer-style thoroughness, backwards compatibility, test integrity and patch mergeability.
3. OSWorld 2.0's late-arriving information, hidden-state inference, conflict disambiguation, streaming and proactive clarification.
4. SFEB's two-sided task validity, isolated browser execution, independent oracle replay, exact-revision evidence, counterfactual repair and unrelated-invariant preservation.

The differentiator is not a larger step count. It is proving that the system reconciles dynamic state correctly and can show independently why each mutation was accepted, rejected, rolled back or deferred for clarification.

## Claim boundary and next steps

The public task-family plan now expands INT-004's oracle, defect and metamorphic families. It remains `not_authored` / `planned_not_runnable`. The next steps are to author the deterministic event contract and reducer independently, build an original fixture/service, create witness/no-op/late-event controls, review the brief blind, then qualify on a pristine runner. No score or browser result is claimed.

Runnable task count remains zero.
