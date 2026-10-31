# POAR-RX Multi-Target Partial-Workflow Benchmark v1

**Date:** 2026-10-31  
**Status:** deterministic research microbenchmark and protocol; not a production transaction engine or external leaderboard result.

## Research question

Can a controller recover a multi-action workflow while preserving target-state invariants, current authority, concurrent external writes, and truthful terminal-state reporting?

The single-action acknowledgement-loss experiment is necessary but insufficient. A workflow may have already committed one branch while another branch is blocked, stale, or ambiguous. Replaying the whole plan can duplicate earlier effects or overwrite valid concurrent changes. Compensation can itself be unauthorized or fail.

The target is not maximum completion at any cost. The controller should prefer an accurately reported partial state or unknown outcome to an unauthorized, destructive, or falsely completed workflow.

## Connection to external work

ParaRecover (Guan, Yin, and Shen, EMNLP 2026; https://aclanthology.org/2026.emnlp-main.975/; preprint https://arxiv.org/abs/2609.12345) evaluates 10,626 DAG-structured instances across 14 error types and two difficulty levels. Its SDE rubric covers Structural Integrity, Diagnostic Reasoning, and Evolutionary Strategy. It is an essential process-level comparator.

The paper also identifies limitations relevant to our design:
- The execution environment is simulated.
- Its workflow abstraction is DAG-structured and does not fully model loops, conditional branching, or asynchronous interactions.
- SDE includes LLM-as-a-judge dimensions, although the authors additionally report judge-free execution metrics.

POAR-RX should complement, not replace, ParaRecover. Adopt its process-level error taxonomy and report diagnostic quality, but score state transition correctness, forbidden effects, authority, recovery and residual world state with deterministic environment predicates and an evaluator separated from the controller.

AgentRx (EMNLP Findings 2026; https://aclanthology.org/2026.findings-emnlp.598/) is complementary for first-critical-failure localization. OSWorld 2.0 is complementary for long-horizon computer-use and checkpoint completion. Use their official evaluators for their own metrics; do not convert unlike scores into POAR-RX points.

## Initial executable slice

The module research/poar-rx-lab/src/workflow.rs models a two-target workflow:

1. First action commits a transition from first-before to first-after.
2. Before/while the second action is attempted, a declared fault may occur.
3. The second action either commits, is rejected, loses its acknowledgement, or encounters a changed precondition.
4. The controller reports verified completion, unverified completion, verified abort, partial state, or unknown outcome.

This first executable slice deliberately serializes the two agent actions (first target, then second target). Concurrent external mutations are injected between them; this is not yet an agent-side parallel DAG executor. Its local world model is both executor and oracle. The benchmark title/path preserves the intended extension toward parallel workflows, but no parallel-execution result is claimed.

### Policies

- blind_workflow_replay: replays both branches following an unacknowledged/failed second action, ignoring pre-state version and current authorization. This is a deliberately unsafe control, not a strong production baseline.
- saga_compensation: uses step acknowledgements and a local compensation policy. It reconciles the ambiguous second branch, but does not read back every workflow postcondition before declaring completion.
- proof_gated_reconcile: reads back both branches, checks the complete goal predicate, preserves unknown outcomes when observation is unavailable, and only compensates after explicit authorization and a matching precondition.

The saga arm is a deliberately bounded conventional comparator. Confirmatory studies must additionally compare a strong, independently implemented saga/orchestrator with documented idempotency, precondition and read-back guarantees. Do not claim superiority over sagas from this toy arm.

### Fault families

- No fault.
- Second action commits; acknowledgement is lost.
- Second action commits; acknowledgement and authoritative read-back are unavailable.
- Permission is revoked before the second action.
- Concurrent writer changes the first target after its initial commit.
- Concurrent writer changes the second target before its expected-version write.
- Second action is definitely rejected.
- Compensation fails after the second action is rejected.

Compensation authorization is independently varied (granted/revoked), for 8 × 2 × 3 = 48 policy/fault/authorization rows.

## Invariants

1. No successful terminal state unless every required target predicate is true.
2. A transport acknowledgement is not by itself proof of a verified workflow postcondition.
3. An unknown action result is not interpreted as non-commit.
4. A permission revocation blocks subsequent unauthorized effects.
5. A concurrent writer's state is not overwritten by reconciliation or compensation.
6. Compensation requires a current authorization grant and a matching state precondition.
7. Failed compensation remains visible as partial state.
8. An uncertain outcome remains unknown when authoritative read-back is unavailable.
9. Any false completion, duplicate logical write, unauthorized effect, or lost concurrent update is independently counted; none is averaged away.

## Metrics

Record, separately:
- Goal predicate completion and verified completion.
- Unverified completion claims and false completion.
- Terminal-state classification (verified complete / verified abort / partial / unknown).
- Duplicate logical writes.
- Unauthorized effects.
- Lost concurrent updates.
- Compensation attempts, success, failure and residual blast radius.
- External writes preserved.
- Read-back count, tool calls, operation count, latency, tokens, CPU/GPU time, memory, energy and monetary cost.

No composite score. Report a safety/correctness/cost Pareto frontier and confidence intervals clustered by task template, tool family and fault family.

## Ablations and external baselines

The next confirmatory design must include:
1. Reactive plan-then-execute.
2. Whole-workflow retry.
3. Conservative stop-on-unknown.
4. Strong saga/compensation controller with idempotency and precondition checks.
5. Generic COG-META / value-of-computation controller.
6. POD-Meta with proof obligations but without recovery journaling.
7. POAR-RX with state reconciliation and proof-gated compensation.
8. Oracle action-selection upper bound on synthetic cases only.

Run mechanism ablations for proof obligations, full-state read-back, authority re-check, compensation gate and semantic-progress stopping.

For a causal comparison, match model/checkpoint, prompts, context, tools, permissions, task seeds, deadlines, retry policy, and total all-in budget. A control that has fewer verification calls but more false completions is not more efficient.

## Known limitations and next gates

This first two-target simulator:
- does not implement durable crash-consistent storage;
- does not model real distributed transactions, real IAM, network partitions, or external APIs;
- does not yet execute agent branches in parallel and does not model multi-agent races, arbitrary DAG/loop/async scheduling;
- uses a public deterministic fault schedule and a visible local oracle;
- does not establish SOTA, production reliability, or general exactly-once semantics.

Next gates:
1. Add crash-point replay and a durable journal with fsync/atomic-write contracts.
2. Add actual multi-target parallel dependency graphs, conditional branches and asynchronous completion.
3. Add hidden, family-held-out fault schedules and protected evaluator separation.
4. Validate the controller against an independently implemented strong saga baseline.
5. Run judge-free state predicate scoring plus optional blinded human/LLM diagnostic annotations; never let LLM-as-judge decide safety-critical ground truth.
6. Publish a contamination card, frozen artifacts, evaluator revision and exact-head run receipts.

## Promotion rule

Promote only when exact-head format, tests and clippy pass; deterministic invariants hold; protected-evaluator controls qualify; and matched ablations demonstrate an advantage without safety regression. A successful microbenchmark is evidence only for the declared simulator contracts.
