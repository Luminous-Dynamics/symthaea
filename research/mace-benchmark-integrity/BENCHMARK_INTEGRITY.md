# MACE benchmark-integrity protocol v0.1

Status: proposed research protocol, not an independently qualified benchmark or certification.
Date: 2026-11-13.

## Thesis

A benchmark score is a claim about a complete evaluation pipeline, not just the agent: task specification, initial state, agent interface, budgets, environment, grader, aggregation and report. MACE should qualify the evaluator and its evidence chain with the same rigor it applies to agent actions.

This matters in practice. Epoch AI's September 28, 2026 OSWorld 2.1 review audited 50 of 108 tasks and found errors in 36; the review estimated potential impact on up to 50.4% of the maximum score in that audited sample, including false positives and false negatives. τ³-Bench maintainers describe correcting more than 50 airline/retail tasks for incorrect expected actions, ambiguity, impossible constraints and missing fallback behavior. The correct conclusion is not that benchmarks are useless; it is that versioned grader/task qualification is required.

SWE-Bench Pro V2 shows several useful practices: two-sided reference-patch/empty-patch gates, pristine-image re-grading, task instruction corrections, and published probe and re-grade artifacts. Its maintainers also disclose residual verifier trust boundaries. MACE should adopt such checks and independently audit the environment and grader rather than treating any one benchmark as ground truth.

## Benchmark run manifest

Every run should pin:

- benchmark name, exact release/commit, split, task IDs, task data digest and provenance;
- exact instructions, initial state, environment image, dependencies and allowed interfaces;
- model/provider/version, agent scaffold commit, prompts/policy digest, memory, tool schemas, sampling parameters and retry policy;
- time, token, cost, tool-call, network and concurrency budgets;
- grader implementation/version, hidden test/rubric digests, aggregation logic and judge model/prompt;
- independent task-state oracle and OS/API/filesystem/network event-ground-truth source;
- raw per-task trajectory, tool calls, side-effect receipts, grader stdout/stderr and final state;
- hardware, OS, runtime/toolchain, exact command and seed;
- exclusions, grader/task mutations, adjudication evidence and report digest.

The run must distinguish `task_result`, `effect_assurance_result` and `benchmark_integrity_result`. These are separate axes; do not collapse them into a single weighted trust score.

## Two-sided and metamorphic grader gates

For each task or task family, build:

1. **Positive control:** an expert/reference solution must pass the task grader and the independent environment oracle.
2. **Negative control:** a no-op or deliberately incorrect solution must fail.
3. **Regression mutation:** inject a controlled regression after an apparently successful outcome; the grader must detect it.
4. **Representation twin:** vary an irrelevant output representation while preserving the actual required state; the grader must still recognize correctness.
5. **Authorization/effect mutation:** preserve the final visible state while changing the actor, target, authority, idempotency or effect history. Task utility may pass, but effect assurance must fail or be indeterminate.
6. **Condition-removed twin:** for each negative security/evaluator vector, remove only the injected defect and require the expected positive result. This shows the check still executes.

Do not assume a hidden-test grader is trustworthy because the gold patch passes. Verify that it fails on no-op, catches known regressions, resists grader tampering and agrees with a separately maintained final-state oracle.

## Evaluator attack corpus

- Agent modifies grader/test files, dependency cache, result artifacts, task state or reference patch.
- Hidden network access or code-host lookup leaks the fixing commit or answer key.
- Test harness reports successful dispatch when no downstream effect occurred, or omits a real side effect.
- Task specification is ambiguous, impossible, internally contradictory or lacks a recovery path.
- Grader has false-positive and false-negative cases, partial-credit loopholes or reward-hacking pathways.
- A retry creates a second effect while the benchmark counts only one logical tool call.
- Final state is correct but the agent achieved it via an unauthorized actor, target or intermediate mutation.
- A failed task is relabeled as a benchmark/setup failure without preserving raw evidence.

## Metrics and target gates

- Task validity and adjudicated ambiguity/impossibility rate.
- Grader false-positive/false-negative rate with confidence intervals and representative examples.
- Reference-positive and no-op-negative gate pass rate.
- Hidden-network, sandbox-escape and grader-tampering acceptance (target: zero accepted in registered fixtures).
- Task-state oracle agreement and effect-observer precision, recall, attribution, ordering and loss.
- Share of reported leaderboard score supported by independently rerunnable raw artifacts.
- Exact reproduction rate across independently operated runners and graders.
- Utility, effect-assurance, benchmark-integrity and overhead reported separately.

Targets: 100% of registered positive, negative and condition-removed fixtures produce their expected typed results; no silent omission of mandatory checks; 100% of task sets pass the two-sided control before leaderboard use; independent rerun agreement on all frozen fixtures. These are future gates, not achieved results.

## Benchmark portfolio and caveats

- `τ³-bench`: stateful tools, domain rules, and pass^k reliability. Pin v1.0.1 or later; the July 2026 banking-knowledge grading fix makes pre/post scores non-comparable for the affected domain.
- `ToolSandbox`: stateful tool execution, dependencies and milestones. Pin exact release and simulator state.
- `AgentDojo`: indirect prompt injection and utility/security trade-offs. Include tool-output mutation and external-effect ground truth, not just policy text.
- `OSWorld 2.1`: long-horizon desktop tasks. Do not use its aggregate as an unqualified headline while the independent September 2026 review's task/grader findings remain unresolved; build an audited, task-pinned subset first.
- `SWE-Bench Pro V2`: long-horizon software-engineering task utility. Use the 642-task 11-repository V2 split and its pristine re-grade artifacts; keep each model/harness/budget configuration distinct. The public leaderboard currently reports different harnesses and inference configurations, so model rank is not a matched causal comparison.

MACE is not competing with these agents on task-solving score. MACE adds an independently verifiable assurance plane around comparable task suites. The right comparison is paired runs with the same model, prompts, tools, budgets and environment: uninstrumented baseline; telemetry-only; and full effect-plan/broker/observer/effect-origin/witness path.

## Sources

- IETF Internet-Draft, [Agent Accountability: Composition and Conformance](https://datatracker.ietf.org/doc/html/draft-mih-sato-agent-accountability-composition-02), 2 November 2026; work in progress, not an RFC.
- Epoch AI, [OSWorld 2.1 benchmark review](https://epoch.ai/benchmarks/osworld-2-1/review), 28 September 2026.
- τ³-Bench maintainers, [Fixing Airline + Retail](https://taubench.com/blog/tau3-task-fixes.html), February 2026.
- τ-bench [v1.0.1 release](https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.1), July 2026.
- Scale AI, [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), updated 22 September 2026.
