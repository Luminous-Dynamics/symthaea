# Computational Engineering Bench (CEB) V0

**Date:** 2026-11-02  
**Status:** Design-only task contract and catalog; no benchmark run or performance claim exists.  
**Machine-readable artifacts:** `benchmarks/ceb/task.schema.json`, `benchmarks/ceb/catalog.v0.json`

## 1. Purpose

CEB measures whether an agent can perform complete computational engineering work: understand a system, diagnose a problem, plan a bounded change, implement it, verify it independently, and produce reproducible evidence.

It is deliberately broader than source-code completion. Its lanes include algorithms, Rust engineering, Nix/build reproducibility, Linux operations, distributed systems, database engineering, cybersecurity, performance, system design and reliability.

CEB is a Symthaea-native evaluation and should complement—not replace—external benchmarks such as SWE-bench Pro V2 and Terminal-Bench 4.0.

## 2. Current status and claim ceiling

The initial catalog contains eight **design-only** task records. They have no runnable fixtures, pinned images/toolchains, reference patches, independent graders, hidden test assets or qualification results.

Therefore:

- They are not executable benchmark tasks yet.
- They must not enter a pass-rate denominator or leaderboard.
- No agent, including Symthaea, has a CEB score.
- A valid JSON record or passing schema validator is not evidence of engineering performance.
- A task becomes `fixture_ready` only after its source snapshot, isolated environment, budgets and independent grader are pinned and validated.
- A task becomes `qualified` only after the grader, reference patch, empty-patch negative control, split custody and isolation controls all pass.

The schema intentionally makes the execution state explicit so planning artifacts cannot be mistaken for benchmark evidence.

## 3. Task contract

Each task is a versioned JSON record governed by `task.schema.json`. Required fields bind:

- Stable task ID, lane, objective and agent-visible prompt.
- Exact source snapshot / fixture URI, revision and SHA-256 tree digest.
- Environment image, toolchain-lock digest, runner profile and network policy.
- Read/write scope, secret exposure, grader visibility and human-approval gates.
- Independent grader identity and digest, reference patch digest, and acceptance checks.
- Wall-clock, turns, tool calls, input/output tokens and cost budget.
- Split partition, sealed status, leakage groups and freshness policy.

The schema accepts null fixture/grader artifacts only while `execution_status=design_only`. A fixture-ready task must use an isolated runner profile and pinned source, environment and grader digests. A qualified task must use a sealed private-holdout or rotating-fresh partition.

### Trust boundaries

The agent must not be able to read or modify hidden tests, reference patches, scoring code or grader credentials. Grader execution must occur in a separate trust domain from the candidate workspace. Task-controlled build scripts, procedural macros, tests and package hooks are untrusted code.

The existing Symthaea sandbox is documented as a dry-run / host-opt-in utility, not a security boundary. CEB must not run third-party or adversarial source on that host runner. See [execution isolation issue #10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064).

## 4. Grader qualification

Every task must pass all of the following before it can be used for a result:

1. **Reference-solution gate:** the adjudicated reference patch passes the full acceptance contract.
2. **Empty-patch gate:** the unchanged starting snapshot fails the task-specific acceptance condition.
3. **Independent regrade:** candidate diffs are re-evaluated from a pristine environment.
4. **No-tampering controls:** deleting tests, editing the grader, modifying caches, hard-coding outputs, changing protected files or fetching hidden solution material must fail.
5. **Isolation controls:** network policy, filesystem boundary, environment/secret removal, process-tree termination and resource limits are tested independently.
6. **Flakiness review:** tests are repeated and unstable checks are repaired or excluded with a recorded reason.
7. **Scope review:** the task instructions and grader measure the same behavior; unrelated pass-to-pass regression tests remain active.

The grader should emit per-check results and a signed/digest-bound evidence receipt. A score is not accepted merely because the agent says its own tests passed.

## 5. Scoring model

Do not compress all engineering quality into one unqualified success bit. Keep a strict resolve gate and publish the following dimensions separately:

| Dimension | Measurement |
|---|---|
| Task resolution | All required independent acceptance checks pass |
| Regression resistance | Existing pass-to-pass and protected invariants remain true |
| Scope discipline | No forbidden or unrelated file changes |
| Security | Negative controls pass; no privilege, secret or isolation violations |
| Evidence quality | Receipt complete, digests correct and verdict reproducible |
| Calibration | Confidence is compared with observed success by claim type |
| Efficiency | Cost, tokens, tool calls, wall-clock and human-review time |
| Maintainability | Readability, API stability, documentation and complexity rubric |

Primary leaderboard metrics:

- Resolve rate (strictly graded tasks / eligible tasks).
- Regression-free resolve rate.
- Security-clean resolve rate.
- Cost per independently accepted task.
- Wall-clock time per independently accepted task.
- Human review minutes per accepted task.
- Calibration error and abstention quality.
- Pareto frontier of resolve rate vs cost vs time.

A task with a critical security or isolation violation is failed regardless of its code tests. A task with incomplete evidence is unqualified, not successful.

## 6. Comparison protocol

Compare four systems under the same task revision and evaluator:

1. Foundation model with a minimal fixed tool harness.
2. The same model with a strong general coding-agent scaffold.
3. The same model with Symthaea's repository twin, HDC/CfC/FEP reasoning, memory and evidence workflow.
4. Symthaea's native generation path without external LLM fallback, where the task is within its declared capability.

Pin model revision, agent scaffold, effort setting, task image, network policy, wall-clock/turn/token/cost budgets and grader. Randomize task order; run stochastic agents repeatedly; report confidence intervals and paired differences.

Measure Symthaea's contribution separately from the foundation model. Do not attribute a stronger model or larger budget to Symthaea's architecture.

## 7. V0 task catalog

The eight initial records in `catalog.v0.json` cover:

| ID | Lane | Task theme | Key evidence |
|---|---|---|---|
| CEB-ALG-001 | Algorithms | Exact streaming top-k under bounded memory | Hidden vectors, invariants and complexity |
| CEB-RUST-001 | Rust engineering | Multi-crate API migration | Hidden callers, compilation and semantic regression |
| CEB-NIX-001 | Nix/build | Reproducible development shell repair | Clean build, pinned inputs and host-independence |
| CEB-LINUX-001 | Linux operations | Service recovery under permissions/resource limits | Service state, logs, scope and recovery invariants |
| CEB-DIST-001 | Distributed systems | Idempotency under retry, crash and replay | Deterministic fault schedule and state-machine properties |
| CEB-SEC-001 | Security | Shell/path injection hardening | Adversarial path and shell-metacharacter controls |
| CEB-PERF-001 | Performance | Measured hot-path optimization | Before/after profiles and semantic equivalence |
| CEB-SYS-001 | Systems design | Requirement-to-evidence traceability | Trace completeness, risk and review rubric |

All eight are `design_only`. The first qualification batch should contain 2–3 independently implemented fixture tasks per lane, not one cherry-picked demonstration.

## 8. Recommended first qualification batch

Start with deterministic, locally reproducible fixtures that do not require a large model or external service:

- Rust: multi-file change, generic API, concurrency/cancellation and property-preserving refactor.
- Nix: locked flake evaluation, package resolution, feature/target mismatch and reproducible build.
- Linux: unit/service diagnosis, permission boundaries, file descriptor exhaustion and rollback.
- Distributed: duplicate delivery, lost acknowledgment, retry storm and crash-recovery.
- Security: path traversal, shell injection, secret inheritance and grader tampering.
- Performance: profiled data structure, allocation reduction and contention hot path.
- Systems design: requirement decomposition, interface contract and safety-case evidence map.

Use synthetic or permissively licensed fixtures initially, with fresh names, hidden variations and rotating seeds. Never reuse the exact Symthaea issues or patches that motivated the benchmark as held-out evaluation tasks; that would contaminate the comparison.

## 9. Statistical and reporting rules

- Freeze task set, grader, model/harness, budget and exclusions before execution.
- Count a task only once in the denominator, even if the agent retries internally.
- Report task-level outcomes, not only the aggregate.
- Use paired runs for model/harness comparisons where possible.
- Bootstrap at the correct independent task or task-family unit; do not treat tests within one task as independent samples.
- Report 95% confidence intervals, sample sizes, costs and missing values.
- Publish every task exclusion and grader correction.
- Maintain separate development, public-validation, private-holdout and rotating-fresh splits.
- Do not train on private holdout tasks or reveal hidden solutions after evaluation.

## 10. Execution plan

1. Validate schema and catalog syntax without claiming task readiness.
2. Build fixture repositories for CEB-ALG-001, CEB-RUST-001 and CEB-SEC-001.
3. Implement independent graders and two-sided gates outside the writable workspace.
4. Establish a real isolated runner before any task-controlled code is executed.
5. Run reference/empty-patch qualification and adversarial tampering controls.
6. Freeze baseline model-agent configurations and budgets.
7. Execute repeated runs only after qualification; publish raw per-task evidence and aggregate uncertainty.
8. Add lanes only when their fixtures and graders meet the same standard.

## 11. External references

- [SWE-bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2): refreshed public task split, two-sided gates and pristine-image regrading.
- [SWE-rebench V2 collection](https://huggingface.co/collections/nebius/swe-rebench-v2): continuously updated task collection; pin the exact dataset revision and leaderboard time window.
- [Terminal-Bench 4.0](https://www.tbench.ai/?version=4.0): 66 frontier terminal tasks; pin the benchmark version.
- [Vals AI Terminal-Bench 4.0 results](https://vals-ai.com/benchmarks/terminal-bench-4): an independent evaluation with model/harness, cost and duration reporting.

**Claim ceiling:** CEB V0 currently defines a task schema and design catalog only. It does not establish Symthaea performance, superiority to commercial agents, security qualification or SOTA.
