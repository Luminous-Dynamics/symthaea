# Computational Engineering Bench (CEB) V0

**Date:** 2026-11-02  
**Status:** Design-only task contract and catalog; no benchmark run or performance claim exists.  
**Machine-readable artifacts:** `benchmarks/ceb/task.schema.json`, `benchmarks/ceb/catalog.v0.json`, `benchmarks/ceb/run-receipt.schema.json`, `benchmarks/ceb/dsse-envelope.schema.json`

## 1. Purpose

CEB measures whether an agent can perform complete computational engineering work: understand a system, diagnose a problem, plan a bounded change, implement it, verify it independently, and produce reproducible evidence.

It is deliberately broader than source-code completion. Its lanes include algorithms, Rust engineering, repository intelligence, Nix/build reproducibility, Linux operations, distributed systems, database engineering, cybersecurity, performance, system design, reliability and benchmark-verifier integrity.

CEB is a Symthaea-native evaluation and should complement—not replace—external benchmarks such as SWE-bench Pro V2 and Terminal-Bench 4.0.

## 2. Current status and claim ceiling

The initial catalog contains ten **design-only** task records. They have no runnable fixtures, pinned images/toolchains, reference patches, independent graders, hidden test assets or qualification results.

Therefore:

- They are not executable benchmark tasks yet.
- They must not enter a pass-rate denominator or leaderboard.
- No agent, including Symthaea, has a CEB score.
- A valid JSON record or passing schema validator is not evidence of engineering performance. The design-only receipt example is not a real run and must not be scored.
- A task becomes `fixture_ready` only after its source snapshot, isolated environment, budgets and independent grader are pinned and validated.
- A task becomes `qualified` only after the grader, reference patch, empty-patch negative control, split custody and isolation controls all pass.

The schema intentionally makes the execution state explicit so planning artifacts cannot be mistaken for benchmark evidence. The design-only receipt sample is intentionally unresolved and simulated; it is a schema fixture, not a benchmark result.

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
8. **Verifier red-team:** attack the reward channel, grader files, test discovery, answer/oracle paths, caches, environment inheritance, network boundaries, symlinks, report serialization and benchmark-version pinning. Include both false-accept and false-reject controls. A valid reference patch must never be rejected solely because it uses an alternate permitted implementation.

The grader should emit per-check results in an in-toto Statement v1, wrapped by a DSSE-signed envelope. The statement binds task, source, candidate patch, environment, grader, checks and resource usage; the envelope authenticates the serialized statement against an independently provisioned trust root. A single grade receipt is not an acceptance: primary and pristine-regrade receipts must bind identical task/source/patch/grader/model/harness inputs, distinct execution identities and independent evidence, with both grades passing. A score is not accepted merely because the agent says its own tests passed. See `docs/research/CEB_VERIFIER_INTEGRITY_PROTOCOL_V0_2026-11-02.md` and issue #10167.

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

Benchmark-verifier correctness is reported on a separate denominator from agent task resolution. Publish false-accept rate (invalid candidate accepted / negative-control attempts), false-reject rate (valid reference or adjudicated equivalent rejected / positive-control attempts), attack-family coverage and fresh-run verdict agreement. Do not merge these into the agent's resolve rate. With zero false accepts in 1,000 independent negative controls, report the one-sided 95% upper confidence bound (approximately 0.3%), not a claim of zero underlying risk. Treat clustered or correlated attacks as such; 1,000 variants of one exploit are not 1,000 independent attack families.

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
| CEB-REPO-001 | Repository intelligence | Construct a typed repository twin and predict change impact | Node/edge gold labels, change-impact precision/recall and incremental invalidation |
| CEB-VER-001 | Benchmark integrity | Audit and harden an adversarial benchmark verifier | Positive/negative controls, exploit resistance and independently reviewed task-level verdicts |

All ten are `design_only`. The receipt and DSSE schemas define evidence shape only; no signing keys, trusted runner identity, executable verifier or accepted receipt has been provisioned. The first qualification batch should contain 2–3 independently implemented fixture tasks per lane, not one cherry-picked demonstration.

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

## 11. Repository-twin evaluation lane

CEB-REPO-001 measures the computational repository twin separately from code-generation success. An initial five-crate synthetic Rust workspace is now committed under `benchmarks/ceb/fixtures/repository-twin-synthetic-v0/`, with package/module/symbol/trait/call/test/feature/target nodes, a fixture-declared gold graph (32 nodes, 37 expected edges) and seven deterministic edit scenarios. It includes public re-exports, generic trait bounds, trait implementations, cross-crate calls, an optional audit feature, a Windows-only module/dependency and integration tests. The fixture is static-only: do not execute repository build scripts, tests or binaries as part of intake.

Gold labels keep syntax-derived facts distinct from compiler/rust-analyzer-resolved facts. The initial fixture explicitly marks its expected semantic edges as not compiler-verified, because no pinned compiler/LSP result has yet been produced. Apply a deterministic edit sequence (rename, move/re-export, trait-impl change, feature gate change, function deletion and test mutation). After each edit, compare incremental invalidation with a full graph rebuild. Score node/edge precision and recall per relation type, change-impact precision/recall for symbols/tests/features/targets, stale-fact rate and provenance coverage.

Proposed targets (not current results): syntax entity P/R ≥0.95; supported compiler-resolved relation P/R ≥0.90; change-impact recall ≥0.95 with precision ≥0.80; zero stale facts after completed invalidation; zero cases where syntax-only facts are labelled compiler-resolved. The gold-label and fixture work is tracked in [issue #10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173).

## 12. External benchmark integrity finding

Terminal-Bench 4.0.0 remains useful as a version-pinned external comparison, but it must not be treated as an unquestioned ground-truth score. Epoch AI's review dated 2026-09-04 labels the release **Flawed** and reports publicly noted scoring defects in 30 of 66 tasks (45.5%), including a directly writable pass signal, answer leakage and false negatives. Terminal-Bench's maintainers say 4.0.0 removed eight tasks and fixed 19, and list verifier improvements such as tamper-resistant graders among the goals for 4.1. These sources describe distinct snapshots and review scopes; retain the exact release, task IDs, grader commit and exclusions with every result.

Accordingly:

- Use Terminal-Bench 4.0.0 as a diagnostic external reference, not a sole success target. Keep the official maintainer leaderboard and independent third-party harness results separate.
- Do not publish a single aggregate as a trustworthy engineering result without task-level verifier review. Where defects are confirmed, report the raw benchmark score and an independently adjudicated clean-subset score separately; never silently delete tasks.
- Add CEB-VER-001 plus rotating verifier attacks. Report false accepts and false rejects separately, by attack family, with one-sided confidence bounds and task-level evidence.
- Prefer a corrected/pinned future release only after independently rechecking its verifier changes; a version bump alone is not proof of validity.

### Current external score context (2026-11-02)

The official Terminal-Bench 4.0 page accessed 2026-11-02 currently renders no populated score rows, although it describes 95% confidence-interval whiskers. A separate Vals AI Mini-SWE-agent evaluation, updated 2026-10-29, reports Opus 5.5 at 65.15%, Sonnet 5.5 at 64.14%, Kimi K3.1 Pro at 63.64%, Gemini 4 Carbon at 63.13%, and GPT-6 Astra at 59.60%. Vals reports three full benchmark runs (avg@3), pass@1 per task, Mini-SWE-agent for all models, and an eight-hour task limit per task; these results are not official maintainer-board scores. Do not pool the two sources or interpret different harnesses as matched comparisons. The Epoch AI defect audit applies to the benchmark release itself, not to a claim that every run exploited a flaw.

## 13. External references

- [SWE-bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2): refreshed public task split, two-sided gates and pristine-image regrading.
- [SWE-rebench V2 collection](https://huggingface.co/collections/nebius/swe-rebench-v2): continuously updated task collection; pin the exact dataset revision and leaderboard time window.
- [Terminal-Bench 4.0](https://www.tbench.ai/?version=4.0): 66 frontier terminal tasks; pin the exact release, audit task-level scoring defects, and separate raw from adjudicated-clean results.
- [Epoch AI review of Terminal-Bench 4.0.0](https://epoch.ai/benchmarks/terminal-bench/review): reports publicly documented scoring defects in 30/66 tasks and labels this release flawed.
- [Terminal-Bench 4.0 maintenance notes](https://www.tbench.ai/news/terminal-bench-4-0): documents task removals/fixes and the 4.1 verifier-hardening direction.
- [Vals AI Terminal-Bench 4.0 results](https://vals.ai/benchmarks/terminal-bench-4): independent Mini-SWE-agent evaluation; keep separate from the official maintainer board.

**Claim ceiling:** CEB V0 currently defines a task schema and design catalog only. It does not establish Symthaea performance, superiority to commercial agents, security qualification or SOTA.


## 14. Existing audit implementation caveat

Two repository-repair audit design documents currently describe Rust binaries under `tools/rust-repair-benchmark-harness/`. A source check on `main` dated 2026-11-02 did not find that manifest or corresponding binary sources; repository code search found only the documentation references. Do not treat those auditors as reusable executable infrastructure until the source is restored and exact-head tests pass. This discrepancy is tracked in [issue #10138](https://github.com/Luminous-Dynamics/symthaea/issues/10138). The benchmark-integrity lane must not inherit unverified implementation claims from design prose.


## 15. Receipt-verifier implementation

CEB task and run statements are now machine-readable in an in-toto Statement v1 + DSSE envelope. `.github/workflows/ceb-contract-validation-v0.yml` validates the Draft 2020-12 schemas, catalog and cross-field receipt invariants with pinned `jsonschema` and negative controls. The workflow has been authored but not run in this conversation. Cryptographic signature verification, trust-root policy and primary/regrade-pair adjudication remain implementation work under [issue #10167](https://github.com/Luminous-Dynamics/symthaea/issues/10167). A schema-valid statement is not proof of an authentic execution.
