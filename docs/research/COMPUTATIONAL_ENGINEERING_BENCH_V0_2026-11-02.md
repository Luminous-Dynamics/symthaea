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

Gold labels keep syntax-derived facts distinct from compiler/rust-analyzer-resolved facts. The initial fixture records a 1-based line number and exact trimmed declaration text for every node. These anchors are source-text provenance only, not complete AST spans or compiler/LSP locations. The initial fixture explicitly marks its expected semantic edges as not compiler-verified, because no pinned compiler/LSP result has yet been produced. Apply a deterministic edit sequence (rename, move/re-export, trait-impl change, feature gate change, function deletion and test mutation). After each edit, compare incremental invalidation with a full graph rebuild. Score node/edge precision and recall per relation type, change-impact precision/recall for symbols/tests/features/targets, stale-fact rate and provenance coverage.

Proposed targets (not current results): syntax entity P/R ≥0.95; supported compiler-resolved relation P/R ≥0.90; change-impact recall ≥0.95 with precision ≥0.80; zero stale facts after completed invalidation; zero cases where syntax-only facts are labelled compiler-resolved. The gold-label and fixture work is tracked in [issue #10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173).

## 12. External benchmark integrity finding

Terminal-Bench 4.0.0 remains useful as a version-pinned external comparison, but it must not be treated as an unquestioned ground-truth score. Epoch AI's review dated 2026-09-04 labels the release **Flawed** and reports publicly noted scoring defects in 30 of 66 tasks (45.5%), including a directly writable pass signal, answer leakage and false negatives. Terminal-Bench's maintainers say 4.0.0 removed eight tasks and fixed 19, and list verifier improvements such as tamper-resistant graders among the goals for 4.1. These sources describe distinct snapshots and review scopes; retain the exact release, task IDs, grader commit and exclusions with every result.

Accordingly:

- Use Terminal-Bench 4.0.0 as a diagnostic external reference, not a sole success target. Keep the official maintainer leaderboard and independent third-party harness results separate.
- Do not publish a single aggregate as a trustworthy engineering result without task-level verifier review. Where defects are confirmed, report the raw benchmark score and an independently adjudicated clean-subset score separately; never silently delete tasks.
- Add CEB-VER-001 plus rotating verifier attacks. Report false accepts and false rejects separately, by attack family, with one-sided confidence bounds and task-level evidence.
- Prefer a corrected/pinned future release only after independently rechecking its verifier changes; a version bump alone is not proof of validity.

### Current external score context (2026-11-03)

Scale's SWE-Bench Pro V2 page reports a 642-task, 11-repository refreshed split after removing 89 invalid tasks. It reports reference-patch/empty-patch two-sided qualification, blind expert resolution of instruction/test contradictions, web-disabled agent execution, and pristine-image regrading. Its displayed frontier scores are Opus 5 + Claude Code 99.40±0.40%, Gemini 4 Argon + mini-swe-agent 99.22±0.45%, Fable 5.1 + Claude Code 99.10±0.50%, and GPT-6 Astra + Codex 96.90±1.10%. The benchmark is highly saturated at the top; a 99% result there alone cannot establish general engineering competence.

The current SWE-rebench leaderboard snapshot displays Fable 5 at 64.5±1.41% ($4.40/problem), Grok 4.5 at 63.8±0.60% ($1.47/problem), Opus 5 at 63.4±1.35% ($3.47/problem), GLM-5.2 at 62.9±1.19% ($1.40/problem), and GPT-5.6 Sol at 62.3±1.83% ($0.85/problem). Its model rows also expose pass@5 and token totals. Treat this as a separate live collection snapshot: record the access date, dataset revision, evaluation window, scaffold and budgets whenever reproducing it. Do not pool it with SWE-Bench Pro V2.

Epoch AI's September 4 review labels Terminal-Bench 4.0.0 flawed and reports publicly documented scoring defects in 30/66 tasks (45.5%). Treat it as an attack-rich diagnostic benchmark with a task-by-task audit, not as an unquestioned leaderboard target. A raw leaderboard score and an independently adjudicated clean-subset score are separate statistics.

These are external reference points, not Symthaea results. No CEB task has been run, no Symthaea-vs-industry comparison has been executed, and no SOTA or superiority claim is justified.

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

CEB task and run statements are machine-readable in an in-toto Statement v1 + DSSE envelope. Each receipt binds exactly three subjects: candidate patch, source tree and result tree. The CEB contract workflow validates Draft 2020-12 schema structure, catalog invariants and cross-field controls with pinned jsonschema; it has not been run in this work session.

A first standalone Rust verifier is now authored in draft PR [#10212](https://github.com/Luminous-Dynamics/symthaea/pull/10212), tracked by [issue #10167](https://github.com/Luminous-Dynamics/symthaea/issues/10167). It includes DSSE Ed25519 verification against an out-of-band pinned trust policy, task-contract hash pinning, check/budget/runtime binding, CEB-TREE-V1 source/result hashing, changed-path and patch-header reconciliation, artifact re-hashing, and primary/regrade independence controls.

The implementation has 18 authored Rust tests and a dedicated Rust build/format/Clippy workflow. They have **not** been run, the Rust toolchain is unavailable in this execution environment, and no Cargo.lock has been generated. The relevant GitHub workflows are queued rather than passed. The verifier cannot establish runner isolation, test semantic correctness, or that the submitted patch transforms the base tree into the result tree. It is not qualified for scored benchmark use. All ten catalog tasks remain design_only.


## 16. Repository fixture and gold-label CI checks

The CEB contract workflow now audits the static repository-twin fixture without compiling or executing it. It checks the fixed fixture revision shape (32 nodes, 37 edges, seven mutation scenarios), unique node and mutation IDs, graph count consistency, all edge endpoints, safe relative source paths, exact one-line source declaration anchors, valid mutation impact references and the explicit absence of compiler-verified claims.

Negative controls inject a dangling edge, duplicate edge, invalid source line, anchor-text mismatch, unsafe path, semantic-authority inflation, forged edge count, unknown mutation impact and duplicate mutation ID. These controls are CI-authored and have not yet been run. They are checks on the gold artifact itself, not evidence that an agent has constructed the graph correctly.

The workflow contains 31 authored negative controls, including result-tree subject digest tampering and a valid-line/wrong-symbol source-anchor attack. It now checks not only that each line exists and matches its anchor text, but also that the anchor text declares the symbol named by the graph node. These checks are authored and have not run. It does not compile the synthetic workspace, execute tests/build scripts, or qualify the runner. The task remains design-only.


## 17. 2026-11-03 verifier and benchmark direction update

The receipt contract now has exactly three in-toto subjects: candidate patch, source tree and result tree. Task contracts declare the CEB-TREE-V1 digest algorithm. The Rust verifier draft independently hashes clean source/result snapshots, checks that receipt changed_paths equal the computed tree delta, and checks that unquoted Git diff file headers name exactly that changed-path set. This is stronger than trusting a runner's claimed result-tree digest, but it is not yet proof that applying the submitted patch to the source produces the result. A safe, independent patch reconstruction/apply stage remains an explicit gate.

### Where to compete, and where not to overfit

- **SWE-Bench Pro V2:** use the locked 642-task protocol as an external software-repair reference. Its top end is near saturation (99.4%); do not optimize solely for that score. Reproduce the same model/harness, task split and limits, and report per-task paired differences. Its pristine regrade and two-sided task gate are practices CEB should inherit.
- **SWE-rebench:** use it to track generalization on a changing task stream and the accuracy/cost frontier. Pin the data snapshot and access date. Its current leading scores around 63–65% are not directly comparable with Pro V2.
- **Terminal-Bench:** use only with independent verifier review and explicit attack-family adjudication. Its 4.0.0 release has a material known defect record; keep official and third-party harnesses separate.
- **CEB:** the differentiator is complete engineering evidence across Rust/Nix/Linux/distributed/security/performance/repository-intelligence work, not only resolving source issues. Require all tasks to pass the same source pin, sealed grader, runner, evidence and regrade contract before a score is published.
- **Repository twin:** measure syntax entities separately from compiler-resolved relations and inferred semantics. Proposed targets remain syntax P/R ≥0.95, supported compiler-resolved relation P/R ≥0.90, impact recall ≥0.95 / precision ≥0.80, and zero stale facts after completed invalidation. Compiler/Rust-Analyzer gold must be independently generated and pinned before using semantic edges as truth.
- **Verifier lane:** aim for zero false accepts in ≥1,000 varied negative controls and zero false rejects in ≥100 independent positive controls, with exact one-sided confidence bounds and attack-family coverage. Those are prospective gates, not results.
- **System-level outcome:** target a Pareto improvement over matched foundation-model baselines in accepted-task rate, cost per independently accepted task, latency and human review time. A reasonable stretch target is at least equal accepted-task quality at 50% lower cost, but it must be earned on paired, frozen tasks; it is not a forecast or an industry-reported benchmark.

### Comparison protocol

For each frozen task family, run (1) a foundation model with a minimal harness, (2) the same model with a strong general coding scaffold, (3) that same model plus Symthaea repository intelligence/memory/evidence workflow, and (4) Symthaea native generation where applicable. Match model revision, prompt, tools, context, network, budgets, runner image and grader. Randomize task order and run repeated trials. Report paired differences with task-family confidence intervals, pass@1 and pass@k separately, cost and token distributions, timeouts, abstentions, safety failures, false accepts/rejects, and human review effort.

Do not claim Symthaea caused a gain unless the same underlying model/scaffold without Symthaea is a matched control. Do not compare Symthaea's native path to a larger hosted model and attribute the entire difference to architecture. Report the architecture's incremental contribution separately.

**Current claim ceiling:** schemas, fixture files, verifier source and authored tests are research artifacts. The Rust tests and schema workflow have not been run, no runner has been qualified, the ten catalog tasks remain design_only, and no performance result exists.


## 18. Repository-intelligence industry baselines (2026-11-03)

CEB-REPO-001 should compare against adjacent production-grade code-intelligence systems, not only against a hand-written gold graph:

- **rust-analyzer** is a direct semantic baseline for Rust. Its architecture models the project as source inputs plus a crate graph containing dependencies and cfg flags, then derives a resolved semantic model with references and types. It supports small incremental input deltas and immutable Analysis snapshots. Pin the rust-analyzer binary/revision, Cargo metadata, toolchain, target and feature configuration used for every gold capture.
- **SCIP 0.10.0** is a language-agnostic indexing protocol for symbol definitions/references/implementations, with a Protobuf schema, Rust bindings and Rust indexer ecosystem. CEB should import/export SCIP and report per-relation coverage and provenance rather than inventing an incompatible symbol interchange format.
- **CodeQL** is a separate security/data-flow baseline. GitHub documents Rust support; CodeQL 2.27.2 (2026-10-09) added Rust extractor and data-flow improvements. Use its alert/data-flow results for security-relevant relations, but do not mistake CodeQL's security graph for a complete semantic repository twin.
- **Cargo metadata** is a manifest/dependency baseline; syntax-only parsers such as Tree-sitter provide a syntax extraction baseline.

A competitive repository twin must combine these sources without merging their authority. Every fact should retain an exact source revision, path/span, cfg/feature/target world, indexer/compiler version and evidence artifact. Suggested authority levels are syntax-declared, manifest-declared, rust-analyzer-resolved, rustc-confirmed, runtime-observed and human-adjudicated. No fact may be promoted to a stronger authority simply because multiple weaker sources agree.

**Paradigm shift:** make the graph temporal and evidence-carrying. Every repository edit produces a graph delta and an invalidation closure. The benchmark compares the incremental graph against a full rebuild after every mutation, including a fresh semantic re-index. This directly measures stale knowledge, invalidation underreach/overreach and computational cost, not just one-shot graph extraction.

Add these metrics to CEB-REPO-001:

- Relation-specific node/edge P/R against pinned rust-analyzer and SCIP outputs.
- cfg/feature/target-world coverage and false activation of mutually exclusive features.
- Change-impact P/R across symbols, tests, public re-exports, dependencies, features and target gates.
- Stale-fact rate and stale-fact lifetime after rename, move, trait-bound, feature or target mutations.
- Incremental invalidation cost as a fraction of a full graph rebuild, plus exact post-update equivalence.
- Provenance coverage and authority-calibration error: every resolved/compiler claim must point to exact toolchain and source inputs.

Prospective targets remain syntax P/R ≥0.95, supported semantic relation P/R ≥0.90, impact recall ≥0.95 with precision ≥0.80, zero stale facts after completed invalidation, and zero authority inflation. These are internal targets, not current results or industry standards.

The current fixture remains design-only. Issue #10064 must qualify the compiler/LSP execution environment before any build script, compiler or language server is launched on untrusted fixture content.
