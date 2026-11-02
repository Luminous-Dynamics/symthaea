# Symthaea Computational Intelligence and Engineering Program

**Date:** 2026-11-02  
**Status:** Research and implementation proposal  
**Scope:** Additive programme for computer science, software engineering, coding agents, and systems engineering. This document complements (does not replace) `docs/CODE_ABILITY_IMPROVEMENT_PLAN.md` and `docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md`.

## Executive thesis

Symthaea should become a **computational engineer**: a system that can build and maintain an executable model of a software or technical system, reason about possible changes, perform controlled interventions, measure the result, and attach reproducible evidence to every important claim.

This is a larger objective than code generation. It includes computer science knowledge, repository comprehension, programming-language semantics, architecture, debugging, operating systems, distributed systems, security, reliability, requirements engineering, verification, and lifecycle operations.

The strategic differentiator should not be “more code emitted per minute.” It should be **more correct engineering outcomes per unit of time, compute, and human review**, with fewer unsafe changes and stronger evidence.

A language model may propose a change. Symthaea should own the model of the codebase, the constraints, the experiment, the evidence, the confidence estimate, and the decision to accept, reject, defer, or ask for review.

## 1. What already exists

A first source-level review of the public repository found substantial foundations. This is not a greenfield coding-agent project.

| Existing component | Current role | Important next step |
|---|---|---|
| `src/language/code_parser.rs` | Language-neutral tree-sitter parser contract; AST entities, spans, relations and diagnostics | Preserve compiler-resolved types and semantic facts separately from syntax-level guesses |
| `src/language/repo_map.rs` | Repository scan, exact symbol spans/snippets, HDC semantic retrieval, compile-error context | Add a versioned, incremental, typed repository graph with explicit invalidation |
| `src/language/code_orchestrator.rs` | Multi-backend synthesis, verification, repository context, LSP and experience hooks | Make task contracts, budgets, provenance and acceptance policy first-class |
| `src/language/rust_lsp.rs` | JSON-RPC client for rust-analyzer | Use semantic queries for types, references, implementations and rename safety; record server/version |
| `src/infrastructure/sandbox.rs` + `src/language/code_executor.rs` | Dry-run and explicit opt-in host command execution | Make the trust mode explicit; add real OS isolation before running untrusted repositories |
| `src/coding_agent/test_generation.rs` | Generates example, boundary and signature-derived checks | Distinguish smoke checks from behavioral oracles; add property, mutation, differential and fuzz validation |
| `examples/coding_agent_benchmark.rs` | 55-task coding-agent suite with name, compilation and assertion layers | Retain as a fast regression suite; add hidden, repo-scale and negative-control evaluations |
| `examples/refactoring_benchmark.rs` | Multi-file refactoring task fixtures | Wire it to the agent and independent compiler/test grader; it currently validates task definitions only |
| `crates/domains/symthaea-broca/src/compiler_trainer.rs` | Nix compiler-feedback training prototype | Audit the reward path: on compiler failure it compares `code` with itself, and its loop does not produce a changed candidate |
| `docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md` | Simulation, digital twin, formal safety and physical engineering roadmap | Connect software and physical-system evidence through shared requirements/provenance without conflating their solvers |

These components show breadth, but file presence is not proof of end-to-end capability. The next review must establish which paths are active under the relevant features, which are exercised in integration tests, which are placeholders, and which are only documented.

### Source-level audit findings to turn into tests

1. **Compiler-grounded reward:** in `CompilerGroundedTrainer::train_step`, the failure path calls `self.scorer.score(&code, &code)`. Self-comparison cannot tell whether a rejected candidate is correct or improved. The same immutable candidate is rechecked on each iteration. Either implement a real repair-candidate interface and score the new candidate against an external acceptance contract, or label this component validation-only and award no correctness reward on failure. A narrow validation-only remediation is proposed in [draft PR #10071](https://github.com/Luminous-Dynamics/symthaea/pull/10071); it is not merged or Cargo-tested.
2. **Test-oracle quality:** signature-derived tests can establish that code parses, has a type-compatible call shape, or does not panic for selected inputs. They cannot, by themselves, establish that the function implements the requested semantics. The current generator emits a comment-only `Option<T>` test that passes without calling the candidate, and its `bool`/`Result` checks assume one `Default::default()` argument irrespective of the actual signature. Mark each generated test by oracle provenance and strength; unsupported inputs must produce no passing test. See [issue #10068](https://github.com/Luminous-Dynamics/symthaea/issues/10068).
3. **Task-level correctness:** keep name presence, formatting, compilation, smoke tests, behavioral tests, hidden tests, mutation score, and human review as separate metrics. Never collapse these into one “correct” bit without retaining the underlying evidence.
4. **Benchmark status:** examples that only validate task definitions or compile a generated snippet must be labelled as such. A benchmark is end-to-end only when the agent must inspect an environment, change it, and pass an independent grader.

## 2. The target architecture: a computational digital twin

Build one connected loop around existing modules, rather than a second disconnected agent.

```text
Engineering request + acceptance criteria
                 |
                 v
        Task / requirements contract
                 |
                 v
     Versioned repository & system twin
   (symbols, types, calls, builds, tests,
    dependencies, runtime, constraints)
                 |
                 v
       Planner + uncertainty estimate
    (alternatives, risks, expected evidence)
                 |
                 v
        Isolated execution environment
                 |
                 v
     Observe compiler, tests, traces,
     resource use, and external tools
                 |
                 v
        Independent verification
   (hidden tests, properties, mutation,
    static analysis, proofs, review gates)
                 |
                 v
       Evidence-carrying change receipt
                 |
          +------+------+
          |             |
        accept       reject / defer
          |             |
          v             v
   reviewed merge   diagnosis + lesson
                         |
                         v
                gated learning memory
```

### 2.1 Repository twin

A repository should be represented as a typed, versioned graph—not just a bag of source chunks.

Minimum node types:

- Repository, workspace, package/crate, module, file and build target.
- Function, method, type, trait, implementation, macro, constant and configuration option.
- Test, benchmark, feature flag, binary, service and deployment unit.
- Dependency, external tool, runtime process, API and persistent data schema.
- Requirement, invariant, hazard, proof obligation and evidence artifact.

Minimum edge types:

- Defines, imports, calls, implements, reads, writes, owns, borrows and constrains.
- Built-by, tested-by, deployed-as, configured-by and depends-on.
- Satisfies, violates, mitigates, proves, observes and supersedes.

Each graph snapshot must be tied to an exact Git tree/commit and relevant toolchain/configuration digests. Incremental indexing must invalidate all affected facts after edits, feature changes, dependency changes or generated-code changes. HDC is appropriate for semantic recall and analogical retrieval; compiler and language-server facts remain authoritative for types, resolution and source locations.

### 2.2 Language and compiler intelligence

Prioritize deep semantic support for Rust and Nix first, because they are central to this workspace. Add Go, C/C++, Python, SQL, Bash and TypeScript through shared parser and tool adapters as the benchmark evidence justifies them.

The language model should distinguish:

- Concrete syntax and AST structure.
- Name resolution, types, traits/interfaces and generic constraints.
- Ownership, borrowing, lifetimes, aliasing and mutation.
- Control flow, data flow, effects, error propagation and concurrency.
- Build configuration, conditional compilation and target platform.
- Runtime behavior, memory/resource costs and failure modes.

Do not try to recreate rustc or rust-analyzer inside Symthaea. Integrate them as trusted semantic oracles and store their versioned outputs with provenance. The same principle applies to other compilers, interpreters, package managers and static analyzers.

### 2.3 Systems model

For software, model process boundaries, threads/tasks, queues, files, sockets, memory, permissions, services, network dependencies, storage and failure domains. Ingest structured logs, spans, metrics, profiles and selected system-call traces.

The system model should answer questions such as:

- What changes when this dependency is upgraded?
- Which services, APIs and data schemas are affected by this edit?
- What is the likely source of this latency or resource regression?
- What happens if this process, disk, network path or peer fails?
- Which invariants must remain true during deployment and recovery?

Use existing operating-system and observability tools as evidence sources. A predicted trace is a hypothesis; an observed trace is evidence. Do not confuse the two.

### 2.4 Systems engineering model

Extend the engineering roadmap's requirement and proof-obligation approach to software delivery:

`stakeholder need → requirement → architecture → component → implementation → test → evidence → operational monitor`

Support explicit constraints, interfaces, trade studies, failure-mode analysis, threat models, verification matrices, change-impact analysis, architecture decision records, rollback plans and lifecycle ownership.

Software and physical engineering can share requirement traceability, uncertainty, provenance and safety-case abstractions. They should not share unvalidated physical models or assume that a successful software test proves physical-system safety.

## 3. Paradigm shifts worth pursuing

### A. From code completion to predictive system modelling

Before editing, Symthaea builds a hypothesis about the current system. It predicts the blast radius, expected behavior and likely failure modes. After editing, it compares predictions with compiler, test, trace and performance evidence. Prediction error becomes a debugging signal.

**Research question:** Does a repository twin improve issue resolution, change-impact recall and regression prevention compared with retrieval-only context?

### B. From plausible patches to evidence-carrying patches

Every accepted patch should carry a machine-readable receipt containing:

- Base and head commit/tree digests.
- Changed-file list and patch digest.
- Model/toolchain/agent versions and execution environment digest.
- Task contract, constraints and acceptance tests.
- Exact commands, exit statuses, logs and test results.
- Static-analysis, property, mutation, fuzz or proof results where applicable.
- Known limitations, confidence and required human approvals.

A reviewer should be able to reproduce the verdict from a clean checkout. A green self-reported status is not sufficient.

### C. From test generation to test-oracle engineering

Generate tests from requirements, specifications, independent reference implementations, invariants and differential behavior—not merely from the implementation's own signature or comments.

Use several complementary techniques:

- Example and boundary tests for quick feedback.
- Property-based and metamorphic tests for families of inputs.
- Differential tests against a trusted implementation where one exists.
- Mutation testing to establish whether tests detect plausible faults.
- Fuzzing and sanitizers for parsing, unsafe code and protocol boundaries.
- Formal proofs/model checking for small, critical invariants.
- Hidden acceptance tests and independent grading for agent benchmarks.

Keep test-generation and test-grading authorities separate. Mutating tests, disabling checks or changing the grader must be an explicit benchmark failure.

### D. From autocomplete retries to causal debugging

Maintain a fault-hypothesis set. Choose the next experiment by expected information gain, cost and risk. Make one controlled intervention where possible, observe the outcome, and update the ranking. This is stronger than repeatedly rewriting code from the same error message.

### E. From one confidence score to calibrated, risk-sensitive autonomy

Confidence must be calibrated against actual outcomes and separated by claim type: syntax, type correctness, behavioral correctness, security, performance and operational safety.

Low-risk changes can be automated under narrow policies. Destructive operations, privilege changes, cryptographic changes, migrations and safety-critical deployments should require stronger evidence and human approval. The system must be able to abstain.

### F. From codebase-specific tricks to transferable engineering knowledge

Store lessons as typed, provenance-linked patterns: precondition, context, attempted intervention, observed result, applicable versions and known counterexamples. Promote lessons only after replay against held-out tasks. A successful patch is not automatically a universally valid rule.

## 4. Computer-science knowledge programme

Build a structured curriculum with both conceptual knowledge and executable exercises.

| Area | What Symthaea should understand | Executable demonstrations |
|---|---|---|
| Foundations | Logic, sets, relations, graphs, probability, information theory and computability | Prove small properties; identify impossible/undecidable requests |
| Algorithms | Complexity, amortized analysis, search, sorting, graph algorithms, dynamic programming and approximation | Correctness tests plus scaling curves |
| Data structures | Trees, heaps, hash tables, indexes, tries, caches, persistent and concurrent structures | Workload-driven correctness and performance comparisons |
| Programming languages | Grammars, type systems, generics, effects, ownership, semantics, interpreters and compilers | Parse, type-check, transform and explain small programs |
| Operating systems | Processes, scheduling, virtual memory, filesystems, permissions, syscalls and isolation | Diagnose services, permissions, resource exhaustion and filesystem failures |
| Architecture | CPU/GPU execution, cache hierarchy, SIMD, NUMA, memory models and accelerators | Profile before/after changes on named hardware |
| Concurrency | Threads, async runtimes, atomics, locks, channels, deadlocks and cancellation | Stress tests, race detection and liveness properties |
| Distributed systems | Partial failure, consistency, replication, consensus, partitioning, time and idempotency | Fault injection, partition tests and recovery verification |
| Databases | Transactions, isolation, indexes, query plans, schema evolution and durability | Migration, crash-recovery and query-plan tasks |
| Networks | TCP/IP, DNS, routing, TLS, congestion, protocols and observability | Packet/trace analysis and controlled outage exercises |
| Security | Threat modelling, memory safety, access control, cryptography, supply-chain security and secure defaults | Exploit regression, negative controls and dependency audits |
| Reliability | SLOs, capacity, backpressure, graceful degradation, disaster recovery and incident response | Failure injection and recovery-time measurement |
| Software engineering | Requirements, architecture, API design, review, testing, versioning and maintenance | Multi-file changes with acceptance and regression suites |
| Formal methods | Invariants, refinement, temporal logic, SMT, model checking and proof construction | Verified algorithms and state-machine models |

Learning should be organized as a concept graph with prerequisites, examples, exercises, counterexamples, references, confidence and evidence. Do not reward memorized definitions as mastery. A concept is mastered when Symthaea can apply it, recognize when it does not apply, and predict the result of an experiment.

## 5. Benchmark programme

### 5.1 External benchmarks

| Benchmark | What it tests | How to use it |
|---|---|---|
| [SWE-bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) | Long-horizon, multi-file professional repository tasks | Primary public repository-repair comparison; preserve the exact dataset and evaluation protocol |
| [SWE-rebench](https://swe-rebench.com/) | Fresh interactive issues and changing task distribution | Repeated decontamination-oriented evaluation and training-data exposure checks |
| [Terminal-Bench 4.0](https://www.tbench.ai/) | Current terminal work: software, tools and system configuration | Pin the exact release and model-agent combination; use 2.0 only for historical, version-matched comparisons |
| [CRUST-bench](https://crust-bench.github.io/) | C-to-Rust translation and repair | Narrow language-semantics and unsafe-to-safe translation probe |
| HumanEval / MBPP / BigCodeBench | Function-level code synthesis | Smoke/regression tests only; never the headline engineering score |

Scale's September 22, 2026 V2 refresh describes 642 tasks across 11 repositories, a two-sided gate (reference patch passes; empty patch fails), pristine-image regrading, and a locked agent phase with web access disabled. On the public V2 leaderboard as accessed 2026-11-02, the leading listed result is 99.40% ±0.40 and GPT-6-Astra (Codex) is listed at 96.90% ±1.10. These are specific model-agent/effort configurations, not a generic base-model score; the page notes that most runs have uncapped cost and a 250-turn limit. Replicate the precise task split and budget before claiming parity or superiority. Its methodology page also acknowledges residual trust boundaries around verifier relays and patch-controlled files, reinforcing the need for an independently isolated grader in our own benchmark. Do not treat SWE-bench Verified as the sole or primary measure. The repository's `benches/BENCHMARK_GUIDE.md` still lists it among external suites; update that registry/documentation to mark it historical and add Pro plus fresh-task evaluation. OpenAI's February 23, 2026 audit reported substantial flawed tests in the audited difficult subset and evidence of training exposure, and recommended SWE-bench Pro instead. SWE-rebench provides a complementary continuously refreshed task pipeline. SWE-bench Pro V2's September 22, 2026 update describes a two-sided task gate (reference patch passes, empty patch fails), pristine-image regrading and a locked agent phase with web access disabled. Those are useful evaluation design principles to adopt internally.

Terminal-Bench's official site now surfaces version 4.0. Epoch AI explains that scores are for model-agent combinations and are generally averaged over repeated runs; therefore, compare the same model, agent scaffold, effort setting, task version and budget. The 84.7% Terminal-Bench 2.0 result reported in April 2026 is historical only and must not be compared with 4.0. The current 4.0 leaderboard page reports 95% confidence intervals but did not expose populated scores when this research was conducted. Establish a reproducible Symthaea baseline first; then target at least +10 percentage points over a matched same-model/same-budget baseline on non-saturated task lanes, followed by top-quartile performance on the frozen version. Do not set a raw 4.0 SOTA claim until independently rerun.

### 5.2 Build a Symthaea-native benchmark: Computational Engineering Bench (CEB)

External benchmarks do not cover enough of our target system. Build a private, versioned task suite with public task specifications and held-back solutions/tests.

| Lane | Example tasks | Independent acceptance |
|---|---|---|
| Algorithmic CS | Implement an algorithm, prove complexity, handle edge cases | Hidden vectors, properties, complexity/scaling bounds |
| Rust engineering | Fix borrow/type/concurrency bugs; add a public API; refactor a crate | Locked build, hidden tests, clippy, mutation and regression checks |
| Repository repair | Resolve issue across multiple crates, features and tests | Fresh checkout, independent tests, unchanged protected files |
| Build/toolchain | Repair Cargo/Nix dependency or feature-resolution failures | Hermetic build, exact toolchain digest, reproducible output |
| Linux operations | Diagnose service failure, permissions, storage pressure, networking or resource leaks | Observable final state plus security and recovery invariants |
| Distributed systems | Handle retries, duplicate messages, partitions, stale state and crash recovery | Deterministic fault injection and state-machine properties |
| Security engineering | Fix a vulnerability without weakening the control | Exploit regression, static analysis, negative controls, dependency provenance |
| Systems design | Translate needs into requirements, interfaces, trade studies and deployment plans | Traceability coverage, expert rubric and scenario consistency |
| Reliability engineering | Investigate a synthetic incident, identify root cause, implement and verify mitigation | Hidden fault labels, recovery tests and SLO impact |
| Performance engineering | Optimize a real bottleneck without changing semantics | Repeated profiles, confidence intervals, correctness equivalence and cost |

Every task must declare difficulty, required tools, allowed network access, time/token/cost budget, initial environment digest, acceptance contract, scoring rubric, hidden tests and known ambiguities.

### 5.3 Grader integrity

The grader must be outside the agent's writable workspace. Re-grade every patch on a pristine environment. Use a two-sided gate: reference solution passes, empty patch fails. Add tests that detect test deletion, grader modification, dependency-cache tampering, hard-coded output, network retrieval of the gold patch, and edits outside the permitted scope.

Use negative controls and adversarial tasks. Publish task exclusions and grader failures; do not silently remove hard cases. Keep a private holdout and rotate fresh tasks to reduce contamination.

## 6. Metrics and initial targets

Report confidence intervals and per-lane distributions, not only one aggregate percentage.

| Dimension | Metric | Initial target / guardrail |
|---|---|---|
| End-to-end task ability | Pass@1 and pass@k on repo-level tasks | Establish a frozen baseline first; then target +10 percentage points over the same foundation model without Symthaea's added reasoning/memory |
| Terminal ability | Terminal-Bench 4.0 resolution rate | Establish a repeated-run, matched model-agent baseline; target ≥+10 pp on non-saturated lanes, then top-quartile on the frozen version |
| Correctness | Independent hidden-test pass rate | No accepted change without independent acceptance evidence |
| Grader integrity | False acceptance on deliberately invalid patches | Zero false accepts on a 1,000-case negative-control suite; report the one-sided confidence bound |
| Regression resistance | Regressions introduced per accepted patch | Trend down; any critical regression blocks promotion |
| Repository understanding | Symbol/type/reference retrieval Recall@10 and stale-index rate | ≥90% Recall@10 on a curated gold graph; zero stale facts after a completed index invalidation cycle |
| Test effectiveness | Mutation score and property/fuzz defect detection | Establish baselines per crate; require measurable improvement before promoting generated tests |
| Calibration | Brier score / expected calibration error by claim type | ECE ≤0.05 on a sufficiently sized held-out calibration set |
| Efficiency | Cost, wall time, tokens, tool calls and compiler invocations per accepted task | Compare at matched quality; optimize cost only after correctness is stable |
| Reproducibility | Re-run agreement from a clean environment | 100% of published results carry the complete environment and evidence receipt |
| Safety | Unauthorized writes, secrets exposure, network escape and grader tampering | Zero tolerated in the controlled evaluation environment |

The 70% and 85% Terminal-Bench values are programme milestones, not a claim that Symthaea currently approaches them. The 85% frontier target is deliberately set just above the 84.7% published reference; beating it requires a same-version, same-protocol, independently reproducible result with uncertainty reported.

For SWE-bench Pro V2 and SWE-rebench, use a paired comparison against the same base model and tool budget. A universal percentage target is less meaningful than a measured improvement over the identical model/harness baseline because benchmark versions, task sets, agent scaffolds and inference budgets differ.

## 7. Fair comparison with industry

Separate three sources of capability:

1. **Foundation model:** the underlying model's language and reasoning abilities.
2. **Agent harness:** planning, memory, tools, execution, retry policy and context management.
3. **Symthaea-specific contribution:** repository/system models, HDC/CfC retrieval and prediction, epistemic calibration, causal experiment selection, evidence receipts and safety policy.

Run at least these conditions:

- Base model with a minimal, fixed tool harness.
- The same model with a leading coding-agent scaffold.
- The same model with Symthaea's engineering loop.
- Symthaea's native generation path where it can operate without an LLM fallback.

Keep the model version, task, initial image, tool permissions, time/token/cost budget and grader fixed. Randomize task order. Repeat stochastic runs. Report paired deltas and confidence intervals.

Compare with leading coding agents available on the evaluation date, but do not attribute a foundation-model improvement to Symthaea. Also compare on dimensions commercial coding agents often under-report: calibrated abstention, reproducible evidence, no-test-tampering, exact repository impact, deployment safety, local/offline operation and cost per independently accepted patch.

## 8. Security and execution policy

A capable systems engineer must also be difficult to trick into damaging the system.

- Default to ephemeral, non-root, resource-bounded execution with a read-only source mount and a separate writable patch area.
- Disable network access by default; allow only explicitly declared endpoints for tasks that require it.
- Do not expose signing keys, cloud credentials, production secrets or user data to code-generation processes.
- Treat repository content, issue text, build scripts, compiler output and dependency metadata as untrusted data, not instructions.
- Keep the grader, baseline and hidden tests outside the agent's write boundary.
- Record every command, file write, network attempt and privilege transition.
- Use explicit approval gates for destructive actions, privilege changes, cryptographic changes, schema migrations and deployment.
- A failed safety check is a failed task, even if the application tests pass.

Use Nix for repeatable environment definitions where appropriate, but do not mistake a reproducible build for a security sandbox. The host isolation boundary must be tested independently.

## 9. Execution sequence: work that does not depend on runners

### Workstream 0 — Audit and truth baseline

1. Inventory all code-generation, code-understanding, repository-map, execution, verification, learning and benchmark entry points.
2. Label each feature as active, compiled-only, experimental, placeholder, disabled or unverified.
3. Trace the exact end-to-end path from user task to accepted patch.
4. Add regression tests for the self-comparison reward path and for weak/generated test-oracle classifications.
5. Classify execution honestly as simulation, trusted host execution or isolated execution; track the isolation work in [issue #10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064).
6. Produce a signed-off baseline report with exact commit, commands, tool versions and missing evidence.

**Exit:** no capability claim depends only on a file existing or a task being accepted by a weak local check.

### Workstream 1 — Benchmark trust layer

1. Refactor the existing 55-task suite into clearly separated smoke, behavioral and repository lanes.
2. Add task manifests, environment digests, hidden grader boundaries and negative controls.
3. Verify the grader itself with reference-patch/empty-patch gates.
4. Add a same-model baseline protocol and cost/latency accounting.
5. Add private original Rust/Nix/Linux tasks and a rotating fresh-task lane.

**Exit:** an independent evaluator can reproduce and audit every reported result.

### Workstream 2 — Repository twin v1

1. Combine Cargo metadata, tree-sitter, rust-analyzer and build/test metadata.
2. Add typed nodes/edges and exact commit/tree provenance.
3. Index feature flags, cfg gates, build scripts, generated code and target platforms.
4. Add incremental invalidation and compare the graph against hand-labelled gold repositories.
5. Route code-orchestrator context through this graph; retain HDC as a retrieval/ranking layer.

**Exit:** high-precision change-impact analysis and reproducible symbol/type retrieval on selected repositories.

### Workstream 3 — Evidence-first patch loop

1. Define a stable task contract and patch receipt.
2. Separate candidate generation, test generation, verification and grading.
3. Add property/differential/mutation/fuzz adapters with per-test provenance.
4. Implement structured repair hypotheses and information-gain-based next experiments.
5. Require pristine re-evaluation and protected-path checks before acceptance.

**Exit:** every accepted patch has independent evidence, and failures produce structured diagnoses.

### Workstream 4 — Systems engineering and operations

1. Extend the shared requirement/evidence model to service, network, storage, build and deployment units.
2. Add Linux/Nix/systemd/networking/storage diagnostic tasks.
3. Add trace/metric/profile ingestion and causal incident experiments.
4. Add failure injection, backup/restore, rollback and recovery verification.
5. Connect engineering review to the existing formal-safety and digital-twin foundations.

**Exit:** Symthaea can make and verify bounded changes to a realistic system, not just produce source files.

### Workstream 5 — Gated continual learning

1. Save only provenance-complete task trajectories.
2. Extract preconditioned, versioned lessons from successful and failed interventions.
3. Replay new lessons against held-out repositories and counterexamples.
4. Measure whether memory improves success, cost or regression rate over a no-memory ablation.
5. Promote a lesson only when the improvement survives independent evaluation.

**Exit:** measurable transfer to unseen tasks without degrading calibration or safety.

## 10. Immediate next experiments

These are small enough to begin locally without waiting for hosted CI:

1. **Reward-path negative control:** feed a syntactically parseable but invalid Nix candidate through the Broca trainer. Verify that it receives no correctness reward. Then implement a repair callback or rename the component to reflect its current validation-only behavior.
2. **Test-oracle classification:** run `TestGenerator` on functions with multiple parameters, non-`Default` parameters, generic types, fallible results and stateful behavior. Record which generated checks compile, which are meaningful, and which are vacuous.
3. **Graph accuracy gold set:** select three Rust crates and manually label symbol definitions, calls, trait implementations, feature gates and tests. Measure parser/LSP/graph precision and recall.
4. **Patch receipt prototype:** define a deterministic JSON receipt for one repository task and verify it from a clean checkout.
5. **Causal debugging microbenchmark:** create 20 seeded compiler/test failures with known root causes. Compare one-shot retry, error-retrieval retry and hypothesis/experiment-driven repair by success, number of experiments and false fixes.

Do not wait for CI runners to design the contracts, create fixtures, audit the reward path, add unit-level negative controls, write the graders or prepare the repository graph schema. Run local commands when available; report unavailable tooling honestly.

## 11. Definition of success

Symthaea should eventually be able to say:

> I understand this software system at the levels of source, semantics, dependencies, execution and operational constraints. I can explain the relevant evidence, propose competing changes, predict their likely effects, perform a bounded experiment, verify the result independently, disclose uncertainty, and produce a reproducible change receipt. I will not claim success when the evidence is missing.

That is the standard to build toward: not a coding chatbot with a large memory, but a verifiable computational engineering system.

## Research and benchmark references

- OpenAI, [Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/), 2026-02-23.
- Deng et al., [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?](https://proceedings.mlr.press/v306/deng26f.html), ICML 2026.
- Scale Labs, [SWE-Bench Pro V2 evaluation update](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), 2026-09-22.
- Badertdinov et al., [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](https://proceedings.neurips.cc/paper_files/paper/2025/hash/21bec6ace947b1b58967b945c8ac0f10-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2025.
- [Terminal-Bench 4.0 official site and leaderboard](https://www.tbench.ai/), accessed 2026-11-02.
- [Epoch AI Terminal-Bench 2.0 methodology and model-agent scoring](https://epoch.ai/benchmarks/terminal-bench), accessed 2026-11-02.
- [CRUST-bench](https://crust-bench.github.io/), C-to-Rust translation benchmark.
