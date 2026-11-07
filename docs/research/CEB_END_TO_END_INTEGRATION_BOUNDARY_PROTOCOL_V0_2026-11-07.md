# CEB End-to-End Integration Boundary Protocol V0 — 2026-11-07

**Status:** Public design and benchmark-triangulation material. No repository task is admitted, no candidate is run, and no CEB performance result is claimed.  
**Owners:** Existing CEB-RESOLUTION / CEB-TEST / CEB-SECURE / CEB-RUST / CEB-CAUSE / CEB-CONTINUITY work; [#10385](https://github.com/Luminous-Dynamics/symthaea/issues/10385), [#10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366), [#10370](https://github.com/Luminous-Dynamics/symthaea/issues/10370), [#10630](https://github.com/Luminous-Dynamics/symthaea/issues/10630).  
**Gates:** #10064 for candidate-controlled execution isolation; #10369 for signer trust and independent pristine regrade.  
**Metric boundary:** Keep the 14-lane / 56-primary-metric matrix. This protocol introduces no new lane, receipt schema, aggregate score, test-count score or execution privilege.

## 1. Research question

Can an engineering agent deliver a change that works across the complete intended system boundary, not merely inside the local package or test harness that motivated the change?

A repository patch may compile and pass unit tests while:
- failing to link with downstream crates or feature combinations;
- violating a public API or wire contract;
- failing at service startup under the declared environment;
- breaking state persistence, replay or restart behavior;
- failing under concurrency, retries or partial failure;
- returning incorrect behavior through the real public interface;
- violating security, tenant isolation or authorization invariants;
- missing calibrated performance or resource constraints;
- requiring an undocumented manual step that the evaluator does not capture.

The central question is **where does the claimed behavior become true?** Every engineering claim has an applicability boundary: package, workspace, target triple, feature set, service process, deployment topology, state history, input distribution, concurrency level or operational horizon. Evidence from a narrower boundary cannot silently certify a broader one.

## 2. Current industry frontier — snapshot 2026-11-07

### 2.1 SWE-Serve: production-path integration can reverse local success

NVIDIA's [SWE-Serve](https://research.nvidia.com/benchmarks/swe-serve) evaluates 53 repository-grounded SGLang production-inference engineering tasks, sourced from recent merged changes and spanning model support, public APIs, caching, GPU kernels and related serving work. The task set has **41 H100 GPU tasks and 12 CPU tasks**. The benchmark evaluates 11 models and 31 model-effort configurations, with three complete runs per configuration under a fixed mini-SWE-agent harness. Its live leaderboard lists leading mean Pass@1 around **75%**, but score, cost and time vary by model and effort. [Paper: arXiv:2609.26777](https://arxiv.org/abs/2609.26777).

The important methodological result is the difference between full and reduced verification on the **19 tasks with end-to-end serving coverage**: patches pass at **69.4%** when the E2E tests are excluded, but only **45.9%** with the full live-serving verifier. Put another way, approximately one-third of patches that pass every other check fail the production-serving tests. The same patch population is being evaluated under two verification scopes; this is stronger evidence about verification-boundary sensitivity than an unrelated model leaderboard comparison. The benchmark also reports lower performance on changes spanning multiple runtime domains. [SWE-Serve leaderboard/methodology](https://research.nvidia.com/benchmarks/swe-serve) · [NVIDIA technical explanation](https://developer.nvidia.com/blog/how-swe-serve-exposes-the-gap-between-local-tests-and-live-serving/).

**Interpretation limit:** SWE-Serve's score applies to its 53 tasks, declared CPU/H100 hardware, closed-book policy, mini-SWE-agent harness and complete verifier. It is not a general-purpose CEB baseline or proof that all systems require a GPU E2E harness.

### 2.2 SWE-Cycle: isolated phases do not establish autonomous completion

[SWE-Cycle](https://arxiv.org/abs/2605.13139) contains **489 rigorously filtered instances** and separates environment reconstruction, implementation and verification-test generation from the integrated FullCycle task. The FullCycle condition requires an agent to work from a bare repository without the preconfigured environment or human scaffolding provided to isolated phases. The paper reports a sharp drop from isolated tasks to the integrated FullCycle condition. Its SWE-Judge combines static review and dynamic execution for end-to-end grading.

**CEB implication:** a system that can implement a change when given a prepared environment has not demonstrated that it can reconstruct the environment, diagnose setup failures, create valid checks and complete the whole workflow. Phase-specific scores must remain separate from FullCycle outcomes.

### 2.3 E2E-SWE: complete repository behavior is a separate construct

[E2E-SWE](https://arxiv.org/abs/2609.38335) evaluates **186 complete-repository generation tasks across 11 languages**. The agent receives a natural-language specification and an empty workspace, builds an installable project and must pass hidden tests after the result is copied into a fresh environment. Across 13 evaluated frontier models, reported Pass@1 ranges from **11.7% to 67.7%**, with four independent runs per model/task configuration in the reported setup.

**CEB implication:** implementation-independent specifications, fresh-environment installation and hidden behavior tests can expose failures that are not visible in a local source tree. Whole-repository generation is adjacent to, not identical with, modifying an existing repository or reviewing a patch.

### 2.4 A common frontier gap

These benchmarks show a recurring problem:

- isolated implementation capability is not environment reconstruction;
- package-local correctness is not downstream integration;
- unit/integration test success is not live service correctness;
- target-specific remediation is not whole-patch safety;
- passing a hidden test is not necessarily acceptable operational behavior.

CEB should adopt the **verification-boundary ladder** as a task and evidence-design principle, without converting levels into a single score.

## 3. Paradigm shift: from test-passing patches to boundary-qualified engineering claims

The proposed engineering standard is:

> A change is only as correct as the strongest independently qualified system boundary that has been exercised for the exact source/configuration revision and the specific claim being made.

This creates a scope-aware evidence ladder:

| Boundary | Example evidence | What it can establish | What it does not establish |
|---|---|---|---|
| B0 — Static/source | source/configuration inspection, diff, declared contract | structure and source-level claims | compile or behavior |
| B1 — Component | function/module/package tests, compiler and linter | selected local semantics and build properties | workspace consumers or service behavior |
| B2 — Workspace | multi-crate build/tests, feature/target matrix, downstream consumer | declared workspace and selected dependency/configuration contracts | deployed process behavior |
| B3 — Interface | API/protocol/schema contract tests, client/server integration | behavior at a named public interface and tested variants | production topology, load or long-running state |
| B4 — System | multi-process/service integration, persistent state, retries, restart and concurrency | behavior across the declared running system and controlled fault cases | all deployment environments or workload distributions |
| B5 — Operational | end-to-end production-shaped path, deployment/configuration, resource/performance/security gates | declared operational requirements for the tested environment and horizon | universal safety or untested operating conditions |
| B6 — Evolution | follow-on commits, upgrades/migrations, evidence continuity and repair survival | persistence of the qualified claim across later revisions/events | correctness outside the sequence and applicability domain |

Levels are **evidence scopes, not an additive quality score**. A task may not need B5 or B6. The applicable boundary is defined by the frozen requirement before agent evaluation. A source-only task can legitimately finish at B0/B1; a serving feature may require B4/B5. A narrower pass must never be upgraded to a broader claim. Narrower evidence cannot silently certify a broader claim.

## 4. Task authoring: paired boundary controls

For each source/defect anchor, author controls that differ by one boundary condition. The expected outcome must be grounded by a task-specific requirement and an independently reviewed oracle plan.

### 4.1 Twelve synthetic scenario blueprints

These are authoring vignettes only; they contain no real injected task code, executable reproducer, hidden answer, task ID or admitted benchmark case.

| ID | Scenario | Minimum discriminating boundary |
|---|---|---|
| E2E-S01 | A crate's unit tests pass, but a downstream workspace consumer uses a changed trait/API differently | B2 |
| E2E-S02 | Default features pass; an optional feature combination exposes a missing capability or type bound | B2 |
| E2E-S03 | x86_64 passes; the declared ARM64 target has a missing cfg/import or unsupported operation | B2 |
| E2E-S04 | Server compiles but fails startup because configuration, migration or required environment variables are inconsistent | B3/B4 |
| E2E-S05 | Client and server individually pass unit tests but disagree on an API/schema version or error semantics | B3 |
| E2E-S06 | A retry after timeout duplicates a side effect despite each request succeeding in isolation | B4 |
| E2E-S07 | A crash/restart loses a state transition that the service contract requires to survive | B4 |
| E2E-S08 | Two tenants share a cache key or authorization result under a concurrency/interleaving condition | B4, secure |
| E2E-S09 | A model/inference feature passes its internal kernel test but fails through the public serving path | B5 |
| E2E-S10 | A change meets functional assertions but violates a predeclared tail-latency, memory or throughput budget | B5, performance |
| E2E-S11 | A repair fixes the target at one revision but a later dependency/API/schema evolution makes its evidence stale or reintroduces the behavior | B6 |
| E2E-S12 | The task specification claims full-system behavior, but only a package-level check is available; abstention/escalation is the only supported decision | boundary/authority control |

Each scenario must record:
- exact source/configuration revision and the declared behavioral requirement;
- the minimum boundary that could establish the claim;
- at least one narrower-boundary positive control and one target-specific negative control;
- cross-package/interface/system dependencies relevant to the claim;
- protected invariants and non-target behavior;
- infrastructure/tool failure and missingness dispositions;
- expected behavior when the required boundary is unavailable;
- the human/compute cost needed to reach the required boundary.

### 4.2 Boundary-rung pairs

Use near-identical claims where only the available boundary changes:

- package-local pass versus downstream consumer fail;
- one feature/target pass versus a different declared feature/target;
- handler test pass versus public API contract failure;
- successful single request versus duplicate side effect under retry;
- fresh process pass versus restart/persistent-state failure;
- target repair pass versus later migration regression;
- local benchmark speedup versus end-to-end latency/throughput violation;
- complete evidence versus a missing/unqualified operational check.

These pairs test whether the agent's confidence and final claim are calibrated to the evidence boundary, rather than whether it always demands the highest level of testing.

## 5. Experimental arms and hypotheses

Reuse existing CEB-CAUSE experiment selection and CEB-TEST/RESOLUTION/Secure lanes. Do not add another scheduler.

### Phase A — Boundary recognition and claim licensing (source-only)

A0: ordinary implementation/review agent.  
A1: agent sees a named acceptance boundary but no explicit claim/evidence contract.  
A2: evidence-aware agent must state the boundary required by each claim and bind existing evidence to exact revisions/configurations.  
A3: A2 plus counterexample/neighbor-boundary analysis and explicit abstention when the requirement cannot be established.

H1: A2/A3 reduce overclaiming from local evidence.  
H2: A3 improves correct identification of the smallest sufficient next boundary action without indiscriminately demanding B5/B6 for every task.  
H3: A3 preserves or improves the existing CEB repair/review quality outcomes at matched reasoning and human cost.

### Phase B — Controlled boundary acquisition (only after #10064 qualifies)

B0: lowest admissible boundary only.  
B1: task-directed escalation to the smallest stronger boundary that discriminates a named competing hypothesis.  
B2: fixed full-boundary suite under a matched total budget.  
B3: unconstrained tool-use comparator.

The agent must state *before the check* what competing hypotheses the check distinguishes and what each possible result would change in the diagnosis/repair decision. More tests and higher boundary levels are not quality by themselves.

### Phase C — Integration boundary removal ablation

For each task with safe and qualified environments, compare:
- local/component checks only;
- component + workspace checks;
- component + workspace + interface/system tests;
- full declared boundary.

The primary question is whether stronger boundary evidence rejects incorrect local successes and whether the additional cost is justified by fewer failed repairs, false accepts, regressions or operational incidents. Reuse #5478 matched ablation/lesion/sham/rescue semantics if the CEB policy itself is being ablated; do not duplicate its general ablation protocol.

## 6. Acceptance authority and task qualification

SWE-Serve provides a useful task-construction pattern:
- the unmodified repository must fail the new-behavior tests while retaining required regressions;
- the frozen reference solution must pass the complete verifier;
- the verifier is challenged by plausible agent patches and adversarial modifications;
- task runtime/hardware and exact environment are declared;
- E2E tests are included only where the requirement demands them.

CEB should reuse #8311 oracle quality, #10680/#10681 verifier-of-verifiers, #10607/#10609 task admission, #10610/#10611 release lifecycle, #10064 isolation, #10369 signer/regrade trust and #10370 evidence convergence. A local green test must not be promoted to system pass if the contract requires a higher boundary.

### Required positive/negative verifier controls

1. reference implementation passes the full declared boundary;
2. empty/no-op patch fails the new-behavior requirement;
3. local-only patch that misses a downstream consumer is rejected when B2 is required;
4. protocol mismatch is rejected when B3 is required;
5. duplicate side-effect under retry is rejected when B4 is required;
6. restart-state loss is rejected when persistence is normative;
7. protected tenant/security invariant is checked across interleavings;
8. performance claims are rejected without calibrated hardware/workload/variance;
9. infrastructure failures and timeouts remain distinct from candidate failures;
10. a task that has no qualified B4/B5 environment remains design-only/unresolved rather than being downgraded to B1 pass.

## 7. Metric mapping — no new primary lane

Map evidence to existing registered CEB metrics:

- CEB-RESOLUTION: independently accepted repair for the task's declared behavior;
- CEB-TEST: validity and effectiveness of the verification oracle;
- CEB-SECURE: protected security/authorization invariant preservation;
- CEB-RUST: cross-crate type/API/feature/target correctness;
- CEB-CAUSE: diagnostic quality and information gain from a boundary-directed experiment;
- CEB-CONTINUITY: survival and evidence revalidation across later source/configuration changes;
- CEB-HUMAN: correct escalation, handoff and human review cost;
- existing cost/resource measures: full model/tool/runner/test/human cost.

The boundary level is a scope descriptor. It does not become an independent score. Never compute a weighted "E2E pass" by averaging local and operational checks; report exact boundary, required checks, denominator and terminal disposition.

## 8. Statistical and resource controls

- Freeze the required boundary by task before treatment assignment.
- Pair agents on the same task/source/environment and preserve the same tool, network and resource policies.
- Run at least three independent trials for nondeterministic agents when execution is later qualified; preserve run-level data rather than treating retries as extra tasks.
- Cluster by repository, original defect/root cause and boundary-pair family.
- Report pass@1 separately from repeated-run reliability; keep all dispositions in denominators.
- Calibrate hardware and infrastructure variance for CPU/GPU, disk, startup, network and stateful services.
- Include the cost of all attempted checks, failed setups and human escalation.
- Do not compare SWE-Serve's SGLang/H100/CPU setup or E2E-SWE from-empty-workspace pass@1 to CEB scores without matched task contracts and budgets.

## 9. Prospective research goals

These are design goals, not registered CEB metrics or achieved results:
- reduce **false boundary promotion** (claiming system behavior from component-only evidence) to zero critical occurrences in the qualified challenge set;
- improve boundary-selection accuracy and avoid unnecessary escalation, with a goal of ≥0.90 correct smallest-sufficient-boundary decisions after pilot calibration;
- preserve existing CEB repair, security and test-oracle quality gates while reducing all-in cost per independently accepted engineering outcome by at least 20%;
- on a matched SWE-Serve-like systems integration slice, exceed the contemporaneous baseline with paired confidence intervals and exact same hardware/harness/verifier—otherwise publish a cost/quality Pareto comparison rather than a superiority claim.

## 10. Runner-independent sequence

1. Map B0–B6 to existing CEB task and evidence authority contracts; no new receipt or task schema.
2. Create a human annotation worksheet for each requirement's minimum sufficient boundary and why narrower evidence is insufficient.
3. Author 12 public synthetic scenarios and boundary-pair controls with no executable payloads or hidden answer keys.
4. Independently review each scenario's required boundary and positive/negative verifier behavior.
5. Extend task admission only after oracle and contamination review.
6. Only after #10064 and #10369 qualification, select a small matched development sample for candidate execution and independent regrade.

## 11. Claim boundary

This protocol introduces a verification-boundary ladder and twelve synthetic authoring blueprints. It does not reproduce SWE-Serve, SWE-Cycle or E2E-SWE; it does not execute any task; it does not claim the proposed B0–B6 ladder is a universal standard; and it does not add a primary metric lane, receipt schema, task admission or SOTA result. Candidate-controlled execution remains blocked by #10064 and independent regrade remains blocked by #10369.
