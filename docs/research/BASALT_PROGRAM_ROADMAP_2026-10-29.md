# BASALT Research Roadmap
## Benchmark for Agentic Software Adaptation, Lifecycle, and Trust

**Status:** Research roadmap / proposed program architecture  
**Date:** 2026-10-29  
**Repository:** `Luminous-Dynamics/symthaea`

BASALT is the working program name for Symthaea's coding-system research. Existing VLEB protocol names and evidence artifacts remain unchanged until a deliberate naming migration is reviewed. A name collision/trademark check is still required before external branding.

## Executive thesis

Move from "generate code and see whether tests pass" toward **contract-directed, evidence-carrying software engineering**:

1. interpret a request and expose ambiguity;
2. compile intent into a typed, versioned change contract;
3. navigate repository structure and prior verified knowledge;
4. synthesize one or more candidate changes;
5. run the appropriate language-native build, test, lint, and security tools in a controlled environment;
6. independently check postconditions, scope restrictions, evidence freshness, and final repository state;
7. adapt when requirements, workspace, dependencies or tools change;
8. report outcome, uncertainty, cost, unresolved obligations, and provenance honestly.

This is a falsifiable research thesis, not a claim of achieved superiority. The generator must never be the sole authority for declaring its own output correct.

## Working name

**BASALT — Benchmark for Agentic Software Adaptation, Lifecycle, and Trust**

BASALT is the wider research and benchmark program. VLEB remains the name of the current analysis/protocol lane while its contracts and trusted runner are being qualified. A broad rename is not part of this roadmap PR.

## Executable evaluation contract artifacts

The roadmap now has a versioned task-level run-record contract:
- [BASALT Evaluation Contract](BASALT_EVALUATION_CONTRACT_2026-10-29.md): primary endpoint, denominator policy, pair invariants, negative controls and promotion gates.
- [Run-record JSON Schema](../../benchmarks/basalt/schema/run-record.schema.json): strict, versioned trial evidence.
- [Suite-manifest schema](../../benchmarks/basalt/schema/suite-manifest.schema.json) and [synthetic manifest](../../benchmarks/basalt/examples/suite-manifest.example.json): frozen task membership, specification digests, and pre-run quality-audit dispositions.
- [Synthetic run record](../../benchmarks/basalt/examples/run-record.example.json): paired fixture only; every task, revision, digest, and outcome is synthetic.
- Rust validator source: [xtask/src/basalt_validate.rs](../../xtask/src/basalt_validate.rs), invoked as `cargo run -p xtask -- basalt-validate --input RUN.json --manifest SUITE-MANIFEST.json`.
- Validator implementation task: [#9021](https://github.com/Luminous-Dynamics/symthaea/issues/9021); signed provenance follow-up: [#9034](https://github.com/Luminous-Dynamics/symthaea/issues/9034).

The validator source checks cross-record pairing, verifier/evaluated-tree digest equality, budget ceilings, outcome/verifier consistency, exact task-set/specification/audit matching, task-specific required verifier obligations, and every trial against the manifest's frozen attempt count and seed schedule. It also bounds file reads and distinguishes required-but-nullable fields from optional-but-non-null fields. It contains 37 authored tests. **Neither compilation nor those tests have been executed in this environment**: no Rust toolchain or local checkout is available. For the current ready-for-review PR head, Showroom Integrity and PR Governance were pending and CI was queued when inspected; earlier attempts were skipped or cancelled. Hash matching does not authenticate the issuer; signed, independently pinned receipts remain a prerequisite for confirmatory scoring.

## Existing work to compose
- Paired task-cluster statistics and cost-quality frontier analysis: [#9057](https://github.com/Luminous-Dynamics/symthaea/issues/9057)
- Leakage-resistant temporal holdouts and task-cohort lifecycle: [#9080](https://github.com/Luminous-Dynamics/symthaea/issues/9080)
- Non-compensatory verified success plus task-specific patch-quality rubrics: [#9084](https://github.com/Luminous-Dynamics/symthaea/issues/9084)
- Fail-closed Rust validation of run records and cross-record evidence invariants: [#9021](https://github.com/Luminous-Dynamics/symthaea/issues/9021)

- VLEB family-stratified paired inference: [#8900](https://github.com/Luminous-Dynamics/symthaea/pull/8900)
- VLEB clean-to-shifted robustness: [#8901](https://github.com/Luminous-Dynamics/symthaea/pull/8901)
- Crossed task/seed uncertainty: [#8879](https://github.com/Luminous-Dynamics/symthaea/pull/8879)
- Metamorphic robustness under validated semantics-preserving transformations: [#9003](https://github.com/Luminous-Dynamics/symthaea/issues/9003)
- Geometry representation bake-off: [#8947](https://github.com/Luminous-Dynamics/symthaea/issues/8947)
- Typed declarative change IR / counterexample-guided synthesis: [#8948](https://github.com/Luminous-Dynamics/symthaea/issues/8948)
- Geodesic-assisted coding plan qualification: [#8975](https://github.com/Luminous-Dynamics/symthaea/issues/8975)
- BASALT program coordination: [#8974](https://github.com/Luminous-Dynamics/symthaea/issues/8974)
- Existing generic state/metric-space and smooth-manifold contracts: [#2017](https://github.com/Luminous-Dynamics/symthaea/issues/2017), [#2018](https://github.com/Luminous-Dynamics/symthaea/issues/2018), [#2024](https://github.com/Luminous-Dynamics/symthaea/issues/2024)
- Multi-file typed change transactions: [#4335](https://github.com/Luminous-Dynamics/symthaea/issues/4335)
- Evidence-scoped code-generation assurance: [#5106](https://github.com/Luminous-Dynamics/symthaea/issues/5106)
- Creative-coding mode / candidate-generation separation: [#5552](https://github.com/Luminous-Dynamics/symthaea/issues/5552)
- Work accounting around geodesic-request/imagination entry points: [#7303](https://github.com/Luminous-Dynamics/symthaea/issues/7303)

Reuse existing code and contracts. Do not create a parallel manifold crate, duplicate verification stack, or unrelated coding orchestrator without evidence that the current abstraction cannot serve the use case.

## Program architecture

### A. Measurement and benchmark validity

Keep benchmark admission, oracle integrity, exact execution identity, evaluator mutation tests, paired statistics, shift-event evidence, and transformation validity distinct from agent generation. Evaluator-controlled inputs and independent reference evidence are mandatory. Add metamorphic pairs only when each transformation is independently validated as semantics-preserving for the declared language/repository; retain meaning-changing negative controls. Treat queued as pending—not pass or fail.

Recent external audits justify ongoing task-quality scrutiny: OpenAI's July 2026 review estimated about 30% of the reviewed SWE-bench Pro tasks had defects, while Epoch AI's September 2026 review flagged scoring issues in 30 of 66 Terminal-Bench 4.0 tasks. The audits differ in method and do not imply every task is invalid. They motivate independent oracle/task qualification, not blanket rejection of external benchmarks.

### B. Declarative change contracts

Extend `CodeIntent`/`CodeSpec` into a canonical, versioned, typed Change Contract IR. Contract fields should capture:

- exact initial repository/source revision and toolchain/build profile;
- desired behavior and explicit preconditions/postconditions;
- allowed edit scope and forbidden changes;
- test, static-analysis, and security obligations;
- resource/network/attempt budgets;
- evidence and verifier requirements;
- ambiguity and unresolved conditions.

A parsed or well-formed contract is not necessarily a correct interpretation of user intent. The system must surface ambiguity, and should return unresolved when a required oracle cannot be established.

### C. Synthesis and counterexample-guided refinement

A candidate backend proposes edits. Independent checks produce typed failures/counterexamples. Symthaea uses those observations to revise the plan and candidate under bounded budgets. Keep generated tests separate from trusted evaluator oracles and include controls for test deletion, verifier mutation, out-of-scope edits, and stale evidence.

### D. Repository intelligence and geometric retrieval

Compare the strongest simple baseline (lexical search / ripgrep or BM25) against syntax-tree retrieval, symbol/dependency graphs, existing HDC/ProgramManifold, and candidate hyperbolic representations. Only combine channels after the ablation shows complementary value.

### E. Geodesic-assisted plan search

Use existing `symthaea-geodesic::ProgramManifold` and HDC geometric operations as candidate retrieval/ranking mechanisms. Repository inspection found an existing `scripts/coding_backend_hard_geodesic_ab.sh` lane that compares geodesic-rejection shadow mode with hard rejection and checks aggregate quality/pass-rate deltas, mean attempts, and shadow false positives. Extend and qualify this experiment before creating another A/B harness.

Important current boundaries:
- `ProgramManifold` is implemented as a discrete set of specification fibers assigned by BinaryHV similarity; it is not, by itself, a general smooth manifold with a graph of executable code-edit transitions.
- `crates/domains/symthaea-geodesic/src/codebase_bridge.rs` indexes Rust `.rs` files today. Cross-language indexing must be a separate, explicit capability claim.
- `ast_bridge::pdg_from_entities` currently accepts extracted entities but delegates to the line-level Rust-source parser rather than using those entities to build the graph. Measure this fidelity boundary before relying on the bridge for AST-aware claims.
- The existing A/B script compares aggregate reports; add exact task-level matched outcomes, frozen task/seed identities, confidence intervals, and change-family/language strata before making broader conclusions.
- `FiberPoint.quality` is an input score clamped to [0,1], not a certificate. Bind it to verifiable provenance if it influences candidate ranking.

Distinguish semantic embedding paths from valid discrete code-change paths. If claiming minimum-cost plans, use a qualified admissible heuristic or exact search reference; otherwise label geodesic values as ranking heuristics. Every proposed change transition requires preconditions, postconditions and replayable evidence.

### F. Language and toolchain adapters

Use a shared adapter contract; preserve native language semantics and tools. Initial priority: Rust, Nix, Python, shell and SQL, with C/C++, Go and TypeScript next according to use. Tree-sitter can provide broad structural parsing, but parser availability alone does not qualify type checking, runtime behavior or repository engineering. Each adapter declares a capability level and passes its matching contract tests.

### G. Cognitive subsystem ablations

Test each addition against simpler alternatives rather than enabling every subsystem by default:

- HDC retrieval / verified repair memory;
- CfC/LTC task-state modeling vs deterministic state tracking;
- prediction error/surprise-driven diagnostics;
- expected-information-gain test and tool selection;
- geodesic or hyperbolic retrieval;
- counterexample-guided refinement;
- language-adapter and independent-verifier components.

The primary output is independently verified software change. Internal activation, geometric coherence or confidence alone is not an engineering result.

## Geometric manifold and geodesic guardrails

Different software structures call for different representations:

- AST hierarchy: hyperbolic geometry is a candidate to test.
- Symbol, import, call and build relationships: graph representations.
- Compositional semantic roles: existing HDC/hyperspherical operations.
- Actual engineering progress: discrete typed repository-state transitions.
- Product/late fusion: only after a controlled study proves each channel contributes information.

Reuse #2017/#2018 and existing HDC geometry. Preserve distinctions between raw vectors, normalized manifold points, tangent vectors at a particular base point, and evidence-bearing state. Handle cut-locus/non-unique geodesics, invalid states and tolerance identities explicitly. No silent normalization of evidence-bearing states.

A geodesic in latent space is not necessarily a compilable program, a low-cost edit plan, a correct semantic transformation, or a proof. A path not found is unknown unless completeness or admissible search supports a stronger conclusion.

## Benchmark plan

### 1. External comparison suite

Use external suites as complementing reference points rather than combine incompatible scores:

- [SWE-bench Multilingual](https://www.swebench.com/multilingual): 300 curated tasks, 42 repositories and 9 languages.
- [Artificial Analysis Coding Agent Index methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking): v2.0 documents 290 tasks across repository engineering, terminal workflows and repository Q&A, with published cost, token, time and integrity methods.
- [Terminal-Bench](https://www.tbench.ai/): terminal and tool-use tasks.
- [AgentSuite/COBA](https://proceedings.mlr.press/v306/suh26a.html): benchmark validity methods.
- [SWE-Shift](https://arxiv.org/abs/2610.32196): clean-to-shifted execution evaluation.
- [OpenAI SWE-bench Pro audit (July 2026)](https://openai.com/index/separating-signal-from-noise-coding-evaluations/): task, ground-truth and grading quality audit.
- [Epoch AI Terminal-Bench 4.0 review (September 2026)](https://epoch.ai/benchmarks/terminal-bench-4/review): reviewer findings about scoring defects.
- [HypeCodeNet / The Natural Geometry of Code](https://proceedings.iclr.cc/paper_files/paper/2026/hash/408598c7e142b607c828f5f769e8c1bd-Abstract-Conference.html): motivation for evaluating hyperbolic structural representations.

Pin versions and license/usage constraints before execution. Public scores are useful reference points, not directly comparable to BASALT unless cohort, environment, budget and verifier are matched.

### 2. BASALT task families

At minimum, define frozen, auditable families for:
- repository understanding and gold-change localization;
- parser / compiler / type-checker repair;
- build and dependency recovery;
- multi-file refactoring and API compatibility;
- SQL migration and transaction integrity;
- runtime recovery and resource lifecycle;
- security fixes and adversarial verification;
- declarative-specification satisfaction;
- tool/session/workspace shifts and stale-evidence invalidation.

Expand only with executable, independently graded fixtures. Do not inflate coverage using non-runnable examples.

### 3. Matched ablation matrix

Compare under the same task split, source revision, model/scaffold, tools, environment, network policy and resource budget:

A. baseline agent and tools;
B. + repository structure retrieval;
C. + HDC/verified-memory retrieval;
D. + geodesic/hyperbolic retrieval;
E. + typed Change Contract IR;
F. + counterexample-guided refinement;
G. + active test/tool selection;
H. full composition.

Run a smaller screening pilot first. Keep an untouched confirmatory set split by repository and patch/task lineage. Adopt only components that improve a preregistered primary endpoint or yield a meaningful cost reduction under a predeclared non-inferiority margin.

### 4. Primary and secondary measures

Primary:
- independently verified issue-resolution rate;
- paired clean/shifted outcomes and the full pass/fail/unresolved transition matrix;
- regression and out-of-scope edit rates;
- stale-evidence acceptance and evaluator-mutant detection;
- per-language/per-task-family effects with uncertainty.

Secondary:
- Recall@k / MRR / nDCG for retrieval;
- repair attempts, compilation/test calls, tool-call count;
- cost per independently verified fix;
- p50/p90 duration, token usage, CPU/GPU and memory;
- unresolved and clarification rates;
- index build/rebuild time and stale-index rejection;
- robustness to file moves, renames, dependency updates, malformed syntax and intervention position.

Report effect sizes and intervals, correct for multiplicity where appropriate, and retain unresolved outcomes in the denominators.

## Current industry comparison and stretch targets (2026-10-30)

Public leaderboard percentages are properties of a full evaluation configuration—task cohort, agent harness, tools, budget, retries, verifier and integrity policy—not of a model name alone.

- [Terminal-Bench 4.0](https://www.tbench.ai/?version=4.0) lists Opus 5.5 + Claude Code at **64.8% ± 3.1%**, Sonnet 5.5 + Claude Code at 61.8% ± 2.9%, and GPT-6 Astra + Codex at 58.2% ± 2.8%. These are benchmark/harness-specific results, not a universal software-engineering score.
- [Artificial Analysis Coding Agent Index v2.0](https://artificialanalysis.ai/methodology/coding-agents-benchmarking) equal-weights FrontierCode v1.1 (100 repository-engineering tasks), Terminal-Bench 4.0 (66 terminal tasks), and SWE-Atlas-QnA-AA (124 repository-Q&A tasks). It runs three attempts per task, reports pooled cost/token/time, and audits trajectories for solution retrieval and test/verifier tampering. FrontierCode uses non-compensatory blockers followed by a bounded quality score when blockers pass.
- [SWE-Shift](https://arxiv.org/abs/2610.32196), published 23 October 2026, evaluates 120 scenarios across 12 change families and nine intervention positions. All 17 evaluated systems had lower shifted than clean success; GPT-6 Astra is reported at 67.35% clean versus 24.79% shifted, with 20.34% paired success. Tool/session changes and stale-evidence scenarios had zero paired success across that evaluated cohort.
- [SWE-bench Multilingual](https://www.swebench.com/multilingual) supplies a cross-language reference of 300 tasks from 42 repositories across nine languages.

These are not matched head-to-head comparisons. BASALT should reproduce selected public setups where permitted, match budgets and execution identities, publish per-task results and uncertainty, and measure the differentiated capabilities separately.

Proposed stretch targets—not achievements or universal standards:
- **Verified engineering:** +3 to +5 percentage points over a strong matched baseline, supported by a predeclared paired interval.
- **Efficiency:** 25% lower cost per strict verified fix under a predeclared non-inferiority margin.
- **Shift adaptation:** at least 50% clean-success retention on a qualified BASALT shift cohort, while reporting paired success and family-level failures separately.
- **Metamorphic robustness:** no more than 5 percentage points of degradation under validated semantics-preserving transformations.
- **Integrity:** zero false accepts across at least 300 genuinely independent adversarial negative controls as an initial gate.
- **Patch quality:** separate blinded, calibrated non-blocking quality rubric; no quality score can compensate for a failed blocker.

Targets require pre-run power/precision analysis and independent cohort qualification. Zero false accepts in 300 controls is an initial hurdle, not proof of zero risk. The SWE-Shift retention target is a BASALT stretch goal, not a directly comparable external result.

## Proposed acceptance gates

These are goals, not achieved metrics or universal industry standards:

1. **Trial accounting:** 100% of locked trials accounted for.
2. **Identity integrity:** zero accepted source/checkpoint/environment mismatches.
3. **Stale-evidence controls:** zero false accepts in at least 300 genuinely independent negative controls as an initial target. This does not prove zero risk; independence must be justified.
4. **Reproducibility:** identical locked inputs and toolchain/analysis identities produce reproducible reports.
5. **Language claims:** each language/capability level passes positive, negative and adversarial contracts.
6. **Geometry:** geodesic/embedding scores never authorize a change; transitions are validated discretely.
7. **Change contracts:** mutation tests show that changing an acceptance-critical condition changes contract identity and that independent verifiers catch deliberate violations.
8. **Performance:** no SOTA claim without matched, held-out, independently audited runs.
9. **Shift robustness:** the runner attests that shifts were applied and that post-shift evidence was refreshed or correctly rejected.
10. **Complexity budget:** additional subsystems must justify their compute, latency and maintenance cost through ablations.

## Execution order that does not depend on queued runs

1. Create the program roadmap and maintain existing VLEB PRs with honest status.
2. Build the Change Contract IR and one end-to-end Rust path.
3. Run the retrieval/geodesic representation pilot independently.
4. Add a discrete task-state graph fixture with known optimal path costs and adversarial transitions.
5. Connect trusted runner receipts to the shifted analyzer; keep scoring ineligible until the evidence boundary is qualified.
6. Expand language adapters and cognitive modules based on measured benefit.

## Risks and explicit non-claims

- A language parser does not imply production competence in that language.
- A compile or test pass does not establish absence of all defects.
- A well-formed specification may misrepresent the user's intent.
- Learned embeddings, HDC similarity and geodesic distances are not correctness proofs.
- A failed search does not prove infeasibility.
- Synthetic geometric competence does not establish real repository or real-world performance.
- No agent-performance score or SOTA superiority is claimed by this roadmap.

## Source references

- Artificial Analysis, Coding Agent Index v2.0 methodology (October 2026): https://artificialanalysis.ai/methodology/coding-agents-benchmarking
- SWE-bench Multilingual: https://www.swebench.com/multilingual
- Tree-sitter: https://tree-sitter.github.io/
- HypeCodeNet, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/408598c7e142b607c828f5f769e8c1bd-Abstract-Conference.html
- AgentSuite/COBA, ICML 2026: https://proceedings.mlr.press/v306/suh26a.html
- SWE-Shift (October 2026): https://arxiv.org/abs/2610.32196
- Counterexample Guided Inductive Synthesis Modulo Theories: https://www.repository.cam.ac.uk/items/e0f2aa2a-af2b-4b9b-9429-6ff0a99a3559
