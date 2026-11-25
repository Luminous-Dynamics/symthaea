# Evidence-Carrying Proof Comprehension Benchmark (ECPCB) — protocol v1

**Issue:** [#13607 — ECPCB-V1](https://github.com/Luminous-Dynamics/symthaea/issues/13607)  
**Parent programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Related assurance capsule:** [#13603 — LEAN-TRUST-CLOSURE-CAPSULE-001](https://github.com/Luminous-Dynamics/symthaea/issues/13603)  
**Status:** benchmark specification and proposed task contract; no gold corpus, matched experiment, human-study outcome, or SOTA result is claimed.

## 1. Executive objective

ECPCB evaluates whether a system can reconstruct, explain and test the mathematical mechanism of a formal proof while preserving exact theorem scope and producing evidence another system can inspect.

The benchmark's central object is not a generated proof string. It is an **evidence-carrying proof explanation** composed of:

1. theorem and context identity;
2. typed formal dependency graph;
3. reconstructed proof-plan graph;
4. local proof-action certificates;
5. global obligation and scope consistency;
6. counterfactual predictions and checked witnesses;
7. explanation claims with evidence and epistemic class;
8. trust-closure assurance vector;
9. human comprehension and delayed-transfer outcomes.

This extends proof success rather than replacing it. A system may solve the theorem but misunderstand its mechanism; retrieve a relevant lemma but apply it outside its scope; or produce a mathematically plausible explanation unsupported by the actual proof. ECPCB measures these failure modes separately.

## 2. Non-goals and claim boundaries

ECPCB is not:
- a new proof kernel or replacement for Lean;
- a theorem-discovery leaderboard by itself;
- a benchmark that treats every dependency edge as a mathematically necessary premise;
- an LLM-judge ranking competition;
- a proof compression contest where shorter is automatically better;
- a human-learning claim based on immediate clarity ratings;
- a single composite mathematical-intelligence score.

Formal proof validity, statement fidelity, explanation faithfulness, human learning and prover-context utility are independent outcomes. Hard trust gates are not averaged against retrieval or presentation scores.

## 3. Task contract

Every benchmark task must be bound to immutable inputs.

### 3.1 Identity and environment

- task ID and version;
- repository, commit, source path and blob digest;
- fully qualified declaration name and declaration kind;
- exact elaborated theorem type identity, including binders, universes and local context;
- Lean toolchain, Lean core revision, Mathlib revision and Lake manifest digest;
- imported module / compiled artifact identities;
- root proof artifact identity and source proof-route provenance;
- trust policy version and observed axiom/admission profile.

A declaration short name or pretty-printed theorem string is not sufficient identity. Source and target changes create a new task or an explicitly paired mutation.

### 3.2 Mathematical content

Each task contains:
- exact theorem statement and all assumptions;
- parameter/domain restrictions;
- conclusion and relevant definitions;
- typed source/dependency graph;
- separately reviewed mathematical-role graph;
- one or more accepted proof routes, where available;
- goal transformations, side conditions, intermediate subgoals and invariants;
- explanation claims tagged as formal_fact, mathematical_interpretation, analogy_with_boundary, or unknown.

The formal dependency graph records what Lean references. The mathematical-role graph records what a reviewer judges each declaration to contribute. A helper, recursor, coercion, typeclass instance or code-generation artifact can be a formal dependency without being the key mathematical idea.

### 3.3 Required system response

For a full ECPCB submission, the system returns:
- exact theorem/scope identity;
- proof-plan graph with typed goal/action/justification/side-condition/subgoal nodes;
- referenced formal dependencies and role hypotheses;
- local action certificates for every action marked kernel-checkable;
- global subgoal and terminal-goal accounting;
- explanation claims with evidence references and epistemic classes;
- counterfactual disposition: still_provable, refuted_by_checked_witness, unknown, ill_formed, or proof_route_changed_only;
- trust-closure capsule with an explicit checker-diversity profile reference (registry digest, profile ID and profile digest; `not_assessed` is explicit when no same-export run exists);
- calibrated uncertainty/abstention;
- runtime, token, tool-call and compute receipts.

A system may return a partial answer. Missing evidence must remain unknown; it must not be silently converted into an empty list, zero dependencies, or a passing result.

## 4. Corpus plan

### 4.1 Initial size

Build 100–200 candidate tasks across 6–8 theorem families. These are candidate counts, not gold labels. Begin with source-pinned examples, then grow only after review throughput and agreement are measured.

Suggested family coverage:
1. propositional and predicate logic;
2. algebraic structures and homomorphisms;
3. elementary and algebraic number theory;
4. combinatorics and finite structures;
5. real / functional analysis;
6. order theory and lattice reasoning;
7. topology / geometry where the Lean corpus and expertise permit;
8. algorithms, semantics and verified reductions.

Each family should contain multiple mechanisms and difficulty levels. Avoid a corpus dominated by one small proof idiom such as And.comm.

### 4.2 Split design

Split by **theorem family and source lineage**, not random individual theorem:
- public development set: source-visible and annotated examples used to develop interfaces;
- public validation set: different theorems and source files, with answer keys held separately;
- hidden test set: entire held-out theorem families or source projects, with no prompt/answer leakage;
- mutation set: paired semantics-preserving and semantics-changing cases, with hidden mutation operators where feasible;
- transfer set: held-out family tasks that test reuse of a mechanism rather than memorization of a declaration.

Freeze task manifests, source hashes and split policy before system evaluation. Any task exposed during model development is marked as development, not hidden test. Record model training cutoff / known benchmark exposure where available; do not claim perfect contamination detection.

### 4.3 Gold qualification

A task remains candidate until:
1. theorem identity and environment are pinned;
2. two independent mathematical reviewers annotate statement, assumptions, proof skeleton and key roles;
3. disagreements are adjudicated by a third reviewer;
4. every formal-fact explanation target is linked to source or kernel evidence;
5. counterfactual answer keys distinguish theorem falsity, proof-route failure, unsupported inference and unknown;
6. the dependency extractor and proof artifacts are independently replayed where the task claims those checks;
7. contamination and licensing disposition are recorded.

Reviewer identifiers and traceability fields do not themselves prove reviewer independence or blinding. The study protocol must manage those separately.

## 5. Matched baseline matrix

Run the following arms on the same eligible tasks. Use the same base language model where possible and disclose when a published external service cannot be reproduced.

| Arm | System | What it isolates |
|---|---|---|
| B0 | Raw source / no retrieval | Minimum-context baseline |
| B1 | BM25 / sparse retrieval | Lexical retrieval |
| B2 | Dense embedding retrieval | Semantic retrieval |
| B3 | LeanSearch v2 | Strong global-premise retrieval baseline |
| B4 | LeanGraphSearch with graph expansion, PPR and reranking | Strong dependency-aware retrieval baseline |
| B5 | LLM-only proof-plan/explanation generation | Generation without external proof graph |
| B6 | LLM + retrieval | Retrieval-augmented generation |
| B7 | LLM + Lean feedback / kernel replay | Formal feedback without the complete evidence-plan pipeline |
| B8 | Lean-native typed dependency graph + reviewed role layer | Structured proof comprehension without Symthaea neural retrieval |
| B9 | Symthaea hybrid graph + HDC | HDC contribution |
| B10 | Symthaea hybrid graph + HDC/CfC | Temporal/dynamical contribution |
| B11 | Full evidence-qualified pipeline | Integrated system: typed graph, proof plan, certificates, mutation checks and trust capsule |

Add published comparator tracks where task licensing and protocol make direct reproduction possible:
- ProofGrid for proof writing/checking/masking/gap-filling and first substantive failure localization;
- ProofRank for conciseness, computational ease, cognitive simplicity, diversity and adaptivity;
- QEDBench rubrics/calibration for human alignment of proof evaluators;
- HumanLean's source/definition/scaffold/proof contract for paper-to-Lean faithfulness;
- APE-Bench's task contracts and dual syntactic/semantic verification;
- FormalMATH / EasyProver / TheoremBench for formal proof-generation and theorem-family coverage.

Do not substitute a published number from a different task distribution for an ECPCB result.

### 5.1 Budget matching

For each model-based arm, freeze and publish:
- model/provider/version and date;
- prompt and system configuration;
- total input/output token budget;
- retrieval corpus and snapshot;
- maximum retrieved candidates and context tokens;
- Lean tool calls, reflection rounds and per-task timeout;
- CPU/GPU, memory and concurrency ceilings;
- online/offline status and network access;
- per-task wall time and estimated monetary cost.

Run both a fixed-compute comparison and a fixed-cost comparison where practical. Report all exceptions. The final output must include raw per-task rows; aggregate means alone are insufficient.

## 6. Metrics

Report a scorecard by task family, difficulty, environment, system arm and evidence eligibility. Do not collapse these into one rank.

### 6.1 Retrieval and route reconstruction

- Recall@k, MRR and nDCG for premise/dependency retrieval;
- CompleteRoute@k: fraction of tasks where all reviewed required proof-route groups are recovered;
- typed-edge precision/recall (type, proof, definition, constructor, recursor, import);
- mathematical-role precision/recall against reviewer labels;
- unknown/abstain rate for unresolved roles.

A dependency's appearance in the source graph is not proof that it is mathematically necessary. Score structural retrieval and reviewed role classification independently.

### 6.2 Theorem identity and scope

- exact theorem/scope identity accuracy;
- target substitution detection;
- binder/universe/local-instance/context preservation;
- Tier-0 exact-scope eligibility and CIRR;
- Tier-1 structural-equivalence CIRR;
- Tier-2 kernel-certified context-transport CIRR;
- Tier-3 dependent-context cases, initially reported as unsupported/unknown until a reviewed transport formalism exists.

Keep each tier in a separate numerator and denominator. Never pool a larger Tier-1/2 denominator into the strict exact-scope CIRR headline.

### 6.3 Proof-plan and local action quality

- proof-plan node/edge precision and recall;
- action-to-goal binding accuracy;
- consumed-hypothesis and side-condition coverage;
- subgoal creation and discharge accounting;
- root-goal coverage;
- local certificate replay rate;
- false local-certificate acceptance;
- acceptance of alternate valid proof routes;
- plan reconstruction reviewer acceptance.

A local action is kernel_checked only when a Lean artifact proves the exact local proposition under the exact recorded context. Role labels remain expert-reviewed interpretations.

Global accounting is a separate sub-scorecard, using [PROOF-OBLIGATION-ACCOUNTING-001](https://github.com/Luminous-Dynamics/symthaea/issues/13641):
- every generated subgoal has exactly one producer action;
- every primary-route goal has exactly one disposition (discharged or transformed);
- every transformed goal has successor subgoals;
- no duplicate discharge, orphan output, or goal-transition cycle;
- an accepted plan has zero unresolved primary-route obligations and exactly one primary alternative;
- candidate alternatives remain isolated from the primary route.

These checks establish plan-graph completeness only. They do not prove an action valid or a goal true. Local terminal discharges still require exact-context Lean certificates.

### 6.4 Counterfactual and metamorphic robustness

Paired transformations:
- reorder independent hypotheses (invariance);
- alpha-rename bound variables (invariance only under a formal alpha-equivalence rule);
- change proof route without changing theorem (invariance of theorem truth; route identity changes);
- remove/strengthen/weaken a necessary assumption (scope sensitivity);
- alter binder type, universe, local instance or target (identity sensitivity);
- remove a side condition or local warrant (plan/certificate failure);
- replace a definition with a non-equivalent near-neighbor (semantic sensitivity).

Metrics:
- semantics-preserving invariance rate;
- semantic-mutation detection rate;
- false equivalence / false comparability rate;
- correct counterfactual disposition;
- checked-counterexample validity;
- correct abstention;
- proof-route-only false-refutation rate.

"Original proof no longer compiles" is not a counterexample to the mutated theorem. A theorem may remain true under a different proof even when the original route fails.

### 6.5 Trust closure and independent replay

Use the multi-axis assurance capsule from [#13603](https://github.com/Luminous-Dynamics/symthaea/issues/13603), including its digest-bound checker-diversity profile reference. The task contract requires an explicit status even when the profile is `not_assessed`; no independent-replay claim can inherit from a synthetic profile.

Report:
- seeded logical dependency recall and false-negative count;
- admission / axiom disclosure precision and recall;
- unresolved dependency handling;
- exact source/toolchain/import/export binding;
- official kernel acceptance;
- independent-checker acceptance of the same exported artifact;
- statement-comparison outcome;
- native-computation classification;
- construction sandbox/provenance completeness;
- checker disagreement and inconclusive rate.

A green final theorem check cannot compensate for an unknown target or undisclosed axiom. LLM/agent agreement is not proof authority.

### 6.5.1 Checker diversity and common-mode risk

Use the source-grounded checker registry and same-export profile from [CHECKER-DIVERSITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/13637). Lean's current `lake comparator --paranoid` adds five external checker paths to the official Lean kernel: `leanchecker-paranoid`, `lean4lean`, `nanoda`, `con-leche`, and `con-ron`. The six paths must not be counted as six independent algorithms.

The conservative current registry groups:
- official Lean kernel, `lean4lean` and `leanchecker-paranoid` in one official-kernel lineage;
- `con-leche` and its verified Rust refinement `con-ron` in one shared algorithm lineage;
- `nanoda` as a separate implementation/algorithm lineage.

It records three algorithm lineages and two distinct independently implemented external algorithm lineages for the complete bundle, while separately recording implementation/language diversity and pairwise relationships that remain unknown. This is an evidence classification, not a scalar independence score.

Report:
- all-checker same-export digest binding;
- checker version, source revision and binary digest;
- accepted, rejected, declined, error, timeout and resource-exhausted outcomes separately;
- distinct implementation families and algorithm lineages;
- independent external algorithm lineages;
- known term representations and parser families;
- unknown pairwise relationships;
- checker disagreement and per-checker resource usage.

A decline, timeout, crash or out-of-memory event is not a theorem refutation. It makes the assurance profile inconclusive. A checker rejection blocks concordance and triggers triage; it does not by itself establish that the informal theorem is false. For a `--paranoid` result, record the complete six-path bundle and preserve the raw per-checker receipts.

The primary reference is the [Lean Lake comparator documentation](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/), which documents the checker bundle, exact statement/axiom checks, sandbox behavior and residual assumptions.

### 6.6 Explanation and human transfer

Separate:
- statement restatement;
- assumption fidelity;
- key-lemma role explanation;
- proof-plan reconstruction;
- proof error detection;
- analogy-boundary recognition;
- immediate near-transfer;
- delayed near-transfer;
- held-out cross-family transfer.

Human study protocol and ethics gates remain in [MATH_PROOF_COMPREHENSION_HUMAN_STUDY_PREREG_V1.md](MATH_PROOF_COMPREHENSION_HUMAN_STUDY_PREREG_V1.md). Delayed transfer is the primary learning outcome candidate; immediate clarity/rating is secondary. No human data is implied by this benchmark protocol.

### 6.7 Efficiency

Report:
- wall-clock median/p95;
- GPU/CPU time and peak memory;
- token/tool-call counts;
- Lean elaboration and kernel-check cost;
- reviewer minutes;
- cost_per_accepted_explanation: total system + checker + review cost divided by independently accepted, fully scoped, evidence-complete explanations;
- cost_per_transfer_gain: total teaching intervention cost divided by estimated causal gain on the preregistered transfer outcome.

These denominators are intentionally demanding. Do not optimize raw theorem throughput at the expense of false acceptance.

## 7. Initial target gates

These are **prospective research targets**, not current measurements. The first phase establishes calibration and confidence intervals; targets must not be presented as achieved.

| Metric | Initial target | Stretch / interpretation |
|---|---:|---|
| Exact theorem identity on eligible controlled cases | 100% | Hard gate; zero silent target drift |
| False acceptance on frozen negative controls | 0 observed | Any false acceptance triggers an audit and blocks the affected release |
| Seeded axiom/admission dependency recall | 100% | Per-declaration-kind and per-Lean-version reporting |
| Unresolved dependencies silently treated as empty | 0 | Hard gate |
| Local certificate replay for actions marked checked | 100% | Exact artifact + exact context |
| Primary-route goal disposition | 100% | Every goal is discharged or transformed exactly once |
| Subgoal producer accounting | 100% | Exactly one producer per generated subgoal |
| Accepted plans with unresolved leaves | 0 | Hard gate |
| Accepted plans with goal-transition cycles | 0 | Hard gate |
| Checker input export digest binding | 100% | Every executed checker uses identical export bytes |
| Full paranoid checker bundle coverage when claimed | 100% | All six execution paths requested and observed |
| Checker outcome classification | 100% | Decline/error/timeout/resource exhaustion never collapsed into rejection |
| Distinct documented checker algorithm lineages | ≥3 | Do not count implementation ports as new algorithms |
| Independent external checker algorithm lineages | ≥2 | Count only supported lineage claims |
| Side-condition / subgoal accounting | ≥95% | Grow only after the gold annotation protocol is stable |
| Mutation sensitivity on scope-changing controls | ≥95% | Report false positives separately |
| Metamorphic invariance on allowed transformations | ≥99% | Only for transformations whose semantics are independently established |
| Proof-plan expert acceptance | ≥80% | Two-reviewer adjudication; report disagreement |
| Retrieval Recall@100 on a reproduced MathlibQR cohort | ≥95% stretch | LeanGraphSearch reports 94.7% on its paper's fair subset; use identical snapshot/protocol before comparison |
| Complete-route coverage@50 on reproduced MathlibMPR | ≥46% stretch | LeanGraphSearch reports 44.2%; never compare across changed cohort/budget |
| Delayed near-transfer | Positive effect vs. preregistered control | Require powered study and confidence interval excluding zero for a superiority claim |
| Minimum-sufficient explanation | ≥25% token reduction at ≤5 pp non-inferiority margin | Only if statement/assumption fidelity and delayed transfer remain non-inferior |

A target of zero observed false acceptance is not proof of a zero true error rate. Publish the number and diversity of negative controls and a confidence interval on the false-acceptance rate.

## 8. Statistical and annotation protocol

- Two independent mathematical reviewers for every gold task.
- Third adjudicator resolves disputed labels; preserve disagreement traces.
- Review proof roles separately from source dependency presence.
- Cluster bootstrap confidence intervals by theorem family/project, not only individual task, to account for correlated tasks.
- Report macro and micro averages; do not allow a large family to dominate the result.
- Use multiple model seeds / repeated runs where nondeterminism is meaningful.
- Freeze exclusions, missing-data rules, task budgets and primary contrasts before test unblinding.
- Do not use the same LLM family both to author all gold labels and to serve as the only evaluator.
- Calibrate judge scores against human labels and report bias, reliability and abstention. LLM judges never set formal theorem truth.

## 9. HDC/CfC ablation design

HDC and CfC are hypotheses to test, not assumed advantages.

Use identical source graphs, theorem splits, prompts, retrieval budgets and language-model backends:
1. BM25 only;
2. dense only;
3. Lean-native typed graph only;
4. LeanGraphSearch/PPR comparator;
5. lexical + dense + graph;
6. graph + HDC;
7. graph + HDC + CfC;
8. full Symthaea evidence pipeline.

Measure both the intermediate retrieval/plan metrics and downstream explanation/transfer metrics. Ablate HDC and CfC independently, not only as a combined stack. Include a no-op/randomized representation control and test performance on unseen theorem families. If HDC/CfC does not beat the strongest matched graph-aware baseline within uncertainty, do not credit it with an advantage on this task.

HDC similarity, temporal dynamics, centrality and agent consensus can prioritize candidate evidence. They cannot authorize a mathematical conclusion.

## 10. Current industry and research anchors

These are **published comparator results on their own protocols**, not head-to-head ECPCB results.

| Work | Published result / capability | ECPCB use |
|---|---|---|
| LeanSearch v2 (May 2026) | On MathlibMPR, 46.1% premise-group recall@10 and 30.4% complete-route coverage@10; standard mode nDCG@10 about 0.62 | Mandatory global-premise retrieval baseline |
| LeanGraphSearch (Oct 2026) | On its fair 810-query MathlibQR subset, Recall@100 94.7%; on MathlibMPR, complete-route coverage 42.8% @20 and 44.2% @50 | Strong dependency-aware retrieval baseline to reproduce |
| FormalMATH (2026) | 5,560 semantically validated Lean 4 statements; DeepSeek-Prover-V2 reported 28.31% Pass@32 | Formal proof generation anchor, not comprehension score |
| ProofGrid (2026) | 15 tasks spanning proof writing/checking/masking/gap filling; instrumented first-failure localization and epistemic-stability analysis | Fine-grained formal reasoning and failure-localization baseline |
| ProofRank (2026) | Proof quality proxies: conciseness, computational ease, cognitive simplicity, diversity and adaptivity | Add quality dimensions without conflating them with validity |
| QEDBench (2026) | 7 judges × 5 solvers, 1,000+ human-evaluation hours; reports meaningful judge/human alignment gaps and positive score inflation | Evaluator calibration and anti-LLM-judge-bias baseline |
| HumanLean (2026) | Staged source/definition/scaffold/proof review for paper-to-Lean fidelity | Reuse source-fidelity contract; human learning is a separate outcome |
| APE-Bench / APE-Harness (2026) | Dual syntactic and semantic verification across proof-engineering task types | Reuse task contracts and dual-verification structure |
| LeanLean (2026) | Repository-scale proof compression evaluation | Compare explanation compression only with matched fidelity/learning measures |
| Lean Kernel Arena / Lake Comparator | Positive/negative proof exports, multiple independent checker implementations, and checker-specific performance / decline behavior | Freeze specific rounds as same-export checker-diversity and differential-failure regression sets |
| RH / Dirichlet-GRH Lean verification (Nov 2026) | 175,187 modules; full export around 61 GB; official kernel, paranoid mode and independent checkers have reported acceptance, with an end-to-end v4.36 comparator stage still running as of 24 Nov | Large-artifact scalability and operational evidence lane; do not treat it as a Symthaea result or human-comprehension benchmark |

Public references:
- LeanSearch v2: https://arxiv.org/abs/2605.13137
- LeanGraphSearch: https://arxiv.org/abs/2610.22339
- FormalMATH: https://aclanthology.org/2026.findings-emnlp.579/
- ProofGrid: https://arxiv.org/abs/2605.12524
- ProofRank: https://arxiv.org/abs/2605.10379
- QEDBench: https://proceedings.mlr.press/v306/gonzalez26a.html
- APE-Bench: https://proceedings.mlr.press/v306/xin26c.html
- HumanLean: https://files.sri.inf.ethz.ch/website/papers/yuhao2026humanlean.pdf
- LeanLean: https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean
- Lean Lake / Comparator validation: https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/
- Lean Kernel Arena: https://arena.lean-lang.org/
- RH / GRH verification status (updated 24 Nov 2026): https://cairn-commons.com/guides/riemann-hypothesis-proof-claim-november-2026
- Independent RH / GRH Lean replay report: https://mrubinst.github.io/grh-lean-verification/

The papers differ in query sets, models, context, proof budgets and scoring. Do not add their percentages together or rank them as though they measured one task.

## 10.1 Large-artifact verification and human-understanding stress case

The 2026 Lean formalization of the Riemann Hypothesis and GRH for Dirichlet L-functions is an unusually valuable external stress case for ECPCB, but its status must be represented precisely.

The 24 November 2026 status page reports 175,187 modules, a full exported proof of about 61 GB, byte-identical sandbox rebuilds, standard axiom dependencies for the five headline theorems, module-by-module Lean kernel replay, acceptance by nanoda and lean4lean, statement comparisons through con-leche/con-ron, and acceptance of the full export by Lean's default and paranoid kernels (48 and 72 hours respectively). The same dated page says the final stage of a fully end-to-end Lake comparator run under Lean v4.36 was still in progress on 24 November. An independent Waterloo report also records complete replay of 175,187 modules and nanoda checking 1,484,729 declarations by 21 November.

Scope caveat: the formalized targets are the classical Riemann zeta hypothesis and GRH for Dirichlet L-functions. This does not extend to Dedekind zeta functions, Hecke or general automorphic L-functions. Nor should one conflate separate versioned checks: v4.33.1 module/kernel/independent checker results, a v4.33 comparator report with a stubbed build, and the v4.36 end-to-end comparator run have different evidential scope.

For ECPCB this is a **stress and comprehension challenge**, not a gold set by default:
- measure source-to-formal statement identity and dependency coverage;
- map proof modules into reviewed mathematical-role clusters;
- reconstruct selected local proof plans and global obligation paths;
- preserve all toolchain, export and module digests;
- measure checker runtime, peak memory, resource exhaustion, declines and replay throughput;
- ask experts to annotate selected modules, then test whether explanations help readers reconstruct the argument;
- keep the formal verification result separate from human understanding and publication status.

A 17-million-line formal proof can pass extensive machine checks while remaining difficult for humans to understand. That gap is central to ECPCB's research purpose, not a reason to diminish the machine-verification evidence.

## 11. Strategy to exceed the state of the art

Do not try to beat the strongest prover on every theorem benchmark at once. Pursue a sequence of credible wins:

1. **Match first:** reproduce a strong LeanSearch v2 / LeanGraphSearch retrieval slice on the exact public snapshot and budget. Establish no regression in the retrieval substrate.
2. **Add evidence:** show improved typed-edge and proof-role fidelity over raw static graph extraction, with expert adjudication.
3. **Add proof-plan correctness:** show that local certificates and global obligation accounting reduce false explanation claims, lost obligations and invalid warrant reuse.
4. **Add adversarial robustness:** outperform LLM-only and LLM+Lean baselines on target mutation, assumption mutation, missing side conditions, double-discharge and goal-cycle controls, and proof-route-only changes.
5. **Add human utility:** demonstrate that evidence-bound proof plans improve delayed transfer, not only immediate explanation ratings.
6. **Add efficiency:** lower the cost per independently accepted explanation while maintaining hard gates.
7. **Ablate architecture:** only then attribute gains to HDC/CfC or Symthaea-specific mechanisms.

The publishable result is not “Symthaea beats LeanGraphSearch at retrieval” unless the matched retrieval test proves it. The stronger possible claim is: **the evidence-carrying pipeline can produce more faithful, counterfactually robust and educationally useful explanations for the same formal proof corpus, with independently replayable evidence and measured cost.**

## 12. Execution sequence and stop conditions

### Phase P0 — contracts and fixtures
- freeze task schema and benchmark protocol;
- pin 6–10 pilot tasks as development-only;
- use existing scope, metamorphic and proof-action controls;
- validate JSON contracts and exact source/toolchain identity.

### Phase P1 — proof reconstruction
- consume Lean-native elaborated extraction as formal graph;
- add reviewed role annotations;
- produce ProofPlanV1 on a small theorem family;
- attach local certificates where formalizable;
- run global obligation accounting and preserve unresolved goal IDs;
- require zero unresolved primary-route obligations before any accepted plan status.

### Phase P2 — matched baselines
- run BM25, dense, LeanSearch v2, LeanGraphSearch, LLM-only and LLM+Lean;
- freeze budgets, corpus and prompts;
- keep per-task logs and exact run receipts.

### Phase P3 — counterfactual and trust
- run scope and metamorphic mutations;
- qualify the 15-root independent axiom audit on its exact head;
- validate the new trust-closure capsule against positive, negative and inconclusive fixtures;
- integrate comparator/export and the complete same-export checker-diversity profile for one small challenge;
- preserve checker declines/timeouts/resource failures as inconclusive, never theorem falsity.

### Phase P4 — human study
- finish expert task gold review;
- conduct feasibility/power analysis and ethics review;
- preregister primary endpoint and analysis;
- run delayed-transfer study only after task answer keys and arm conditions are frozen.

Stop or mark inconclusive if statement identity, proof artifacts, axiom policy, reviewer agreement, source licensing, contamination, or mutation answer keys cannot be established.

## 13. Claim ceiling

ECPCB v1 is a benchmark specification. It is not evidence of:
- improved human comprehension or delayed transfer;
- SOTA theorem proving or retrieval;
- a complete Lean trust audit;
- independent acceptance of any Symthaea-produced proof artifact;
- mathematical correctness of an explanation without adjudication.

Every benchmark release must expose what was run, what was checked, what is only reviewer interpretation, and what remains unknown.
