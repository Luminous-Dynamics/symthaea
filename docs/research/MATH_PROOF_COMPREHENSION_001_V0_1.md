# MATH-COMPREHENSION-001 — Proof-to-concept reconstruction and human transfer

**Status:** Research design, v0.1  
**Issue:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Branch:** `research/math-proof-comprehension-001`  
**Base snapshot:** `main @ 4c198f4fb610736ecbd2d5829fc01ff38ba4ee32` (2026-11-22)  
**Claim status:** Design only. No new theorem, Lean proof, benchmark score, or HDC/CfC advantage is claimed.

## 1. Thesis

A mathematical system should be evaluated on more than whether it can emit a proof term. It should also demonstrate that it can recover the mathematical claim, assumptions, definitions, inferential structure, and conceptual role of important constructions—and communicate these accurately enough that another person can use the ideas on a new problem.

We call this capability **proof comprehension**.

The proposed pipeline is:

`formal artifact → trusted theorem identity → dependency graph → concept graph → explanation claims → human transfer`

The direction must also work in reverse:

`informal claim / insight → candidate formal statement → Lean elaboration → kernel-checked proof or explicit unknown`

Each arrow has its own acceptance criteria. A successful downstream stage must not silently certify an upstream semantic correspondence that was never checked.

## 2. Why not another theorem prover?

The current benchmark landscape already contains strong theorem-completion and proof-engineering systems:

| Benchmark | Published scale / result | What it measures | What it does not establish |
|---|---|---|---|
| FormalMATH | 5,560 semantically validated Lean statements; strongest reported DeepSeek-Prover-V2 (671B): 28.31% Pass@32 | Formal mathematical proof generation | Human comprehension or explanation faithfulness |
| VeriSoftBench | 500 tasks, 23 repositories; best reported 41% curated-context and 35% full-repository (Pass@8) | Repository-context proof obligations | Conceptual reconstruction or reader transfer |
| LeanLean | 64 repositories; Opus 5 reports 48.3% average compression within 12 hours | Repository-scale proof compression | Whether the compressed proof is easier to understand |
| Discover-and-Prove | Hard-Mode MiniF2F/FIMO; 36 PutnamBench theorems formally proved in Hard Mode | Answer discovery before formal proof | Explanatory fidelity and human learning |
| TheoremBench | Nearly 100 classical theorem families with extracted supporting tasks | Subtheorem coverage and token efficiency | Whether people can reconstruct and transfer the proof idea |
| Formal Conjectures | 2,615 Lean statements, including 1,029 open conjectures and 836 solved problems for autoformalization | Research-level formal discovery and statement formalization | Multi-level explanation and measured human transfer |
| Aegis / EasyProver | 2,045 Rust–Lean program pairs, 11,533 candidate specifications; 52.7% proved in a stratified 2,000-statement sample under 30 minutes per statement | Implementation-derived verification | Mathematical proof pedagogy or concept transfer |

These figures come from different datasets, models, splits, attempt budgets, and evaluation protocols. They are a landscape—not a single comparable leaderboard.

**Research gap to test:** evaluate the chain from formal proof identity through conceptual explanation to human transfer, while making every formal assertion traceable.

## 3. Trust model

### 3.1 Evidence layers

Record the following independently:

1. **Source identity:** exact repository, revision, file paths, declaration names, toolchain and dependency lock.
2. **Statement identity:** elaborated theorem type, universes, binders, implicit parameters, namespaces and a canonical digest.
3. **Proof integrity:** proof artifact digest, axiom profile, admissions / `sorry` scan and checker receipts.
4. **Semantic fidelity:** whether the formal proposition matches the intended natural-language claim, including quantifiers, assumptions, domains and conclusion.
5. **Conceptual interpretation:** human-readable account of why definitions and lemmas matter, linked to supporting subgraphs and reviewed separately.
6. **Human utility:** whether a reader can correctly restate, critique and transfer the idea.

Kernel acceptance settles only the formal proposition under the accepted environment and trust model. It does not, by itself, validate a paper-to-theorem correspondence or the pedagogical correctness of an explanation.

### 3.2 Claim classes

Every explanation statement must be classified:

- `formal_fact`: exact source or declaration evidence is mandatory;
- `interpretation`: an interpretation of a proof subgraph, requiring review;
- `analogy`: analogy boundary and known failure modes must be stated;
- `unknown`: unresolved content is explicit; no completion is fabricated.

No model-confidence value, multi-agent vote, HDC similarity, or proof-text resemblance is evidence of theorem validity.

## 4. Typed proof-to-concept graph

### Node kinds

`theorem`, `hypothesis`, `definition`, `structure`, `notation`, `lemma`, `imported_result`, `proof_obligation`, `proof_term`, `tactic_block`, `transformation`, `source_claim`, `explanation_claim`, `example`, `counterexample`, `counterfactual`, `review`.

### Edge kinds

`depends_on`, `uses`, `unfolds_to`, `rewrites_to`, `specializes`, `generalizes`, `proves`, `formalizes`, `derives_from`, `equivalent_to`, `explains`, `illustrates`, `challenges`, `refutes`, `requires_assumption`, `removes_assumption`, `transfers_to`.

Edges such as `equivalent_to` require an explicit evidence record and reviewer disposition. Never infer them from lexical or vector similarity.

The graph is an explanation/indexing layer over existing Lean and proof-memory infrastructure, not a new proof authority or competing theorem store.

## 5. Architecture proposal

### A. Lean-native extraction

Use Lean's elaborated environment and declaration metadata rather than trying to infer proof structure from raw tactic text alone. Capture:
- declaration name and elaborated type;
- universe and binder information;
- constant references and dependency closure;
- imported module versions and source spans;
- proof-term identity and checker metadata;
- notation / coercion / typeclass context needed to interpret a declaration.

Tactic syntax may be retained as provenance, but the semantic dependency graph should be extracted from elaborated declarations.

### B. Mathematical normalization

Create a deterministic intermediate representation for:
- quantifiers and binder scope;
- definitions and typeclass assumptions;
- algebraic / analytic transformations;
- hypotheses and domain constraints;
- theorem conclusions;
- witness and counterexample obligations.

Do not normalize away distinctions that affect theorem meaning. Preserve a link from every normalized object to the original Lean declaration and source.

### C. Concept retrieval

Compare:
- BM25 / lexical search;
- dense embedding retrieval;
- dependency-graph traversal;
- symbol- and type-directed retrieval;
- graph + lexical + dense hybrid;
- hybrid plus HDC;
- hybrid plus HDC and CfC.

HDC is a candidate structural index for relational patterns, not a proof checker. CfC is a candidate for maintaining investigation context across long tasks, not a mathematical inference oracle. Retain either only when matched-budget ablations show a reproducible improvement on held-out theorem families.

### D. Explanation synthesis

Generate multiple levels from the same verified source:
- one-paragraph orientation;
- undergraduate explanation;
- graduate proof sketch;
- specialist dependency account;
- Lean declaration / proof trace.

Every level should preserve the same theorem identity and assumptions. Each factual mathematical claim should cite its supporting nodes. Interpretations, analogies and unresolved steps remain visibly distinct.

### E. Counterfactual learning

Introduce controlled mutations:
- remove one hypothesis;
- weaken a bound;
- change a domain or endpoint;
- substitute a nearby definition;
- replace a lemma with a weaker result;
- change a reduction guarantee from parsimonious to decision-preserving.

The system predicts what breaks, then the prediction is checked with Lean, a finite counterexample, symbolic derivation, or independent mathematical review. A failed build of the old proof is not itself a counterexample to the changed theorem.

## 6. Initial case studies

### 6.1 Alpöge–Claude RH formalization

The Lean and Mathlib communities issued a joint statement dated 2026-11-21 certifying the RH statement and proof after build, statement-comparison, axiom and independent-checker review. The archive is reported as 175,187 modules and roughly 17 million lines.

Use this as a **comprehension** test, not a re-proof task. Michael Rubinstein's 164-page working guide is useful context but explicitly leaves a central weighted-prime inequality (M) to the Lean development. The first RH slice should therefore:
1. pin the published archive and exact theorem declarations;
2. select one bounded dependency subgraph around a named bottleneck;
3. map the guide's prose to exact Lean declarations;
4. mark what the guide explains versus what it imports from Lean;
5. obtain independent mathematical review of the reconstructed explanation.

Do not imply that a small slice explains all of RH or that Symthaea independently checked the upstream proof.

### 6.2 ETH / exact-counting slice

Use one transformation from #10811 and one complexity-transfer rule from #10835. Demonstrate:
- why the active task needs decision, counting, unique-extension, approximation, or parameter preservation;
- where that guarantee appears in the reduction proof;
- how the target-size and parameter maps affect the complexity claim;
- how an intentionally invalid guarantee upgrade is detected.

This gives the P-vs-NP-adjacent programme a concrete explanatory deliverable without confusing an ETH result with P=NP or P≠NP.

### 6.3 Small theorem pilot

Before the RH-scale task, select 3–5 small Mathlib theorem dependency subgraphs with expert-reviewed informal proofs. This validates the schema, extraction and human evaluation protocol at tractable scale.

## 7. Evaluation protocol

Keep tracks separate; do not hide them in a single score.

| Track | Measurement |
|---|---|
| Statement fidelity | Exact quantifiers, hypotheses, domains, conclusion |
| Dependency comprehension | Precision / recall of material dependency edges and lemma roles |
| Proof-plan reconstruction | Expert rating of the mathematical skeleton |
| Definition understanding | Correct unfolding, examples and boundary cases |
| Counterfactual diagnosis | Accuracy on mutated assumptions, definitions and reductions |
| Explanation faithfulness | Fraction of factual claims with exact supporting evidence; false-fact rate |
| Human comprehension | Blinded pre/post understanding and time-to-correct-understanding |
| Near transfer | Ability to solve a related but unseen problem after explanation |
| Cross-family transfer | Apply a concept learned in one theorem family to a held-out family |
| Efficiency | Wall time, CPU/GPU, memory, tokens, calls, reviewer minutes and monetary cost |

### Baseline arms

Use identical models, frozen prompts, tool access, token limits and wall-clock budgets for:
1. raw Lean source only;
2. source-only LLM summary;
3. lexical RAG;
4. dense RAG;
5. dependency graph only;
6. symbolic / type-directed retrieval;
7. graph + lexical + dense hybrid;
8. hybrid + HDC;
9. hybrid + HDC + CfC.

Use family-level and time-forward holdouts. Keep all model-generated explanations out of gold annotations. Human reviewers should be blind to system arm where practical. Use two independent annotators and adjudicate disagreements rather than averaging them away.

### Hard safety / correctness gates

- Zero accepted theorem-statement substitutions or weakened claims.
- Zero unsupported statements labelled as formal facts.
- Zero proof verdicts inferred from HDC, model confidence or agent consensus.
- `unknown` is allowed and must score better than a fabricated explanation.
- Every formal acceptance binds to source revision, elaborated theorem type and checker receipt.

### Prospective targets

These are target thresholds, not measurements:
- **T0:** exact identity binding on every task; zero false-positive proof / statement acceptance on all negative controls.
- **T1:** ≥90% precision and ≥85% recall on expert-labelled material dependency edges; ≥90% support coverage for formal-fact explanation claims.
- **T2:** ≥80% expert acceptance of proof skeletons on held-out theorem families.
- **T3:** ≥20 percentage-point improvement in reader comprehension or near-transfer over raw-source and source-only baselines, preregistered and replicated.
- **T4:** ≥10 percentage-point transfer improvement on unseen theorem families without increased false-fact rate.
- **T5:** an HDC/CfC gain only if matched ablations reproduce it; otherwise use the simpler system.
- **T6:** lower explanation cost without reducing fidelity, comprehension or transfer.

First publish baseline performance; do not claim SOTA based on aspirational targets.

## 8. Work plan and stop gates

| Stage | Deliverable | Stop / pass condition |
|---|---|---|
| P0 | Proof identity and trust receipt | Exact source, theorem, type, axiom and checker identity bound |
| P1 | Lean dependency extractor | Deterministic extraction; mutation tests catch stale or missing declarations |
| P2 | Typed concept graph + claim schema | All graph nodes and edges have source provenance |
| P3 | 3–5 theorem comprehension pilot | Independent reviewer accepts identity and core proof skeleton |
| P4 | Blind human study | Preregistered protocol; no cherry-picked readers or tasks |
| P5 | RH slice | Only after source pins and small-pilot gates pass |
| P6 | ETH reduction slice | Invalid guarantee upgrade is rejected |
| P7 | HDC/CfC ablations | Keep only causally useful components |

Do not wait for a large runner to start the research: source inspection, schema validation, task design, annotation rubrics and small exact counterexamples can proceed independently. However, no Lean compile / kernel result or Rust test result is claimed until it has actually run.

## 9. Required receipt

The companion JSON Schema is `MATH_PROOF_COMPREHENSION_RECEIPT_V1.schema.json`. Every receipt must bind:
- source and toolchain;
- theorem identity and proof artifact;
- axiom and admission profile;
- checker verdicts and raw-output digests;
- concept graph nodes and typed edges;
- explanation claims and evidence references;
- reviewer decisions;
- human evaluation protocol and raw outcomes;
- known limitations and unresolved obligations.

## 10. External reference points

- Alpöge proof verification, Lean / Mathlib community statement (2026-11-21): https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
- Rubinstein, working RH explanation (2026-11-21): https://www.math.uwaterloo.ca/~mrubinst/alpoge_claude_grh/
- FormalMATH (EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.579/
- VeriSoftBench: https://utopia-group.github.io/VeriSoftBench/
- LeanLean: https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean
- Discover and Prove (ACL 2026): https://aclanthology.org/2026.acl-long.3/
- TheoremBench: https://arxiv.org/abs/2606.09450
- Formal Conjectures: https://arxiv.org/abs/2605.13171
- EasyProver / Aegis (2026-10-28): https://arxiv.org/abs/2610.38174


## 11. SAT / ETH implementation readiness audit (2026-11-22)

A source-level inspection of the current public issues and draft PRs gives the following readiness boundary:

| Component | Current source-level state | What remains before it can support a promoted result |
|---|---|---|
| #10801 / PR #10801 — reference CNF semantics | Draft change correctly adds OR-over-AND distribution, explicit TRUE/FALSE clause conventions, normalization and branch-state isolation in the diff | Rust format/tests, exact-head CI, independent review; keep this narrow repair distinct from solver scalability |
| #10809 / PR #10809 — Tseitin | Draft source adds biconditional gate constraints, root assertion, deterministic collision-checked auxiliary names, and small-formula unique-extension tests | Rust compile/tests and independent review; bounded input depth/output size; complete integer-ID / projection contract from #10804 |
| #10810 — typed decision outcomes | Open specification correctly separates SAT model, UNSAT decision, theorem validity and independently checkable proof | Migrate callers and ensure SAT model validation / total variable universe; UNKNOWN, timeout, resource limit and error stay distinct |
| #10849 — bounded CNF conversion | Open source-review issue identifies unbounded Cartesian expansion and unsafe allocation sizing in the distributive path | Checked size arithmetic, deterministic budgets, fallible reserve and typed incomplete outcomes; never return partial CNF as complete |
| #10803 — independent UNSAT verification | Open plan for DIMACS, pinned Kissat/CaDiCaL adapters and DRAT/LRAT-style independent proof checking | Actual parser/adapters/checker and executable negative controls |
| #10805 / #10784 — benchmarks | Public plan pins SAT Competition 2026 data and defines reproducible solver comparisons | Executed same-machine, per-instance baselines and proof/model validation; no solver score until then |
| #10049 / #10090 — ETH / exact counting | Research programme and exact-counting benchmark specifications are open | Reconstruct the full algorithm/proof, bit complexity, independent implementation and matched exact-counter comparisons |

### Specific source-level caution

The current `dpll_sat` implementation still calls the distributive `to_cnf` path rather than the Tseitin API. Thus PR #10809, even if merged, would not by itself make the DPLL execution path scalable. The distributive path's Cartesian-product capacity calculation uses saturating multiplication, which prevents arithmetic wraparound but does not impose a safe resource budget. The current decision API also returns a legacy `ProofResult` for SAT/UNSAT; its trace is not an independently checkable UNSAT certificate.

These observations are source-level review findings, not runtime test results. They should be resolved through the existing #10810, #10849, #10803 and #10804 work items rather than creating competing APIs here.

### ETH/P-vs-NP interpretation

The exact-counting programme is an ETH-adjacent research programme. A deterministic subexponential algorithm for a suitable #SAT class would have consequences for ETH, but it is not a polynomial-time SAT algorithm and does not establish P=NP. Keep these claims separate:
- formal theorem status;
- paper-to-Lean correspondence;
- machine / encoding translation;
- exact implementation correctness;
- competition performance;
- worst-case asymptotic bound.

For practical SAT performance, compare against the pinned 2026 competition winners and specialist solvers using official per-instance data. For exact counting, compare against the matching Model Counting Competition track and specialized counters. Do not compare #SAT results to SAT solved counts or approximate counters to exact counters.


## 12. Current frontier correction and benchmark strategy (2026-11-22)

### 12.1 RH verification versus Clay prize status

The Lean and Mathlib communities' joint statement (2026-11-21) says they have full confidence in Alpöge's proof, certifies the statement and proof in their formal-verification scope, and describes the multiple independent checks. Andrew Sutherland's MIT verification page was updated 2026-11-22 16:20 UTC. It records compilation, axiom inspection, module-by-module replay, acceptance by nanoda and con-leche, acceptance by lean4lean, and theorem comparisons by con-leche / con-ron. At that update, the full `lake comparator` run and one full Lean-kernel replay of the exported proof were still in progress.

Separately, the Clay Mathematics Institute still labels the Riemann Hypothesis **Unsolved** on its official problem page. This is not a contradiction in the evidence: formal proof checking and the Clay prize procedure are different questions. The 2018 Clay rules require publication in a qualifying outlet, at least two years after publication, and general acceptance in the global mathematics community before the Institute considers a prize. Do not describe the prize as awarded or the Clay status as changed.

The GRH claim covered by the Lean development is for Dirichlet L-functions. Do not silently expand it to every generalized Riemann hypothesis for all L-function families.

Sources:
- Lean / Mathlib community statement: https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
- MIT verification ledger (status is time-stamped and may change): https://math.mit.edu/~drew/grh/
- Official Clay RH status: https://www.claymath.org/millennium/riemann-hypothesis/
- Official Clay prize rules: https://www.claymath.org/millennium-problems/rules/
- Rubinstein working explanation and declared gap at inequality (M): https://www.math.uwaterloo.ca/~mrubinst/alpoge_claude_grh/

Our programme should therefore record five separate fields: formal statement identity, checker outcomes, independent mathematical reconstruction, community acceptance, and Clay prize status.

### 12.2 Industry benchmark snapshot

Do not compare these scores as if they measured the same task.

| System / benchmark | Current published reference | Research lesson for Symthaea |
|---|---|---|
| FormalMATH | 5,560 semantically validated Lean statements; DeepSeek-Prover-V2 671B reports 28.31% Pass@32 | Statement-supplied proof generation remains difficult; report domain strata and false-proof controls |
| EasyProver / Aegis | 669/672 PutnamBench; 1,054/2,000 sampled implementation-derived specifications proved within 30 minutes | Easy contest sets are near saturation; implementation-derived and mutation-sensitive tasks are more discriminative |
| VeriSoftBench | 500 tasks / 23 repositories; reported best 41% with curated dependency context and 35% with full repository context | Dependency curation matters; graph extraction and context selection are measurable engineering targets |
| LeanLean | 64 repositories, up to 174k lines; Opus 5 reports 48.3% average compression within 12 hours | Whole-repository proof restructuring is now a benchmarked capability; compression must be evaluated alongside compute and proof readability |
| TheoremBench | Nearly 100 classical theorems, with main and supporting-subtheorem tasks | Measure coverage across the proof structure, not just final-theorem pass/fail |
| Formal Conjectures | The May 2026 paper snapshot reports 2,615 statements; the live public site currently advertises a larger evolving collection | Pin immutable snapshots and preserve source-lineage / contamination metadata |

These results are taken from the benchmark papers or live benchmark pages and have different models, budgets, splits and protocols:
- https://aclanthology.org/2026.findings-emnlp.579/
- https://arxiv.org/abs/2610.38174
- https://utopia-group.github.io/VeriSoftBench/
- https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean
- https://arxiv.org/abs/2606.09450
- https://arxiv.org/abs/2605.13171
- Live Formal Conjectures collection: https://google-deepmind.github.io/formal-conjectures/

### 12.3 Additional paradigm shifts to test

**Minimum Sufficient Explanation (MSE) frontier.** Instead of asking only whether an explanation is correct, search for the smallest source-linked proof/concept subgraph and explanation that still lets a reader pass a preregistered comprehension and transfer test. Report the Pareto frontier of explanation tokens, dependency coverage, expert faithfulness and human transfer. Do not equate shortest explanation with best explanation.

**Proof-to-human-to-Lean round trip.** A reader learns from a proof explanation, writes a paraphrased theorem or proof sketch, and a separate system formalizes it. Lean checks whether the resulting statement is equivalent to the source statement and whether the proof is valid. This gives a measurable bridge from comprehension to formalizable knowledge.

**Counterfactual proof understanding.** Change one assumption, quantifier, endpoint or dependency. Ask the system to predict what fails and provide a minimal counterexample or a corrected theorem. Score truth of the diagnosis, not whether the original proof happens to stop compiling.

**Mechanism transfer.** Hold out entire theorem families. Ask whether a method learned from one proof (for example, implication-based redundancy, finite case analysis, interpolation, low-rank factorization, or witness reconstruction) helps with a different family. This is the strongest evidence that the model acquired reusable mathematical knowledge rather than memorized a proof.

**Explanation evidence accounting.** Each mathematical sentence is linked to a Lean declaration, an elaborated dependency, a formal derivation, a counterexample or an explicit interpretation label. An explanation cannot silently promote a conjecture, analogy or intuition into a proved fact.

### 12.4 Experimental design improvements

- Freeze benchmark source snapshots, model and Lean versions, prompts, tools, token budgets, wall-clock budgets and random seeds.
- Separate public development tasks, sealed family-level test tasks and prospective future tasks.
- Use two independent human annotators and an adjudication log. Do not treat model agreement as ground truth.
- Preregister the human study and run a power analysis before confirmatory claims. Use a small pilot only to estimate feasibility and variance.
- Report participant background, pre-test scores, post-test scores, transfer accuracy, time-to-correct-understanding, confidence calibration and attrition.
- Report compute and money alongside quality: CPU/GPU time, memory, tokens, tool calls, Lean compile time and expert-review minutes.
- Preserve all failures: wrong statements, false explanations, unsupported analogies, timeouts, non-compiles, abstentions and reviewer disagreements.
- Run HDC/CfC ablations at identical retrieval corpus, context size and compute budget. If there is no robust gain, remove these components from the proof-comprehension path.

### 12.5 Refined target ladder

In addition to the existing goals, evaluate:
- **MSE-1:** at least 95% of formal explanation claims trace to exact declarations or explicitly labelled interpretation sources.
- **MSE-2:** reduce explanation token budget by at least 25% at matched human comprehension and transfer—not by shortening text alone.
- **Round-trip-1:** at least 90% statement-equivalence accuracy on controlled paraphrases, with zero acceptance of weakened or altered statements.
- **Counterfactual-1:** at least 90% correct disposition on held-out assumption-removal mutations; exact minimal counterexample required when the modified proposition is false.
- **Transfer-1:** at least 10 percentage-point improvement on theorem-family holdouts with no increase in false-fact rate.
- **Efficiency-1:** publish quality–compute–human-time Pareto curves, not a single aggregate score.

All thresholds are prospective. First measure reproducible baselines and confirm task difficulty. A hard gate failure cannot be averaged away.
