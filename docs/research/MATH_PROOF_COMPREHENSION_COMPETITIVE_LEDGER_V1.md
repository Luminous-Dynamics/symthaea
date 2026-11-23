# Mathematical proof comprehension — competitive landscape and benchmark ledger v1

**Parent:** [#13293 MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Date:** 2026-11-22  
**Status:** literature baseline, not Symthaea benchmark results.

## Executive positioning

Symthaea should not compete on a single blended theorem-proving score. The field's current benchmarks measure different capabilities:

- formal completion on a supplied statement;
- discovery of the theorem before proving it;
- repository-scale proof obligations and dependency retrieval;
- proof compression / restructuring;
- formalization of implementations;
- theorem-family transfer;
- human comprehension and explanation fidelity.

The last two, when linked to a formal evidence graph and a blind human study, are the proposed differentiation. We do not claim no one has studied explanation; we claim this exact joined evaluation—formal identity, proof-mechanism reconstruction, explanation support, counterfactual understanding and human transfer—needs to be benchmarked as a coherent capability.

## 1. Benchmark ledger

| Benchmark / work | Public reference | Published scale / headline result | Capability measured | Strategic implication |
|---|---|---|---|---|
| FormalMATH (EMNLP Findings 2026) | https://aclanthology.org/2026.findings-emnlp.579/ | 5,560 semantically validated Lean 4 statements; DeepSeek-Prover-V2 671B: 28.31% Pass@32 | Formal proof generation from formal statements | Use as a proving baseline, but measure theorem-family transfer, claim fidelity and explanation quality separately |
| EasyProver / Aegis (Oct 28, 2026 preprint) | https://arxiv.org/abs/2610.38174 | 669/672 PutnamBench; all six IMO 2026 formalizations; 597/600 NTP4VC; 189/189 VERINA; Aegis has 2,045 Rust–Lean program pairs and 11,533 candidate specs; 1,054/2,000 sampled specs proved within 30 minutes | Multi-agent Lean proving and implementation-derived verification | Contest theorem accuracy alone is saturated; add implementation semantics, mutations and hidden specifications |
| VeriSoftBench | https://utopia-group.github.io/VeriSoftBench/ | 500 tasks from 23 Lean repositories; 41% best with curated context, 35% full raw-repository context | Repository-context software verification | Dependency selection/context construction is a measurable bottleneck and useful baseline |
| LeanLean (2026) | https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean | 64 repositories up to 174k LOC; Opus 5 reports 48.3% average proof-size compression within 12 hours | Repository-scale proof compression and restructuring | Add an explanation-compression frontier, but do not equate shorter code with better mathematical understanding |
| TheoremBench (Jun 8, 2026) | https://arxiv.org/abs/2606.09450 | Nearly 100 classical theorems; plain and premised versions with subtheorem families | Main theorem and supporting-subtheorem coverage, token efficiency | Compare proof dependency coverage, not just final theorem pass/fail |
| Formal Conjectures (May 13, 2026 paper snapshot) | https://arxiv.org/abs/2605.13171 | 2,615 statements in the paper snapshot; 1,029 open research conjectures and 836 solved tasks for autoformalization | Research conjectures and formalization | Use immutable time-stamped family holdouts and contamination controls |
| Discover-and-Prove (ACL 2026) | https://aclanthology.org/2026.acl-long.3/ | Hard-Mode MiniF2F/FIMO; reports 36 PutnamBench theorems formally proved in Hard Mode | Discovery before proof generation | Use as discovery baseline; add conceptual and cross-family transfer |
| Aegis implementation lane | https://arxiv.org/abs/2610.38174 | Real Rust-to-Lean program pairs and candidate specifications | Correctness, safety and semantics of code | Useful transfer target for formal explanation and human-readable specification reasoning |

Scores and budgets are not directly comparable across rows. Cite the paper's own protocol. Do not create an aggregate rank across these tasks.

## 2. Where Symthaea should compete

### Track A — proof identity and formal trust

Baselines: Lean kernel, theorem prover output, #11362's claim-to-Lean fidelity checks.

Targets:
- exact theorem/source/toolchain identity on every task;
- zero false-positive theorem acceptance on changed/weakening/namespace-shadowing controls;
- explicit axioms, admissions, unknown and timeout;
- no HDC similarity or model confidence as proof evidence.

This is a hard gate, not a score that can be averaged.

### Track B — repository dependency comprehension

Baselines:
- raw repository context;
- curated context (VeriSoftBench-style);
- lexical/BM25 retrieval;
- dense retrieval;
- Lean dependency graph;
- symbolic/type-directed retrieval;
- hybrid graph + lexical + dense;
- hybrid plus HDC.

Targets:
- ≥90% precision / ≥85% recall on expert-labelled material dependencies;
- distinguish type dependency, proof dependency, imported infrastructure, inductive/recursor edges and mathematical role;
- do not claim every syntactic dependency is mathematically essential.

### Track C — proof mechanism reconstruction

For each proof slice, require:
- exact theorem claim and all assumptions;
- each key lemma's role and the direction in which it is used;
- a derivation skeleton in mathematical prose;
- one worked example / counterexample if available;
- a list of quoted or unresolved obligations;
- per-sentence evidence links for formal facts.

Target: ≥80% independent expert acceptance on whole held-out theorem families, with reviewer disagreement retained.

### Track D — counterfactual understanding

Mutate one binder, assumption, endpoint, definition or reduction guarantee.

Require:
- predict whether the changed statement is still true;
- provide a concrete counterexample when false;
- distinguish "proof no longer typechecks" from "the mathematical proposition is false";
- detect theorem substitutions and weakened claims.

Targets:
- ≥90% correct disposition on held-out controlled mutations;
- zero acceptance of altered statement as original;
- abstention is preferred over fabrication.

### Track E — human comprehension and transfer

Compare raw Lean/source, generic LLM summary, dependency graph only, and source-linked concept explanation.

Primary outcomes:
- exact restatement of theorem and assumptions;
- explanation of selected construction / lemma role;
- near-transfer problem accuracy;
- confidence calibration.

Secondary:
- time-to-correct-understanding;
- ability to detect a deliberately planted missing assumption;
- ability to write a formalizable paraphrase;
- explanation length / token and reviewer cost.

Targets:
- ≥20 percentage-point improvement in comprehension or near-transfer over raw-source and source-only baselines;
- ≥10 percentage-point improvement on unseen theorem families;
- no increase in false-fact rate.

### Track F — explanation compression (Minimum Sufficient Explanation)

Treat explanation length as a constrained optimization objective, not as a standalone score.

Find the Pareto frontier of:
- token count;
- dependency coverage;
- formal-fact support coverage;
- expert faithfulness;
- human comprehension;
- near-transfer.

Target: ≥25% fewer explanation tokens at matched comprehension within a preregistered 5 percentage-point non-inferiority margin. The frontier must include full-source and dependency-only baselines. A shorter explanation that drops assumptions or hurts transfer is a failure.

### Track G — mathematical mechanism transfer

Hold out entire theorem families. Train/tune on separate families. Ask whether a mechanism learned in one proof—recursion, quotient induction, local-to-global composition, interpolation, valuation arguments, symmetry reduction, witness reconstruction—helps solve or explain an unseen instance in another family.

Target: at least +10 percentage points over a source-only baseline without increased false-fact rate. Require a task-level explanation of the transferred mechanism.

## 3. HDC / CfC competitive ablations

Use identical corpus, model, prompt, context budget and compute:
1. BM25;
2. dense embeddings;
3. dependency graph only;
4. symbolic / type-directed retrieval;
5. lexical + dense + graph hybrid;
6. hybrid + HDC;
7. hybrid + HDC + CfC.

Report recall@k, MRR, material-edge precision/recall, statement fidelity, explanation support, transfer, wall-clock time, memory and tokens. Do not claim HDC/CfC advantage from one benchmark or a similarity score. If the improvement fails to reproduce on held-out theorem families, remove the component from this capability.

## 4. Industry comparison strategy

We do not have to beat EasyProver at PutnamBench to make a meaningful contribution. We should:
- reproduce published proof-generation baselines under compatible budgets;
- compare repository comprehension against VeriSoftBench-style context conditions;
- compare proof simplification against LeanLean under matched resource ceilings;
- compare subtheorem/coverage behavior against TheoremBench;
- compare research discovery against Formal Conjectures / Discover-and-Prove;
- introduce the proof comprehension + human-transfer suite as a separate track.

The new capability should be evaluated with open data, pinned environments, raw per-task rows and independent human review. We should publish negative results and false-positive rates. The ambition is to become a strong complementary research system, not to misrepresent one result as beating all of industry.

## 5. Claim policy

A leaderboard improvement on a supplied theorem is proof-generation evidence.  
A smaller file is proof-compression evidence.  
A correct dependency graph is structural evidence.  
A human-readable explanation is only explanation evidence when reviewed.  
A human transfer improvement is a learning/utility result.

Do not merge these into a single "mathematical intelligence score" until the component metrics are independently reliable, and even then keep the hard trust gates separate.
## 6. Newly identified adjacent state of the art (2026-11-22)

The first version of the ledger understated three directly relevant developments. They must become explicit baselines / interoperability targets.

### HumanLean — faithful paper-to-Lean formalization

**Paper:** Yuhao et al., *Faithful Lean Formalization for Human Proof Review*, under review (2026). Public paper PDF: https://files.sri.inf.ethz.ch/website/papers/yuhao2026humanlean.pdf

HumanLean's key contribution is a structured formalization contract with staged source / definition / scaffold / proof review. It separates accepted goals, excluded goals, external dependencies and proof routes; external facts not formalized in Mathlib can be declared as axioms in the scaffold, with that boundary made explicit. It also constrains the Lean layout so one main theorem per goal and helper lemmas are isolated for review.

On nine theoretical papers:
- GPT-6-Astra baseline: 56/150 accepted goals, 56 formalized, 55 judged faithful; HumanLean: 60/150 accepted, 60 formalized, 60 judged faithful.
- Aggregate faithfulness: 98.2% baseline vs 100% HumanLean for GPT-6-Astra.
- Two independent human reviewers with Lean/math expertise confirmed the 100% faithfulness claim for HumanLean/GPT-6-Astra. They spent 5–20 minutes per goal, mostly recovering relevant paper context.
- HumanLean reports reducing axiom exposure from 0.20 to 0.07 per formalization with GPT-6-Astra, and custom-definition verbosity from 11.80 to 1.13 per formalization.
- Cost increased: $6.7 to $8.7 per formalized goal for GPT-6-Astra. This is a real quality/review-burden trade-off, not a free improvement.

**Implication:** We should reuse HumanLean's claim/goal contract, source audit discipline and independent review protocol where possible. Do not build a competing paper-to-Lean formalization pipeline. Our work begins from already formalized artifacts and measures conceptual reconstruction, explanation faithfulness, counterfactual understanding and whether humans can use what they learned on a new problem. HumanLean's two-reviewer paper-faithfulness score is not a human-learning/transfer result.

Important limitations: this is a paper-level result under review, with nine paper case studies; its faithfulness is bounded by the selected goals and explicit external-dependency policy. Axiomatized external facts must not be presented as internally proved. It does not establish that readers learn or transfer mathematical mechanisms after reading an explanation.

### APE-Bench / APE-Harness — repository-scale proof engineering

**Paper:** Xin et al., ICML 2026 / PMLR 306: https://proceedings.mlr.press/v306/xin26c.html  
**Public implementation:** https://github.com/xinhjBrant/APE-Bench  
**Code snapshot:** commit `20017343657a7ce77fc67467c3dc9e443c81c075`, MIT license.

APE-Bench introduces dual verification: a task must satisfy both syntactic compilation and semantic requirement checks in a pinned library environment. The public implementation documents:
- 100 proof-engineering tasks;
- 64 semantic-judge validation tasks;
- theorem proving, proof engineering, judgment, instruction synthesis and library-annotation task types;
- multi-version execution and version-scoped retrieval;
- task contracts, isolated runtimes, and reference scaffolds for APE-Agent, Claude Code and Codex.

**Implication:** Adopt dual verification and the task-contract abstraction for our machine evaluation. Use APE-Bench proof-engineering and semantic-judgment tasks as external regression lanes where licensing/data conditions allow. Do not duplicate its execution harness or call our prose-only pilot an APE benchmark.

### LeanGraphSearch — strong dependency-aware retrieval baseline

**Paper:** Xu et al., *Lean Graph Search for Formal Theorem Proving* (Oct 15, 2026): https://arxiv.org/abs/2610.22339

It builds a Mathlib declaration graph using jixia static analysis and performs query-personalized PageRank from semantically retrieved seeds. Its reported results include:
- MathlibQR Recall@100: 94.7%, versus 82.7% for LeanSearch v2 on the 810-query fair subset;
- MathlibMPR complete-route coverage: 42.8% vs 39.1% at @20, and 44.2% vs 41.3% at @50;
- downstream proof pass-rate gains up to 40% over a vanilla prover and up to 10% over LeanSearch v2 across the paper's evaluated benchmarks and budgets.

**Implication:** A generic "dependency graph retrieval" arm is not enough. Explicitly reproduce LeanGraphSearch's seed retrieval + PPR + reranking baseline, or use its pinned implementation / reported results with compatible caveats. Compare our Lean-native extracted graph, jixia-style static graph and upstream kernel-export graph; measure edge differences rather than assuming these methods produce identical graphs.

### Updated competitive positioning

| Layer | Strong existing comparator | What Symthaea must add |
|---|---|---|
| Paper ↔ Lean fidelity | HumanLean | Explanation of an existing formal proof and measured reader understanding / transfer |
| Repository proof engineering | APE-Bench / APE-Harness | Conceptual dependency map and proof mechanism reconstruction |
| Dependency retrieval | LeanGraphSearch / LeanSearch v2 | Role-aware explanations, counterfactuals and human transfer; HDC only if an ablation beats graph retrieval |
| Proof compression | LeanLean | Minimum-sufficient explanation at matched comprehension, not just shorter proof code |
| Formal theorem proving | FormalMATH / EasyProver / Discover-and-Prove | Separate held-out mechanism transfer and human outcomes; do not imply it replaces these systems |

## 7. Revised interoperation and claim policy

1. **HumanLean compatibility:** treat its source/definition/scaffold/proof contract as an upstream input. Our claim classes should align with its source-supported facts, formalized goals, excluded goals, and explicitly axiomatized external dependencies. Preserve external-axiom status; never convert an axiom into a proved lemma.
2. **APE-Harness compatibility:** shape tasks and receipts so an APE-style harness can check source identity, task contract, build, semantic requirements and exact output. The human explanation and learning outcomes remain separate outputs.
3. **LeanGraphSearch compatibility:** include graph-based retrieval with PPR and reranking as a required baseline. Our own HDC/CfC variant must beat this strong graph-aware baseline on held-out tasks before we claim retrieval advantage.
4. **Human evaluation is a distinct outcome:** reviewers confirming that a formalization matches a paper is not the same as readers learning the proof, restating its assumptions, or solving a near-transfer task. We must measure these directly.
5. **No novelty inflation:** do not describe claim-to-Lean fidelity, dependency graph search, proof engineering dual verification or proof compression as new inventions of Symthaea. Our proposed contribution is a reproducible bridge among formal proof structure, faithful conceptual explanation, counterfactual diagnosis and measured human transfer.

## 8. Revised baseline arms

The next frozen experiment should include:
- raw Lean/source;
- generic LLM explanation;
- HumanLean-style source/definition/scaffold/proof contract and staged faithfulness review;
- APE-style dual-verification / task-contract baseline;
- lexical / dense premise retrieval;
- LeanGraphSearch-style dependency graph + personalized PageRank + reranker;
- source/type/value Lean-native dependency extraction;
- graph-aware explanation and concept reconstruction;
- graph-aware explanation plus HDC;
- graph-aware explanation plus HDC/CfC.

Match context budget, model, tools, token budget and compute. Compare outputs using proof identity, dependency precision/recall, explanation fact support, reviewer burden, reader comprehension and near-transfer. If HDC/CfC fails against LeanGraphSearch, remove it from this pathway.


## 9. TheoremGraph / LeanGraph — formal and informal graph scale is now an external baseline (2026-11-23)

**References:** [TheoremGraph paper](https://arxiv.org/abs/2606.25363), [TheoremSearch project](https://github.com/uw-math-ai/TheoremSearch), [Math-Graph dataset](https://huggingface.co/datasets/uw-math-ai/math-graph), [LeanGraph extractor](https://github.com/aurasoph/lean-graph), [API documentation](https://www.theoremsearch.com/docs).

TheoremGraph reports 11.7M informal mathematical statement nodes, 18.3M candidate informal dependency edges, 388,105 Lean declaration nodes and 11.3M typed formal edges across 25 Lean projects. LeanGraph records distinct signature, proof and definition edges, plus structure inheritance, field-composition and documentation-reference relations. Signature metadata includes binder position/kind, function/argument role and projection use.

### Match-quality caveat

The 47,952 judge-affirmed cross-formality links must not be reported as 47,952 exact formalizations. The November 19, 2026 author rebuttal divides them into 4,765 exact and 43,187 inexact/related pairs. The latter include specializations, generalizations and one-way implications; they are potentially useful for retrieval, but are not proposition equivalence. The reported expert audit is small (20 pairs reviewed by one expert), while a separate 199-pair sample was judged by two LLMs. Judge agreement/κ is repeatability, not semantic truth. Treat all such labels as candidate annotations until independent human adjudication.

### LeanGraph graph-mode caveat

The TheoremSearch corpus records LeanGraph submodule commit `9823e3f8f45036b124905e541c7dc92c39620283`, using Lean 4.29.0. At that revision, the unified graph is filtered and expanded: signature edges are filtered / sometimes parent-redirected; proof and definition dependencies use transitive expansion; structure, field and docstring overlays are added. A later LeanGraph commit `3bf57c70f0326e45b9e84345c8fe7cf520919358` adds a raw graph before the readable-view pass. These are not interchangeable outputs.

### Competitive consequence

Symthaea should not claim novelty for large-scale dependency extraction, semantic theorem retrieval, or graph-based informal/formal linking. LeanGraph / TheoremGraph become mandatory external comparators. The candidate Symthaea contribution is the layer beyond graph construction:

1. source-, toolchain-, universe-, value-availability- and axiom-bound evidence receipts;
2. explicit distinction between direct kernel edges, filtered/expanded edges, structural overlays and external candidate links;
3. relation-aware informal/formal matching with direction and exactness (not one generic match score);
4. independently reviewed mathematical-role graphs;
5. explanations whose formal facts have claim-level evidence links;
6. counterfactual diagnosis that separates proof-route sensitivity from theorem truth;
7. measured human comprehension, proof validation, immediate/delayed transfer and cross-family transfer.

### Required benchmark arms

For the first four Mathlib + six Lean-core pilot declarations, compare historical TheoremGraph graph output, current LeanGraph raw output where compatible, Symthaea's local typed extractor, and manually reviewed role annotations. For retrieval/explanation, include BM25, dense semantic search, LeanSearch v2, LeanGraphSearch-style semantic seeds + dependency PPR + reranking, symbolic/type-directed retrieval, and graph-aware explanation with and without HDC/CfC. Match context and compute budgets.

Report edge precision/recall by edge class, exact-vs-inexact relation classification, theorem identity fidelity, axiom/opaque handling, proof-route-vs-truth disposition, explanation fact support, counterexample validity, human outcomes and resource cost separately. Do not combine them into one aggregate mathematical-intelligence score.

The interoperability work is tracked in [#13325 — MATH-GRAPH-INTEROP-001](https://github.com/Luminous-Dynamics/symthaea/issues/13325). Its figures are external published/project figures, not Symthaea results. No local differential or performance improvement is claimed until executed on pinned artifacts.


## 10. Proof-plan reconstruction and human-learning evidence (2026-11-23)

A further distinction is now explicit in the benchmark design: a dependency graph, an explanation, a proof presentation, a proof plan and delayed transfer are not interchangeable measures.

Hamami & Morris (2024) argue that understanding a proof involves rationally reconstructing its underlying plan ([DOI](https://doi.org/10.1111/nous.12489)). Gao & Evans' November 6, 2026 study of 143 undergraduates found that proof-presentation performance was only moderately associated with later proof construction on a delayed near-transfer task ([DOI](https://doi.org/10.1007/s40753-026-00319-4)). Thus Symthaea must measure at least four distinct reader capabilities: (1) statement/assumption comprehension, (2) proof validation, (3) proof-plan reconstruction or presentation, and (4) immediate and delayed transfer.

### New benchmark lane: MATH-PROOF-PLAN-001

[#13326](https://github.com/Luminous-Dynamics/symthaea/issues/13326) defines a typed plan graph of goal states, actions, justifications, side conditions, subgoals, witnesses, invariants and unresolved steps. It requires support per action and permits alternative valid plans. It adds plan perturbation and alternative-route recognition instead of scoring only whether a natural-language summary is fluent.

Prospective targets include at least 80% expert acceptance of held-out proof-plan skeletons, 90% side-condition/subgoal coverage and a 10 percentage-point improvement in delayed or cross-family transfer over the strongest matched baseline. These are goals, not current results.

The existing study preregistration is being extended to separate the treatment arms and outcomes. No participants have been recruited and no human study has been run. Expert annotation, ethics determination/review, consent, feasibility pilot and a power analysis remain prerequisites to confirmatory data collection.


## 11. ProofGrid, generative verifiers and local-to-global epistemic stability (2026-11-23)

Three additional baselines sharpen the evaluation beyond theorem pass rate and explanation quality.

### ProofGrid — machine-checkable proof competence

Arkoudas & Batzoglou, [*Stress-Testing the Reasoning Competence of LLMs With Proofs Under Minimal Formalism*](https://arxiv.org/abs/2605.12524), introduces 15 tasks spanning proof writing, proof checking, proof masking and proof gap-filling in compact natural-deduction notation. Its instrumented checker localizes the first substantive proof failure rather than collapsing everything into a binary pass/fail. The paper also introduces an Epistemic Stability Index and uses 2PL item-response analysis / task-discrimination measures.

**Use in Symthaea:** ProofGrid is a baseline for local inference checking, proof gap detection and failure localization. It is not a human-transfer benchmark and does not replace Lean's kernel. Add a separate local-to-global consistency experiment: measure whether a model that rejects an invalid inference when isolated also avoids using it in a complete proof plan. Preserve the paper's ESI as its own metric; do not rename a different formula as ESI.

### Scaling Generative Verifiers — evaluator calibration

Mahdavi et al., [*Scaling Generative Verifiers For Natural Language Mathematical Proof Verification And Selection*](https://proceedings.mlr.press/v306/mahdavi26a.html), ICML 2026, reports that combining GenSelect and LLM-as-a-Judge improves proof verification/selection, while judge-prompt choice materially affects performance. It also reports that gains in proof-level metrics do not necessarily improve final-answer precision.

**Use in Symthaea:** any LLM reviewer must be evaluated across prompt variants and calibrated against human labels. Report false acceptance, false rejection, abstention and calibration. Do not let a generative verifier become the authority for a formal fact.

### FormalMATH — explanation can interfere with proving

FormalMATH's 2026 paper reports that providing human-written natural-language solution references can measurably degrade proof generation in its evaluated settings ([paper](https://aclanthology.org/2026.findings-emnlp.579/)). This is a warning against assuming that more explanation is always better or that a teaching-oriented explanation is necessarily the best prover context.

**Use in Symthaea:** test two different interventions: explanation delivered to a human learner, and structured plan context delivered to a proof-generating model. Report their effects separately. A proof plan may improve human transfer while having no benefit—or even a cost—for Lean proof synthesis.

### New benchmark outcomes

Add the following as separate results, not a combined score:
- first-invalid-step localization accuracy and distance from the actual first invalid inference;
- local-to-global inference consistency under matched adversarial corruptions;
- judge prompt sensitivity, calibration and false-acceptance rate;
- proof-plan structural/evidence quality;
- reader plan reconstruction and delayed transfer;
- proof-generation utility when the same plan is supplied as model context.

Prospective target: at least 95% precision for any system-labelled invalid inference used to block acceptance; zero false acceptance of an altered theorem as the original; report local-to-global consistency with confidence intervals and per-mutation family results. These are goals, not measured Symthaea results.

The executable schema and initial fixture are tracked in [#13338 — MATH-PROOF-PLAN-SCHEMA-001](https://github.com/Luminous-Dynamics/symthaea/issues/13338). The validator's structural result must remain distinct from Lean kernel, expert and human-study results.


## 12. Paired local-to-global proof consistency experiment (2026-11-23)

ProofGrid motivates a distinct local-to-global consistency question: a model may reject an invalid inference in isolation yet reuse the same inference in a complete proof. Symthaea's proposed metric is **Context-Induced Reuse Rate (CIRR)**, not ProofGrid's Epistemic Stability Index:

`CIRR = locally rejected, independently invalid warrants reused in global plans / eligible locally rejected, independently invalid warrants`.

The paired protocol binds both stages to the same theorem identity, exact hypothesis set, proposition identities and warrant. Global context may change order, distractors, narrative framing and plan length, but may not silently add assumptions that make the warrant valid. Scope-change cases are a separate control. CIRR is undefined for a zero denominator; report numerator, denominator, cluster-aware intervals, per-condition strata, local false acceptance, global invalid reuse, valid-warrant false rejection, and first-failure localization separately.

The key intervention compares a structured immutable local verdict/evidence envelope supplied to the global planner against ordinary natural-language self-critique or no local-verdict handoff. This tests whether evidence-carrying reasoning state resists contextual regression better than conversational memory.

The full protocol is [PROOF_PLAN_LOCAL_GLOBAL_CONSISTENCY_V1.md](PROOF_PLAN_LOCAL_GLOBAL_CONSISTENCY_V1.md), tracked under [#13326](https://github.com/Luminous-Dynamics/symthaea/issues/13326). No CIRR measurement has been executed; the metric and prospective thresholds are research targets only.
