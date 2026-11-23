# Mathematical proof comprehension — benchmark matrix v1

**Date:** 2026-11-23  
**Parent programme:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Status:** research design and prospective targets; no SOTA result is claimed.

## Executive decision

Do not compete with the strongest theorem-proving systems on a blended proof score, and do not claim novelty for dependency-graph extraction or graph-based retrieval. Existing systems already cover large-scale typed Lean dependency graphs, high-recall premise retrieval, repository-scale proof compression, formal theorem generation, theorem-statement semantic tests, and proof-quality grading.

The proposed differentiation is an evidence-bound capability chain:

**formal identity → proof state and typed evidence → intervention and counterfactual response → cross-context consistency → independently evaluated explanation → human delayed transfer.**

Each link must be measured separately. A system can retrieve the right lemma without understanding it, prove the wrong formalized statement, explain a proof without improving human comprehension, or reject a local inference and later rely on it globally. The benchmark must expose those cases rather than collapse them into one score.

## 1. Current public comparison points

| System / benchmark | Publicly reported scope or result | What it already measures | Symthaea response |
|---|---|---|---|
| [TheoremGraph / LeanGraph](https://arxiv.org/abs/2606.25363) | 11.7M informal statements, 18.3M candidate informal edges, 388,105 Lean declarations and 11.3M typed formal edges across 25 projects; six formal edge categories | Large-scale formal/informal statement graph and cross-formality matching | Interoperate; measure proof-plan roles, source-bound warrants, intervention outcomes and human transfer. The four-theorem extractor is a small pilot, not a competitor to this corpus |
| [LeanGraphSearch](https://arxiv.org/abs/2610.22339) | MathlibQR shared subset: Recall@100 94.7% vs LeanSearch v2 Standard 82.7%; MathlibMPR Covered@20 42.8% vs LeanSearch v2 Reasoning 39.1% | High-recall multi-hop premise retrieval and proof integration | Mandatory retrieval baseline. Add role correctness, valid proof-plan use, and learner reconstruction, not just recall |
| [FormalMATH](https://aclanthology.org/2026.findings-emnlp.579/) | 5,560 semantically validated Lean 4 statements; DeepSeek-Prover-V2 671B reports 28.31% Pass@32 | Formal proof generation across undergraduate and competition mathematics | Match prover budgets, but evaluate statement fidelity, explanation support and delayed transfer separately. Reader and prover contexts need isolated ablations |
| [LPBench](https://github.com/lp-bench/lp-bench) | Current README: 118 research-level Lean 4 problems: 32 open, 65 literature and 21 calibration; hidden theorem sandwiches | Research-level proving and theorem-statement fidelity | Reuse public task tier and hidden-sandwich discipline; never expose evaluator-side bounds or use proving score as a comprehension metric |
| [T²](https://aclanthology.org/2026.acl-industry.150/) | 2,206 problems from five Lean 4 repositories, about 41 successor theorems per problem on average; best reported full-set testing accuracy 38.9% | Downstream semantic regression; compile dependent successors, not only the generated statement | Add mutation classes and unrelated-successor controls; separate semantic mismatch from maintenance failure |
| [LeanLean](https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean) | 64 repositories, up to 174k LOC; best reported mean compression score 48.3% within a 12-hour limit | Repository-scale proof compression and dependency restructuring | Compare minimum-sufficient explanations at matched comprehension; compression alone is not understanding |
| [ProofGrid](https://arxiv.org/abs/2605.12524) | 15 tasks across proof writing, checking, masking and gap filling, with first-invalid-step instrumentation | Fine-grained proof competence and failure localization | Reuse task dimensions and diagnosis instrumentation; keep its ESI distinct from CIRR |
| [ProofBench / ProofGrader](https://proceedings.iclr.cc/paper_files/paper/2026/hash/764d4fc73ab69289b1e10f15089274f3-Abstract-Conference.html) | Expert-rated natural-language proof quality over competition problems | Fine-grained grading of readable proof quality | Complement reviewer scores with plan fidelity, source coverage, counterfactual response and delayed human transfer |

These figures come from different datasets, model versions, tool access and resource budgets. They are reference points for matched local replications, not a single comparable leaderboard.

## 2. Paradigm shifts worth pursuing

### A. Proofs as executable evidence-carrying state

Represent each proof plan as typed state transitions. Every node binds to a goal, exact assumptions, consumed premises, produced subgoals, proof action, side conditions and source evidence. Every claim must trace to a formal declaration, an independently checked inference, or an explicitly uncertain annotation. Model confidence and graph centrality never count as evidence.

The plan is not a replacement for Lean's proof term. It is a human-auditable and falsifiable account of why the term should establish the target.

### B. Intervention semantics instead of static explanation

For each theorem family, mutate one semantic dimension at a time: remove or add a hypothesis; reverse a lemma direction; replace an injective embedding with a non-injective function; change a binder, domain or instance; replace only the proof route while preserving the proposition; or change only presentation, order and distractors.

The oracle should distinguish theorem falsity, entailment-direction failure, route invalidation, changed scope, presentation-only changes and unknown. A failed proof route is not a false theorem. A changed assumption is not a valid primary CIRR example. An unknown result is neither a pass nor a fail.

### C. Local-to-global epistemic consistency

Pair a local warrant evaluation with a later global proof-planning stage under identical source, theorem, premise, conclusion and hypothesis preimages. Measure definite material reuse of independently invalid, locally rejected warrants. Keep hidden evaluator labels out of candidate context. Report CIRR numerator, denominator, unresolved bounds and confidence intervals; zero eligible denominator means undefined.

### D. Dual-context explanation

Maintain two isolated contexts: reader context (proof plan, conceptual explanation, side conditions and counterfactuals) and prover context (only premises and formal fragments allowed by the proving protocol). Test whether an explanation helps a human without contaminating theorem proving. FormalMATH reports a counterintuitive negative effect from natural-language solution references on formal proof generation, so this split is a required ablation.

### E. Human comprehension as a primary outcome

Measure immediate reconstruction, first-invalid-step localization, assumption recognition, counterfactual prediction and delayed cross-family transfer. Compression is constrained optimization: fewer tokens count as progress only when comprehension remains non-inferior.

## 3. Proposed benchmark tiers and prospective targets

| Tier | Task | Metrics | Prospective target |
|---|---|---|---|
| T0 — finite semantic controls | Bounded propositional entailment and countermodel search | Exact truth-table result, deterministic witness, budget abstention | 100% correctness on exhaustive controls; never report budget exhaustion as proof |
| T1 — formal identity and extraction | Source-pinned Lean environment extraction | Source/theorem/hypothesis identity, type/proof edge precision and recall, closure reconciliation | 100% identity match; 100% exact closure reconciliation on supported declaration kinds |
| T2 — proof-plan reconstruction | Recover goal decomposition, key warrants, side conditions and alternatives | Expert plan acceptance, required-subgoal coverage, evidence coverage, first-invalid-step | ≥80% expert acceptance; ≥90% side-condition and subgoal coverage |
| T3 — intervention and theorem semantics | Theorem sandwiches, counterfactual mutations and successor tests | Entailment direction, mutation detection, downstream successor accuracy and false alarms | ≥90% disposition accuracy; ≥95% precision and ≥90% recall on seeded statement-defect families |
| T4 — local/global consistency | Paired warrant receipts | CIRR, unresolved bounds, false local acceptance, valid-warrant rejection | CIRR ≤5% only with nonzero eligible denominator; report cluster-aware intervals |
| T5 — human transfer | Blind proof reconstruction and delayed/cross-family transfer | Immediate comprehension, delayed transfer, localization and reviewer time | +10 percentage points delayed transfer vs strongest matched control |
| T6 — explanation efficiency | Minimum-sufficient proof plan | Token/word reduction at matched comprehension | ≥25% fewer tokens with comprehension non-inferior within a 5-point margin |

All thresholds are prospective engineering targets, not current results.

## 4. Required comparison arms

Freeze theorem, source revision, proof assistant/toolchain, imports, context, decoding parameters and resource budget for every eligible task. Compare:

1. Raw source/proof expression.
2. BM25 or lexical retrieval.
3. Dense semantic retrieval.
4. LeanSearch v2.
5. LeanGraphSearch / PPR.
6. Source plus direct dependency graph.
7. Source plus typed graph plus structured proof plan.
8. Structured plan plus evidence receipts plus counterfactual oracle.
9. HDC and HDC+CfC as ablations, not presumed improvements.

For human studies, compare raw proof, strongest baseline explanation, structured plan, and structured plan with counterfactual exercises. Use the strongest matched control as the primary baseline.

## 5. Statistical and integrity protocol

- Split by theorem family and source repository, not random paraphrase, to limit leakage.
- Pin exact source commit, Lean toolchain, imports, generated artifacts, model build, prompt, decoding configuration, tool access and budgets.
- Use hidden theorem-sandwich obligations and held-out countermodels; candidate-visible inputs must exclude evaluator labels and witnesses.
- Preserve task-level receipts and all negative controls.
- Report per-family results, paired confidence intervals, abstentions, invalid-task exclusions, token/tool/latency cost and reviewer time.
- Use two independent mathematical reviewers for gold plan roles, record disagreements, and allow unknown.
- Publish benchmark scores only after matched baselines have been rerun. A paper's headline number is a replication target, not a directly comparable result.

## 6. Immediate execution sequence

1. Finish and test the bounded finite propositional oracle; keep it explicitly outside the Lean proof-authority boundary.
2. Complete the LeanGraph/TheoremGraph differential on the four exact Mathlib theorems, mapping direct proof/type dependencies and documenting edge-class exceptions.
3. Add an inductive/constructor/recursor calibration family and a tactic-generated proof family.
4. Obtain two independent expert proof-plan annotations before expanding the dataset.
5. Implement theorem-sandwich and downstream-successor mutation fixtures from issue #13357.
6. Preregister a human delayed-transfer study only after task selection, ethics review and power analysis.

## Claim boundary

The current extracted Mathlib seed contains four theorems and demonstrates deterministic, source-pinned extraction and merged-closure reconciliation. It does not establish that edges are mathematically essential, plans are expert-correct, models perform well, CIRR is low, or humans learn more. Those claims require the later benchmark tiers.

### Context-aware formalization and translation validity

The theorem-sandwich evaluator treats theorem statements as sequents \`Γ ⊢ P\`, not only as propositions \`P\`. Formal assumptions are a typed AST list and are conjoined into the sequent semantics; free-text assumption descriptions do not participate in entailment. A separate evaluator-only translation-alignment receipt prevents a formally valid but semantically mis-translated AST from being counted as a statement match. Misalignment becomes \`incomparable_or_translation_mismatch\`; partial, unknown or unreviewed translation remains \`unknown\`.

This is still only a bounded propositional control. Lean's dependent binder contexts, typeclass parameters, universe parameters, implicit arguments and non-propositional structures require elaborated Lean-level comparison and independent review.

### Representation-sensitive graph differential

Report raw direct type/proof edge changes and generated-helper-expanded overlap separately. The Mathlib v4.29/v4.37 pilot shows why: Nat.choose_pos direct proof-value Jaccard is 0.265, while helper-expanded overlap is about 0.946; much of the apparent drift is generated-helper encapsulation. In contrast, Set.countable_union changes its elaborated operation instance from Union (Set α) to Max (Set α), changing type references without changing the visible theorem expression or underlying Set.union definition. Neither direct nor helper-expanded Jaccard is semantic equivalence; source and elaboration provenance must explain each difference.

The translation review is a post-hoc two-reviewer gate: all reviewers must agree on alignment for a task to be counted as aligned; explicit disagreement is reported as disputed/unknown, and two misaligned reviews block semantic acceptance. Reviewer IDs and evidence references are structurally distinct, but actual independence, expertise and blinding must be established by the study process, not inferred from distinct strings.

The theorem-sandwich validation set also includes incomparable statements, equivalent surface rewrites (\`p → q\` versus \`¬p ∨ q\`), and context-order permutations. These are exact finite propositional controls for classification correctness; they are not substitutes for Lean's dependent-context equivalence or expert translation review.

