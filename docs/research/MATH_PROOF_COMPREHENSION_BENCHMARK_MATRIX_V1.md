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


## 7. SOTA-oriented stretch targets and matched replication rules

The following are **prospective stretch targets**, not achieved results. They intentionally target the published task/cutoff where a direct comparison is possible. A result on a different subset, index, candidate pool or budget must be reported as non-comparable.

| External comparison | Published reference | Symthaea stretch target | Comparison condition |
|---|---:|---:|---|
| MathlibQR premise retrieval, 810-query fair subset | LeanGraphSearch Recall@100 94.7% | ≥96.0% Recall@100 | Same 810 queries, same declaration inventory and candidate pool; matched index/update date; report Recall@10/50/100 and nDCG as well |
| MathlibMPR complete proof-route coverage | LeanGraphSearch Covered@20 42.8%; Covered@50 44.2% | ≥45.0% Covered@20; ≥47.0% Covered@50 | Same annotated proof routes, same query split and retrieval budget; route coverage requires every premise group in at least one gold route |
| TheoremGraph formal concept retrieval | Reported name/signature + graph expansion Recall@10 77.5%, LeanSearch v2 78.0% | ≥80.0% Recall@10 | Same TheoremGraph evaluation split, graph revision, candidate pool and ranking budget; compare directly to the provided baseline rather than the MathlibQR figures |
| T² downstream semantic testing | Best reported full-set testing accuracy 38.9% | ≥43.0% accuracy, plus ≥95% precision on planted defect flags | Exact T² split and task protocol; stratify statement defects, unrelated successor failures and compilation-only successes |
| Human delayed transfer | No directly equivalent result in these prover/retrieval benchmarks | +10 percentage points vs strongest matched control | Preregistered primary endpoint, held-out theorem family, blinded scoring, powered sample size and task/participant-clustered confidence interval |
| Local-to-global consistency | No directly equivalent published metric | CIRR ≤5% | Nonzero eligible denominator; independent invalid-warrant adjudication; publish raw numerator, denominator, unknown-outcome bounds and interval |

### Hard guardrails that outrank score maximization

- Zero false formal-proof acceptance on the complete seeded vacuity, axiom-substitution, wrong-theorem and missing-assumption negative-control suite. A finite zero count is not a universal guarantee; report the denominator and an upper confidence bound.
- 100% exact theorem, source revision, declaration and formal-context identity for all tasks entering the primary semantic and CIRR denominators.
- No acceptance of an unreviewed or disputed natural-language-to-AST translation. Two distinct reviewer records are a schema requirement; actual reviewer independence, expertise and blinding must be established operationally.
- Unknown is a valid outcome. Report risk-versus-coverage curves, calibration and selective error; never optimize a score by silently dropping abstentions or zero-denominator tasks.
- Separate reader-facing explanation and prover-facing context. FormalMATH reports natural-language solution references can degrade proof generation in its evaluated settings, so teaching and proving utility must be measured independently.

### Matched-cost reporting

Every comparison must pin and report: theorem/source/toolchain identity; graph or corpus version; train/dev/test contamination checks; candidate pool; prompt and model revision; decoding; allowed tools; max tokens; wall-clock; hardware/runtime; search/index budget; proof-checker version; abstention handling; and reviewer minutes. Publish paired per-task differences, task-family breakdowns, confidence intervals and all negative controls. Do not infer superiority from cross-paper headline numbers.

## 8. What would constitute a real paradigm shift?

1. **Intervention-complete explanations:** explanations expose which assumptions and transformations are causally necessary, and predict the result of controlled edits before the oracle runs.
2. **Proof-plan invariance under representation changes:** equivalent statement rewrites, alpha-renaming, reordered contexts and generated-helper repackaging should preserve the appropriate semantic outcomes while raw provenance differences remain visible.
3. **Authority-separated evidence:** the system proposes plans, but formal kernels, checked countermodels, source hashes, independent reviewers and blinded human outcomes each retain separate authority.
4. **Cross-context epistemic reliability:** a rejected warrant is not merely flagged locally; the system must avoid reusing it implicitly in a later global plan under the same immutable scope.
5. **Learning-aware compression:** optimize minimum-sufficient explanations on a comprehension/transfer frontier, not token reduction alone.

Those are falsifiable contributions. A graph with more nodes, a higher embedding score, a fluent explanation or a better single prover Pass@K score is not by itself evidence of proof comprehension.

### HDC / CfC ablations

The proof-comprehension programme has a dedicated HDC/CfC ablation protocol in [MATH_PROOF_COMPREHENSION_HDC_CFC_ABLATION_V1.md](MATH_PROOF_COMPREHENSION_HDC_CFC_ABLATION_V1.md) and issue [#13440](https://github.com/Luminous-Dynamics/symthaea/issues/13440). HDC is tested for role binding under renaming/compositional changes; CfC is tested for proof-state transition reconstruction; hybrid systems are compared against BM25, dense, LeanSearch v2, LeanGraphSearch/PPR and symbolic baselines under matched data/compute/resource budgets. These are hypotheses, not assumed advantages.

## 9. Updated external baselines — interactive comprehension, reformulation, proof quality (2026-11-24)

The novelty claim must account for work published by November 24, 2026. These systems overlap parts of the proposed capability; the table below narrows the differentiated claim rather than treating proof comprehension as an untouched field.

| Work | Public result | What is already demonstrated | Required Symthaea comparison / remaining distinction |
|---|---|---|---|
| [Explorable Proofs](https://arxiv.org/abs/2604.02598), UIST 2026 | Lean-linked proof reading with worked examples, proof-state inspection and dependency tracing; lab study n=16 found better, more correct and more detailed proof descriptions than a chatbot baseline | Interactive, executable proof explanations can improve immediate proof-reading comprehension | Direct interface baseline where feasible. Extend evaluation to exact theorem/assumption receipts, checked counterfactual interventions, plan reconstruction, long/research-level proofs and delayed/cross-family transfer. The paper explicitly identifies AI translation mismatch, short-proof/undergraduate-sample limits and the absence of long-term retention/transfer measures |
| [MathAdv](https://arxiv.org/abs/2608.25449) | 13 undergraduate/graduate domains; formal proving plus knowledge, informal reasoning and expert-authored equivalent reformulations | Multi-dimensional mathematical diagnosis and reformulation robustness are established benchmark dimensions | Reuse its transformation categories and holdout discipline. Add typed proof-plan evidence, local/global consistency and source-bound explanation claims |
| [ProofRank](https://arxiv.org/abs/2605.10379) | 382 challenging competition tasks; conciseness, computational ease, cognitive simplicity, diversity and adaptivity | Proof quality is broader than correctness, and quality dimensions can trade off | Report the relevant quality proxies alongside—never instead of—expert plan fidelity and human transfer |
| [ProofOptimizer](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60917970c4257e50aff60d1d7bde2edf-Abstract-Conference.html) | Reported proof-length reductions of 87% on miniF2F, 57% on PutnamBench and 50% on Seed-Prover IMO 2025 proofs | Automated proof compression is already a strong capability | Do not compete on token/line reduction alone. Measure minimum-sufficient explanations at matched comprehension and delayed transfer; preserve assumptions, side conditions and source evidence |
| [Legendre Mathlib search leaderboard](https://www.legendre-leaderboard.com/) | On its 894 resolved MathlibQR queries, Lean Finder v2 reports Recall@10 89.4%, Recall@100 97.3%, nDCG@10 0.725 | A stronger, current public retrieval reference than relying on older paper cutoffs alone | Add the exact 894-query benchmark as a separate mandatory replication. Do not compare its 97.3% directly with LeanGraphSearch's 94.7% on the distinct 810-query fair subset; run both systems on the same candidate pool and queries |
 
### Updated retrieval targets

Keep the distinct cohorts separate:

| Cohort | Current public reference | Prospective Symthaea stretch target | Required match |
|---|---:|---:|---|
| MathlibQR 810-query fair subset | LeanGraphSearch Recall@100 94.7% | ≥96.0% Recall@100 | Same 810 queries, 171 shared declarations, same candidate pool and index timestamp |
| Legendre 894 resolved queries | Lean Finder v2 R@10 89.4%, R@100 97.3%, nDCG@10 0.725 | ≥90.5% R@10, ≥98.0% R@100, ≥0.74 nDCG@10 | Same 894 queries, renderer, full-name gold resolution, candidate revision, and live-service/index snapshot; re-query the baseline as part of the experiment |
| MathlibMPR complete proof routes | LeanGraphSearch Covered@20 42.8%, @50 44.2% | ≥45.0% @20, ≥47.0% @50 | Same annotated route groups and query split; full route coverage requires every premise group in at least one gold route |

These are prospective targets. No Symthaea retrieval results on either MathlibQR retrieval cohort have been produced by the four-declaration extraction pilot.

### Revised core claim

The differentiated research contribution is not simply "explorable proofs" or "graphs explain proofs." It is a joined, auditable evaluation of:

1. **Formal identity and statement fidelity:** exact source/toolchain/type/assumption identity, plus hidden theorem-sandwich and downstream-successor controls.
2. **Evidence-bound proof-plan reconstruction:** reconstruct goal transformations, consumed warrants, side conditions, subgoals, invariants and alternative routes—not tactic names or a flat dependency list.
3. **Intervention completeness:** predict the effect of assumption, definition, inference-direction and route-only mutations before consulting the oracle; distinguish theorem falsity from proof-route failure.
4. **Cross-context consistency:** test whether a locally rejected invalid warrant returns as material global support under identical scope, with independent adjudication and unresolved-outcome bounds.
5. **Human learning rather than interface fluency:** compare immediate comprehension, delayed near transfer and cross-family transfer; score false-fact rate and statement fidelity as hard gates.
6. **Evidence-preserving compression:** optimize explanation length only subject to non-inferiority in delayed transfer and exact assumption/side-condition preservation.

### Counterexample quality diagnostic

For the bounded classical propositional tier, record three separate outcomes:
- **witness validity:** every premise evaluates true and the conclusion false under the exact candidate sequent;
- **support size:** number of true atoms relative to the declared all-false baseline;
- **minimum-support gap:** emitted witness support size minus the oracle's minimum-support countermodel size.

The minimum-support oracle is a deterministic witness-selection aid over at most 16 atoms, not a general measure of mathematical simplicity. For dependent types and non-propositional mathematics, use checker-backed witnesses or explicit unknown. This diagnostic should never replace semantic correctness or proof-plan validity.

## 10. Evaluator calibration and durable-learning baselines (2026-11-24)

### Evaluator calibration is a separate benchmark axis

[ProofBench / ProofGrader](https://proceedings.iclr.cc/paper_files/paper/2026/hash/764d4fc73ab69289b1e10f15089274f3-Abstract-Conference.html) reports expert-rated proof quality over 145 competition tasks and 435 model solutions, with 0.926 MAE for its best reported evaluator. [QEDBench](https://proceedings.mlr.press/v306/gonzalez26a.html) shows why we should not rely on a single LLM-as-judge score: on its university-level proof rubric, an evaluated judge had positive score inflation up to +0.28 and domain-specific failures.

Symthaea therefore needs three separate evaluator reports: (a) formal / finite checker results where the task is in scope, (b) independent human mathematical review and adjudication, and (c) calibrated LLM-evaluator agreement with humans. The LLM judge is an auxiliary localizer, never the authority that sets theorem truth or primary benchmark eligibility. The detailed protocol is [MATH_PROOF_COMPREHENSION_EVALUATOR_CALIBRATION_V1.md](MATH_PROOF_COMPREHENSION_EVALUATOR_CALIBRATION_V1.md), tracked in [#13476](https://github.com/Luminous-Dynamics/symthaea/issues/13476).

### Human proof presentation is not durable transfer

A 2026 study of 143 undergraduate mathematics students found that presentation of a supplied proof was only moderately associated with later proof construction on a related irrationality task after approximately three months. This supports making delayed near-transfer the confirmatory human endpoint; immediate explanation quality, rubric score and self-reported clarity remain separate outcomes. See the [study](https://link.springer.com/article/10.1007/s40753-026-00319-4).

The [Explorable Proofs UIST 2026 study](https://arxiv.org/abs/2604.02598) is an adjacent immediate-comprehension/interface baseline (n=16), not an unstudied capability. Where its task/interface is reproducible, include it as an explicit human-study comparator. Do not treat its immediate proof-description improvement as evidence of long-term retention or transfer.

### SOTA comparison policy

- Retrieval comparisons must include LeanGraphSearch and Lean Finder V2, each on its own published query cohort and then on a matched local rerun.
- Statement semantics must include theorem sandwiches and successor-theorem tests; compilation alone is not a statement-fidelity metric.
- Proof explanation evaluation must include independent expert ratings and the LLM-judge alignment gap.
- Human comparisons must distinguish immediate comprehension from delayed cross-family transfer.
- No SOTA win is claimed from the four-declaration extraction sample, synthetic 1/1 CIRR arithmetic, or an unpowered human feasibility pilot.

## 11. HDC/CfC readiness and comparison requirements

The inspected current source snapshot Luminous-Dynamics/symthaea@bc915c485402eaf955370eb09cffb6e4fad09e82 contains a general-purpose HDC CodeHDEncoder and a stateful CfC network with forward(input, dt) / BPTT APIs. It does not establish that a Lean elaborated ProofPlanV1 adapter or proof-transition CfC head already exists: the inspected Broca StructuralGenerator is still a mock and describes its CfC head as future work. The cfc_gpu.rs WGPU path is a placeholder (is_gpu_available false; Auto resolves to CPU), so initial benchmarks must report CPU execution.

The HDC/CfC study must introduce a typed Lean/ProofPlan adapter, binder-normalized alpha-equivalent AST inputs, exact evidence IDs outside vector similarity, and a declared fixed-dimension projection from HDC to CfC input. CfC must be compared against parameter-matched MLP, GRU, LSTM and temporal-convolution baselines. Use unit dt in the primary discrete proof-plan task; irregular dt is a separate controlled experiment, not wall-clock timing leakage. See the implementation-readiness and adapter contract in MATH_PROOF_COMPREHENSION_HDC_CFC_ABLATION_V1.md, issue #13440.

These constraints materially change the claim boundary: available HDC/CfC components are not evidence of an integrated mathematical proof-comprehension system, and no accuracy/latency/GPU advantage is claimed.

## 12. Updated theorem-proving frontier and benchmark saturation audit (2026-11-24)

This update separates **formal proof construction**, **answer discovery**, **premise retrieval**, **proof grading**, **statement fidelity**, and **human proof comprehension**. These are related capabilities, not one leaderboard.

### Current external reference points

| System / source | Reported result or scope | What to compare | Claim boundary |
|---|---|---|---|
| [Goedel-Architect](https://arxiv.org/abs/2606.06468) | 99.2% MiniF2F-test pass@1 and 75.6% PutnamBench pass@1; optional natural-language blueprint seeding raises the paper's report to 100% MiniF2F and 88.8% PutnamBench (597/672), plus 4/6 IMO 2025, 11/12 Putnam 2025 and 3/6 USAMO 2026 | Blueprint decomposition/refinement, multi-lemma parallel proof closure, formal proof outcomes and cost under the same exact task/compute protocol | Authors' framework result. It already uses a formal lemma-dependency blueprint, so dependency graphs or proof decomposition alone are not our novelty |
| [Goedel-Prover-V2 (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) | 8B model: 84.6% MiniF2F pass@32; 32B: 88.1% pass@32 standard and 90.4% with self-correction; 86 PutnamBench problems at pass@184 | Self-correction, scaffolded synthesis, Lean feedback, pass@k and total compute | Different test-time budgets; pass@1 and pass@32/184 are not interchangeable |
| [Discover and Prove / DAP (ACL 2026)](https://aclanthology.org/2026.acl-long.3/) | 10 CombiBench solves versus 7 previous SOTA at Pass@16; first reported 36 formal PutnamBench Hard proofs; its paper reports >80% answer accuracy on the same hard problems where formal proving remains below 10% | Hard-mode answer discovery before formal proof, and answer-to-formalization gap | A key comparator for our own discovery-versus-proof split; don't count answer accuracy as verified proof success |
| [IMO-Bench](https://imobench.github.io/) | Four tasks: 400 answer questions, 60 natural-language proof tasks, 1,000 grading examples and 60 Lean proof tasks. Its public Advanced ProofBench leaderboard lists Aletheia at 91.9% and GPT-5.5 Pro at 88.1% in the displayed snapshot | Independent axes for answer correctness, proof writing, grading and Lean proving | Distinct task sets and grading. A natural-language proof score cannot be compared to a Lean pass rate |
| [FormalProofBench](https://arxiv.org/abs/2603.26996) | Graduate-level theorem suite; the paper reports 33.5% for its best foundation model under its agentic harness | Formal proof generation beyond olympiad-level tasks; cost, tool use and domain breakdown | Private benchmark; only a published reference until a matched evaluation is available |
| [LemmaBench](https://arxiv.org/abs/2602.24173) | Live benchmark of extracted research-paper lemmas, with current reported pass@1 around 10–15% | Research-level theorem proving on a rolling time-based holdout | Its automatic extraction and assumption reconstruction need a separate formalization-quality audit |
| [Formal Conjectures](https://arxiv.org/abs/2605.13171) | 2,615 Lean 4 statements, including 1,029 open research conjectures and 836 solved problems in the described snapshot | Rolling research-problem discovery, conjecture proving, and contamination-resistant evaluation | Dataset coverage is not a result; each task's formal statement and proof status must be version-pinned |
| [Lean Finder v2 / Legendre leaderboard](https://www.legendre-leaderboard.com/) | On 894 resolved MathlibQR queries: Recall@10 89.4%, Recall@100 97.3%, nDCG@10 0.725 | Premise retrieval against a common query cohort | Not the same as LeanGraphSearch's 810-query fair subset. Keep cohort metrics separate |
| [LeanGraphSearch](https://arxiv.org/abs/2610.22339) | Its MathlibQR fair subset reports Recall@100 94.7% against LeanSearch v2 82.7% | Multi-hop dependency-aware retrieval and reranking | A distinct 810-query comparison set; do not blend with the 894-query Legendre leaderboard |
| [PutnamBench leaderboard](https://trishullab.github.io/PutnamBench/leaderboard) | The current featured Lean ranking displays all 672 problems solved. The page separates answer-placeholder-filled and no-answer variants; 346/672 tasks contain an answer placeholder that must be inferred in no-answer mode | Benchmark saturation, answer-discovery versus proof-construction, exact benchmark commit and compute | The page notes that some later-added problems have not been rerun across all methods, and that compute budgets differ. A 672/672 headline is not a uniform, no-answer, same-budget comparison |

The Stanford [AI Index 2026 technical performance chapter](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_2_technical.pdf) also reports IMO-ProofBench scores including Aletheia 91.9%, Gemini 3 Deep Think 76.7% and GPT-5.2 Thinking (high) 35.7%, while its proof auto-grader correlates with expert scores at Pearson 0.96 on basic and 0.93 on advanced solutions. These remain benchmark-specific results, not evidence that an automatic judge can authorize mathematical truth.

### What this changes in our competitive strategy

1. **Raw theorem-proving is now an external baseline, not the primary novelty claim.** Goedel-Architect already uses a formally stated blueprint dependency graph; DAP explicitly separates hard-mode answer discovery from proof construction; Goedel-Prover-V2 shows strong self-correction and test-time scaling. Symthaea should interoperate with these styles of prover rather than claim to replace them based on a four-theorem pilot.
2. **Track answer discovery, statement formalization, proof construction and proof comprehension as separate stages.** A system may identify the right answer but fail to formalize or prove it; it may prove a weakened statement; it may produce a formal term but fail to reconstruct why its steps work. Hard-mode and theorem-sandwich tests expose these gaps.
3. **Use live research corpora for rolling held-out evaluation.** LemmaBench and Formal Conjectures suggest an evolving task source rather than another fixed contest-only test. Preserve temporal cutoffs, source commit, theorem identity, hidden test families and adjudicated formalization quality.
4. **Treat theorem-statement faithfulness as a first-class reliability gate.** For each natural-language/formal pair, preserve the exact assumptions, quantify upper/lower theorem-sandwich obligations, and run downstream successor tests. A kernel-accepted proof of the wrong proposition remains a benchmark failure.
5. **Treat proof-plan comprehension as the differentiated layer.** Formal dependency extraction (type, value, helper, recursor, trust closure) is evidence input. The plan layer must explain which mathematical transformation each dependency supports, what side conditions are consumed, what subgoals are created, and which alternatives remain valid. Expert role labels and counterfactual tests are required; graph centrality or lemma retrieval is not proof understanding.
6. **Measure durable learning, not just interactive fluency.** Keep immediate comprehension, plan reconstruction, proof validation, near transfer and delayed cross-family transfer distinct. A strong interface or fluent explanation does not establish retention.

### Revised benchmark targets

All values below are prospective targets. No Symthaea score is implied.

| Axis | First rigorous milestone | Stretch target |
|---|---:|---:|
| Theorem/source/hypothesis identity | 100% exact on all admitted tasks | 100% with tamper-evident preimage and environment replay |
| False formal-proof or false statement acceptance | 0 observed on frozen negative controls | 0 observed across rolling temporal holdouts, with confidence bounds |
| Premise retrieval | Matched replication on the exact 894-query Lean Finder v2 cohort and separate 810-query LeanGraphSearch cohort | Beat the strongest matched baseline in Recall@k or nDCG without loss of complete-route coverage |
| Proof-plan role classification | ≥90% precision / ≥85% recall on independently reviewed roles | ≥95% / ≥90% |
| Plan reconstruction | ≥80% two-expert acceptance on held-out theorem families | ≥90%, with reviewer agreement and route alternatives reported |
| Side-condition / subgoal coverage | ≥90% | ≥95% |
| Counterfactual disposition | ≥90% on seeded mutations, with valid route-only negatives | ≥95%; 100% validity of all supplied finite witnesses |
| CIRR | ≤5% on ≥100 eligible pairs, nonzero denominator, report 95% CI and unresolved bounds | ≤2% on ≥500 pairs across theorem-family holdouts |
| Evaluator calibration | ≥95% defect precision, ≥90% defect recall; report Brier/ECE and risk-coverage | Positive bias ≤0.10 rubric points and matched ProofGrader/QEDBench evaluator agreement |
| Human learning | Powered pilot; report immediate, near and delayed effects separately | ≥10 percentage-point delayed near-transfer improvement and positive cross-family transfer versus strongest matched comparator |
| Efficiency | Exact token/tool/wall-clock/memory and reviewer-time accounting | Better accuracy/transfer-cost Pareto frontier than the matched baseline, not merely fewer generated tokens |

For retrieval, run the 894-query and 810-query cohorts independently. For theorem proving, record pass@1 and pass@k, candidate count, model snapshot, Lean version, context, search/parallelism, tool budgets and total inference/verification cost. For hard mode, publish both answer accuracy and Lean proof success. For live research benchmarks, use a time-based hidden holdout and freeze snapshots; never claim a SOTA result from random splits or a public task after it has entered training.

### Specific implementation priorities

- Add a **hard-mode task adapter** that records natural-language answer discovery, candidate formal statement, theorem-sandwich result, Lean proof outcome and human-review outcome as separate linked receipts.
- Add a **live-lemma intake protocol** sourced from LemmaBench / Formal Conjectures, but do not ingest their gold evaluator artifacts into model-visible context.
- Extend theorem-sandwich mutation controls to cover equivalent reformulations, strengthened statements, weakened statements, missing hypotheses, vacuous axioms/admissions, and namespace/notation changes.
- Add a **proof-plan bridge evaluation**: given a formal proof term, a model must reconstruct alternative valid plans, dependency roles and conditions, then survive plan-node mutations without claiming that a proof-route change refutes the theorem.
- Re-run strong public baselines on precisely the same held-out theorem families before any “better than SOTA” statement. Cross-paper table values are context, not a leaderboard victory.

### Claim boundary

The external results above establish that current theorem-proving and retrieval performance is substantially ahead of our small proof-comprehension pilot. They do not establish that those systems solve proof comprehension, delayed transfer or local-to-global warrant consistency. Conversely, our current typed extraction and finite-receipt tests do not establish better theorem-proving accuracy. The defensible contribution is a new, independently testable reliability-and-comprehension layer composed with existing provers—provided the future matched experiments show benefit.

## 13. PutnamBench saturation and retrieval cohort correction (2026-11-24)

A fresh live leaderboard inspection changes the interpretation of the hard-mode baseline:

- The featured Lean PutnamBench ranking currently shows **672/672 solved** by multiple pipelines.
- The separate **Lean no-answer** ranking (346/672 tasks require answer inference) also shows **672/672** for Midas Prover, with the listed average cost of $6.17 and 31 minutes per problem.
- The answer-provided ranking is distinct from the no-answer ranking. A result in one must not be reported as a result in the other.
- Goedel-Architect's public project page describes 99.2% MiniF2F-test and 75.6% PutnamBench pass@1, with optional natural-language proof seeding; its project site reports 88.8% (597/672) at pass@4 and $985.67 total API cost for that PutnamBench run. These are distinct modes and budgets, not one result.
- Goedel-Prover-V2's ICLR abstract reports 86 PutnamBench problems at pass@184; the project README has updated figures/modes (86 at pass@192). Keep the venue paper result and live project result as separate provenance rows.
- Lean Finder v2's live Legendre comparison uses 894 gold-resolved MathlibQR queries (from the original 946; 11 gold declarations absent at pinned mathlib-4280 revision and unresolved, not misses). It reports R@1 55.0%, R@10 89.4%, R@100 97.3%, nDCG@10 0.725. LeanGraphSearch's 810-query fair subset is a distinct cohort and remains a separate comparison.

### Strategic consequence

PutnamBench is now useful as an integration/smoke test, but is not a discriminative SOTA target unless we use a harder, independently adjudicated, rolling, no-answer holdout with exact budget matching. Do not optimize toward a saturated public leaderboard and then claim superior mathematics.

The harder research frontier for this programme should be:
1. theorem statement reconstruction and equivalence/entailment direction;
2. dependency role correctness, not dependency retrieval alone;
3. proof-plan action, warrant, subgoal and side-condition fidelity;
4. mutation-based explanation sensitivity and valid-route invariance;
5. local rejection persistence under global planning (CIRR);
6. delayed human near- and cross-family transfer;
7. confidence calibration and review burden at a fixed risk ceiling.

For any future Putnam comparison, publish separate answer-provided and no-answer rows, exact task revision, run timestamp, model/prompt/toolchain revision, candidate count, total inference budget, per-problem cost/time distribution, and failed/unresolved tasks. A 672/672 ceiling means the primary scientific comparison should move to those undermeasured reliability and comprehension axes.
