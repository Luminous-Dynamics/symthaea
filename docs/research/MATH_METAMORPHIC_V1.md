# Mathematical invariance and intervention sensitivity — benchmark protocol v1

**Issue:** [#13547 — MATH-METAMORPHIC-001](https://github.com/Luminous-Dynamics/symthaea/issues/13547)  
**Programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Status:** prospective benchmark. No Symthaea metamorphic score or model result is claimed.

## 1. Thesis

A mathematical intelligence system should satisfy two complementary requirements:

1. **Invariance:** preserve mathematical disposition and proof-plan meaning when the representation changes without changing the mathematics.
2. **Sensitivity:** correctly detect and respond when an intervention changes the theorem, assumptions, binder, domain, quantifier, warrant or proof obligation.

A model that changes its answer under harmless alpha-renaming is brittle. A model that ignores a removed positivity assumption or a reversed implication is dangerously insensitive. Aggregate proof accuracy can hide both failure modes.

This protocol tests the paired property directly.

## 2. Current research and competitive landscape

The programme should interoperate with the strongest adjacent work rather than claim that it is replacing it.

| Work | Published scope/result | What it measures | What this programme adds |
|---|---|---|---|
| [FormalMATH (EMNLP 2026 Findings)](https://aclanthology.org/2026.findings-emnlp.579/) | 5,560 semantically validated Lean 4 statements; DeepSeek-Prover-V2 reports 28.31% Pass@32 | Formal theorem proving | Paired representational invariance, changed-scope sensitivity, and proof-plan fidelity |
| [LeanGraphSearch](https://pluslabnlp.github.io/LeanGraphSearch/) | MathlibQR shared 810-query subset: Recall@100 94.7%; MathlibMPR: Covered@20 42.8% | Multi-hop premise retrieval and complete proof-route coverage | Whether retrieved dependencies support the exact warrant and survive interventions |
| [MathAdv](https://arxiv.org/abs/2608.25449) | 13 mathematical domains; includes expert-crafted equivalent problem transformations | Robustness to equivalent reformulation, knowledge, informal reasoning and Lean proving | Paired invariance and sensitivity across statement, context, warrant and proof-plan semantics |
| [ProofGrid](https://arxiv.org/abs/2605.12524) | 15 tasks for proof writing/checking/masking/gap filling; first-substantive-failure instrumentation | Fine-grained proof competence and epistemic instability | Paired metamorphic relation labels, scope identity, and distinction between harmless rewrites and semantic interventions |
| [QEDBench (ICML 2026)](https://proceedings.mlr.press/v306/gonzalez26a.html) | 7 judges × 5 solvers; 1,000+ hours of human evaluation; some judge configurations show up to +0.28 mean-score inflation | Automated proof-grader alignment with experts | Evidence-bound adjudication and non-compensable false-acceptance metrics |
| [TheoremBench](https://arxiv.org/abs/2606.09450) | Nearly 100 classical theorems, with premised subtheorem families and theorem-level coverage/token-efficiency | Long-form, dependency-rich theorem proving | Whether proof plans correctly explain the role and scope of dependencies |
| [Proof understanding theory](https://doi.org/10.1111/nous.12489) | Argues that understanding a proof involves rationally reconstructing its underlying plan | Conceptual account of proof understanding | Machine-readable, independently evaluated plan and intervention receipts |
| [Gao & Evans (2026)](https://link.springer.com/article/10.1007/s40753-026-00319-4) | 143 undergraduates; delayed near-transfer about three months later; presentation score was only moderately associated with proof construction | Human proof presentation and delayed transfer | Measure retained mechanism transfer, not explanation fluency alone |

These are different tasks, cohorts, models, access conditions and resource budgets. Their published results are comparison anchors, not directly comparable Symthaea scores.

## 3. Core artifact: MathMetamorphicPairV1

Every paired item binds:
- exact source theorem, repository revision, declaration, elaborated type and proof environment;
- exact hypotheses, premises, target, binder/context and warrant identities;
- candidate-visible original and transformed tasks;
- versioned transformation specification and deterministic seed;
- expected semantic relation: equivalent, changed-scope, changed-statement, proof-route-only, ill-formed or unknown;
- independent oracle authority and evaluator-only evidence;
- candidate model, prompt, decoding, tool policy, budget and run identity;
- proof-plan / explanation outputs and raw-output digests;
- evaluator disposition, defect location and evidence references;
- schema, normalizer, source, environment and artifact digests.

Candidate-facing input and evaluator-only gold labels/witnesses must remain separate. The evaluator must never derive truth or eligibility from candidate self-report.

## 4. Transformation matrix

### A. Semantics-preserving controls

- Alpha-renaming of bound variables.
- Reordering independent hypotheses.
- Equivalent connective rewrites, including implication versus disjunction with negation.
- Definitional unfolding/folding in a pinned environment.
- Harmless notation or formatting changes.
- Repackaging generated/private helpers while preserving the theorem.
- Alternative valid proof routes preserving the same target.
- Reordering surrounding contextual text without changing the mathematical scope.

Expected behavior: preserve theorem identity/equivalence and mathematical disposition. A valid proof plan may take a different route; the benchmark must not require identical tactic traces or graph topology.

### B. Semantics-changing controls

- Add, remove, strengthen or weaken an assumption.
- Reverse an implication or quantifier.
- Change a binder type, domain, local instance or universe.
- Replace an injective map with an arbitrary map.
- Change a cutoff, parameter, exponent or index.
- Substitute a per-object inequality for an averaged inequality or vice versa.
- Remove a side condition used by a proof action.

Expected behavior: identify the changed mathematical scope and obtain a supported disposition for the transformed statement. Never inherit the original proof status automatically.

### C. Proof-route-only controls

- Remove or replace a supporting helper while leaving the theorem statement identical.
- Change tactic structure or generated-helper packaging.
- Replace a dependency with a distinct valid proof route.

Expected behavior: distinguish route invalidation from theorem falsity. Compilation failure and failed proof search are not counterexamples.

### D. Unsupported / unknown controls

- Dependent telescope transformations outside the supported equivalence adapter.
- Typeclass/local-instance changes without a formal context transport.
- Opaque definitions without a certified correspondence.
- Unresolved informal-to-formal translation.
- Incomplete source, toolchain or kernel receipts.

Expected behavior: unknown or ineligible, never inferred equivalent through similarity.

## 5. Oracle and authority policy

- **Finite propositional tasks:** bounded truth-table decision procedure; atom-budget exhaustion is unknown.
- **Lean tasks:** exact elaborated theorem/context identity, and kernel-checked proof or equivalence transport where required.
- **Mathematical interpretation:** two independent reviewers plus adjudication.
- **Automated evaluator:** can locate candidate defects and organize evidence, but cannot set gold mathematical truth.
- **Proof-route status:** separately records original proof replay, alternative proof replay, proof failure or incomplete search.
- **Theorem truth status:** separately records proved, refuted by a checked witness, unknown or ill-formed.

A failed proof attempt is never a refutation. A model vote is never a proof. A relevant dependency citation is not sufficient unless the cited evidence supports the exact claim and direction.

## 6. Metrics

Report each metric independently:

1. **Semantics-Preserving Invariance Accuracy (SPIA):** equivalent pairs for which theorem disposition and required proof-plan semantics remain equivalent.
2. **Semantics-Changing Sensitivity Accuracy (SCSA):** semantic mutations correctly identified and correctly classified under the independent oracle.
3. **False Stability Rate (FSR):** semantic mutations incorrectly treated as equivalent or assigned inherited status.
4. **False Sensitivity Rate (FSeR):** harmless transformations incorrectly treated as mathematical changes.
5. **Proof-Route Confusion Rate (PRCR):** route-only failure incorrectly called theorem refutation or statement change.
6. **Scope identity accuracy:** exact source, target, binder and assumption correspondence.
7. **First-invalid-step localization:** only when a unique first invalid step is independently defined.
8. **Proof-plan fidelity:** side-condition coverage, warrant validity, subgoal discharge, invariant preservation and alternative-route recognition.
9. **Selective risk / abstention:** coverage and error at each confidence/abstention threshold.
10. **Cost:** tokens, checker calls, latency, Lean elaboration time, memory, reviewer minutes and adjudication rate.

### Prospective targets

| Metric | Target | Guard |
|---|---:|---|
| SPIA on automatable frozen equivalence controls | ≥99% | No required route identity |
| SCSA on independently adjudicated held-out mutations | ≥95% | Family-level confidence intervals |
| FSR on high-risk negative controls | 0 observed | Publish numerator/denominator; not a universal guarantee |
| FSeR on exact metamorphic controls | ≤1% | Stratify transformation class |
| PRCR on proof-route-only controls | 0 observed | Route failure must not imply theorem falsity |
| Kernel-proved claims | 100% exact kernel replay | Exact target/environment receipt |
| Gold-label coverage | 100% adjudicated or marked unknown | No silent exclusion |

All targets are prospective. An aggregate score cannot compensate for a false mathematical acceptance.

## 7. Baseline arms

At matched task, source, prompt context and compute budget, compare:

1. Raw source/proof context.
2. Generic LLM reasoning without a dependency graph.
3. Lean type/dependency graph only.
4. LeanGraphSearch/PPR retrieval plus source.
5. Typed ProofPlanV1 plus evidence references.
6. ProofPlanV1 plus finite/Lean counterfactual oracle.
7. Full Symthaea pipeline.
8. HDC, CfC and HDC+CfC ablations only after proof-specific adapters and matched baselines are validated.

For proving, report answer-provided and no-answer modes separately. For proof understanding, report plan reconstruction and learner transfer independently from Lean pass@k.

## 8. First bounded corpus

Start by reusing existing assets:
- implication-converse finite countermodel;
- provisional And.comm ProofPlanV1 seed;
- four pinned Mathlib declarations: Nat.choose_pos, Set.countable_union, Multiset.map_id, Finset.card_map;
- four structural Lean calibration declarations: List, List.nil, List.cons, List.rec.

For each supported theorem family, add positive equivalence transformations, semantic mutations, route-only controls, and unknown controls. Begin with five theorem families and hold out entire families, not paraphrases. Two reviewers independently audit mathematical labels and proof-plan roles.

The strict exact-scope local/global CIRR from [#13345](https://github.com/Luminous-Dynamics/symthaea/issues/13345) remains primary. Certified-equivalent comparability from [#13537](https://github.com/Luminous-Dynamics/symthaea/issues/13537) is a separate cohort. Do not pool them.

## 9. Execution sequence

1. Freeze the current finite controls and exact-scope receipt regression fixtures.
2. Define deterministic typed transformation manifests and expected semantic relations.
3. Implement positive, negative, route-only and unknown fixture families.
4. Reuse the current local/global receipt validator and add metamorphic pair validation.
5. Run a baseline matrix before any model tuning.
6. Have two independent reviewers adjudicate task labels.
7. Add a theorem-family holdout and publish per-family paired outcomes.
8. Connect successful proof-plan tasks to a separately governed human delayed-transfer study.

## 10. Claim ceiling

- Invariance is not theorem validity.
- Sensitivity is not mathematical understanding by itself.
- Correct semantic relation does not imply a correct proof plan.
- LLM judge agreement is not an oracle.
- A route failure is not theorem refutation.
- A finite semantic oracle is complete only for its declared AST and atom budget.
- No Symthaea model-evaluation result has been collected; the current exact-head run qualifies only the finite semantic fixtures and validator.


## 11. Initial executable finite contract (2026-11-24)

The first bounded validator has now been added to the standalone Rust proof-plan contract crate:
- `scripts/research/proof-plan-contract/src/metamorphic.rs` — typed candidate/evaluator pair, domain-separated sequent digest recomputation, finite semantic relation derivation, and transformation-class consistency checks.
- `scripts/research/proof-plan-contract/src/bin/math-metamorphic.rs` — CLI that validates a fixture dataset and emits one machine-readable line per pair.
- `docs/research/MATH_METAMORPHIC_PAIR_V1.schema.json` — strict JSON Schema for the typed pair dataset.
- `docs/research/MATH_METAMORPHIC_PAIR_V1.json` — five controls: equivalent implication/disjunction rewrite, removed-hypothesis mutation, proof-route-only change, independent-hypothesis order permutation, and incomparable conclusions.

The finite checker maps a typed sequent to the implication from the conjunction of its hypotheses and premises to its conclusion, then exhaustively checks entailment in both directions. For this bounded Boolean AST:
- equivalent connective rewrites must be bidirectionally entailed;
- reordering independent hypotheses must preserve entailment even though the raw sequent digest changes;
- removing a hypothesis can produce a directional entailment change and cannot be labelled semantics-preserving;
- unrelated atomic conclusions are incomparable;
- a proof-route-only control must retain exact sequent digest while carrying two distinct, non-empty route digests.

Candidate serialization omits evaluator-only gold relation and authority. The validator recomputes both typed-sequent digests and rejects a mismatch between the declared transformation class and the derived finite relation.

**Five-pair exact-head qualification:** the finite-reference job passed on code-bearing head `a8e907025c4021ab417abb35c6bb29d353191359` (workflow run [45465358113](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45465358113)): formatting passed, the standalone crate reported **102 passed / 0 failed**, and the CLI validated all five pairs (two equivalent controls, one directional entailment, one route-identity-only control and one incomparable pair). The independent Lean 4.29 and 4.37 extraction/differential lanes were still in progress at that update. Subsequent branch commits were documentation-only relative to the code-bearing head.

The runtime environment used for this audit did not contain Cargo/Rust, so Rust execution was performed by the pinned GitHub workflow, not locally. This finite contract is not Lean kernel replay, does not evaluate arbitrary mathematical propositions, and is not a model evaluation or CIRR score.


### Route-only control qualification

The finite synthetic route-only fixture currently checks that the *declared route identity digest* changes while the exact typed sequent digest remains unchanged. This is a useful anti-confusion control, but the digest alone does **not** establish that either route artifact exists, is a valid plan, or is kernel accepted.

For a real Lean route-only benchmark pair, the next receipt extension must bind both proof-plan artifacts, their exact target type digests, Lean/Mathlib/Lake environment identities, and separate kernel replay receipts. The theorem target must remain identical. Route replay failure must remain a route outcome, not a theorem refutation. Until those receipts exist, report the synthetic result as **route-identity-only**, not evidence of alternative-proof validity.
