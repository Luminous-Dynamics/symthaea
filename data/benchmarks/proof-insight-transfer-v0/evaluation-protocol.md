# Proof Insight Transfer: pre-registered evaluation protocol v0

**Status:** protocol proposal; no benchmark runs or results yet.  
**Applies to:** `data/benchmarks/proof-insight-transfer-v0/` and the `symthaea-proof-insights` research track.  
**Frozen source snapshot:** OpenAI `math` commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`.  
**Purpose:** determine whether typed, evidence-linked proof insights improve verified transfer—not whether they sound mathematically persuasive.

## 1. Claims this protocol permits

Keep three claims distinct:

1. **Index reproducibility:** source fingerprints, family membership, and split assignments match the frozen manifest.
2. **Within-benchmark utility:** an intervention improves the pre-registered outcome on eligible held-out targets under the fixed protocol.
3. **External/SOTA generalization:** the improvement replicates on a contamination-resistant external set under a comparable task, checker, and compute budget.

Passing (1) does not establish (2); passing (2) does not establish (3). Never describe a random family split of public source material as proof of pretraining independence. The source is public and may have been seen during model training; exposure is generally unknown.

The indexed 372-family split is a **pre-screen assignment**, not 372 eligible theorem tasks. Preserve all eligible, excluded, unknown, missing-artifact, and not-attempted counts. Do not silently remove hard, incomplete, or awkward records.

## 2. Eligibility and evidence gates

An item enters the *confirmatory formal-transfer denominator* only after a human-auditable eligibility record establishes:

- pinned source commit and artifact SHA-256;
- the target's exact Lean declaration and its source locator;
- pinned Lean toolchain, Mathlib revision, imports, and checker command;
- an explicit axiom policy and successful proof-audit receipt;
- statement correspondence review against the intended source claim;
- the intended target family and all known companion/alternative-proof family links;
- dependent-successor tests where meaningful downstream declarations exist;
- frozen mutation controls that show the semantic checks can reject deliberately incorrect nearby targets.

If statement correspondence is unresolved, a formal artifact is missing, or the checker is unavailable, record the reason and status; do not count it as a verified success or quietly discard it. Compilation by itself is not semantic validation. This is material: the T² ACL 2026 study tested generated declarations against dependent successor theorems across 2,206 problems and reported only 38.9% Testing Accuracy for its best model in the cited setup despite substantially higher compilation success ([paper](https://aclanthology.org/2026.acl-industry.150/), [implementation](https://github.com/ldilab/T2)).

## 3. Fixed comparison arms

Use one common task set and a single frozen base model/checker setup for the confirmatory *retrieval ablation*. The only intended difference among arms is the information supplied by the retrieval/insight intervention.

| Arm | Information supplied | What it tests |
| --- | --- | --- |
| A. No retrieval | Target statement and fixed task context only | Base solve rate |
| B. BM25 / lexical | Same-source-corpus lexical retrieval | Strong, inexpensive baseline |
| C. Typed syntax | Lean statement / AST retrieval | Structural similarity without a natural-language insight |
| D. Dependency graph | Retrieval by declarations, imports, premises, and dependency links | Formal neighbourhood signal |
| E. Raw proof retrieval | Retrieved source proofs, without extracted insight records | Whether any gain is merely proof exposure |
| F. Typed insight | Insight contract only: obstacle, move, preconditions, state transition, evidence, failure boundary | Primary treatment |
| G. Insight + proof | Typed insight plus its source proof context | Complementarity / upper-bound ablation |
| H. HDC / embedding | Vector similarity retrieval, separately reported | Whether representation adds value beyond simple baselines |

Select retrieval parameters and the strongest simple baseline using **validation families only**. Freeze them before a test run. Do not change prompts, hyperparameters, retrieval cutoffs, or eligibility after seeing test outcomes. Do not choose a baseline retrospectively to make the treatment look better. Prior Symthaea results found a surface-token HDC classifier matched the majority baseline (22/31), so vector methods remain ablations until their incremental value is demonstrated.

For all arms, pin the model identifier/version, system prompt, tool permissions, Lean/Mathlib revisions, retry policy, random seeds where exposed, and time of run. Match task-level budgets: model-call cap, token cap, wall-clock limit, and monetary/API limit must be written into the run manifest before evaluation. Cap context at the same token budget across arms; report actual tokens and cost too. Any arm that receives more compute or privileged hints is a separate, clearly labelled experiment.

### Typed transfer-contract gate

Every transfer candidate must carry a machine-readable `directional_contract`: compatibility class, mismatch codes, source-move requirements, and target-goal requirements. Allowed mismatch categories include object-type mismatch, unavailable premise, inequality-direction mismatch, quantifier-scope mismatch, missing preconditions, domain mismatch, and proof-dependency mismatch. The schema prevents `verified_transfer` unless compatibility is `compatible`; a curated `rejected` candidate must be `incompatible` or `partial`.

Treat this as a testable filter rather than decoration. The static pilot dispositions are documented in [`transfer-adjudications.md`](transfer-adjudications.md): a finite support-face obstruction for a geometric covering problem does not construct the graph witness required by the Hadwiger counterexample, and an upper bound on independence number cannot be reversed to prove a lower bound on independence number. These are curator judgments, not model-run results.

Before claiming transfer utility, require 100% rejection of the frozen finite negative-control suite, including incompatible inequality direction, wrong object type, missing premise, incompatible quantifier scope, and missing dependency. Also report false rejection on expert-labelled positive controls; a matcher that rejects everything does not pass. Report transfer acceptance/rejection precision and recall separately from verified solve rate.

## 4. Primary endpoint and scoring

**Primary endpoint:** family-clustered, paired absolute difference in *verified exact-target proof success* between F (typed insight) and the strongest simple baseline selected on validation (A–E). The exact primary contrast and direction must be written to the run manifest before touching the held-out test partition.

A target is a success only if all applicable checks pass:

1. the exact pinned target declaration is proven by the configured checker;
2. no `sorry`, `admit`, disabled trust check, or disallowed axiom is used;
3. the proof-audit receipt identifies the exact statement and dependency/axiom policy;
4. the informal/formal statement mapping has passed the predeclared review gate;
5. meaningful dependent-successor checks pass and mutation controls demonstrate that the gate rejects the frozen negative fixtures.

Report both raw checker success and semantic-gate success. The latter is primary. If a target has no meaningful successor theorem, report it as not applicable with the reason; never fabricate a proxy test.

**Required secondary outcomes:** (i) verified transfer rate across target families, (ii) time and cost per verified solve, (iii) counterexample / false-conjecture rejection rate, (iv) statement mismatch and mutation-detection rate, (v) axiom-policy violations, (vi) failure and timeout rates, and (vii) proof-insight evidence fidelity under blinded annotation. Report attempted, not-attempted, tool-unavailable, timeout, and out-of-scope separately. A time-out is not a proof failure for every scientific question, but is a non-solve for the primary fixed-budget endpoint.

### Constructive problem-solving is separate

When the answer/witness is unknown, do not reuse the theorem-proving success flag. Report at least four separate states: candidate proposed, witness typechecks, every requested proof obligation is discharged, and answer equivalence passes the predeclared RPE/formal/human check. A plausible answer or typechecked witness alone is not solved. Use FormalMath500, MiniF2F-Solving, or PutnamBench-Solving only after confirming the chosen release, license, task conversion, checker policy, and scoring protocol; published D-FPS numbers are setup-specific context, not a directly comparable Symthaea result ([ICML 2026 paper](https://proceedings.mlr.press/v306/liu26gl.html), [released implementation/data](https://github.com/Purewhite2019/formal_problem_solving_main)).

Research-level informal proof quality is another separate track. Blindly score correctness, exposition, citation/attribution, and abstention. First Proof Batch 3's 10-problem, expert-reviewed scores and its cost differences illustrate important dimensions, but are not comparable with Lean exact-target pass rates ([results, 14 October 2026](https://1stproof.org/third-batch-press-release.html)).

## 5. Statistical analysis (lock before the first confirmatory run)

- Define a trial as a target-level, matched arm comparison; define the related-result family as the resampling cluster. Do not treat several theorems or retries from one family as independent families.
- Use the absolute percentage-point paired effect as the principal reported effect. Use a predeclared family-cluster bootstrap with 10,000 resamples and a fixed, recorded bootstrap seed for its 95% confidence interval. Publish per-family aggregates so the calculation can be independently reproduced.
- Claim a practically meaningful transfer improvement only if the point estimate is at least **+10 percentage points** versus the selected validation baseline **and** the 95% interval excludes zero on the positive side. Otherwise report the estimate and interval without a win claim.
- For secondary confirmatory comparisons, correct for multiple comparisons (Holm). Label unplanned analyses exploratory. Report denominators, exclusions, missingness, ties, per-domain slices, and raw counts alongside percentages. If the effective number of eligible test families is too small for an informative interval, call the result underpowered; do not weaken the gate.
- For insight-quality labels, use at least two independent annotators blind to system arm and outcome, adjudicate disagreements, and report agreement (e.g. Krippendorff's alpha) rather than hiding disagreement. A suggested reliability target is alpha >= 0.80; below it, treat subjective labels as exploratory or revise the rubric before the confirmatory run.
- Precompute statistical power once eligible sample size and baseline rate are known. If the study cannot distinguish the target effect at reasonable power, expand the eligible external set or report a pilot—not a benchmark victory.

This is a protocol template; the alpha level, power assumptions, eligibility denominator, model budget, and any deviation must be recorded in the final run manifest **before** inspecting confirmatory test results.

## 6. Leakage and external validity

Family-level splitting blocks only direct within-family exposure under the catalog's declared mapping. It does **not** block:

- mathematically equivalent claims or shared lemmas across catalog families;
- copied paragraphs, near-duplicate proofs, or theorem variants under different IDs;
- common Mathlib declarations and imported proof dependencies;
- pretraining or post-training exposure to public manuscripts and benchmarks;
- model-specific memorization or contamination in retrieval caches.

Before calling a result generalization evidence, add a leakage audit: normalized theorem-statement and proof fingerprints; cross-family dependency/near-duplicate edges; source-to-target family overlap review; and exact retrieval logs showing every item surfaced for each target. Freeze this audit before scoring. Keep a separate *external challenge set* with later, unpublished, or otherwise exposure-resistant targets, adjudicated for rights and feasibility. Report unknown exposure explicitly; do not claim that hashing alone proves non-contamination.

## 7. Mutation suite and negative controls

Freeze negative fixtures before running the model. Include (where meaningful): tautological statement substitution; dropped/strengthened/weakened critical hypothesis; mismatched statement digest; unrelated successor test that does not depend on the candidate; dependency/axiom-policy violation; `sorry`/placeholder proof; corrupt source digest; duplicate or cross-family source path; and a source revision mismatch; incompatible source/target object types; and an upper-bound premise offered for a lower-bound goal (or vice versa).

The gate should reject **100% of this finite, versioned fixture suite** before a benchmark run is accepted. This is a finite test-suite requirement, not a claim of 100% semantic correctness on arbitrary mathematics. Each fixture has an expected failure code, and each observed result must be stored with a receipt/hash.

## 8. Go / no-go decisions

**Proceed to a larger study** only when the indexer reproduces the frozen split, source/statement fingerprints and proof-audit receipts match, the schema validates with an independent JSON Schema implementation, and the mutation suite passes.

**Claim typed-insight utility** only after the primary paired comparison passes the predeclared effect/interval rule, semantic-gate success improves (not only raw compilation), and no axiom, statement-fidelity, or negative-control regression is observed.

**Keep HDC / embedding out of the critical path** unless its ablation adds repeatable value over the strongest simple structural baseline under the same budget.

**Stop or redesign** if the treatment only improves lexical-match metrics, if the gain disappears against raw-proof retrieval, if statement/mutation gates miss known bad variants, if gains require extra compute not charged to the arm, or if external-set performance does not replicate. A null result is informative and must remain in the record.

## 9. Run manifest minimum

Include [`exposure-audit.md`](exposure-audit.md) in the run manifest. Family `089` was inspected during exploratory source selection before its `test` assignment was checked; it is known-exposed for this workflow and must remain excluded from the confirmatory test denominator without rewriting the frozen split. Report original and adjusted counts.

Each run must persist: protocol version and digest; source commit and family split digest; eligible/excluded/unknown item accounting; target statement digest; related-family ID; proof-insight record digests and source locators; model/tool versions and all budgets; prompt/retrieval configuration; random seeds; per-arm inputs actually surfaced; checker output; axiom/dependency receipt; semantic/mutation test results; wall-clock/tokens/cost; failures and abstentions; annotator rubric/reliability; analysis code version; raw per-target outcomes; and the final analysis receipt. Freeze config and hash the manifest before running; append results rather than rewriting history.

## References

- Kim, Han & Hwang, [Benchmarking Testing in Automated Theorem Proving](https://aclanthology.org/2026.acl-industry.150/), ACL 2026 Industry Track.
- [T² source and experiment harness](https://github.com/ldilab/T2).
- Liu et al., [Formal Problem-Solving / D-FPS](https://proceedings.mlr.press/v306/liu26gl.html), ICML 2026; [implementation/data](https://github.com/Purewhite2019/formal_problem_solving_main).
- [First Proof Batch 3 results](https://1stproof.org/third-batch-press-release.html), 14 October 2026.
- [OpenAI math source snapshot](https://github.com/openai/math/tree/1581d4cbdd69a0faf34b2089614fe2a814316a6a).
