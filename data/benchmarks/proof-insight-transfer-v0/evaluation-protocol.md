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

The indexed 372-family split is a **pre-screen assignment**, not 372 eligible theorem tasks. Preserve all eligible, excluded, unknown, missing-artifact, and not-attempted counts. Every family assigned to the frozen validation and test splits must have at least one explicit outcome row; when a family has no eligible formal target, retain an `excluded` or `unknown` row with the reason. The baseline analyzer recomputes A–E/I scores from the raw validation outcomes and verifies the validation JSONL digest in the receipt; it does not trust a manually asserted score table. Do not silently omit hard, incomplete, awkward, or unavailable validation/test families from either analysis input.

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

### Benchmark-row validity audit

A checked Lean proof establishes its formal statement; it does not by itself establish that the statement matches the intended mathematics or that a benchmark harness excludes trivial/vacuous targets. This is now an explicit dataset-quality track informed by *Faults in Our Formal Benchmarking* (June 2026), which reported 4,833 findings across 10,318 problems from five benchmark families, including 398 mechanically certified issues such as counterexamples, vacuous theorems, and unsound axioms ([paper and released audit artifacts](https://paperswithlean.com/paper/2606.29493.html)).

Before any target contributes to a confirmatory denominator:

- Preserve the exact source problem, formal statement, source commit, and every dependency/axiom receipt. Audit the full dependency closure, not just the final declaration text.
- Obtain two independently authored statement-correspondence reviews blinded to model outcomes. Record each reviewer decision and its evidence digest separately. Unresolved disagreement stays `unknown` and is excluded from verified-success claims, while remaining in raw/unknown accounting.
- Run pinned static checks for placeholders (`sorry`/`admit`), unexpected axioms, invalid/counterexample-susceptible statements where mechanically detectable, vacuity/trivialization, and evaluation-harness shortcuts. Record each check as passed, failed, unavailable, or not applicable; never infer a universal absence of defects from a finite checker suite.
- For each target class, include nearby statement mutations that alter a critical hypothesis, conclusion, quantifier, or logical direction. Verify that the semantic/statement gate catches the intended mutation before scoring the associated target. Publish detected/undetected controls by failure class.
- Report raw rows, eligible rows, excluded rows, unknown rows, confirmed defect counts by severity, unresolved semantic-review fraction, and mutation detection. Corrections create a new content-addressed artifact and a new audit receipt; never silently repair or remove a defective row from the frozen source.

The benchmark-row validity report is a prerequisite for calling a formal-transfer score a qualified result. Its defect rates are an audit report, not a weighted component of the proof-success metric. Static checks may miss defects; human review may disagree; both limitations must remain explicit.

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
| I. Strong premise retrieval | Frozen LeanSearch v2 adapter when it supports the exact eligible target, otherwise a preregistered same-corpus, dependency-aware global-premise retriever with pinned code/config | Whether typed insight transfer beats a strong retrieval system rather than only generic lexical baselines |

Select retrieval parameters and the strongest baseline candidate among A–E and I using **validation families only**. Freeze them before a test run. Arm I must use **one fixed implementation for the entire run** and identify its exact version/commit, corpus snapshot, index digest, candidate budget, and any remote service model/version. Prefer the pinned open LeanSearch v2 implementation if it supports the full eligible task set. If it cannot, choose a single named, pinned same-corpus global-premise implementation for all targets in this run. Do not switch implementations per target inside arm I; unsupported targets are recorded as tool-unavailable/out-of-scope and count as non-solves, with reasons. Never imply a same-corpus implementation reproduces LeanSearch v2's paper scores. Do not change prompts, hyperparameters, retrieval cutoffs, or eligibility after seeing test outcomes. Do not choose a baseline retrospectively to make the treatment look better. Prior Symthaea results found a surface-token HDC classifier matched the majority baseline (22/31), so vector methods remain ablations until their incremental value is demonstrated.

For all arms, pin the model identifier/version, system prompt, tool permissions, Lean/Mathlib revisions, retry policy, random seeds where exposed, and time of run. Match task-level budgets: model-call cap, token cap, wall-clock limit, and monetary/API limit must be written into the run manifest before evaluation. Cap context at the same token budget across arms; report actual tokens and cost too. Any arm that receives more compute or privileged hints is a separate, clearly labelled experiment.

### Typed transfer-contract gate

Every transfer candidate must carry a machine-readable `directional_contract`: compatibility class, mismatch codes, move kind, source/target domains and object types, predicates, relation directions, quantifier scopes, preconditions, and human-readable source/target requirements. Allowed mismatch categories include object-type, predicate, inequality-direction, unavailable-premise, quantifier-scope, precondition, domain, and proof-dependency mismatches. The schema prevents `verified_transfer` unless compatibility is `compatible`; a curated `rejected` candidate must be `incompatible` or `partial`. The Rust contract validator additionally rejects a `representation_reduction` marked compatible across domain/object-type boundaries and an `inequality_transfer` marked compatible when the same predicate has opposite upper/lower-bound direction.

Treat this as a testable filter rather than decoration. The static pilot dispositions are documented in [`transfer-adjudications.md`](transfer-adjudications.md): a finite support-face obstruction for a geometric covering problem does not construct the graph witness required by the Hadwiger counterexample, and an upper bound on independence number cannot be reversed to prove a lower bound on independence number. These are curator judgments, not model-run results.

Before claiming transfer utility, require 100% rejection of the frozen finite negative-control suite, including incompatible inequality direction, wrong object type, missing premise, incompatible quantifier scope, and missing dependency. Also report false rejection on expert-labelled positive controls; a matcher that rejects everything does not pass. Report transfer acceptance/rejection precision and recall separately from verified solve rate.

## 4. Primary endpoint and scoring

**Primary endpoint:** family-clustered, paired absolute difference in *verified exact-target proof success* between F (typed insight) and the strongest preregistered baseline candidate selected on validation (A–E or I). The exact primary contrast and direction must be written to the run manifest before touching the held-out test partition.

The target transfer has its own verification record, separate from the source theorem's verification. A `verified_transfer` label requires a pinned target source commit and artifact hash, exact target statement digest, checker/version and receipt, explicit axiom policy, independent review that the formal target matches the intended claim, a passed dependent-successor test (or a documented not-applicable decision), passed mutation controls, and a compatible directional contract with no mismatch codes. Source-proof verification alone does not discharge these target-side obligations.

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
- Use the absolute percentage-point paired effect as the principal reported effect. Baseline selection must score A–E and I on the same frozen validation targets and use semantic-gate success rate, with alphabetical arm ID as the tie-break. Publish the I adapter's retrieval-only recall/coverage and end-to-end success separately; the published LeanSearch v2 paper's external metrics are context, not a baseline score unless reproduced under this run's pinned task/corpus/budget. Use a predeclared family-cluster bootstrap with 10,000 resamples and a fixed, recorded bootstrap seed for its 95% confidence interval. Publish per-family aggregates so the calculation can be independently reproduced.
- Claim a practically meaningful transfer improvement only if the point estimate is at least **+10 percentage points** versus the selected validation baseline **and** the 95% interval excludes zero on the positive side. Otherwise report the estimate and interval without a win claim.
- For secondary confirmatory comparisons, correct for multiple comparisons (Holm). Label unplanned analyses exploratory. Report denominators, exclusions, missingness, ties, per-domain slices, and raw counts alongside percentages. If the effective number of eligible test families is too small for an informative interval, call the result underpowered; do not weaken the gate.
- For insight-quality labels, use at least two independent annotators blind to system arm and outcome, adjudicate disagreements, and report agreement (e.g. Krippendorff's alpha) rather than hiding disagreement. A suggested reliability target is alpha >= 0.80; below it, treat subjective labels as exploratory or revise the rubric before the confirmatory run.
- Precompute statistical power once eligible sample size and baseline rate are known. Use a preregistered effect grid of **0, +10, +15, and +20 percentage points** and family-clustering scenarios of **ICC 0, 0.05, 0.10, and 0.20**, preserving the eligible-family size distribution observed in validation. Report the simulated probability of passing the complete utility gate at each scenario and Monte Carlo uncertainty.
- Because the utility gate requires both an estimated gain of at least +10 percentage points **and** a positive 95% interval, planning power at a true effect exactly equal to +10 points is structurally limited: even when the interval is positive, sampling noise puts the estimate below the threshold about half the time in a well-behaved symmetric approximation. Therefore do **not** treat 80% power at exactly +10 points as a coherent design requirement. The confirmatory design should attain a **95% Monte Carlo lower confidence bound of at least 80%** for gate-pass probability under a true +20-point effect at the highest preregistered plausible ICC (0.20), and a **95% Monte Carlo upper confidence bound of at most 5%** for the zero-effect false-claim probability. Report the +10 and +15 cases regardless of outcome. If it cannot meet those gates, expand the separately sourced eligible set or report an underpowered pilot—not a benchmark victory.
- **PITB v0 minimum-cluster floor:** require at least 20 eligible independent test families before even considering a confirmatory claim. Below 20, label the result an underpowered pilot regardless of the point estimate or bootstrap interval. This is a conservative project-specific floor, not a universal theorem about cluster counts and not a substitute for power analysis. Few-cluster inference depends on the number and balance of clusters; cluster-randomized trial evidence has found low power in some settings with fewer than 20 clusters, while general cluster-inference guidance warns that simple cluster-count rules can fail when cluster sizes are imbalanced ([small-cluster trial analysis](https://academic.oup.com/ije/article/47/1/321/4091562); [Cameron, Gelbach & Miller / empirical cluster-inference guidance](https://doi.org/10.1016/j.jeconom.2022.04.001)).

This is a protocol template; the alpha level, power assumptions, eligibility denominator, model budget, and any deviation must be recorded in the final run manifest **before** inspecting confirmatory test results.

## Protocol amendment log
- **2026-10-21, before confirmatory outcomes were collected or inspected:** elevated benchmark-row validity from a broad semantic-gate principle to an explicit audit track. Confirmatory targets require two independent statement-correspondence reviews, full dependency/axiom receipts, pinned static defect screens, nearby statement mutations, and raw/eligible/excluded/unknown/defect accounting. This follows the June 2026 published audit of five Lean benchmarks that found 4,833 findings across 10,318 problems, including 398 mechanically certified issues. These requirements do not claim that finite static checks prove universal benchmark validity. All future frozen run configs must hash this protocol revision.

- **2026-10-21, before confirmatory outcomes were collected or inspected:** made `compatibility_probability` a distinct nullable field from action confidence in the transfer-selectivity schema and defined it as probability of mathematical compatibility independent of accept/reject/abstain. The Rust analyzer reports probability coverage, Brier score, ten-bin reliability data, ECE, and component-clustered 95% bootstrap intervals for Brier/ECE when at least two scored components are present; nulls are not imputed. Calibration outputs remain descriptive in v0 because no calibration acceptance threshold is preregistered. Any run config must hash the current protocol bytes.

- **2026-10-21, before any confirmatory outcomes were collected or inspected:** added a separate insight-component intervention track. It preregisters full insight versus a high-similarity incompatible decoy, precondition/failure-boundary/evidence ablations, and a no-insight control under matched task budgets. This is a secondary causal experiment and does not change the primary A-I retrieval contrast. Confirmatory component claims require same-token matching, paired target outcomes, family/component-clustered uncertainty, and content-addressed condition/input receipts. No ablation or transfer result is claimed; any run config must hash the current protocol bytes.


- **2026-10-21, before any confirmatory outcomes were collected or inspected:** added a secondary transfer-selectivity track with blinded, dual-review compatibility labels, positive and hard-negative controls, source/target-family leakage prevention, pair-graph component-level uncertainty, and separate false-accept/recall/abstention/coverage reporting. This does not change the primary F-versus-validation-selected-baseline endpoint. It closes a reject-everything loophole in transfer-gate evaluation: the finite negative mutation suite is an integrity check, while positive-control recall and held-out false-accept rate test whether transfer judgments are useful. The track's sample-size and recall targets are proposed acceptance gates; they are not measured results. The updated protocol bytes must be hashed into any future frozen run config.


- **2026-10-20, before any confirmatory model outcomes were collected or inspected:** added a conservative minimum of 20 eligible independent test-family clusters for a confirmatory utility claim and made simulation-based power planning mandatory. This is an additional no-go condition, not a relaxation of the +10 percentage-point effect or positive 95% interval gates. The floor is project-specific and not itself evidence of adequate power. The same pre-outcome amendment requires explicit outcome/exclusion rows for every frozen validation and test family to prevent silent denominator shrinkage, requires the baseline scores to be recomputed from content-hashed raw validation outcomes, and makes power planning a gate: simulate the combined utility rule across true effects 0/+10/+15/+20pp and ICC 0/0.05/0.10/0.20, with ≥80% power at +20pp and ICC 0.20 and ≤5% false claims at zero effect. This specifically avoids demanding 80% power at the same +10pp threshold the observed point estimate must exceed. The Monte Carlo confidence bounds prevent an under-replicated simulation from being declared adequate merely because its point estimate passes. It also adds arm I, a strong premise-retrieval comparator, because the May 2026 LeanSearch v2 paper reported 46.1% recovery of ground-truth premise groups within 10 candidates on its 69-query research-Mathlib benchmark and 20% downstream proof success in its own fixed-prover setup; those numbers are not treated as directly comparable PITB results. The current file bytes, not an earlier commit of this document, must be hashed into the frozen run configuration. Git history preserves the prior draft and this amendment.

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

## Transfer-selectivity secondary track

This track evaluates whether the typed transfer gate distinguishes genuinely reusable strategies from superficially similar but incompatible targets. It is **secondary** to the primary exact-target solve comparison: a good rejection score cannot compensate for failing to improve verified target solves, and a high solve score cannot conceal unsafe transfer judgments.

Construct source-insight/target-obligation pairs with labels assigned before any model or retrieval output is inspected. Two independent reviewers classify each pair as `compatible`, `incompatible`, `partial`, or `unknown`; preserve both labels and adjudication. If disagreement remains unresolved, retain `unknown` rather than forcing a binary ground truth. Each label must cite the source proof move, target obligation, required preconditions, and a concrete reason code.

The pair set must contain both compatible positive controls and incompatible negatives. Include hard negatives with high lexical or topical overlap, not only obvious out-of-domain pairs. Cover at least: missing preconditions, object/domain mismatch, reversed inequality direction, predicate mismatch, changed quantifier scope, unavailable target premise, and irrelevant near-neighbour. Keep the existing finite mutation suite as an integrity gate; 100% rejection of that finite suite is not a population-level false-accept estimate.

Prevent family leakage in this track separately from the theorem-solve split: a source insight may not appear in more than one partition, and source/target family edges must not cross partitions. If the transfer-pair graph has connected components that make this impossible under the intended split, keep each whole component in one partition and use those components—not individual pairs—as the independent statistical units. Report raw pair counts, source-family counts, target-family counts, component counts, unknown labels, and all exclusions.

For every pair, the system must emit one of `accept`, `reject`, or `abstain`, plus specific precondition checks and mismatch codes. Report (a) false-accept rate among adjudicated incompatible pairs, (b) recall among compatible pairs, (c) false-reject rate, (d) abstention and accepted coverage, and (e) selective risk as coverage varies. Every pair outcome distinguishes action confidence from `prediction.compatibility_probability`, which estimates the probability the transfer is compatible regardless of whether the system accepts, rejects, or abstains. When that probability is available, report its coverage, Brier score, and ten-bin reliability data; never impute a missing probability or infer calibration from cosine similarity. The v0 Rust analyzer currently reports Brier/ECE/reliability descriptively, while component-clustered 95% intervals apply to the primary selectivity rates; a confirmatory calibration claim requires a separately reviewed uncertainty analysis and adequate probability coverage.

Proposed transfer-gate adoption targets are: reject every item in the frozen finite negative-control suite; at least 90% point recall on compatible held-out positive controls; and at least 20 independent pair-graph components in each of the compatible and incompatible classes before any confirmatory transfer-selectivity claim. Report component-clustered 95% intervals for rates. These are conservative project-specific targets, not achieved results or a substitute for power analysis. If the sample-size floor is unmet, report a pilot and extend the independently adjudicated pair set without tuning against the held-out labels.

## Insight-component intervention track

This is a separate, pre-registered secondary experiment. It does not add arms to or retroactively alter the primary A-I retrieval ablation. Its purpose is to test **which parts of a typed insight cause measurable utility**, rather than infer causal value from a correlation between record quality and success.

### Conditions

For each eligible target, preserve the target statement, checker, model, system prompt, retrieval corpus/candidates, tool permissions, retry policy and budget. Randomize condition order per target with a frozen seed, disable cross-condition conversation/cache carryover where possible, and retain the exact material shown to the solver.

1. **Full insight:** strategy, preconditions, state transition, evidence locators, and failure boundaries.
2. **Surface-matched decoy:** a high lexical/topical-similarity source insight adjudicated incompatible with the target. This is not a fabricated false theorem; the mismatch and source provenance are reviewed in advance.
3. **Precondition ablation:** preserve the strategy and evidence, but mask explicit applicability preconditions.
4. **Failure-boundary ablation:** preserve the strategy and preconditions, but mask known non-applicability conditions.
5. **Evidence ablation:** preserve the paraphrased move and contracts, but remove source locators/evidence anchors.
6. **No-insight control:** identical target context with no insight record.

Generate every ablation mechanically from one immutable full record. Each condition must be bound to a condition-manifest digest and derived-input digest; do not hand-edit an individual test prompt after inspecting a result. Use a fixed model tokenizer to replace masked content with neutral placeholders of the same token count. If exact token matching cannot be achieved for the selected model, report actual input tokens and treat component contrasts as exploratory, not confirmatory. All conditions retain the same hard token/call/time/cost limits.

### Outcomes and causal interpretation

The primary secondary-track contrast is **full insight versus surface-matched decoy** on exact-target semantic-gate-verified success. Report the paired percentage-point effect and family-clustered 95% interval. Secondary contrasts compare full insight against each ablation and against no insight. Also report (a) exact checker success, (b) semantic-gate success, (c) transfer accept/reject/abstain quality on adjudicated pairs, (d) failure mode, and (e) cost/time/tokens per verified solve. Do not combine these outcomes into an opaque weighted score.

Use the frozen target/family set in every condition and retain timeouts, tool-unavailable outcomes, abstentions, and exclusions. Resample independent family or pair-graph components, not individual correlated targets. A causal component claim requires the predeclared contrast, no budget/token drift, all receipts, and a 95% interval whose lower bound is above zero; a proposed practical-value target is +10 percentage points. A component result below that threshold, a null interval, or inability to match tokens is reported as inconclusive/exploratory, not repaired by changing prompts or thresholds after the run.

This design operationalizes three falsifiable questions: does a semantically compatible strategy help beyond a topical near-miss; do explicit preconditions reduce false acceptance or improve verified success; and do failure boundaries and provenance add measurable value? It is still a design until the conditions, receipts and outcomes actually exist.

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
