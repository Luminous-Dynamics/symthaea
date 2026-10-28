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


### Proof-progress traces: distinguish exploration from verified advancement

The terminal exact-target semantic-gate outcome remains the primary endpoint. A separate optional trace records the retained execution path: state fingerprints, obligation-instance lineage, action and checker events, monotonic resource counters, and content-addressed receipts. This supports diagnosis and resource attribution without promoting a progress proxy into proof success.

- Every obligation instance has an ID distinct from its statement digest. This matters because two branches may contain the same formula but represent different proof obligations.
- A `split` or `refined` transition is **not** a verified closure. Count progress only when an obligation is explicitly `discharged` with a checker receipt; a decomposed parent is not counted as a solved leaf.
- For `verified_success`, every terminal obligation leaf must be receipt-backed discharged, the exact target must pass the kernel, semantic gates must pass, and successor checks must pass or be explicitly not applicable.
- Event sequence, event identities, elapsed time, cumulative model calls/tokens/cost, and declared budgets are validated. A duplicate close, orphan obligation, non-monotonic counter, missing receipt, unsupported final success, or budget overrun is a trace-integrity failure.
- For all arms, report terminal semantic-gate success as the primary outcome. Proposed diagnostics are receipt-backed leaf closures per 1,000 generated tokens, tokens/time to first receipt-backed closure, terminal leaf-closure fraction, and the distribution of no-closure tactic events. No progress diagnostic is currently a confirmatory score; no-progress attribution across backtracking branches is out of scope until a branch-aware trace is designed.

The schema and CI fixture are a **contract and synthetic guardrail**, not a running Lean telemetry collector and not measured proof-progress results. The v0 fixture deliberately splits one root obligation into two children, then closes both with distinct receipts; the validator must count exactly two closures, never count the split itself, and reject mutated traces. Future instrumentation should emit the trace from the prover/checker boundary rather than reconstruct it from model narration.

This track follows two complementary findings: *Discover and Prove* separates answer discovery from formal proof in Hard Mode, while OptProver explicitly penalizes valid but non-progressing proof steps in its training objective ([DAP / ACL 2026](https://aclanthology.org/2026.acl-long.3/); [OptProver / ICML 2026](https://proceedings.mlr.press/v306/li26ek.html)). These studies motivate the experiment design; their scores are not PITB baselines.


### Global failure recovery and cost-aware action routing

The proof-progress trace now has explicit event types for `failure_diagnosed`, `lemma_invalidated`, `subtree_retracted`, `global_replan`, `action_routed`, and `automation_dispatched`. Failure diagnoses use a bounded taxonomy (`insufficient_guidance`, `false_lemma_or_specification`, `decomposition_error`, `missing_dependency`, `tooling_failure`, `budget_exhaustion`, `unknown`); replan / route / retraction events require decision receipts. An `action_routed` event also records the proposed action class, its pre-action success-probability estimate, and estimated micro-USD cost. The schema is a telemetry contract; no current PITB solver emits these events yet.

This introduces an important distinction: a local failed tactic is not necessarily a local problem. The strategy could need better guidance, the candidate lemma may be false, a previous decomposition may be wrong, or the proof's dependency structure may be missing a bridge. A coordinator should be able to invalidate a false lemma or retract a flawed subtree and replan globally, instead of repeatedly decomposing the same failed direction. Action routing should weigh estimated verified-success probability against the cost of another specialist invocation, but record estimates *before* the action and link them to an actual receipt-backed outcome. Do not grade a router on its own self-reported probability.

#### Experimental arms and measurements

On the same frozen target set and checker, compare:

1. **Minimal shell-agent baseline:** a general coding agent with filesystem/shell access and a separate trusted evaluator; this tests how much value requires any specialised harness.
2. **Fixed local recovery:** retrieve / attempt / compiler-feedback repair, with a preregistered uniform retry policy.
3. **Typed-insight local recovery:** the same fixed loop with an insight and its explicit preconditions.
4. **Global recovery coordinator:** full proof-dependency graph, classified failure evidence, and explicit ability to retract helper lemmas/subtrees and replan.
5. **Cost-aware coordinator:** arm 4 plus a router choosing between automation, direct tactic, helper-lemma construction, retrieval, or specialist invocation according to estimated success and cost.

Freeze model/tool versions, all route/action budgets, action prices, input contexts, retry and cache policy. On counterfactual pairs, keep the model, prompt template, configuration, checker policy and caps matched; the target artifacts necessarily differ only by the independently adjudicated mutation. Report both the primary semantic-gate solve endpoint and the **success–cost frontier**—verified solves at fixed spending caps and cost at fixed verified-solve rates. Additional diagnostics include cost/time to first receipt-backed leaf closure, cost per semantic-gate success, retraction/replan frequency, verified recovery rate after a diagnosed false lemma or decomposition error, and calibration of pre-action success/cost estimates. The proof DAG / lemma dependency graph may be the right routing substrate; an arbitrary proxy count of closed subgoals is not.

Industry comparators now include cost-aware graph routing. CoCo-Prover reports a better success–cost frontier and up to 30.9% lower cost than its strongest baseline in its own five-benchmark Lean program-verification setup ([paper](https://arxiv.org/abs/2610.09681)). Maestro studies failure diagnosis, global coordination, flawed-lemma correction, subtree retraction and replanning, with reported strong results on PutnamBench and other benchmarks ([paper](https://arxiv.org/abs/2610.25920)). A minimal bash-only Lean agent also provides a useful control for determining whether a specialised harness adds value ([paper](https://arxiv.org/abs/2610.17331)). These results are **not comparable PITB scores**; they identify baselines and capabilities we should test.

Pre-register the cost target separately from the primary utility gate. Proposed advancement criteria: (a) preserve the primary solve-rate requirement under matched budgets; (b) at fixed semantic-gate success, aim for at least 25% lower cost than the strongest validated fixed-local-recovery baseline, with component-clustered uncertainty reported; and (c) improve verified recovery after false-lemma/decomposition diagnoses without raising the false-accept rate. The 25% target is aspirational, informed by the published cost-efficiency frontier; it is not a measured PITB result. If the sample is too small to estimate the frontier, report it as exploratory and collect more independent components rather than relaxing the gate.


### Frozen multi-budget success–cost frontier plan

Use `success-cost-frontier-plan.schema.json` to freeze a complete **target × method arm × budget cap** schedule before observing outcomes. A valid plan declares at least three resource-cap levels whose model-call, token, wall-time and micro-USD caps each strictly increase; every target/method/cap cell must occur exactly once. Run IDs, target/component links and arm/budget references must be unique and complete. The synthetic validation-only fixture exercises two targets, three distinct orchestration arms and three caps (18 assigned runs); it is a schedule-contract fixture, not an execution dataset.

Each target appears unchanged in every arm and budget tier, with the same checker/toolchain and frozen target-manifest digest. Arms must freeze model/version, prompt/config digests, checker/version, retry policy and cache policy. When comparing coordinator variants, the intended difference is the orchestration or routing intervention, not a hidden model, prompt or retry-budget change. All missing, timed-out, tool-unavailable and non-solve assignments remain in the execution ledger.

The descriptive implementation is `crates/bridges/symthaea-proof-insights/src/bin/analyze_success_cost_frontier.rs`, which consumes the frozen plan and a complete JSONL outcome ledger (one row per planned target × arm × budget assignment). Each row is joined on run ID, target, component, arm, budget, and exact input digest; the target statement digest checked by the checker must match the plan. Attempted rows require the four retained run/checker/semantic/successor receipts, while not-attempted/tool-unavailable/out-of-scope rows have zero reported usage and remain non-solves. The analyzer independently enforces model/checker/config identity and every assigned resource cap. Its report binds the raw plan, raw outcome ledger, and analyzer source hashes.

At every budget cap it emits exact-target semantic-gate success rate over all assigned targets, actual usage and cost-per-assigned-target/cost-per-verified-solve, plus paired same-budget arm differences with component-clustered percentile intervals. It derives a nondominated success-rate/cost frontier from observed points only. Resample independent target/family components, not individual budget-tier attempts; repeated runs on one target are correlated. Do not linearly interpolate across unmeasured budget regions. The primary +10-percentage-point verified-utility gate and the additional aspirational 25% cost target remain separate. The frontier report's confirmatory decision is deliberately fixed to `not_evaluated_*`; the analyzer neither opens the contents of checker receipts nor substitutes for the primary preregistered analysis and power gate. The 18-row synthetic ledger is all `not_attempted`, and the synthetic report must have no curves, paired effects, or Pareto points. A complete plan/report contract proves neither that model runs occurred nor that a cost reduction was achieved.

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
- **2026-10-22, before confirmatory outcomes were collected or inspected:** split the semantic-preserving counterfactual class into syntax-only no-op and changed-surface equivalent reformulation. This reflects recent global/local perturbation research and prevents an identical-fragment control from standing in for invariance to a changed surface form. Added one synthetic matched pair, outcome joins, class-specific stability output, and fail-closed mutation controls; no mathematical robustness result is claimed. Future run configs must hash this revision.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** added a frozen multi-budget success–cost frontier plan schema. CI requires complete target × arm × budget coverage, three strictly increasing common resource caps, unique run IDs and component-consistent assignments. This is experiment scheduling only; no outcome matrix or frontier is claimed. Future run configs must hash this protocol revision.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** extended the optional trace schema to record classified proof failures, lemma invalidation, subtree retraction, global replanning and budget-aware action routing with decision receipts. Added a separate success–cost frontier experiment; it does not replace the primary exact-target semantic-gate endpoint. No coordinator or router result is claimed. Future run configs must hash this protocol revision.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** added a distinct counterfactual artifact path alongside each variant artifact digest. This prevents a paired row from claiming different bytes at one path. The current paths are synthetic placeholders; production evaluation must verify both artifact bytes inside a pinned bundle. No theorem-prover result is claimed.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** separated source-artifact identity and base/counterfactual target-artifact identities in the paired sensitivity track; the analyzer binds each side to its own artifact digest and rejects one shared identity for byte-distinct variants. It also checks full-suite split/component leakage and matched inference configurations. This is a provenance-integrity amendment only, not a performance result.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** added a counterfactual precondition-sensitivity track with paired base/variant target digests, mutation-receipt and reviewer contracts, stable outcome-row join keys, split/component isolation, and explicit stability/sensitivity/false-accept/false-reject metrics. The seven-pair checked-in fixture is synthetic and no system output has been scored. Future run configs must hash this protocol revision.
- **2026-10-22, before confirmatory outcomes were collected or inspected:** added an optional proof-progress trace contract. It records obligation-instance lineage and receipt-backed discharges but explicitly excludes goal splits from progress, retains the terminal semantic-gate endpoint as primary, and adds synthetic relational invariants / anti-gaming controls. No trace collector or target-level progress result is claimed. Future run configs must hash this protocol revision.
- **2026-10-21, before confirmatory outcomes were collected or inspected:** elevated benchmark-row validity from a broad semantic-gate principle to an explicit audit track. Confirmatory targets require two independent statement-correspondence reviews, full dependency/axiom receipts, pinned static defect screens, nearby statement mutations, and raw/eligible/excluded/unknown/defect accounting. This follows the June 2026 published audit of five Lean benchmarks that found 4,833 findings across 10,318 problems, including 398 mechanically certified issues. These requirements do not claim that finite static checks prove universal benchmark validity. All future frozen run configs must hash this protocol revision.

- **2026-10-21, before confirmatory outcomes were collected or inspected:** made `compatibility_probability` a distinct nullable field from action confidence in the transfer-selectivity schema and defined it as probability of mathematical compatibility independent of accept/reject/abstain. The Rust analyzer reports probability coverage, Brier score, ten-bin reliability data, ECE, and component-clustered 95% bootstrap intervals for Brier/ECE when at least two scored components are present; nulls are not imputed. Calibration outputs remain descriptive in v0 because no calibration acceptance threshold is preregistered. Any run config must hash the current protocol bytes.

- **2026-10-21, before any confirmatory outcomes were collected or inspected:** added a separate insight-component intervention track. It preregisters full insight versus a high-similarity incompatible decoy, precondition/failure-boundary/evidence ablations, and a no-insight control under matched task budgets. This is a secondary causal experiment and does not change the primary A-I retrieval contrast. Confirmatory component claims require same-token matching, paired target outcomes, family/component-clustered uncertainty, and content-addressed condition/input receipts. No ablation or transfer result is claimed; any run config must hash the current protocol bytes.


- **2026-10-21, before any confirmatory outcomes were collected or inspected:** added a secondary transfer-selectivity track with blinded, dual-review compatibility labels, positive and hard-negative controls, source/target-family leakage prevention, pair-graph component-level uncertainty, and separate false-accept/recall/abstention/coverage reporting. This does not change the primary F-versus-validation-selected-baseline endpoint. It closes a reject-everything loophole in transfer-gate evaluation: the finite negative mutation suite is an integrity check, while positive-control recall and held-out false-accept rate test whether transfer judgments are useful. The track's sample-size and recall targets are proposed acceptance gates; they are not measured results. The updated protocol bytes must be hashed into any future frozen run config.


- **2026-10-20, before any confirmatory model outcomes were collected or inspected:** added a conservative minimum of 20 eligible independent test-family clusters for a confirmatory utility claim and made simulation-based power planning mandatory. This is an additional no-go condition, not a relaxation of the +10 percentage-point effect or positive 95% interval gates. The floor is project-specific and not itself evidence of adequate power. The same pre-outcome amendment requires explicit outcome/exclusion rows for every frozen validation and test family to prevent silent denominator shrinkage, requires the baseline scores to be recomputed from content-hashed raw validation outcomes, and makes power planning a gate: simulate the combined utility rule across true effects 0/+10/+15/+20pp and ICC 0/0.05/0.10/0.20, with ≥80% power at +20pp and ICC 0.20 and ≤5% false claims at zero effect. This specifically avoids demanding 80% power at the same +10pp threshold the observed point estimate must exceed. The Monte Carlo confidence bounds prevent an under-replicated simulation from being declared adequate merely because its point estimate passes. It also adds arm I, a strong premise-retrieval comparator, because the May 2026 LeanSearch v2 paper reported 46.1% recovery of ground-truth premise groups within 10 candidates on its 69-query research-Mathlib benchmark and 20% downstream proof success in its own fixed-prover setup; those numbers are not treated as directly comparable PITB results. The current file bytes, not an earlier commit of this document, must be hashed into the frozen run configuration. Git history preserves the prior draft and this amendment.

- **2026-10-21, before confirmatory outcomes were collected or inspected:** preregistered separate external challenge strata for frozen-infrastructure transfer, formal-infrastructure construction, and dependency-rich repository tasks. Added an adversarial proof-critic calibration design inspired by FormalRewardBench error-injection methodology. These are evaluation-design changes only: they do not alter the primary v0 endpoint, import external benchmark scores into PITB, or claim transfer gains. All run configs must hash this protocol revision.

## 6. Leakage and external validity

Family-level splitting blocks only direct within-family exposure under the catalog's declared mapping. It does **not** block:

- mathematically equivalent claims or shared lemmas across catalog families;
- copied paragraphs, near-duplicate proofs, or theorem variants under different IDs;
- common Mathlib declarations and imported proof dependencies;
- pretraining or post-training exposure to public manuscripts and benchmarks;
- model-specific memorization or contamination in retrieval caches.

Before calling a result generalization evidence, add a leakage audit: normalized theorem-statement and proof fingerprints; cross-family dependency/near-duplicate edges; source-to-target family overlap review; and exact retrieval logs showing every item surfaced for each target. Freeze this audit before scoring. Keep a separate *external challenge set* with later, unpublished, or otherwise exposure-resistant targets, adjudicated for rights and feasibility. Report unknown exposure explicitly; do not claim that hashing alone proves non-contamination.


### External challenge strata: transfer versus infrastructure construction

The v0 primary endpoint remains **exact-target proof success with the target development and checker context frozen**. Do not pool infrastructure-building tasks into that endpoint. For external generalization, report these strata separately:

1. **Frozen-infrastructure transfer:** target definitions, imports and dependency closure are supplied; only the requested proof obligation is changed.
2. **Infrastructure construction:** a vetted informal argument is supplied, but the agent must create missing formal definitions, intermediate lemmas and proof scaffolding before reaching the target. AutoformBench (released 13 October 2026) is a relevant comparator design: 341 Lean 4 formalization tasks from 24 advanced-mathematics sources, with human-formalized and audited target statements ([benchmark and results](https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/)). Its headline task success must not be numerically compared to PITB exact-target success because the work and denominator differ.
3. **Dependency-rich repository transfer:** project-local definitions, imports and successor obligations are preserved, as in repository-scale evaluation; report dependency closure size/depth and context regime.

For the infrastructure stratum, record whether the target proof succeeds, whether required definitions/helper lemmas were constructed, the number of accepted intermediate declarations, downstream successor-test results, and total calls/tokens/time/cost. The hypothesis is that typed proof insights may be most valuable at *infrastructure bottlenecks*; this remains untested and must be evaluated as a separate interaction/stratified effect, not asserted as a result.

### Adversarial proof-critic calibration

Proof generation and proof criticism are separate capabilities. FormalRewardBench evaluates 250 Lean proof preference pairs with five curated error-injection families; its paper reports 59.8% for its best frontier LLM evaluator and 24.4% for specialized theorem-prover models in that study's setup ([paper](https://arxiv.org/abs/2605.10141)). Those scores are **not** transfer-compatibility results and are not PITB baselines. Borrow the experimental method, not the score: add contrastive pairs where one minimal change breaks a key hypothesis, type, quantifier, premise or inference direction while a style-only change preserves the proof obligation. Two blind reviewers should label whether the *strategy applies to the target*, independently of whether a proposed proof compiles.

Report false acceptance, false rejection, abstention, probability coverage and calibration by error class. Preserve the positive controls: a critic that rejects every candidate is not useful. Keep these critic outcomes secondary to verified target-solve utility and do not count proof-critic accuracy as proof success.

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


## Mutation-qualification prerequisite
### Kernel-checkable calibration pilot

The isolated package at [`lean-pilot/`](lean-pilot/PITBCounterfactualPilot.lean) provides a small, pinned formal calibration set that is independent of the synthetic JSON fixture:

- `alpha_renaming_same_type`: a proof by reflexivity that alpha-renamed binders define equal proposition terms.
- `identity_statement_iff_true_function`: a kernel-checked propositionally-equivalent reformulation with an explicit `↔` proof.
- `successor_monotonicity_with_premise`: the exact proof move `Nat.succ_le_succ h` when the required order premise is present.
- `premise_removed_variant_is_false`: a kernel-checked contradiction demonstrating that the unconditional successor-order statement is false.
- `same_goal_after_hypothesis_clear`: an instrumented Lean proof traces the goal/local context before and after `clear h`; the target type remains unchanged, `h : a ≤ b` disappears, and an alternate equality hypothesis closes the same goal. CI re-elaborates this file separately, checks the two structured trace blocks, and hashes their snapshots/context digests into the receipt. This is one formal calibration example, not a model run or a general replay engine.

The isolated workflow pins Lean 4.34.1, builds the package, invokes Nanoda with `sorry` disallowed and an axiom allowlist audit, then directly re-elaborates the source to capture structured base/variant state traces. It checks the same goal type in both states, confirms that `h : a ≤ b` appears only in the base state, and hashes the state blocks and normalized local contexts into a content-addressed receipt bound to the commit, case manifest, toolchain, Lean/Lake versions, compiled `.olean`, build log, and trace log. The receipt is evidence only for these toy declarations under this pinned environment. It is not an outcome row for an AI system; the model has not attempted these pairs, and the pilot does not establish transfer utility or benchmark-scale sensitivity.

**Promotion rule:** after this workflow produces a successful exact-head artifact, use it to test the mutation-qualification data path and reviewer rubric before authoring a real train/validation/test set. Do not backfill the synthetic suite with the Lean pilot and call it a scored result; keep fixtures, formal harness examples, and model outcome datasets separate.


Before counterfactual pairs enter a confirmatory denominator, apply [the mutation adequacy protocol](counterfactual-mutation-adequacy.md) and validate a versioned [mutation-qualification record](counterfactual-mutation-qualification.schema.json). This record separates elaboration validity, formal statement relation, strategy-contract applicability, theorem truth evidence, and actual checked strategy outcome. In particular, a parser/elaborator failure is an invalid mutant—not evidence that the system correctly rejected an inapplicable mathematical strategy. The checked-in qualification fixture and CI controls are synthetic only. Freeze the schema, rubric, receipts, and current protocol digests in every run manifest; no confirmatory claim is permitted until a real pinned Lean pilot supplies the corresponding evidence.

## Counterfactual precondition-sensitivity track

This separate track tests whether a typed insight is applied because its mathematical preconditions hold, rather than because the target looks similar. It does **not** change the primary F-versus-validation-selected exact-target endpoint or the existing transfer-selectivity metrics.

### Pair construction

Each contrast links two target-outcome IDs for the same source insight:

- **Base control:** an independently reviewed target on which the source insight is applicable.
- **Counterfactual:** a controlled statement variant or altered premise that tests one declared applicability boundary.
- **Semantics-preserving control:** a byte-level variant whose formatting/normalization changes, but whose intended mathematical statement remains equivalent.

For each pair, preserve the source artifact digest and insight digest, separate base and counterfactual target-artifact paths and digests, separate statement digests, mutation-fragment digests and mutation receipt, mutation class, separate adjudication/reviewer-evidence receipts, and stable `base_outcome_pair_id` / `counterfactual_outcome_pair_id` join keys. The analyzer must bind each outcome to the right variant's exact path, artifact digest, and statement digest; a shared path or artifact hash for two byte-distinct variants is an integrity failure. In a real run, both paths must resolve within a pinned, auditable artifact bundle; the checked-in synthetic paths are only contract fixtures. A change type such as reversed inequality is not automatically a gold label for every problem: expert review must verify that the particular mutation actually breaks the strategy's required contract. Use two independent reviewers, with the same reviewer set assessing both sides of a contrast; preserve unresolved decisions as unknown.

The draft schema distinguishes two semantic-preserving controls: (a) `no_semantic_change`, a syntax-only no-op whose tracked fragment digest stays identical, and (b) `equivalent_reformulation`, where the surface fragment digest changes but blind review confirms the intended formal claim remains equivalent. It separately covers removed/strengthened preconditions, inequality-direction reversal, changed object type, changed quantifier scope, removed premises, changed predicates, and weakened conclusions. The checked-in nine-pair suite is explicitly synthetic, including one pending unknown adjudication. Its placeholder labels demonstrate structure and CI guardrails only; it is **not** a mathematically adjudicated dataset.

### Global-equivalence robustness versus local semantic sensitivity

A 16 October study of Lean autoformalization robustness separates **global perturbations** (a theorem/proof pair is paraphrased and should preserve intended meaning) from **local perturbations** (a value, symbol or proof step changes and the output must faithfully reflect that change). MathAdv also tests equivalent reformulations across 13 mathematical domains and reports substantial robustness limitations in contemporary provers ([robustness study](https://arxiv.org/abs/2606.14867); [MathAdv](https://arxiv.org/abs/2608.25449)).

PITB's `equivalent_reformulation` class adapts this distinction to *formal insight transfer*: the surface fragment digest changes, the target artifact and statement digests remain pinned, but independent adjudication confirms the applicability label remains compatible. This is **not** a reproduction of either benchmark's natural-language task. Local semantic mutations remain incompatible only when blind reviewers establish that the actual edit breaks the strategy's requirement. The goal is to measure strategy invariance to equivalent rewrites separately from sensitivity to broken preconditions and faithfulness to locally changed targets.

### Metrics and adoption gates

Report each dimension separately; do not collapse them into one score:

1. **Paired contract adherence:** the system accepts the compatible base and rejects or abstains on the independently verified incompatible variant.
2. **Mutation sensitivity:** among reviewed mutations that invalidate the required contract, the fraction rejected or abstained on, conditional on the base control being handled correctly.
3. **Semantics-preserving stability:** the decision remains stable on equivalent rewrites; also report absolute change in the predicted compatibility probability rather than treating threshold stability as perfect calibration.
4. **False rejection:** compatible bases and semantics-preserving variants incorrectly rejected.
5. **False acceptance:** incompatible variants accepted despite the changed requirement; break this down by mutation class.
6. **Outcome linkage and cost:** join each side to its actual transfer-selectivity outcome row and report time, tokens and verified proof utility for both versions.

Proposed gates for a future confirmatory version are at least 90% mutation sensitivity alongside at least 90% compatible-positive recall, at least 95% decision stability on adjudicated semantics-preserving variants, rejection of 100% of a frozen finite integrity suite, and at least 20 independent pair-graph components in each primary class plus simulation-based power planning. These are targets, not observed results. A reject-everything strategy fails positive recall; a strategy that accepts every near-match fails the mutation controls.

### CI contract currently implemented

`counterfactual-precondition-suite.schema.json` and `fixtures/counterfactual-precondition-suite.synthetic.json` validate typed mutation records, distinct base/counterfactual statement digests, unique outcome-row join keys, paired reviewer identities and labels, and component/split isolation. CI applies 13 intentionally malformed synthetic variants. This checks the dataset contract, not mathematical correctness and not a model's behavior. Real evaluations additionally need independently reviewed Lean statements, executed mutation receipts, outcome rows and target-kernel evidence.

This design is motivated by three separate ideas in the literature: T² tests formal statements through dependent successor theorems; FormalRewardBench uses expert-curated proof error injections; effect-grounded tactic representations study the structural edit a tactic induces rather than only its surface text ([T²](https://aclanthology.org/2026.acl-industry.150/), [FormalRewardBench](https://arxiv.org/abs/2605.10141), [effect-grounded tactic representations](https://aclanthology.org/2025.mathnlp-main.12/)). These works motivate the test design, not a claim that the counterfactual track already improves PITB.


### Executable paired counterfactual analysis

The Rust `analyze_counterfactual_sensitivity` binary first validates the suite's strict nested envelope, dual-review agreement/evidence, and consistency between mutation class, semantic-change flag, and before/after fragment digests. It then joins the base and counterfactual outcome-row IDs to the selected-split transfer-selectivity JSONL. It fails closed on unreferenced/missing/duplicate rows; source-artifact, insight, variant-specific target path/artifact or statement digest drift; component or adjudication mismatch; reviewer evidence/reason mismatch; matched model/config/prompt/budget violations; and declared-budget overrun. It audits split and family-component isolation across the entire suite, including partitions not scored in the current invocation.

Report at minimum: compatible-base acceptance recall; compatible counterfactual acceptance recall; rejection-or-abstention sensitivity and false acceptance on incompatible variants; paired contract adherence; exact decision stability on semantics-preserving variants; compatibility-probability Brier score; mean absolute probability shift for no-op variants; and sensitivity/false acceptance by mutation type. It bootstraps connected pair components rather than individual edges and records the exact suite/outcome content digests and seed. Also report actual model calls, tokens, wall-clock milliseconds and micro-USD by side, their per-pair means, variant/base ratios, and signed variant-minus-base deltas with component-clustered intervals. These are paired descriptive costs for two target variants; a full success–cost frontier requires preregistered runs at multiple matched spending caps and must report verified solves at fixed cost and cost at fixed solve rate. Do not infer cost savings from a single synthetic budget.

The checked-in synthetic fixture has eight pairs and sixteen outcome rows, including one pending/unknown contrast that is tracked but excluded from scored-pair metrics. CI asserts synthetic paired metrics and resource totals, validates the aggregate report against an independent Draft 2020-12 engine, and confirms failure for missing paired rows, source provenance drift, reviewer receipt disagreement, missing measured usage and over-budget outcomes. The fixture intentionally encodes strong, known answers so the analyzer path is testable; its perfect metrics and equal synthetic costs are **not** benchmark evidence, and the output always retains `overall_confirmatory_decision = not_evaluated_...`. Real benchmark claims require independently reviewed mathematical labels and actual target-checker receipts. The primary exact-target solve endpoint remains unchanged.

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
- Arnal et al., [AutoformBench: A Lean Formalization Benchmark for Advanced Mathematics](https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/), 13 October 2026.
- Uluşan et al., [FormalRewardBench: A Benchmark for Formal Theorem Proving Reward Models](https://arxiv.org/abs/2605.10141), May 2026.
- [OpenAI math source snapshot](https://github.com/openai/math/tree/1581d4cbdd69a0faf34b2089614fe2a814316a6a).


- **2026-10-23, before outcomes were generated or inspected:** implemented `analyze_success_cost_frontier.rs` with a strict outcome schema and report schema, matched-join and budget checks, exact checked-target digest binding, paired same-budget component-bootstrap differences, and an observed nondominated success/cost frontier. Added an 18-cell synthetic no-run ledger and CI schema/report/source-digest checks plus a synthetic-execution forgery negative control. The implementation always leaves the confirmatory decision unassessed and cannot substitute for the primary utility/power gate. The new tests are authored but not yet confirmed by a current exact-head Rust run; no model scores or cost reductions are claimed. All future frozen run configs must hash the updated protocol bytes.


### 2026-10-24 amendment — enforce fixed-target pair identity at analyzer boundary

The outcome analyzer now validates the suite contract before joining outcomes: both base and variant replay states must match the target descriptor's source commit, theorem-statement digest, elaborated-goal digest, environment digest, and Lean toolchain. The two states must also agree on goal count. Only local proof context/snapshot evidence may vary for this fixed-theorem intervention track. This is a fail-closed integrity gate; it does not prove that the captured state digests or replay receipts truthfully describe the underlying Lean process. A regression test mutates each invariant independently and expects rejection.


### 2026-10-24 amendment — validate intervention evidence, not just intervention labels

The Rust analyzer now checks that the declared operation matches the intervention kind and that the before/after declaration receipts and precondition contract agree. A cleared required hypothesis must be present before, absent after, type-matched to the insight's required type, and paired with an available→absent precondition status and a move-forbidden/replan policy. Alpha-renaming must preserve declaration type/position while changing the name. Irrelevant removal and independent-declaration reordering have separate structural contracts. This validates receipt consistency; it does not replace Lean replay or independent adjudication of whether a precondition is logically necessary.


### 2026-10-24 amendment — semantic-gate-qualified primary endpoint

Executed outcome rows must carry a semantic-gate record bound to the exact target statement. The record separately captures (i) statement correspondence with at least two independent reviewers for a pass, (ii) axiom-policy compliance and receipt, (iii) dependent-successor testing or a reasoned not-applicable decision, and (iv) mutation-control status and suite receipt. The analyzer derives `passed` / `failed` / `unknown` from the component evidence and rejects contradictions. The primary exact-target solve rate now requires both checker success and semantic-gate pass; raw checker success is reported separately.

The semantic-gate digests remain references, not self-authenticating proof of reviewer identity or receipt truth. Independent review and verification of receipt contents remain required. This changes the evidence contract, not the result of an experiment.


### 2026-10-24 amendment — preserve individual statement-review evidence

A semantic-gate pass now depends on at least two separately identified reviewer records, each with an individual decision and evidence digest. The analyzer rejects blank or duplicate reviewer identities, invalid evidence digests, and aggregate correspondence statuses that disagree with the individual decisions. A split decision remains `unknown`; it is not converted to a majority-vote label. The aggregate review-receipt digest remains as an additional envelope binding, not a substitute for the individual records.


### 2026-10-24 amendment — cross-family leakage is a confirmatory gate

The frozen family split is an assignment rule, not evidence that the holdout is independent. Before confirmatory scoring, prepare a versioned relationship inventory tied to the exact partition digest. Classify candidate cross-family links as exact duplicate, shared source artifact, formal dependency, reused lemma, equivalent statement, conceptual similarity, or citation-only; attach evidence locators/digests and preserve unresolved assessments.

For quarantine, an exact duplicate, shared source artifact, formal dependency, reused lemma, or equivalent statement is a hard relationship. A confirmed or unresolved hard edge joins its endpoints into a connected component; if that component spans train/validation/test, quarantine the whole component under a frozen, pre-outcome rule. Dismissed edges require two distinct reviewers. Conceptual similarity and citation-only links are inventoried but do not automatically count as leakage, because the benchmark's purpose is to measure safe transfer across related mathematical ideas.

The new `analyze_family_leakage` tool re-derives the registered split assignment, checks the upstream pin, 372-family count, 70/15/15 counts and exact partition digest, rejects unknown/duplicate family links, derives cross-split connected components, and reports original plus known-quarantined family counts without changing the frozen split. Evidence and coverage-receipt hashes are references, not byte-level authentication.

**Current status: not qualified.** The checked-in relationship fixture is synthetic control data only; the real independently reviewed relationship inventory is not yet present. An incomplete inventory must never certify a clean holdout. Before running model outcomes, reviewers must inspect the real evidence, complete and sign off the relationship-coverage audit, independently verify cited artifact bytes, preserve all exclusions in the run manifest, and freeze the adjusted denominator. If that leaves too few independent components, report an underpowered/exploratory result; do not reshuffle the split or relax this gate after viewing outcomes.

### 2026-10-24 amendment — deterministic title-only relationship triage

Added `discover_family_leakage_candidates`, which validates the same pinned 372-family partition and emits an explainable, content-addressed list of candidate cross-family relationships based on title token overlap. Its candidate report has a strict Draft 2020-12 schema, fixed tokenizer/stopword/rarity thresholds, shared-token explanations and split labels. CI runs the generator twice against the frozen manifest and byte-compares the serialized outputs, then validates the report and reconciles total, cross-split and train/test counts.

**Every output remains `title_similarity_only` and `unreviewed`.** The discovery output is never auto-converted into a hard dependency edge and never changes the split. The heuristic is designed to prioritize human review; it is not proof of mathematical dependence or actual leakage, and it may miss dependencies with dissimilar titles.

The generated report is produced in runner temporary storage rather than checked in as benchmark evidence. The actual dependency inventory remains incomplete; neither this new tool nor its synthetic CI controls qualify the benchmark holdout.
### 2026-10-24 amendment — independent artifact-path candidate channel

Added `discover_artifact_path_candidates`, which scans the pinned family's linked artifact paths independently from the title-token triage. Its output includes split labels, source titles, matching paths, shared/rare tokens, overlap score, rarity score and a content digest. A dedicated Draft 2020-12 schema and CI byte-for-byte reproducibility control protect its report.

All path-derived candidates remain `artifact_path_similarity_only` and `unreviewed`; a matching slug does not prove shared artifact bytes, mathematical dependence or contamination. Do not merge path and title scores or tune thresholds on the held-out results without a pre-outcome calibration plan. An independently reviewed real relationship inventory remains mandatory for confirmatory scoring.

### 2026-10-24 amendment — simulation-based power planning

Added `plan_pitb_power.rs` and `power-planning-report.schema.json` to support pre-outcome power sensitivity analysis. The deterministic simulator takes an explicit component-size vector, baseline success rate, configured treatment gain, minimum observed gain, intracluster correlation (ICC), simulation count, and seed. It models shared family difficulty with a beta-binomial cluster effect, simulates paired control/treatment outcomes, and estimates how often a study would meet the registered gate (observed gain at least the selected minimum and a cluster-level 95% interval lower bound above zero). Its report includes a Wilson interval for Monte Carlo uncertainty, degenerate-interval counts, and the generated marginal gain after probability clipping.

The planner is **not** a benchmark result, does not infer ICC from data, and is not a substitute for the final preregistered analysis or an independent statistical review. The interval is an approximation using a cluster-level t critical value approximation; sensitivity runs must span plausible baseline rates, gains and ICC values. Run it on the actual eligible component-size vector only after the real relationship/quarantine audit is complete. The 20-cluster floor remains a floor, not a power guarantee. CI is configured to check deterministic output, report-schema validity, count reconciliation and rejection of an impossible power estimate, but no runner pass is claimed until an exact-head result is observed.


### 2026-10-24 amendment — paired outcomes and heterogeneous transfer effects

A code audit found that the first power-planner draft used shared family difficulty but conditionally independent control/treatment outcomes within each target and a constant treatment gain across families. That assumption does not adequately represent a paired comparison where both arms solve the same target and the intervention's effect may differ by mathematical family.

The planner now includes two explicit sensitivity parameters in addition to the family-level baseline ICC:

- `cluster_effect_sd`: the probability-scale standard deviation of a normally distributed family-specific treatment-effect deviation. Effects can be negative as well as positive; the two success probabilities are clipped to [0,1], and the report gives the resulting generated marginal gain.
- `pair_association`: a value in [0,1] interpolating the joint-success probability from independence to the Frechet upper bound. This is a coupling parameter, **not** a Pearson correlation estimate. Zero represents independent control/treatment potential outcomes conditional on the family probabilities; one gives the strongest feasible positive joint-success association.

The workflow now validates the regular paired scenario and a second heterogeneous-effect/independent-pairing scenario, along with deterministic repeat output and schema/denominator controls. Unit tests cover the paired Bernoulli marginals, the Frechet upper-bound condition, parser range checks, determinism, and rejection of fewer than 20 components.

These additions make the planning assumptions more explicit; they do not establish which ICC, pair association, or family-effect variation best describes PITB. The actual relationship inventory remains incomplete, the genuine component-size vector is unknown, and no exact-head runner pass is claimed. Before confirmatory use, preregister a grid spanning plausible values for all these parameters, use the frozen eligible component sizes, and obtain independent statistical review of the planning and final analysis methods. The current cluster-t interval remains an approximation; this planner must not be treated as the definitive power-analysis authority.


### 2026-10-24 amendment — minimum components do not imply adequate power

Added an exploratory power-sensitivity note under `docs/research/pitb-power-planning-sensitivity-illustration-2026-10-24.md`. A separate Monte Carlo implementation of the planning equations illustrates why component count, cluster-size distribution, baseline rate, pair outcome association and family treatment-effect heterogeneity must all enter a preregistered sensitivity analysis. Under one synthetic configuration (47 targets across 20 components, baseline 0.35, ICC 0.05, family-effect SD 0.05, pair-association fraction 0.5), a configured +15 percentage-point marginal gain had around 44% probability of passing the combined observed-effect/interval gate. About 50 components under a repeated synthetic size pattern produced around 82% estimated gate power in that one model.

These are not Rust program outputs, not CI results, not empirical PITB parameters and not a selected sample-size recommendation. The note displays Monte Carlo Wilson intervals, explicitly limited to Monte Carlo uncertainty. The real component distribution remains unknown until the cross-family relation audit is complete. Before confirmation, decide a desired power threshold before unsealing test outcomes, run the frozen actual component vector over a preregistered multi-parameter grid, and get an independent statistical review. The conventional 80% target is suggested for consideration, not silently added as a binding PITB gate.


### 2026-10-24 correction — the power planner now uses the paired component-bootstrap interval

The previous amendment described a cluster-level t-interval approximation. That implementation was superseded after a consistency audit: PITB's success-cost frontier analyzer uses a component-clustered percentile bootstrap for the paired treatment-minus-control delta. The power planner now resamples whole components with replacement, recomputes the target-weighted paired delta, and applies the same interpolated 2.5th-percentile lower-bound rule. Bootstrap replicate count is an explicit input (minimum 1,000), is embedded in the report, and is schema-validated. A degenerate interval remains fail-closed and cannot pass the planning gate.

The sensitivity note's numerical tables have been recalculated using a separate NumPy implementation of the paired component-bootstrap estimator (10,000 simulated studies, 1,000 bootstrap replicates per study). Those estimates are still exploratory, not Rust binary output, not CI evidence, and not an empirical PITB result. The underlying statistical model remains an unvalidated sensitivity model, and its parameters are not estimated from the held-out data.

This alignment reduces estimator mismatch but does not make the power model exact: generative assumptions, cluster-size inputs, bootstrap Monte Carlo error, and the independent statistical review still matter. Use the actual frozen eligible component sizes and a preregistered sensitivity grid before a confirmatory experiment.

### 2026-10-28 amendment — degenerate primary intervals cannot establish utility

A code audit found that `analyze_pitb_results` computed `passed` from `lower > 0`, an observed effect of at least 0.10, and at least 20 family clusters, but did not require a nonzero bootstrap interval width. A synthetic case where every family has the same positive delta yields a degenerate interval `[a, a]`; the old predicate could therefore pass it despite the protocol's explicit prohibition on zero-width intervals counting as a win.

The primary gate now records `confidence_interval_informative = (upper - lower > f64::EPSILON)` separately from the literal `confidence_interval_excludes_zero_positive = (lower > 0)`. The gate requires both facts, the +10-percentage-point effect threshold, and the minimum number of eligible family clusters. A degenerate interval returns `no_claim_degenerate_bootstrap_interval`; insufficient independent clusters retains the underpowered disposition. A regression test covers the positive-degenerate case and controls for an informative positive interval, an effect below threshold, and too few clusters.

This is an unverified implementation change until Rust tests and exact-head CI execute. The new tests were statically inspected but not compiled in the authoring environment. It does not constitute a transfer result or SOTA claim.
#### Power-planner parity follow-up

The embedded `analyze_pitb_results --mode power` simulator now uses the same `confirmatory_utility_gate` predicate as the final analysis. It obtains both bootstrap endpoints, counts zero-width intervals in each scenario's `degenerate_interval_trials`, and never counts those trials as gate passes. Analysis mode independently replays the fixed-seed simulation and checks the per-scenario degenerate count as well as power, uncertainty, and mean-effect summaries. This closes the previous mismatch where the power simulator used only the lower endpoint.

A regression test sends 20 identical positive family deltas through the actual deterministic bootstrap, verifies the returned interval is `[1.0, 1.0]`, and verifies the primary gate rejects it with `no_claim_degenerate_bootstrap_interval`. A companion simulation-helper test checks both endpoints and reuses the same gate decision. **These tests have not been compiled or executed in this environment; this section documents intended source behavior, pending exact-head Actions evidence.**

### 2026-10-28 amendment — weak-null family-effect heterogeneity enters the power gate

The prior power planner tested the strict null by setting every family effect to zero. That is insufficient for a target-weighted average-effect estimand: individual families can benefit or regress while the average true effect is zero. The weak-null amendment therefore adds a separate **36-cell heterogeneity grid**. A further paired-association consistency audit has versioned the primary report as `pitb.power-report.v2` and expanded the strict-null/positive-effect grid to 48 cells: four true effects × four ICCs × three paired-outcome association levels.

- Heterogeneity SDs requested: **5, 10 and 20 percentage points**.
- Baseline family ICCs: **0, 0.05, 0.10 and 0.20**.
- Paired-outcome association fractions: **0, 0.5 and 0.9**. Here 0 is conditional independence and 1 is the Frechet upper bound; the fraction interpolates joint-success probability and is not a Pearson correlation.
- For every weak-null simulated study, draw baseline family rates from the beta-binomial difficulty model; draw family-effect offsets; subtract their target-count-weighted mean so the finite-design target-weighted mean effect is zero; and apply one common scale factor only as necessary to keep all treatment probabilities in [0,1]. The report records the mean realized family-effect SD because boundary feasibility can attenuate the requested SD. Sample paired binary potential outcomes from a bivariate Bernoulli coupling that preserves both arm marginals at each association-fraction cell. The strict-null and positive-effect power grid now uses the same coupling at association fractions 0, 0.5 and 0.9 instead of assuming conditional independence only.
- Compute each simulated study's paired family-bootstrap interval and apply the exact same `confirmatory_utility_gate` as the primary analysis. Zero-width intervals remain non-passing and count toward `degenerate_interval_trials`.
- Emit the estimated false-claim rate, ordinary per-cell Wilson bounds, a simultaneous adjusted upper bound, the mean observed effect, the realized heterogeneity diagnostic, and raw false-claim/degeneracy counts. The report verifier must rerun every registered weak-null scenario from validation-derived family sizes and the frozen seed and compare all summary statistics and counts.
- Confirmatory scoring is blocked unless both the strict-null grid and the 36-cell weak-null heterogeneity/association grid meet the registered 5% simultaneous upper-bound screen, along with the +20pp/ICC 0.20 power gate and the independent-family floor.

**Multiplicity correction:** across the 12 strict-null cells (four ICCs × three paired-outcome association levels) and 36 weak-null cells, the scorer computes a one-sided Wilson upper bound using z = 3.08, conservatively above the one-sided Bonferroni critical value for alpha = 0.05/48. The power/no-go gate uses the maximum of these adjusted upper bounds and requires it to be at most 5%; the ordinary per-cell 95% Wilson bounds remain descriptive. The +20pp/ICC 0.20 power criterion uses the minimum power estimate and minimum Wilson lower bound across all three registered association levels. Regression tests cover the strict-null association grid and paired Bernoulli marginal/joint probabilities.

**Remaining limitation:** both the strict-null/power grid and Gaussian centered heterogeneous-effect weak-null grid now vary pair association, but neither currently includes selective missingness, outcome/family-dependent attrition, or alternative family-size-generating mechanisms. The pair-association fractions interpolate from conditional independence to the Frechet upper bound and are not empirical Pearson correlations. This is a model-conditional simultaneous screen, not a universal Type I error guarantee. The next step remains independent statistical review plus explicit missingness and alternative family-size stress scenarios.

Source regression tests cover (i) exact target-weighted zero-mean centering under unequal family sizes, (ii) treatment-probability feasibility and realized SD bounds, (iii) deterministic replay of weak-null and strict/power reports, (iv) the conservative 48-cell simultaneous Wilson bound, (v) strict-null pairing-grid coverage, and (vi) the rule that degenerate intervals cannot pass. **These Rust tests are authored but uncompiled in the current work session; this source update is not a verified runner result.**