# Proof Insight Transfer: research frontier and falsifiable path beyond current benchmarks

**Date:** 2026-10-23  
**Status:** research and evaluation plan; no PITB model-performance result is claimed.  
**Scope:** Symthaea's Proof Insight Transfer Benchmark (PITB), especially typed insight reuse under proof-state interventions.

## Executive conclusion

The strategic opportunity is not to claim a new theorem prover SOTA from synthetic fixtures. It is to evaluate a different capability: **when should a system reuse an old proof insight, when should it adapt the move, and when must it refuse or replan because a supporting condition no longer holds?** This combines verified usefulness with counterfactual safety and robustness to representation changes.

Current PITB assets establish useful *contracts* and a small Lean calibration. They do not yet establish a replay dataset, independently adjudicated transfer labels, model outcomes, or measured improvement. Keep that distinction visible in code, CI, and reporting.

## What current systems demonstrate—and what their headline metrics do not mean

These numbers are intentionally not on a shared leaderboard. They use different tasks, models, budgets, and metrics.

| System or evaluation | Reported result | Why it matters for PITB | Comparability limit |
|---|---|---|---|
| Pythagoras-Prover (arXiv:2606.12594; June 2026) | The authors report 89.8% MiniF2F-Test Pass@32 and 93.0% Pass@2048 for 32B, 93/672 PutnamBench problems solved, and 86.1% MiniF2F-Test Pass@32 for 4B | Current open-prover comparator; ALF structured mutation and the separate MiniF2F-ALF challenge are relevant to contamination-sensitive testing | Pass@2048 is a much larger sampling budget than Goedel's reported Pass@32/184 results and is not a PITB transfer score. Treat the numbers as paper-reported until independently reproduced. [Paper](https://arxiv.org/abs/2606.12594), [project/released challenge](https://github.com/Pythagoras-LM/Pythagoras-Prover) |
| Goedel-Prover-V2 (ICLR 2026) | 32B model: 88.1% MiniF2F Pass@32 in standard mode and 90.4% in self-correction mode; 8B model: 84.6% Pass@32 | Strong reference for end-to-end proof search, self-correction, and test-time exploration | MiniF2F Pass@32 is not transfer-selectivity, same-goal context intervention, or Pass@1 at a matched budget. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) |
| SorryDB, January 2026 snapshot | The displayed leaderboard reports 57.2% Pass@1 for Aleph Prover on a fixed 1,000-task slice | Real open obligations are an important engineering benchmark, not just curated olympiad tasks | Snapshot, task eligibility, model access, and benchmark setup differ from PITB; refresh before any external leaderboard claim. [Leaderboard](https://sorrydb.org/) |
| LeanSearch v2 (May 2026 preprint) | nDCG@10 0.62 vs 0.53 next-best in its standard retrieval evaluation; 46.1% macro premise-group recall@10 on 69 research-level theorems; downstream proof success 20% vs 16% for the next-best retriever and 4% without retrieval in the reported setting | Establishes a strong retrieval baseline and highlights that multi-premise/global retrieval is different from finding a vaguely similar theorem | Retrieval quality and downstream solve rate are separate tasks. PITB must pin the adapter, corpus/index digest, candidate budget, and downstream prover loop. [Paper and open implementation](https://arxiv.org/abs/2605.13137) |
| T² (ACL 2026 Industry Track) | On 2,206 Lean problems with about 41 successor theorems per problem on average, the strongest reported setup achieved 38.9% Testing Accuracy | A compiled theorem can still be semantically wrong; dependent-successor behavior is a scalable additional signal | Successor testing is not a universal semantic oracle and is not equivalent to PITB's source-insight compatibility judgment. [Paper](https://aclanthology.org/2026.acl-industry.150/), [implementation](https://github.com/ldilab/T2) |
| FormalML (ICLR 2026) | 4,937 Lean subgoal-completion problems in machine-learning theory, including complex research-level contexts and premise-retrieval demands | A practical assistant must work inside an existing proof, not only solve isolated olympiad theorems | Its task setup and model scores should be reproduced from the paper before numerical comparison. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d4e6e1a0557d3d9e3140966c1db533-Abstract-Conference.html) |
| Discover and Prove / Hard Mode (ACL 2026) | Reports 36 PutnamBench theorems proved in Hard Mode and 10 CombiBench solutions versus 7 for the prior cited result | Separating answer discovery from proof construction addresses a weakness in benchmarks where the statement exposes the answer | This is answer discovery plus formal proof, not same-goal insight transfer. [Paper](https://aclanthology.org/2026.acl-long.3/) |
| ProofOptimizer (ICLR 2026) | Reports proof-length reductions of 87% on MiniF2F, 57% on PutnamBench, and 50% on the cited IMO 2025 proofs | Verified proofs should also be maintainable, comprehensible, and reusable | Shorter proof text is not itself evidence of greater transfer correctness or human usefulness. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60917970c4257e50aff60d1d7bde2edf-Abstract-Conference.html) |
| TheoremBench (June 2026 preprint) | Nearly 100 classical theorems are available in plain and premised variants; the paper adds theorem-level coverage and token-efficiency metrics and finds explicit premises substantially improve performance | Explicitly measure insight/premise contribution, theorem-level coverage and token efficiency rather than one solve-rate average | The premise intervention is not PITB's local-context counterfactual; do not compare headline scores directly. [Paper](https://arxiv.org/abs/2606.09450) |
| VeriSoftBench (2026) | 500 tasks from 23 real-world Lean verification repositories; best reported system solves 41% with curated context versus 35% with the raw repository | Repository-aware context selection and multi-hop dependency depth matter in engineering proofs | Software-verification completion is not an insight-transfer label; use it as an external deployment-oriented benchmark. [Paper](https://arxiv.org/abs/2602.18307), [benchmark](https://utopia-group.github.io/VeriSoftBench/) |
| FVSpec (2026, arXiv v4) | 9,415 Lean challenges were produced from 2,623 of 7,413 distinct deduplicated property-based tests drawn from 497 repositories; evaluated models averaged 70% success on easy and 49% on hard tasks | A future external challenge set can include specifications derived from everyday engineering tests, not only theorem libraries | The paper explicitly notes that a Lean kernel pass does not ensure faithfulness to the original property-based test; pin dataset version because the benchmark website and current paper describe different corpus counts. [Paper v4](https://arxiv.org/html/2606.01008), [benchmark](https://fvspec.galois.com/) |

A second baseline-quality warning is the 2026 study *Faults in Our Formal Benchmarking*, which reports 4,833 findings across 10,318 problems, including mechanically certified issues. PITB therefore needs a statement-audit and counterexample-oriented mutation track in addition to kernel checking. [Paper and artifacts](https://proceedings.mlr.press/v306/ammanamanchi26a.html).

## Paradigm shifts to prioritize

### 1. Treat an insight as a typed, falsifiable contract—not a paragraph to retrieve

An insight record should identify the obstacle, proposed move, required local declarations and type-level preconditions, state transition, evidence, expected benefit, known failure boundary, and replan/abstain policy. A candidate insight is not applicable merely because its theorem or prose looks similar. Separate three states: *eligible to try*, *formally verified outcome*, and *causally useful compared with a baseline*.

### 2. Evaluate interventions on a fixed theorem and exact goal

Hold the declaration, target statement, Lean environment, and elaborated goal type fixed; change one local-context fact at a time. Include:
- semantics-preserving changes: alpha-renaming, irrelevant declaration removal, and reordering independent declarations;
- precondition withdrawal: remove a declaration the strategy actually requires;
- redundant-route controls: withdraw one hypothesis while preserving an independent route to the same goal;
- adversarial near misses: same surface symbols but changed type, direction, quantifier scope, or required premise.

The expected behavior differs by intervention: preserve the decision for validated equivalent states, adapt when a redundant route exists, and reject or replan when a required precondition disappears. This makes *failure to notice a changed proof state* measurable instead of hidden inside aggregate solve rate.

### 3. Make the evidence path end-to-end and content-addressed

Bind source commit, exact declaration and statement, environment/toolchain, local-context and goal digests, intervention operation, raw prover trace, checker/audit receipts, outcome rows, and final report. A digest proves identity, not truth; a schema proves structure, not mathematics; Lean kernel checking proves the elaborated proposition, not that it faithfully represents the intended informal claim. Keep each claim separate.

### 4. Use paired causal ablations and a multi-budget success–cost frontier

Run the same frozen targets against: no retrieval, BM25/lexical, syntax/AST retrieval, dependency graph, raw proof retrieval, strong global-premise retrieval (prefer pinned LeanSearch v2 if task coverage permits), typed insight alone, and typed insight plus source proof context. Select configuration on validation components only. Report test results once the plan is frozen.

At minimum measure exact-target checked solve rate, Pass@1, success by shared budget cap, false-accept rate on incompatible transfers, recall on compatible transfers, abstention quality, equivalence decision stability, success-versus-cost frontier, tokens/calls/wall time/monetary cost, Lean checking time, and proof size. Report raw numerator/denominator and unknown/excluded/not-run rows alongside every percentage.

### 5. Make contamination and benchmark validity explicit

Public Lean repositories may have appeared in model pretraining; a repository-family split cannot prove training-data independence. Keep a temporal or post-cutoff external challenge set, identify exact snapshots, group all companion/alternative proofs into connected components, and avoid tuning on test outcomes. Independently review statement correspondence; retain unresolved cases as unknown, not as easy negatives or silently removed rows.

### 6. Evolve the benchmark adversarially—but keep the evaluator independent

A promising next frontier is a generator–prover–auditor loop rather than a static bag of counterfactuals. GAR (ICLR 2026) jointly trains a task composer and prover in an adversarial curriculum; the paper reports average relative Pass@32 improvement of 4.20% across two evaluated prover configurations on MiniF2F, and a DeepSeek-Prover-V2 ProofNet-Test change from 22.58% to 25.81%. This is evidence that adaptive task generation is worth evaluating, not evidence that an equivalent PITB mechanism already exists. [GAR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html).

For PITB, the generator should propose *minimal proof-state interventions near an insight's applicability boundary*: remove a premise, change a type or inequality direction, alter quantifier scope, reorder independent declarations, or remove one of several redundant routes. A separate checker/reviewer must classify the mutation and establish whether the target statement stayed fixed and whether the intervention preserves or breaks the precondition. Do not let the same generator that creates a mutation certify its label.

Track a development-only **boundary-finding yield** (independently qualified new failure modes per 100 attempted mutations), distinct from model accuracy. Deduplicate by mutation family, audit false mutation labels, then freeze an independently held-out external challenge set. Otherwise, a powerful generator can simply overfit the evaluator to recognizable adversarial patterns.

The key architectural separation is:

- **Generator:** proposes context changes or nearby counterexamples, with source and seed receipts.
- **Independent semantic auditor:** determines which target and preconditions changed, and leaves uncertain cases unlabelled.
- **Candidate system:** predicts reuse, adaptation, replan, or abstention and attempts the proof.
- **Kernel / exact-target verifier:** checks the resulting Lean proof and any declared axiom policy.
- **Scorekeeper:** joins hashes, reports the paired outcomes, and cannot modify eligibility or labels after seeing predictions.

The Pythagoras-Prover work is an important adjacent result: its authors report a structured Augmented Lean Formalisation (ALF) pipeline and release MiniF2F-ALF, a mutation-sensitive challenge on which all evaluated models lose accuracy ([paper](https://arxiv.org/abs/2606.12594), [released code and challenge](https://github.com/Pythagoras-LM/Pythagoras-Prover)). PITB should borrow the idea of systematically varying formal statements and contexts, but not a weak label-generation assumption: on confirmatory PITB rows, each mutation's semantic relation and insight-contract effect must be independently adjudicated; invalid or ambiguous mutants do not enter binary accuracy denominators, but remain fully accounted for. An augmentation filter may be adequate to create training signal; it is not sufficient evidence that a counterfactual benchmark label is true.

This structure could make the challenge set grow where systems fail while keeping evaluation claims fail-closed. It is a proposed architecture; no generator loop or boundary-finding score is implemented or measured by this PR yet.

## A staged scoreboard

These are **proposed gates**, not measured PITB achievements or universal definitions of intelligence.

### Gate A — harness integrity

- Every trace is bound to the exact triggering commit and input artifacts.
- All expected negative controls are rejected; stale/mismatched digests, duplicate IDs, cross-split family leakage, missing checker receipts, and unknown labels fail closed.
- The Lean pilot compiles and passes its independent checking/axiom policy; its structured trace preserves the target goal and proves the intended local declaration is absent after intervention.
- No model score is published from synthetic fixtures.

### Gate B — real pilot, descriptive only

Build a small, expert-reviewed pilot with real proof states and actual model outcomes. Include independent reviewers, paired target IDs, recorded run budgets, and every failed/not-attempted/unknown row. Run at least three arms before interpreting treatment effects: no retrieval, strongest pinned retrieval baseline, and typed insight. Publish an error taxonomy and examples of false accept, false reject, correct abstention, and correct replan.

### Gate C — confirmatory evaluation

Freeze the target set, protocol, strongest baseline, analysis code and random seeds before scoring test outcomes. Use cluster-aware intervals at the connected-component/family level. Determine sample size from a prospective power analysis; the existing target of 20 independent components per class is a minimum exploratory floor, not a guarantee of adequate power.

Suggested initial success criteria to preregister and challenge:
- exact-target checked solve rate improves by at least **10 percentage points** over the validation-selected strongest baseline, with the component-clustered 95% interval for the paired difference above zero;
- at least **90%** recall for adjudicated compatible transfer and at least **90%** sensitivity to reviewed critical-precondition-breaking cases;
- at least **95%** decision stability on adjudicated semantics-preserving interventions;
- **100% rejection of the finite, pre-declared structural and semantic negative-control suite** as a harness integrity gate;
- publish the cost frontier; do not hide a large compute premium behind a solve-rate point estimate.

These thresholds may be revised before data collection if power, reviewer agreement, or task eligibility makes them unsuitable. After the test is opened, changes must be labelled exploratory and require a new held-out evaluation for confirmation.

## Implementation progress: paired proof-state outcome contract

PITB now adds `proof-state-intervention-outcome.schema.json`, `proof-state-intervention-analysis-report.schema.json`, and `fixtures/proof-state-intervention-outcomes.synthetic.json`, with a Rust scorekeeper at `crates/bridges/symthaea-proof-insights/src/bin/analyze_proof_state_intervention.rs`.

A review of the paired outcome semantics exposed an important issue: the expected decision had been copied from the pair-level contract to both states. That incorrectly marked the **base** state ineligible even though the required hypothesis was present. The outcome contract now separates the expected applicability judgment from the action policy per state: the base is `eligible / preserve_eligibility`, while the variant after required-hypothesis removal is `not_applicable / move_forbidden_or_replan`. Semantics-preserving variants remain eligible. The workflow and Rust analyzer independently derive these expected labels from intervention kind and state role rather than trusting a row's claim.

An exact-target evidence gap is also closed in the contract: every executed row must include `checked_target_statement_sha256`, and the Rust analyzer rejects any checker receipt that names a target digest different from the paired suite statement. A proof digest without the exact target identity is not enough to count as verified target success. The generated report also records the analyzer source SHA-256, which CI cross-checks against the checked-in source.

The Rust analyzer joins each outcome to the exact source insight, pair, state snapshot, statement, goal/environment and local-context digests. It reports assigned-row exact-target solve rate, decision coverage, applicability accuracy, eligible recall, inapplicability sensitivity, false-eligibility and unsafe-reuse rates, forbidden-move compliance, semantics-preserving stability, required-precondition withdrawal accuracy, and resource totals. Proportions use deterministic component-clustered bootstrap intervals. It now reports the point estimate and independent-component count but suppresses the interval below 20 independent components; this is a descriptive reporting floor, not a substitute for prospective power analysis. Each requested split is restricted to a single model/configuration identity per comparison arm; unrun cases remain visible in the assigned denominator and are counted separately. Synthetic data is never scored: the checked-in eight rows are all `not_run`, the analyzer emits no model metrics, and the confirmatory decision remains unassessed.

New Rust regression tests cover the fixture no-score path, missing paired state rows, incorrect role-aware applicability labels, synthetic execution fraud, mismatched checker-target digest, deterministic bootstrap, and the one-component uncertainty guard. The dedicated workflow now runs these tests, executes the analyzer against the four-pair/eight-row synthetic suite, validates the output schema, and confirms that no synthetic metrics were emitted. These are authored checks, **not yet reported test results**: the environment has no local Rust toolchain, and the exact-head workflow has not reported a pass. The earlier JavaScript simulation only validates cross-field joins; it is not a Rust compile or an independent Draft 2020-12 validator result.

## What “beat SOTA” should mean here

First reproduce a relevant baseline on the same eligible tasks, checker/toolchain, context budget, attempt budget, and model availability. Then show incremental value from the insight representation against the strongest eligible retrieval baseline, with paired uncertainty intervals, safety slices, and cost reporting. A result on PITB's task is **a PITB result** until independently replicated; it does not imply that Symthaea beats Goedel-Prover-V2 on MiniF2F, LeanSearch v2 on premise retrieval, or a SorryDB leaderboard entry.

The most defensible breakthrough claim would be: *a typed proof insight improves verified downstream completion at a matched budget while maintaining calibrated refusal on counterfactual states and stability under semantics-preserving changes.* That is narrower than “general reasoning solved,” but also much more scientifically testable and operationally useful.

## Immediate next steps

1. Strictly schema-validate the Lean pilot receipt, including commit identity, input-schema digest, fixed-goal/context trace, and model-outcome not_run status. This is a harness improvement, not a theorem-prover result.
2. Continue the analysis independently of CI, but never treat queued jobs as evidence. Build a reproducible local/CI-independent check for the receipt format and run the checked-in synthetic negative controls where tools permit.
3. Extend the real Lean calibration to more than one precondition pattern and independently reachable alternative route before recruiting models.
4. Populate a reviewed pilot with real state pairs and actual runs only after the evidence and split contracts are stable.


## 2026-10-23 update: external benchmark lessons and executable cost frontier

This update adds recent published evidence and an executable descriptive evaluator; neither changes the primary PITB endpoint nor represents measured PITB results.

### Fresh external comparators and what they change

**First Proof Batch 3 (published 14 October 2026)** is a useful stress test for claims about “research-level mathematical reasoning”. The project assessed five systems on ten unpublished research-mathematics problems with 24 hours per system and no human input, using around 30 expert reviewers and separate scores for correctness, exposition, and attribution. Its release reports 9/10 problems answered correctly by the ProofCouncil multi-model harness versus 7/10 for the direct GPT-6 Astra submission; every problem was solved by at least one system. The release also reports $1,888 total cost for ProofCouncil versus $36 for the direct Astra run, adequate citations on 7/10 ProofCouncil submissions versus 4/10 direct Astra submissions, and negative average exposition scores for about one third of ProofCouncil responses. The open-weight Kimi K3 harness's submissions all contained major errors in this setup. See the [First Proof official release](https://1stproof.org/third-batch-press-release.html) and [published benchmark material](https://1stproof.org/third-batch.html).

That is not a head-to-head result against PITB: it is expert-graded informal research mathematics, with different targets and scoring from Lean kernel checking and typed insight-transfer interventions. It does, however, reinforce a design decision: **never collapse mathematical correctness, evidence attribution, exposition, abstention, and cost into one vague reasoning score.** A future PITB research-assistant track could evaluate citation/source fidelity, explanation quality, uncertainty-aware abstention, and expert judgment separately after the formal-transfer track is operational.

**Cost-aware proof orchestration is now a first-class SOTA comparison axis.** The October 2026 CoCo-Prover preprint describes action selection over proof and lemma-dependency graphs and reports a better success/cost frontier, with up to 30.9% lower cost than its strongest baseline in its own five-benchmark Lean program-verification setup ([paper](https://arxiv.org/abs/2610.09681)). The October 19 Maestro preprint focuses on global failure diagnosis, correcting false lemmas, retracting flawed subtrees, and replanning; it reports 669/672 PutnamBench and 59/60 Lean-IMO-Bench solves under its own setup ([paper](https://arxiv.org/abs/2610.25920)). These claims suggest strong features and baselines to reproduce, not results we can transfer numerically to PITB. In particular, cost gains must be compared on the same fixed target set, checker, and resource caps; a different benchmark's percentage is not our baseline.

### Engineering increment now committed to PITB

The branch now contains:
- `crates/bridges/symthaea-proof-insights/src/bin/analyze_success_cost_frontier.rs`: a descriptive evaluator for the preregistered target × method × budget matrix. It validates full cell coverage and joins results against run/target/component/arm/budget/input identity; checks frozen model/checker/config identities and actual resource caps; binds an executed check to the exact planned theorem-statement digest; computes success curves with the assigned-target denominator, same-budget paired differences with component-clustered bootstrap intervals, actual cost per assigned target / verified solve, and the observed non-dominated success-cost frontier.
- `success-cost-frontier-outcome.schema.json` and `success-cost-frontier-analysis-report.schema.json`: strict Draft 2020-12 contracts for run rows and aggregate output.
- `fixtures/success-cost-frontier-outcomes.synthetic.jsonl`: all 18 planned cells explicitly `not_attempted`, with zero usage and null evidence.
- An added workflow stage that runs the analyzer in the isolated research crate, validates both schemas with pinned `jsonschema[format]==4.26.0`, checks report/source/plan/outcome digests, and mutates one row to try to pass a forged execution in a synthetic fixture.

The new analyzer deliberately emits no curves, paired effects, or frontier points for a synthetic dataset and permanently reports an unassessed confirmatory decision. It does not inspect receipt contents or determine theorem truth; the primary `analyze_pitb_results` power and utility gate remains authoritative. The frontier report can diagnose cost efficiency but cannot make a primary utility/SOTA claim.

### Remaining integrity work before performance experiments

1. Execute the exact-head Rust/Clippy/workflow checks; no latest-head Rust pass is asserted in this note.
2. Run independent JSON Schema validation through the newly added workflow step; this session's local structural checks were JavaScript parsing/joins, not an independent Draft 2020-12 engine.
3. Create and independently audit a small set of real target bundles, theorem correspondence and successor receipts; preserve unresolved items as unknown.
4. Produce actual outcomes under the frozen baseline, matched configurations and at least three preregistered caps. Include every assignment, failure, timeout, tool-unavailable and non-attempt.
5. Report (a) primary exact-target improvement with preregistered power/cluster uncertainty, (b) false acceptance and compatible recall on counterfactual transfer, and (c) cost frontier separately. Do not transfer First Proof / CoCo-Prover / Maestro headline metrics to PITB or describe the proposed +10pp solve-rate / aspirational 25% cost gates as achieved.



### 2026-10-24 update: suppress low-cluster bootstrap intervals

A consistency audit found that the proof-state analyzer could emit percentile-bootstrap intervals with only two independent components, even though the evaluation protocol calls for at least 20 independent pair-graph components per primary class and a prospective power analysis. The analyzer now keeps a descriptive point estimate and exposes `independent_components`, but suppresses the interval below 20 clusters with an explicit status. At or above the floor, intervals remain descriptive and do not independently establish adequate power or validate reviewer labels.

This is a conservative reporting guardrail only. No experiment was run, no model result changed, and current-head compilation/CI remains unverified until an exact-head run completes.


### 2026-10-24 update: enforce paired resource budgets

The proof-state outcome contract now requires declared caps for model calls, input/output tokens, wall time and micro-USD cost on every executed row. The Rust analyzer rejects any reported usage above a cap and rejects paired base/variant rows that use different cap vectors. Synthetic `not_run` rows must carry a null cap field; they do not fabricate an execution budget. This closes a loophole where a paired intervention could be described as matched while one side had a larger declared resource ceiling.

The cap checks validate declared metadata and reported counters; they do not independently attest the underlying provider logs or receipt content. The primary experiment still needs matched receipts, exact model/prompt/checker configuration, and prospective power analysis. Rust execution and independent Draft 2020-12 validation remain unclaimed pending the exact-head workflow.


### 2026-10-24 SOTA refresh: add Pythagoras-Prover and ALF

The comparator table now includes Pythagoras-Prover. Its paper reports 89.8% on MiniF2F-Test at Pass@32 and 93.0% at Pass@2048 for its 32B model, plus 93 solved of 672 PutnamBench problems; it also reports 86.1% MiniF2F-Test Pass@32 for its 4B model, compared with 82.4% for DeepSeek-Prover-V2-671B in the stated setup. These are different attempt budgets and not PITB outcomes. The project also releases MiniF2F-ALF, a mutation-sensitive test split, while using ALF variants as training augmentation. PITB's distinct contribution must be verified, independently labelled transfer sensitivity: no model performance claim follows from having a mutation generator or schema alone.

Strategic consequence: keep three scoreboards separate—(1) end-to-end prover Pass@N on fixed external tasks, (2) transfer-contract accuracy under independently adjudicated proof-state interventions, and (3) verified solve/cost frontier at fixed multi-budget caps. To demonstrate value beyond current systems, PITB should show incremental transfer safety/usefulness against strong retrieval and coordinated-proof baselines, rather than trying to compare its small intervention suite against headline MiniF2F percentages.


### 2026-10-24 update: validate the fixed-target identity across both states

A source audit found a contract/enforcement gap: the suite schema described fixed theorem, goal, environment, toolchain and source identity, but the Rust analyzer had only joined each outcome to the suite's declared state fields. It did not require the target descriptor and both state descriptors to agree on all those invariant fields. The analyzer now checks source commit, statement digest, goal-type digest, environment digest, toolchain and goal count before accepting the pair. Regression controls mutate those fields and require fail-closed rejection.

This protects against malformed paired tasks and label contamination where the intervention silently changes more than local context. It does not independently attest the replay receipts or the truthfulness of the state collector, and it is not a proof-transfer performance result.


### 2026-10-24 update: validate counterfactual intervention semantics

The paired-state analyzer now cross-checks the intervention kind against the declared operation, before/after declaration identity, presence/type/position fields, and expected precondition/action policy. This prevents a row from being labelled "required hypothesis cleared" when the operation or declaration receipt says otherwise. The synthetic contract fixture was corrected so its required-type digest matches the actual declared required-precondition type. Regression tests include operation mismatch, type mismatch, a falsely retained cleared declaration, and an end-to-end analyzer rejection.

This is a consistency guard only: it cannot establish that the intervention was actually performed by Lean, that a precondition is mathematically necessary, or that a system successfully transferred an insight.


### 2026-10-24 update: semantic-gate-qualified primary success

The proof-state analyzer now separates raw checker success from the primary, semantically qualified exact-target solve rate. Executed rows require a target-bound semantic-gate record covering dual statement-correspondence review, axiom-policy compliance, dependent-successor evidence or reasoned not-applicable status, and mutation controls. The analyzer derives overall gate state from these sub-results rather than trusting a self-reported `passed` label. The report now counts passed, failed, and unknown gates.

This design is informed by the T² study's gap between compilation and dependent-successor testing, and by the 2026 formal-benchmark audit. It is only an evaluator contract until real receipts and model outcomes exist; it does not establish semantic correctness or improve a prover score by itself.


### 2026-10-24 update: individual review decisions are part of the gate

The semantic-gate contract now records each statement-correspondence review separately: reviewer ID, decision, and evidence digest. The Rust analyzer rejects duplicate identities, malformed digests, and disagreement hidden behind a claimed consensus status. Disagreement is unresolved unless a separately preregistered adjudication procedure is implemented. This is stricter than trusting an integer such as `independent_reviewer_count: 2`, which alone does not describe who reviewed the statement or whether their decisions agree.
