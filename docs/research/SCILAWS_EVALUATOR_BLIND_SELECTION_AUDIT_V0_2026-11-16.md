# SciLaws evaluator-blind candidate-selection audit v0

**Date:** 2026-11-16  
**Parent:** [#11201 — SciLaws-Bench adapter](https://github.com/Luminous-Dynamics/symthaea/issues/11201)  
**Work item:** [#12489 — evaluator-blind candidate selection](https://github.com/Luminous-Dynamics/symthaea/issues/12489)  
**Implementation:** [PR #12492](https://github.com/Luminous-Dynamics/symthaea/pull/12492)  
**Status:** draft implementation; isolated exact-source harness is being qualified

## 1. Research problem

Scientific model generation and scientific model selection are separate capabilities.

A research system may generate a candidate that is better than the model it eventually submits. Conversely, a candidate with strong in-sample or held-out numerical fit may violate scientific constraints or fail to recover the underlying structure. A benchmark that scores only the selected candidate can hide whether the bottleneck is generation, selection, or both.

The SciLaws-Bench project page checked on 2026-11-16 reports 14 model configurations under a fixed ReAct/Python harness. Its current GPT-6 aggregate is:
- Real numerical fit S_N: 51.33;
- Real scientific validity S_V: 82.91;
- Parallel hidden-structure recovery S_S: 72.46.

These scores are deliberately kept separate. The project reports 53.6% fit–validity concordance across unequal-fit candidate pairs and a GPT-6 task-level fit/structure correlation of r=0.06. It also reports a best-generated vs submitted-candidate gap of 0.03–0.14 S_N for several configurations, an oracle best-of-N gain of +4.5 S_N and +5.3 S_V when increasing candidate pools from five to twenty, and evaluator-Pareto domination in 60% of selected winners in the displayed five-candidate groups. These are upstream benchmark observations, not Symthaea results.

Sources:
- https://yiyihum.github.io/SciLaws-Bench/
- https://arxiv.org/abs/2609.01552

## 2. Upstream version, dataset and evaluator custody

Code compatibility audit pin:
- Repository: yiyihum/SciLaws-Bench
- Code commit: 8174c7b5289fd3e414892f20efca322cb1e0e5e9
- README blob: bd9a2cde3473fbf10b464e5645ea33c767cc3032
- dataset/LICENSES.md blob: 4b2e7bbdf1d8b522a811e99ec34c7ae40a08b0e1
- dataset/task_index.csv blob: 0a4894bcee366e5272f1164e7c4f31aae76f88b1
- harness/evaluate_numeric.py blob: 0b860d7e565ae66c5e0e561e7968d4cbb4980518
- harness/evaluate_validity.py blob: 616952f4eec74bcb766507edacfd6a48faf5a9b2
- baseline_agent/all_results.md blob: 3d5010b49d4bfed4d8c9ac753459a8b4331bfbc8
- harness/AGENT_INTERFACE.md blob: cce0d46359900d5e7044e473f41b064c4f12bf0d

Important snapshot boundary:
- The pinned code commit contains an older nine-model baseline table (GPT-5.5 S_N=50.73, S_V=81.84, S_S=58.26).
- The current 14-model website page is a different snapshot. Do not combine its scores with the older file or claim the current page is reproduced from that code commit.
- The current website does not expose the matching task-level result bundle/task-set digest, so exact head-to-head reproduction still requires a pinned dataset revision, scorer, task split and raw records.

The ~1.1 GB dataset is hosted separately on Hugging Face (RealSR/SciLaws-Bench). Per-task licensing varies, including non-commercial, GPL, academic-use and custom terms. The upstream MIT code license does not relicense every dataset row. Do not download or redistribute tasks until the exact dataset revision and task-specific license have been reviewed.

The upstream eval/ directories contain reference anchors and scientific-validity rubrics; Parallel simulator state contains hidden target laws. These remain evaluator-only. The policy receives only permitted public task data and mediated simulator responses. No eval/ answer keys or hidden simulator state may be placed in the policy-visible workspace.

## 3. Architecture and data boundary

The implementation in crates/domains/symthaea-architecture-engineering/src/scilaws_candidate_selection.rs is a post-submission auditor. It does not choose a candidate.

### Live selection packet

The live packet accepts only:
- task manifest digest and development-split digest;
- selector policy ID and policy digest;
- selection trace digest and resource-receipt digest;
- candidate IDs and canonical model-artifact digests;
- development evidence digest, development RMSE, symbolic complexity and declared assumption-check IDs;
- actual submitted candidate ID, or explicit abstention.

It uses strict unknown-field rejection. Evaluator-only S_N, S_V, S_S, held-out scores, rubric/judge outputs, hidden target labels and oracle best-of-N metrics are not accepted.

### Evaluator-only result packet

A separate type holds the post-submission Real evaluation:
- model and evaluation-evidence digests;
- complete / failed / not-evaluable status;
- native S_N and S_V if and only if the attempt is complete;
- submission-contract and anti-hacking gate outcomes.

A complete candidate with either metric missing is rejected. Failed and not-evaluable attempts must carry no placeholder metric values; they remain in the denominator, not imputed to zero.

### Identity and custody

The auditor requires:
- exact campaign and task-manifest identity match;
- exact candidate-ID set match;
- exact candidate model-artifact digest match;
- valid SHA-256 formatting on all bound artifacts;
- evaluator-manifest digest in the final audit.

These digests prove identity binding only. They do not prove who generated the record, when it was generated, or that selection chronologically preceded sealed evaluation. Chronology, signer identity and evidence custody remain with the existing Mycelix/attestation and independent evaluator owners. The selector trace must be externally attested before official benchmark outcomes are made visible.

## 4. Real-track Pareto audit

The first executable slice handles SciLaws-Real only.

It reuses the existing architecture-engineering Pareto objective implementation:
- objective 1: maximize native S_N;
- objective 2: maximize native S_V;
- no arbitrary scalar weight or blended scientific score;
- submission-contract and anti-hacking gates are separate hard feasibility constraints.

The audit reports:
- complete, failed, not-evaluable and infeasible counts;
- all per-candidate attempts and evaluator evidence digests;
- exact Pareto front and a deterministic dominated-by witness;
- submitted candidate disposition: abstained, non-evaluable, infeasible, strictly dominated, or Pareto-nondominated;
- best available S_N and best available S_V among all complete, feasible candidates;
- submitted S_N and submitted S_V;
- separate coordinate-wise regret: best S_N minus submitted S_N, and best S_V minus submitted S_V.

Coordinate-wise regret is not a single utility or oracle "winner". A candidate may trade numerical fit against validity and remain Pareto-nondominated. The audit preserves that distinction.

Duplicate candidate IDs fail closed. Duplicate canonical model-artifact digests are retained and reported as byte-identical artifact groups; a matching digest is not automatically semantic equivalence. Failed attempts and multiple IDs are not silently discarded.

Parallel S_S is not included in this Real two-objective comparison. It requires a separate Parallel task contract and hidden-structure scorer boundary. Combining S_S with S_N/S_V in one unqualified Pareto plane would conflate different task settings and evaluation constructs.

## 5. Benchmark comparison and targets

### SciLaws-Bench

Current external anchor (14-model public page):
- GPT-6: S_N 51.33, S_V 82.91, S_S 72.46.
- The same model may rank differently across fit, validity and hidden-structure recovery.

Proposed first target:
- reproduce one exact task/data/scorer snapshot from raw records;
- reduce evaluator-dominated submissions by at least 50% relative to the frozen Symthaea selector on a preregistered confirmatory subset;
- no degradation of S_N, S_V or S_S beyond paired, task-cluster uncertainty;
- report per-task metrics and complete attempt denominator.

This is not yet an achieved result. A claim against the website's GPT-6 numbers requires a matching task set, dataset revision, harness, attempt/budget, model/tool access and scorer. The old code-file leaderboard is not a substitute for that missing result bundle.

### ASI-Bench

The current public 26-configuration snapshot (updated 2026-11-10) reports 60 tasks across 11 domains. Twenty-four configurations use three-run means; two are one-run exceptions. The published evaluation used no external tools and the scored task-set identifier is not supplied. Its B1/B2/B3/B4 aggregate means are 55.60 / 33.84 / 31.40 / 31.48. The biggest drop is B1→B2 (21.76 points), consistent with a method-operationalization bottleneck.

Proposed Symthaea target:
- close at least 10 percentage points of the B1→B2 loss versus a matched direct-implementation baseline;
- no B1 regression;
- equal total model calls, tokens, context, data, tools, runtime and human intervention;
- confirm on held-out task families, not only the six-task development subset.

B2 method operationalization is distinct from B3 method selection. Improve them separately.

### Autonomous Science Bench

AS-Bench 0.1 currently has six pilot campaigns in biology, chemistry and materials. Its current public leaderboard lists GPT-6 Astra(max)/Codex at 72.2% resolution ($190.23 / 557 min) and Opus 5.5(max)/Claude Code at 66.7% ($172.91 / 1142 min). A resolution requires passing both the discovery gate (quality of selected measured designs) and learning gate (ranking of unmeasured designs). Its authors caution that small pilot ranking gaps are not statistically meaningful.

Proposed Symthaea target:
- first reproduce all six native campaigns and their two gates, costs, failures and timing;
- then achieve a +5 percentage-point resolution improvement over the best compatible baseline at equal or lower all-in cost, without losing either native gate;
- use paired trials and report task-level uncertainty before superiority claims.

### FrontierPhysics

The current public leaderboard reports 56 expert-authored physics tasks across nine fields. Opus 5.5 leads at 27.4% Pass@1; GPT-6 Astra is 23.2%; GPT-6.1 Sol is 19.6%. A pass requires all automated checks and blocking rubric criteria.

Proposed target:
- exceed 30% Pass@1 under the exact same task release, harness, trials and interventions;
- then target 35% with held-out expert task families and independent blocker adjudication.

This is a stretch target, not a forecast. A result on one physics benchmark does not establish general scientific autonomy.

### Unsatz

Unsatz evaluates structure-first recovery without a supplied ansatz across 80 systems and 20 domains. The paper reports nine models solving 21–60%. Its findings include that agents often identify variables but fail to recover interactions, and prefer training-error reduction over better held-out prediction. Mechanism reuse helped only when the new system shared the mechanism and the source-system discovery had succeeded.

Proposed target:
- exceed the strongest compatible baseline by at least 5 percentage points on an identical task release and budget;
- preserve equivalence classes and abstention on non-identifiable cases;
- report interaction recovery and out-of-support predictions separately from variable detection;
- include false transfer and negative-transfer rates.

## 6. Paradigm shifts to pursue

1. **Generation ≠ selection.** Preserve the entire candidate pool, freeze the selected candidate, then score hidden outcomes independently. Measure where the gap occurs.
2. **Fit ≠ validity ≠ structure.** Keep numerical prediction, source-grounded scientific validity and hidden mechanism recovery as independent claims and endpoints.
3. **Model adequacy is a budgeted obligation.** Use #12375 and #12454 to ask not only which named models disagree, but whether material alternatives were actually detectable under the available actions and budget.
4. **Transfer must be conditional and falsifiable.** Reuse discovered mechanisms only after verifying source success and target applicability; report negative transfer and false-reuse rates.
5. **Methods should become executable contracts.** The #12471 method-operationalization slice should turn B2 methodological guidance into typed task-native steps, declared preconditions, resource accounting and independently checked artifacts.
6. **Benchmark lifecycle must resist saturation.** Preserve immutable anchors and add sealed refresh sets with audited family separation. Never silently rewrite the score distribution.
7. **Knowledge is a qualified object, not a confidence number.** Every claim should carry model identity, scope, assumptions, evidence, contradiction state, independent verification and replication lineage.

## 7. Industry strategy

OpenAI, Anthropic and other frontier developers have a structural advantage in foundation-model scale, compute and specialized tool access. Luminous should not try to reproduce their entire infrastructure or claim superiority from a handful of synthetic experiments.

The competitive opportunity is the full evidence-qualified scientific loop:
- Symthaea generates and maintains competing mechanisms.
- Symtropy tests predictions and numerical/physical adequacy independently.
- Mycelix records provenance and replication without making signatures a truth oracle.
- Sol-Atlas applies only models whose validity envelope matches the engineering case.

External foundation models can contribute candidate generation and method proposals. They are not the final scientific authority. The most defensible differentiation is a system that can show not only what it believes, but why it is identifiable, what could falsify it, what was not tested, and whether the claim survives independent reproduction.

## 8. Verification status and next work

The full `symthaea-architecture-engineering` package test command on the stacked branch currently fails in unrelated modules (including event_tape, synthetic_search, analytical_benchmark and direction_board), before isolating the SciLaws module. The branch also has a Cargo lock-resolution mismatch when this package test is run with `--locked`. Those are repository/base-branch issues, not evidence that this module passes or fails.

To avoid changing unrelated modules or weakening workspace gates, PR #12492 now has an isolated harness at `research/scilaws-candidate-selection-audit-harness`. Its `src/lib.rs` imports the exact two repository sources by path:
- `crates/domains/symthaea-architecture-engineering/src/pareto_objectives.rs`;
- `crates/domains/symthaea-architecture-engineering/src/scilaws_candidate_selection.rs`.

This does not copy or fork the implementations. The harness has its own pinned `Cargo.lock`, Rust 1.96 format/test/Clippy gates, and is deliberately narrower than a whole-package/workspace claim. Its exact-head workflow has passed once before the latest lockfile pin and fixture-cleanup commits; the current exact head is being requalified with `--locked` enabled for both tests and Clippy. Until the latest head passes, do not claim current-head qualification.

No SciLaws dataset has been imported. No official SciLaws run or leaderboard result has been reproduced. The next increments are:
1. pass the newest exact-head isolated harness with locked dependencies;
2. connect the frozen live packet to the existing candidate/anytime history pipeline;
3. implement the separate Parallel S_S audit under #12522 without mixing score spaces;
4. pin the Hugging Face dataset revision and review per-task licenses;
5. run matched candidate-generation/selection arms on frozen candidate pools, then request official evaluator scoring.

## 9. Claim ceiling

A passing synthetic/contract fixture qualifies only the packet boundary, score join, Pareto audit, coordinate-regret calculation and denominator semantics. It does not establish improved live Symthaea selection, SciLaws reproduction, SOTA, scientific novelty, causal truth, physical validity or autonomous laboratory execution.
