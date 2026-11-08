# CEB Review Benchmark Triangulation V0 — 2026-11-07

**Status:** Public comparative research and design material. No benchmark task is admitted, no model has been evaluated, no hidden holdout is created, and no CEB result is claimed.  
**Scope:** Existing CEB-REVIEW / CEB-HUMAN metrics and current review/causal-debugging protocols.  
**Owners:** [#10723 — disconfirm-first review and human-agent complementarity](https://github.com/Luminous-Dynamics/symthaea/issues/10723); [#10419 — engineering review scope]; [#10367 — causal debugging and human-agent complementarity].  
**Primary CEB matrix:** 14 lanes / 56 primary metrics; CEB-REVIEW remains three primary metrics: `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`. Do not create a parallel review lane or change the measurement matrix here.

## 1. Main finding

The November 2026 code-review benchmark frontier is not one leaderboard. It is a set of complementary measurement designs that answer different questions:

1. **Did the agent identify a known target defect?** (CRBugBench, CR-Bench-v2, SecPRBench; with different task and scoring contracts.)
2. **Did the review comment identify a true, actionable issue, including one absent from the original gold set?** (ReviewBench grounded and augmented evaluation; CR-Bench-v2 Usefulness Rate).
3. **Did the reviewer explain the correct causal mechanism, and does that mechanism remain present in the reviewed source revision?** (SecPRBench's mechanism-based diagnosis criterion.)
4. **Did the review reduce the human team's review burden without hiding critical omissions, noise or unauthorized actions?** (ReviewBench's precision/recall/cost production framing; CRBugBench's human complementarity; CEB-REVIEW's reviewer_burden.)

These are related but distinct. CEB should preserve the current primary metrics, but make the adjudication protocol capable of distinguishing all four questions. Never collapse them into an opaque composite.

## 2. Current external evidence — snapshot 2026-11-07

### 2.1 ReviewBench — multi-source gold and grounded versus augmented findings

GitHub introduced ReviewBench on October 5, 2026. Its public dataset contains **219 pull requests across 187 public repositories and 19 languages**, selected after examining distributions from 103.9 million GitHub pull requests. It reports grounded precision/recall/F1 against the existing gold set, and augmented precision/recall/F1 that separately adjudicate findings not represented in that set. Grounded recall is the headline cross-agent comparison because augmented recall changes its denominator according to what each system discovers. Senior engineers independently re-labeled the ground-truth findings and agreed with the benchmark's true/false-positive judgments **96.6%** of the time. GitHub reports that offline ReviewBench movement anticipated the direction of a production experiment; in one lite-tier experiment, addressed rate rose 8.0%, recall rose 13.6%, comment volume rose 61%, and cost per review fell 8.0% relative to the control. This is GitHub's own reported production comparison, not an independent replication. Sources: [GitHub launch and method (October 5, 2026)](https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/) · [live leaderboard](https://review-bench.ai/).

The live high-severity grounded-F1 leaderboard observed November 7 lists:
- OhMyBug: **83.0 F1 / 92.0 precision / 75.6 recall** (October 25 run).
- intent-review-agent: **82.2 / 92.0 / 74.4** (October 20).
- Copilot code review Max: **75.7 / 96.8 / 62.2** (October 16).
- Laude RevEngine with Claude Opus 5.5: **73.3 / 92.8 / 60.6** (October 14).
- Copilot code review Balanced: **72.7 / 96.9 / 58.2** (October 16).

The leaderboard states that grounded scores count only findings people reviewed. These figures are specific to the high-severity grounded-F1 view, not overall code-review accuracy, all-severity review, or CEB results. Sources: [ReviewBench live leaderboard](https://review-bench.ai/) · [GitHub methodology and metric explanation](https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/).

**Important metric pitfall from the published methodology:** let GTP be fixed-gold true positives, MTP the candidate's matched gold findings, UTP the candidate's unmatched findings independently judged true positives, and C all candidate findings. Augmented precision is (MTP + UTP) / C. Augmented recall is (matched GTP + UTP) / (GTP + UTP). Because UTP is candidate-specific and appears in both numerator and denominator, augmented recall does not use a common denominator across agents and can rise as an agent surfaces more unmatched findings. The methodology itself warns against using augmented recall as the sole basis for cross-agent ranking; use grounded recall for common-gold comparisons and report novel TP count / augmented precision as complementary per-agent diagnostics. An unmatched finding must still be adjudicated; it is not gold merely because a candidate produced it. [ReviewBench methodology, augmented metrics](https://github.com/review-bench/ReviewBench/blob/main/docs/METHODOLOGY.md) · [online methodology](https://review-bench.ai/methodology/).

The benchmark's scoring pipeline uses a Claude Sonnet 5 classifier to label candidate findings, including unmatched findings. GitHub reports an independent senior-engineer re-label audit of the gold set, but this is not a fully model-independent adjudication of every candidate-specific augmented finding. CEB should retain a human-audited stratified matcher/utility sample and report classifier lineage.

**CEB use:** adopt the *principle* of multi-source candidate discovery, semantic deduplication, common rubric and independent expert audit. Keep a strict known-gold comparison and a separate discovery channel for valid findings absent from the original gold set. Do not automatically add unmatched model findings to gold truth. Version the rubric, matcher, judge, dataset and expert-audit scope.

**Second live capture — November 7, 2026, 20:14 SAST:** a later retrieval of the same high-severity grounded-F1 view showed 16 rows and listed intent-review-agent at **82.2 F1 / 92.0 precision / 74.4 recall** (October 20 run) as the top row. The earlier 83.0 OhMyBug entry above is retained as the earlier same-day snapshot, not silently overwritten. For comparisons made after this capture, use the live board's contemporaneous ranking and record the exact filter, capture time, dataset/judge/matcher version and model run date. A mutable leaderboard row is not a permanent SOTA constant. [Live leaderboard capture](https://review-bench.ai/).

### 2.2 SecPRBench — correct causal diagnosis matters more than pointing at the right file

The October 24, 2026 preprint introduces **265 real-world vulnerability-review tasks** from nine widely deployed systems projects, including Firefox, V8 and OpenSSL, in C, C++ and Rust. For each task the paper validates the pre-introduction tree, introducing tree and repaired tree; it replays the same PoC/invocation and requires the target failure to be absent, present and absent respectively. Of 265 cases, 217 pass mechanical checks; 48 require human adjudication of source and execution evidence. [Y. Yang et al., SecPRBench (arXiv:2610.33509)](https://arxiv.org/abs/2610.33509).

Across eight direct-review model+harness configurations, target recall ranges from **32.5% to 78.1%**. Four leading agents—Grok 4.6, GPT-6 sol, GLM 5.3 and Claude Opus 5.5—are statistically tied; Grok is numerically highest at 78.1%, while Opus 5.5 reports 74.0% at an estimated $0.44 and 1.4 minutes per scan, versus Grok's 78.1% at $3.20 and 12.6 minutes. These are paper-specific cost estimates, not current API quotes. The top numerical result is not automatically the best operating point.

The key diagnosis result: **68% of pooled direct-review misses** contain a finding in a root-cause file but describe a neighboring or incorrect mechanism. Recall also falls as the mechanism spans more root-cause files: 69.6% for one file versus 38.5% for five or more. A file/line hit can therefore coexist with a causal diagnosis failure. The judge asks whether the finding explains the trigger, unsafe operation and violated security condition, then checks whether the mechanism remains live in the reviewed source. Sources: [paper abstract/results](https://arxiv.org/abs/2610.33509) · [Section 4 and Table 2](https://arxiv.org/abs/2610.33509) · [limitations](https://arxiv.org/abs/2610.33509).

**Limits:** the tasks are known, publicly disclosed vulnerabilities in nine systems-code projects, up to 30 files / 2,000 changed lines. Every buggy task has a validated target, so this is target detection rather than ordinary-PR prevalence or full precision. Fixed companions control only for the target; they do not establish whole-PR safety or report precision. The judge is Claude Opus 5.5, also an evaluated model, while Sonnet 5 drafted most references. Judge agreement is high (96.9% over 2,143 nonempty reports), but the 25+25 manual audit was diagnostic rather than a representative accuracy estimate. Each agent has one run per task and budgets are not matched. Cost excludes construction and judging. Public disclosure/training contamination is possible. [SecPRBench limitations](https://arxiv.org/abs/2610.33509).

**CEB use:** add mechanism-level adjudication and fixed-target controls to the existing review protocol. Do not confuse localization with diagnosis. Keep causal claims bound to the reviewed source revision and the authority of the evidence.

### 2.3 CR-Bench-v2 — known-defect detection and comment usefulness are orthogonal

The November 2 preprint (v2 revised November 3) builds 584 defect-centric PR review instances, including a 174-instance CR-Bench-Verified slice from SWE-Bench Verified. The verified subset covers 11 Python repositories and is concentrated in Django (63 tasks). Each instance has one labeled known defect. It measures Known-Defect Detection Rate (KDDR) separately from Usefulness Rate (the share of comments classified as Bug Hit or Valid Suggestion rather than Noise). [K. Pereira et al., CR-Bench-v2](https://arxiv.org/abs/2611.02566).

On the verified slice, GPT-5.2 Reflexion reports 33.0% KDDR / 89.8% usefulness; GPT-5-mini Reflexion 33.7% / 72.6%; Gemini Code Assist 8.1% / 78.0%. The three judges agree more on comparative direction than absolute level; Claude Sonnet 4.5 matches the human majority on 71.7% of 60 labels from one system, while expert-expert agreement ranges 0.48–0.75. Reference comments and taxonomy labels are generated by Claude Opus 4.5, from the same model family as the primary judge. The agents tested in the paper do not execute code/use tools; cost and latency are not reported. Public SWE-Bench data may be in training sets. Other true PR defects are outside the single labeled KDDR target and may count as valid suggestions instead. [CR-Bench-v2 results/limitations](https://arxiv.org/abs/2611.02566).

**CEB use:** report defect Recall@5 and actionable precision independently. A useful maintainability suggestion is not a target-defect hit. An unlabelled but source-grounded defect is not automatically noise. Human-audit the semantic finding-to-defect matcher.

### 2.4 CRBugBench — scale and human complementarity

CRBugBench pairs human-verified bugs with introducing PRs: 1,000 public tasks across 98 repositories plus 800 proprietary tasks. Its best reported configuration, Claude Opus 5 + Copilot CLI, detects 71.9% of human-verified bugs within its top five comments. The paper reports weaker performance on complex and non-local bugs and a sharp drop in proprietary repositories; exploration-nudging prompts add up to 5.8 percentage points at approximately 10% higher cost. The October 13 PREreview summarizes human-only / agent / human+agent results as 50.0% / 71.9% / 82.1%; that combined comparison is reviewer-reported and must retain its own experimental context. [Microsoft Research overview](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/) · [arXiv paper](https://arxiv.org/abs/2610.13668) · [PREreview](https://prereview.org/reviews/23335998).

**CEB use:** use as a scale and human-complementarity contextual comparator. Its top-five detection rate is not precision, and should not be equated with ReviewBench F1 or SecPRBench all-findings recall.

**Research protocol:** the proposed [CEB Review Causal-Diagnosis Preregistration V0](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md) turns these insights into a staged, paired, counterfactual experiment. It includes P/I/F/N/M/IRR source-state controls, matched reviewer arms, gold/matcher adjudication, cost accounting and a power-derived sample-size rule. This is a design proposal, not a task set or evaluation result.

### 2.5 CodeReviewBench — small replay benchmark and human-gold ceiling

Kodus maintains CodeReviewBench: **30 merged pull requests and 95 human-authored golden bugs**, drawn from five repositories (cal.com, Discourse, Grafana, Keycloak and Sentry) and five languages (Go, Java, Python, Ruby and TypeScript). Its current leaderboard has 13 model configurations, run through the Kodus harness with Claude Haiku 4.5 as judge. The top row is DeepSeek V4 Pro: **43.9 F1 / 43.6% precision / 44.2% recall**, finding 42 of 95 golden bugs at an estimated **$0.30 per PR / $0.21 per bug found**. The per-model page reports a 95% PR-bootstrap interval of **32.3–56.3 recall points** over 2,000 resamples across 30 PRs. It separately reports three judge re-scores of the same findings (43.9 ±0.6 recall points); this estimates judge noise, not model run-to-run variance. Every model entry had one agent pass. [Live leaderboard](https://www.codereviewbench.com/leaderboard) · [DeepSeek V4 Pro scorecard](https://www.codereviewbench.com/model/deepseek-v4-pro) · [maintainer's recall discussion, August 21](https://www.codereviewbench.com/blog/ai-code-review-recall-2026).

**Critical limitation:** the benchmark's gold set consists of bugs already reported by human reviewers before merge. Its maintainers explicitly say defects that escaped the original human review are absent, so measured recall is recovery of human-found bugs—not exhaustive defect discovery. The benchmark is useful for model/harness replay, inspectable per-PR misses, precision/recall trade-offs and historical cost, but has only 30 PRs, one harness and one judge family. Some model entries have different evaluated coverage; compare each scorecard's denominator. [CodeReviewBench scope and limitations](https://www.codereviewbench.com/blog/ai-code-review-recall-2026) · [per-model artifacts](https://github.com/kodustech/codereviewbench).

**CEB implication:** retain it as a contextual comparator, not a direct Recall@5 equivalent. Its most important contribution is making the *human-gold ceiling* explicit. CEB should add blind audits of zero-finding PRs and multi-producer "clean" consensus, so defects missed by the original review population have a route to independent discovery. Do not retroactively add newly discovered defects to a frozen task denominator; report them through a separate, blinded open-set channel.

### 2.6 Martian Code Review Benchmark — production adoption signal at scale

Martian's online Code Review Benchmark analyzes production review-bot activity and developer response timelines: bot suggestions, author responses, follow-up changes and resolved conversations. Its current dashboard (accessed November 7, 2026) reports a recent-period view of **17,768 scored PRs**. Its visible overall table lists Greptile at **65.2 F1 / 82.9% precision / 53.7% recall**, Cubic Dev AI at 63.2 / 68.6 / 58.7, and Qodo Code Review at 62.3 / 75.3 / 53.1. The tool-specific recent-period sample sizes shown separately are 1,600, 1,525 and 2,435 PRs, respectively; visible all-time PR counters are much larger and are not the same denominators. [Live dashboard](https://www.codereviewbenchmark.com/) · [dashboard alternate URL](https://codereview.withmartian.com/?fscore_group=day).

The benchmark's distinctive contribution is operational: it examines whether review-bot suggestions are followed by developer responses, code changes and resolved threads, using LLM-powered analysis of the review timeline. That is a useful human-workflow signal which fixed-gold replay benchmarks do not directly measure.

**Critical distinction:** suggestion adoption is not ground-truth correctness. A developer can accept a correct comment, reject a correct comment for timing/scope reasons, accept a mistaken comment for unrelated reasons, or make the same code change independently. Follow-up code is evidence about workflow impact, not automatic proof of the finding's factual or causal validity. The table's pooled/recent view and agent-specific samples are also not a common paired denominator.

**CEB implication — separate truth from uptake.** Keep four outcomes distinct: (1) independently adjudicated correctness/causal diagnosis, (2) actionability/usefulness, (3) human uptake or disposition, and (4) downstream engineering outcome such as recurrence/rework where observable. Do not add a single adoption score or use uptake to rewrite frozen gold. Record uptake as observed process data and investigate attribution; report silence, defer, correction and rejection separately from acceptance.

### 2.7 SWE-sweep — open-world repository discovery and multi-bug repair (adjacent, not review-equivalent)

SWE-sweep changes the task from reviewing a supplied PR to discovering and repairing multiple bugs in a repository without hints about bug type or location. Its October 1, 2026 paper covers **4,068 bugs across 100 repositories and 22 languages**. Each repository state is selected to maximize concurrent known bugs; candidate patches are checked against hidden fail-to-pass tests for each bug and the original test suite for regressions. The paper's main evaluated table reports its best system at **4.7% of all bugs resolved**; the authors conclude the hardest step is discovery rather than repair. [Paper](https://swesweep.com/paper).

The live benchmark website's leaderboard snapshot viewed November 7, 2026 lists **Astra 6 + mini-SWE-agent at 9.5%**, followed by Sol 5.6 xhigh at 4.7% and Claude Opus 5.5 xhigh at 3.6%. This live board is a different model/harness snapshot from the paper's original seven-model evaluation. Do not splice the 9.5% live row into the October paper's evaluation table or infer an improvement without matching the exact model, harness, dataset snapshot and scoring run. [Live benchmark/leaderboard](https://swesweep.com/).

**CEB relevance:** this is an adjacent frontier for open-world repository comprehension, proactive defect discovery and multi-issue planning, not a code-review metric comparator. It demonstrates the distinction between issue-conditioned repair and deciding what work exists. CEB should preserve a separate open-set discovery channel and evaluate it with fixed source snapshots, independently validated multi-bug gold, interaction controls between defects, and total discovery/repair cost. Do not translate its bug-resolved percentage into CEB Recall@5, PR review precision or a direct superiority claim.

### 2.8 LoopControlBench — deterministic completion authority (adjacent, narrow population)

LoopControlBench evaluates real merged Python library bug-fix PRs. Its gate checks that the maintainer's test fails on the buggy parent and passes after the real fix, with the scoring environment matching task certification. The published reference is **46/70 = 65.7% solved** for GPT-5-mini, one attempt per task, no escalation or repair loop. The maintainer says the pool subsequently grew to 89 tasks, while the 65.7% score remains scoped to the original 70. It explicitly limits the population to reproducible, maintainer-tested Python library bugs under a modern shared environment; it is not 65.7% of software bugs generally. [Repository/readme](https://github.com/kltownsend/loopcontrolbench).

Its strongest design lesson is that the **validator, not the model, decides done**: test edits are prohibited, source paths are allowlisted, a named fail-to-pass test is the acceptance oracle, and setup/harness compatibility is checked at admission rather than silently removed mid-run. The default runner uses a disposable, network-disabled container with dropped Linux capabilities. The readme separately warns that its optional agentic path runs install/tests on the host and must only be used in a disposable VM. This makes it a useful methodological reference for explicit admission, deterministic outcome authority and precise denominator control, not a broad SOTA leaderboard. [Security and reference-score details](https://github.com/kltownsend/loopcontrolbench).

**CEB relevance:** compose the deterministic-oracle principle with CEB's stronger provenance, evidence authority, hidden-test fidelity and independent regrade gates. A single fail-to-pass test is strong evidence for one declared behavior, not whole-repository correctness, causal uniqueness, broad language coverage or independent pristine regrade. Keep the Python-only 70-task denominator attached to the result.

### 2.9 Saving SWE-Bench — realistic user framing as an evaluation treatment

Garg, Steenhoek and Huang's paper, *Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation*, was submitted October 10, 2025, revised January 23, 2026 (v4), and accepted to CAIN 2026. It converts formal GitHub issue descriptions into realistic user-style queries using observed developer interaction patterns with chat-based coding assistants. The authors report that formal public benchmarks overestimated performance for some evaluated models by **more than 50%**, and report approximately **10–16%** differences on their internal benchmark. These effects are specific to the models and benchmark subsets they tested; they are not a universal correction factor. [Microsoft Research overview](https://www.microsoft.com/en-us/research/publication/saving-swe-bench-a-benchmark-mutation-approach-for-realistic-agent-evaluation/) · [arXiv:2510.08996](https://arxiv.org/abs/2510.08996).

**CEB implication:** prompt/task framing is an experimental variable. A system may repair a formal issue but fail to clarify an ordinary developer request; a failing test adds information, not merely a different phrasing. Use paired shared-source/defect anchors across formal issue, realistic user request, diff-only review, repository-only discovery and failing-test-guided repair. Separate wording-only contrasts from information-package contrasts. Keep the outcomes in existing CEB-REVIEW, CEB-RESOLUTION, CEB-REPO and CEB-HUMAN lanes; do not average them into a composite engineering-intelligence score.

The [CEB Review Cross-Framing Anchor Protocol V0](CEB_REVIEW_CROSS_FRAMING_ANCHOR_PROTOCOL_V0_2026-11-07.md) defines the six framing conditions, source/defect anchors, leakage controls and paired analysis. It is a design proposal, not a task set or performance result.

### 2.11 Causal software root-cause analysis — SDRCA and UniVal

Causal fault-localization work is a necessary adjacent comparator: CEB should not present causal graphs and interventions as if no software-engineering literature had explored them.

**SDRCA (Journal of Systems and Software, October 2026)** is a runtime defect root-cause framework. Its reported method combines information-theoretic anomaly scoring across heterogeneous monitoring signals, causal graphs, in-distribution counterfactual interventions and Shapley-style allocation of multiple-variable contributions. It is evaluated on PetShop, RE1 and RE3 and reports Recall@1 / Recall@3 improvements over correlation and causal baselines. Its domain is runtime anomaly localization with monitored signals, not source-only pre-merge review. A Shapley value allocates modeled contribution under a specified coalition game; it does not, by itself, prove causal identification or replace intervention validity. [Sun et al., SDRCA](https://doi.org/10.1016/j.jss.2026.112931).

**UniVal (ICSE 2021 Distinguished Paper)** combines predicate outcomes and variable-value profiles in a causal statistical fault-localization method intended to reduce confounding in spectrum-based localization. The paper reports evaluation over 835 Defects4J program versions containing real faults. It is an execution-profile fault-localization comparator, not an agentic PR-review benchmark and not a source-counterfactual diagnosis evaluation. [Paper](https://arxiv.org/abs/2102.06292) · [ICSE distinguished paper record](https://2021.icse-conferences.org/details/icse-2021-papers/72/Improving-Fault-Localization-by-Integrating-Valueand-Predicate-Based-Causal-Inference-Techniques).

**CEB implication:** evaluate source-level mechanism diagnosis separately from runtime root-cause localization. The new [causal-topology protocol](CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md) tests AND/OR path structure, partial temporal order, interaction conditions and residual paths. Compare with SDRCA/UniVal only on an explicitly matched runtime evidence slice if their data and intervention contracts permit it; never compare their Recall@1/3 directly to CEB Recall@5.

### 2.13 Causality-Based Fault Localization / SoftReX — actual causality under execution

Navaratnarajah et al.'s **Causality-Based Fault Localization** preprint (October 31, 2026; tool: SoftReX) frames fault localization using actual-causality / causal-responsibility semantics rather than only suspiciousness correlations. It models code elements as variables and mutations as interventions, and explicitly discusses overdetermination, pre-emption and masked faults. On Defects4J, its main comparison evaluates 827 bugs across 17 Java projects after excluding 27 OOM/timeout cases. It reports 103 Top-1 and 301 Top-10 localizations, with MRR 0.349, versus BLMu's 89 Top-1, 270 Top-10 and MRR 0.301. Its guided search reports about 2.6× fewer mutants than Major in its matched subset while retaining competitive localization accuracy. [Paper](https://arxiv.org/abs/2611.01292).

**Scope and limitations:** this is execution-grounded fault localization on Java test suites and mutant sets. It does not evaluate source-only PR comments, human-actionability, repository-context ambiguity or reviewer burden. The paper's main experimental result depends on the supplied test suite and mutation semantics; its structural model does not yet integrate runtime data-flow/value state in the same way as a complete dynamic causal model. The Defects4J exclusions, task split, mutant budget and ranking metrics must remain attached to every result.

**CEB implication:** use SoftReX as the closest current actual-causality localization method, and include its problem classes—overdetermination, pre-emption and masking—in the synthetic diagnosis corpus. Do not equate its Top-1/Top-10 localization counts with CEB defect Recall@5. CEB's identifiability work asks the additional question: when does the available evidence distinguish the candidate causal models at all?

### 2.14 CausalRepair — minimal causal context through dual slicing

Wu et al.'s **CausalRepair** (PACMSE / ISSTA 2026) combines test-semantic static slicing with runtime trace-based dynamic slicing to construct a compact causal context for automated repair. The paper reports 313 correct fixes on Defects4J across V1.2 and V2.0, compared with 236 for its second-ranked ReinFix baseline, and 289 on Defects4J-Trans. It reports two experienced Java researchers independently review patch semantics, with a third reviewer resolving disagreements. The authors also evaluate a more realistic fault-localization setting, where CausalRepair fixes 232 bugs with noisy initial localization. [Paper](https://arxiv.org/abs/2608.10613) · [artifact](https://doi.org/10.5281/zenodo.20953899).

**Scope and limitations:** CausalRepair is automated repair, not pre-merge code review. Its strongest context contains actual execution traces and a failure-triggering test. It uses Defects4J Java projects and one principal backbone configuration; its success count is not an all-language or production prevalence estimate. Patch equivalence is independently reviewed, but remains tied to the paper's evaluation contract.

**CEB implication:** compose source topology with a minimal causal-context discipline. Where execution is allowed, a dynamic slice may remove static over-approximation; where execution is unavailable or unauthorized, preserve possible paths and mark which are merely source-possible. A compact context is useful only if it does not delete an alternative sufficient path or hide an unobserved caller.

### 2.15 Partially Observed Structural Causal Models — identifiability under latent context

Orujlu et al.'s **Partially Observed Structural Causal Models** (UAI 2026, PMLR 337) formalizes settings where unobserved upstream context co-determines both the interaction structure and downstream mechanisms. It defines intervention hierarchies over node-level and edge-local mechanisms and gives identifiability results for which interventions can separate structure formation from mechanism behavior. Its experiments demonstrate non-identifiability under latent context and structure–mechanism confounding under latent edges. [Paper](https://proceedings.mlr.press/v337/orujlu26a.html).

**Scope and limitations:** this is a general causal-modeling theory and simulator evaluation, not a code-review benchmark or software RCA leaderboard. Its identifiability results depend on its formal assumptions and intervention family; CEB must not claim to implement POSCMs merely by adding a latent-node annotation.

**CEB implication:** this is the theoretical basis for the new [Causal-Identifiability Protocol V0](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PROTOCOL_V0_2026-11-07.md). A code reviewer should preserve material evidence-compatible models, identify a query on which they disagree, and request the minimum safe observation that distinguishes them. If no admissible discriminating evidence exists, the correct output is a bounded or non-identifiable claim—not an invented consensus graph.

### 2.12A Causal claim types — source possibility is not incident attribution

The current literature spans different causal targets. CEB should label the target claim before comparing methods:

| Claim type | What the method can establish under its contract | Key limitation |
|---|---|---|
| Source structural possibility | A source path can reach a state under stated conditions | Does not show that the build selects it or a run executes it |
| Build-world selection | Which feature/target-conditioned implementation is in an exact artifact | Does not show that the path was exercised |
| Execution realization | A qualified trace/slice records a path in a particular run | Does not establish that the path was the actual cause |
| Actual-cause attribution | A realized factor is a cause under a declared context-specific causal model/semantics | Depends on model, actual context and counterfactual criterion |
| Interventional-effect identifiability | All admissible evidence-compatible models agree on the effect of a specified intervention | Does not establish that a historical incident occurred |
| Incident attribution | Source/artifact/event identity and causal evidence bind the mechanism to a named incident | Does not automatically generalize beyond that incident's scope |

This distinction matters for the three nearby research strands: SoftReX's actual-causality/responsibility ranking is an execution-/mutation-grounded localization target; CausalRepair's dynamic slice is an executed dependency context for a failing test; POSCMs' identifiability theory asks when interventional claims are uniquely determined across compatible partially observed models. They are complementary, not interchangeable. Do not rank their metrics together or claim that a source-only reviewer has established an incident cause merely because it produced a plausible topology.

### 2.12 AACR-Bench — holistic repository context

AACR-Bench reports **200 real PRs from 50 active projects across 10 languages**, with full repository context and an AI-assisted, expert-verified annotation pipeline. The authors report a 285% increase in annotated defect coverage relative to their prior comparison baseline; that is a dataset-coverage claim, not evidence that a model improved by 285%. It is especially relevant to multi-file reasoning, caller discovery, cross-file tracking and the impact of context granularity/retrieval. The benchmark does not itself establish causal intervention prediction or residual-path correctness. [Paper](https://arxiv.org/abs/2601.19494) · [benchmark repository](https://github.com/alibaba/aacr-bench).

**CEB implication:** AACR-Bench motivates repository-level context and multilingual/cross-file stratification, while SecPRBench motivates mechanism correctness and the causal-topology extension tests whether reviewers can represent the missing path/interactions. Do not infer causal diagnosis from a repository-context score.

### 2.10 Kodus State of AI Code Review 2026 — suggestion implementation is not correctness

Kodus separately publishes platform-level analytics from its production code-review pipeline, distinct from its 30-PR CodeReviewBench replay above. The report analyzes **22,743 AI-authored PRs** and **180,739 review suggestions** across its supported GitHub/GitLab/Bitbucket/Azure DevOps pipeline, with a data window beginning September 1, 2025 and a snapshot ending June 8, 2026. It reports **33% implemented/adapted** overall, rising from 25% to 48% by PR-created month across the cited eight-month window; the newest months are right-censored and June is partial. Bugs have a reported 34.5% implementation/adaptation rate; security 31.2%. It also reports that **71.8% of 13,609 flagged merged PRs had at least one open flag**, while 60% of security flags and 64% of critical flags were unaddressed. The report explicitly states the open-flag result includes false positives and accepted risk. [Kodus State of AI Code Review 2026](https://kodus.io/data), methodology and snapshot disclosed in the report.

**Composition and attribution caveats:** the report warns that platform-wide adoption/merge trends are heavily compositional. Its fixed 24-team panel shows that platform-wide PR size and merge-time trends largely disappear when comparing the same teams at both ends. Its “implemented” rule is based on later changes to flagged lines; that does not prove the review comment caused the edit or that the edit corrected the defect. Silent ignores count in the implementation outcome. These are descriptive analyses from Kodus' own commercial review pipeline, not a randomized intervention or independent adjudication of every suggestion.

**CEB implication:** use these data to motivate a separate uptake/outcome protocol, not as correctness evidence. The production-impact experiment must freeze truth/actionability before exposure; distinguish accepted, adapted, rejected, deferred, independently reproduced and no-response; verify repairs and regressions separately; control team composition and censoring; and never treat a later diff as causal proof. See [CEB Review Production-Impact Protocol V0](CEB_REVIEW_PRODUCTION_IMPACT_PROTOCOL_V0_2026-11-07.md). Uptake stays contextual under existing CEB-HUMAN/CEB-CONTINUITY ownership; no new metric lane or acceptance score is created.


### 2.17 CGARF — causality-guided repair pipeline

Yuan et al.'s **CGARF** (Empirical Software Engineering; published August 22, 2026) composes three components: Causal Graph-Augmented Multi-Agent Debate for location re-ranking, Self-Reflection Consistency Distillation for more stable patch generation, and Two-Stage Patch Filtering using functional tests, causal matching and group-similarity selection. It evaluates 12 configurations across three localization methods (Agentless, OrcaLoca and CoSIL) and four models on SWE-bench Lite and Verified. The paper reports **2–4 percentage-point repair-success gains** and about **3 percentage points average patch-applicability gain** across the evaluated configurations. [Paper and results](https://doi.org/10.1007/s10664-026-10940-1).

**Scope and limitations:** CGARF evaluates repair, not source-only PR-review comments. The reported gains are aggregated across configurations with different localizers/models; task denominators, exact per-configuration intervals and effect heterogeneity must be retrieved from the full paper before quantitative replication. Its causal graph debate and filtering pipeline motivate a useful I1/I2 ablation, but a higher patch-success rate does not prove that the model correctly identified which cause was identifiable from the evidence or that its explanation was accurate.

**CEB implication:** borrow the separation of localization, causal argument, patch generation and patch verification. Evaluate each stage with its own outcome and provenance; do not let a correct patch retroactively validate an unsupported diagnosis. The three primary CEB-REVIEW metrics remain distinct from repair pass rate and patch applicability.

### 2.18 SecRepoBench — repository-level secure code completion

**SecRepoBench** contains **318 code-completion tasks** from 27 C/C++ repositories covering 15 CWEs. A model or agent completes a masked region with full-repository context; correctness is checked with developer tests and security with OSS-Fuzz-derived PoCs. The visible leaderboard snapshot accessed November 7, 2026 lists an ImageCodex + GPT-5.1-Codex-Max review setup at **60.1 secure-pass@1** and **76.4 pass@1**. [Benchmark and leaderboard](https://secrepobench.github.io/) · [Google Research summary](https://research.google/pubs/secrepobench-benchmarking-code-agents-for-secure-code-completion-in-real-world-repositories/).

**Scope and limitations:** this is secure code completion, not review of a vulnerability-introducing pull request, causal diagnosis, or reviewer burden. Its test/PoC contract and denominator must remain attached to each score. “Secure completion” is not automatically equivalent to CEB's joint secure-functional pass or a review agent's security finding precision.

**CEB implication:** use it as a downstream secure-code generation comparator and possible cross-lane robustness context, not a direct code-review SOTA row. A future matched experiment could compare whether diagnoses from CEB produce safer repairs, but only after freezing the same source snapshot, security contract, tests/PoCs, model/harness, budget and independent oracle.



### 2.19 Requirement-conformance over-correction and Fix-guided Verification Filter

Jin and Chen's **Are LLMs reliable code reviewers? systematic overcorrection in requirement conformance judgement** (Automated Software Engineering, published June 26, 2026) constructs paired canonical/buggy implementations from HumanEval, MBPP and QuixBugs: over 700 tasks and over 1,400 instances. It evaluates five models under Direct, Direct+Explain and Full (verdict, rationale, proposed fix) prompting. It reports that richer explanation/repair prompts can substantially increase false rejection of correct implementations. For GPT-4o, FNR rises from 26.2% to 73.2% on HumanEval and 35.9% to 87.9% on MBPP when moving from Direct to Full prompting. Across models under Full prompting, the Fix-guided Verification Filter reduces average FNR from 54.8% to 16.3% on HumanEval, 69.0% to 28.9% on MBPP, and 51.0% to 24.0% on QuixBugs. [Paper](https://doi.org/10.1007/s10515-026-00638-5).

The filter checks the original and proposed fixed implementation against benchmark and specification-constrained tests. Its four outcomes distinguish: both pass/equivalent (the fix was unnecessary), both fail (oracle augmentation or repair remains unresolved), original passes but patch fails (over-repair/regression), and original fails while patch passes (repair-supported). The authors report small FPR increases in some settings; QuixBugs' shallower tests limit the filter's discriminative power. They explicitly acknowledge that canonical solutions and benchmark tests may not fully define the natural-language specification, and prompt semantics can alter the implied review rubric. These caveats matter: an executable patch is a hypothesis, not an independent oracle.

**CEB extension:** require a target witness and a preservation witness for repair-bearing findings, with an independently frozen contract/oracle. Test both false rejection of canonical behavior and false acceptance of mutant behavior. A patch cannot be called verified solely because it passes tests generated from its own rationale.

### 2.20 ReviewGrounder — draft then ground, but in academic peer review

Li et al.'s **ReviewGrounder** (ACL 2026) separates review drafting from an evidence-grounding stage, using explicit paper-specific rubrics and targeted evidence consolidation. Its reported framework uses a Phi-4-14B-based drafter and GPT-OSS-120B-based grounding stage, and outperforms larger backbone baselines on human-judgment alignment and rubric-based review quality across eight dimensions on an academic-paper review benchmark. [ACL Anthology](https://aclanthology.org/2026.acl-long.1477/).

**Scope and limitation:** this is academic peer review, not source-code pull-request review, and its eight rubric dimensions are not CEB metrics. It supports a transferable *pipeline* principle—draft, ground each claim, then adjudicate—not a claim of software-review SOTA.

**CEB extension:** make the grounding stage bind every finding to exact source revision, contract clause, causal claim kind and independent oracle. Keep grounding evidence distinct from the model's own rationale and patch.

### 2.21 Selective conformal judging — calibrate accepted claims, not verbal confidence

The SCOPE work summarized by the HyperMatrix Research Lab uses bidirectional preference entropy (querying both pairwise response orders) and conformal calibration to control the error rate among non-abstained LLM judgments under exchangeability. Its summary reports empirical FDR around 0.097–0.099 at a target alpha of 0.10 and up to 2.4× as many accepted judgments as vanilla baselines. [Research summary](https://hypermatrix.cs.dal.ca/selective-prediction/).

**Scope and limitation:** SCOPE concerns pairwise LLM judging, not code review or causal diagnosis. Its risk guarantee depends on the calibration assumptions and task population; it cannot be imported unaltered across repositories, languages, severities or claim kinds.

**CEB extension:** calibrate acceptance/abstention separately by claim type and severity on a repository-/defect-family-disjoint calibration set. Measure selective risk versus coverage and report group-wise gaps. A judge's confidence, consensus or stable rationale is not itself a finite-sample guarantee. See [Risk-Controlled Selective Claim Acceptance Protocol V0](CEB_REVIEW_RISK_CONTROLLED_SELECTIVE_CLAIM_ACCEPTANCE_PROTOCOL_V0_2026-11-08.md) for disjoint development/calibration/confirmatory roles, a one-sided Clopper–Pearson risk bound, sample-size illustrations and escalation rules. It makes no current CEB risk guarantee.



### 2.22 Incremental build graphs — Bazel Skyframe

Bazel's **Skyframe** documents a dependency graph in which outputs are functions of their declared inputs. A changed input invalidates the reverse transitive closure of dependent nodes. Its incremental algorithm also supports *change pruning*: after recomputation, if an invalidated node's value is unchanged, dependents can be resurrected rather than rebuilt. The documentation explicitly describes the trade-off between bottom-up invalidation and top-down checking, and the difficulty of proving an incremental mutation equivalent to a clean rebuild. [Skyframe reference](https://bazel.googlesource.com/bazel/+/refs/heads/master/site/en/reference/skyframe.md).

**Scope and limit:** Skyframe is an incremental build-engine dependency model for deterministic derived values. It is not a general claim-validity or causal-inference system. A source file's output may be unchanged while a policy, contract, trust root, verifier or incident attribution changes the decision-use validity of an engineering claim.

**CEB implication:** reuse the reverse-dependency and change-pruning intuition, but do not assume all engineering claims behave like deterministic build nodes. A retained claim requires an unaffectedness witness and a complete declared support closure. Where equivalence cannot be established, revalidate or leave unresolved; do not resurrect solely because a diff looks irrelevant.

### 2.23 Artifact provenance — SLSA v1.1 and in-toto attestations

The current **SLSA v1.1** provenance guidance binds produced artifacts to a builder identity, build definition, external parameters, resolved dependencies and output subjects. It recommends recording dependencies that can affect the build and verifying provenance against expected builder/source/build-type/parameter values. It also makes the builder's trust boundary explicit. [SLSA v1.1](https://slsa.dev/spec/v1.1/) · [Provenance](https://slsa.dev/spec/v1.1/provenance) · [Artifact verification](https://slsa.dev/spec/v1.1/verifying-artifacts).

The **in-toto Attestation Framework v1.2** separates predicate, statement, envelope and bundle, with the statement binding a typed predicate to one or more subject artifacts. [Specification](https://github.com/in-toto/attestation/blob/main/spec/README.md).

**Scope and limit:** SLSA/in-toto answer provenance and artifact-authenticity questions; a valid attestation does not by itself establish that a code-review claim is true, that a repair preserves behavior, or that an old approval remains authorized after policy revocation. Nor does a provenance record prove completeness of every semantic dependency unless the builder/format guarantees that scope.

**CEB implication:** bind support-closure leaves to exact digests and existing receipt/attestation subjects where available. Keep signer identity, builder identity, trust roots and current authorization as separate dependencies. Do not create a parallel CEB receipt or replace #10314/#10370; use their canonical identity and trust contracts.



### 2.24 Incremental Program Analysis in the Wild — empirical dependency-reuse limits

Wang et al.'s ASE 2025 study constructs a real-world Java program-change benchmark from **4,084 commits across 20 open-source projects** and evaluates two representative incremental program analysis tools. The authors report that incremental analysis can substantially improve efficiency over exhaustive analysis, but the evaluated tools exhibit inconsistencies and markedly higher peak-memory use. They identify gaps in standardized real-world program-change benchmarks and in balanced evaluation metrics. [IEEE publication](https://doi.org/10.1109/ASE63991.2025.00194) · [ASE research-program entry](https://conf.researchr.org/details/ase-2025/ase-2025-papers/211/Incremental-Program-Analysis-in-the-Wild-An-Empirical-Study-on-Real-World-Program-Ch).

**Scope and limitation:** this is incremental program analysis, not engineering-claim validity or CEB-CONTINUITY. Its population is Java; its measured outcomes concern analysis result maintenance, efficiency, consistency and memory. Do not map incremental-analysis efficiency onto stale-evidence safety or claim-disposition accuracy.

**CEB implication:** include consistency and resource costs when studying support-closure updates. An incremental closure implementation must be compared against a clean full recomputation on sampled sequences; a fast update that misses a dependency is worse than an expensive rebuild. Report peak memory and worst-case closure growth, not only mean update time.

### 2.25 DiffCI Open Evidence Study 2026 — change-aware test selection against a cheap path rule

DiffCI's public open study reports a pre-registered historical replay of **2,000 commit deltas across 20 repositories**. Only 29.8% of deltas offered a selective-test opportunity; across the full corpus, the reported test reduction against a path-rule baseline was **4.2%**. The study reports **96.4% versus 70.7% job-level recall** of 280 historical CI failures for DiffCI versus a path rule. Its repository-clustered 95% interval for aggregate reduction was **−4.8% to +47.9%**, crossing zero; the report explicitly says the sample cannot rule out a null population effect. In measurable mutation cases it reports 32/32 detections but states this is not a reliability estimate. It also reports that only three of seven attempted repositories passed canonical suite-green qualification. [Open study and data](https://diffci.com/research/diffci-open-evidence-2026).

**Scope and limitation:** this is a self-published change-aware test-selection study, not an independent CEB benchmark. Historical test/job selection recall is not evidence-disposition recall, claim invalidation precision, or correct selective retain. Its 4.2% whole-corpus reduction is more informative economically than the 94.3% win rate within the 29.8% opportunity bucket. The repository-clustered interval and limited reach must accompany any quoted result.

**CEB implication:** use a cheap path/change heuristic as a required baseline for support-closure computation. C2 must demonstrate that typed semantic dependencies improve *claim validity decisions* over C0 path overlap and C1 continuity dispositions, not simply produce a larger graph or a higher within-opportunity win rate. Include full recomputation as a correctness oracle on a sampled subset; measure incremental cost net of graph construction and evidence retrieval. The comparison is C0 path heuristic, C1 existing #10630 continuity, C2 support-closure extension; measure closure build/update, evidence ingestion, retrieval, peak memory and revalidation action cost. Charge the full cost and compare C2 with a sampled clean full-recomputation oracle before claiming safe incrementality.


## 3. Metric correspondence to the existing CEB-REVIEW lane

The current CEB-REVIEW lane retains exactly three primary metrics. This triangulation is interpretive; it does not mutate their definitions.

| External construct | Relationship to CEB | Permitted use | Prohibited shortcut |
|---|---|---|---|
| CRBugBench top-five human-verified defect detection | Close / contextual to `defect_recall_at_5` | Frontier baseline under its reported task contract | Treat 71.9% as actionable precision or direct CEB score |
| ReviewBench grounded recall | Related; top-five ranking, task mix and severity scope must be matched before exactness | Candidate matched replication | Call high-severity grounded F1 an all-PR top-five score |
| ReviewBench grounded precision | Related to `actionable_precision`; common human rubric and unit needed | Calibration of precision expectations | Equate a benchmark LLM judge to blinded CEB adjudication |
| ReviewBench augmented precision/recall | Complementary discovery audit beyond fixed gold | Detect valid issues missing from initial gold | Mix dynamic per-agent augmented denominator into a fixed leaderboard rank |
| SecPRBench target recall | Related to defect detection, but all reported findings and security-only domain differ | Mechanism-diagnosis and systems-security stress slice | Treat target recall as top-five recall or report precision |
| CR-Bench-v2 KDDR | Related target-detection construct; one known defect / instance and review-only agents | Known-target detection stress test | Treat a valid suggestion as a target hit |
| CR-Bench-v2 Usefulness Rate | Related to comment utility, not identical to actionable precision until rubric/denominator are harmonized | Comment-quality diagnostic | Infer defect recall from usefulness |
| CodeReviewBench human-flagged bug recall | Related but not top-five constrained; gold contains only prior human-detected bugs | Small replay/cost reference and a warning about human-gold completeness | Treat 44.2% as CEB Recall@5 or exhaustive bug recall |
| Martian production review adoption | Related but distinct: production developer follow-up and LLM-derived adoption analysis | Human-workflow uptake signal; compare only after matching scope and human adjudication | Treat an accepted code change as proof that a finding was correct, or pool non-paired per-agent PR samples |
| Saving SWE-Bench prompt mutation | Methodological reference for realistic user-query framing versus formal GitHub issues | Paired shared-anchor framing experiment; report information content and task scope | Apply a universal percentage discount to CEB results or treat framing effect as a new metric |
| Human minutes / adjudication time | Exact only if accounting boundaries match | CEB `reviewer_burden` comparison | Exclude second reviewer, judge audit or context reconstruction time from assisted arm |

## 4. A diagnosis ladder for CEB review

The review protocol should adjudicate each finding at five stages. This is an annotation rubric and evaluation decomposition, not a new task schema or five new primary metrics.

1. **Discovery:** does the reviewer identify an issue worth investigating?
2. **Localization:** does the cited source span/path intersect a relevant implementation path?
3. **Mechanism:** does it explain the correct trigger, state/path, unsafe operation and violated requirement/invariant?
4. **Evidence:** are the claims grounded in the correct source revision and the appropriate `source_declared`, `syntax_derived`, `compiler_resolved`, `runtime_observed`, `intervention_observed` or `inferred/unresolved` authority?
5. **Decision utility:** is the finding actionable, correctly prioritized, non-duplicative, within scope and worth the human attention/cost it imposes?

A finding that localizes correctly but gives the wrong mechanism is a **diagnosis miss**, not a hit. A finding with a correct causal chain but stale source evidence is **not current evidence**. A valid new defect absent from the original target is **discovery requiring independent adjudication**, not an automatic false positive. A helpful suggestion that does not identify the target remains useful but does not satisfy target recall.

### Failure taxonomy for annotation

- `DISCOVERY_MISS`: target not identified or no relevant finding produced.
- `LOCALIZATION_MISS`: finding points to an unrelated region/path.
- `MECHANISM_MISDIAGNOSIS`: location is relevant, causal explanation is wrong or describes a neighbor.
- `SCOPE_OR_REVISION_ERROR`: claim is outside task scope or evidence belongs to a different revision/configuration.
- `AUTHORITY_OVERCLAIM`: inference/comment/type-check is represented as stronger runtime/intervention evidence.
- `USEFUL_NON_TARGET_FINDING`: independently valid issue but not the labeled target.
- `UNSUPPORTED_OR_NOISY_FINDING`: no sufficient evidence or actionable content after adjudication.
- `UNRESOLVED_ADJUDICATION`: insufficient/conflicting evidence; do not force binary truth.
- `EVALUATOR_MISMATCH`: finding-to-defect matcher or judge gets semantic identity wrong.

Keep these labels at the finding/defect adjudication layer; they do not expand CEB's primary metric registry.

**Annotation instrument:** use the copyable [CEB Review Finding Adjudication Card V0](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md) for each candidate finding. It records the five-link causal explanation, exact source revision, evidence authority/freshness, target versus non-target status, adjudicator uncertainty and judge/matcher audit. It is an annotation aid, not a parallel task or receipt schema.

**Two additional runner-independent protocols now make the comparison actionable:**

- [CEB Review Gold-Completeness and Open-Set Discovery V0](CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md) separates frozen target truth, independently adjudicated novel findings and the unknown unobserved defect remainder. It adds blind sampling of zero-finding PRs and multi-producer "clean" cases. Naïve capture–recapture is explicitly prohibited because reviewer/agent producers are dependent.
- [CEB Review Sequential Evidence Acquisition V0](CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md) compares static-first, unconstrained tool use, transparent evidence-directed heuristics, calibrated sequential value-of-information and ask/abstain. Each action must name competing hypotheses, a discriminating observation, authority, safety and total cost. No tool-call activity score is added.

## 5. Proposed experimental design

Use the same frozen PR/task, same revision, same requirements, equal total budget, same tool permissions, and fixed output budget where feasible:

A. **Single-pass reviewer:** one run, normal review prompt.  
B. **Reflexion reviewer:** self-critique before finalizing, as in CR-Bench-v2.  
C. **Exploration-nudged reviewer:** explicit prompt to inspect non-local dependencies, call paths and policy/spec context; test the CRBugBench result under matched cost.  
D. **Evidence-contract reviewer:** every consequential finding binds the relevant claim to source revision, scope and authority class.  
E. **Causal-diagnosis reviewer:** must explain trigger → path/state → unsafe operation → violated condition; may abstain when the chain is not established.  
F. **Disconfirm-first reviewer:** sees the initial review, attempts to refute each claim, and independently searches for high-severity omissions.  
G. **Blind independent reviewer:** receives the same source but not the first review; distinguish it from F.  
H. **Human-only / human-assisted:** equivalent task, scope and effort accounting.

For each arm preserve both candidate review and evidence artifacts. Do not give an execution bonus. Tool use can be disallowed for a source-only arm and enabled for a qualified targeted-check arm, but both should be judged under the same claim authority rules. Dynamic tests/reproductions only after #10064 qualifies isolation.

### Required controls

- Clean/no-known-defect PRs to measure false-positive noise.
- Fixed/repaired companions to check whether target-specific causal claims disappear.
- Neighboring-mechanism controls in the same file/function.
- Same-line but wrong-mechanism controls.
- Non-local multi-file cases with cross-file invariants.
- Stale revision, wrong receipt subject, skipped test and infrastructure-error controls.
- Valid alternative implementations and newly discovered non-target defects.
- High-severity and security-focused subsets with adequate sample counts.
- Human audit of accepted, rejected and unresolved judge/matcher cases, stratified by model family and severity.

A fixed companion is a target-specific negative control; it does not prove the entire PR is safe. A judge's high inter-rater agreement does not prove unbiased accuracy. Report both agreement and a representative human-audit error estimate with uncertainty.

## 6. Prospective targets, matched to CEB rather than inflated from raw leaderboards

The current matrix's prospective CEB-REVIEW pilot gates remain the governing targets:
- `defect_recall_at_5 >= 0.85` on a sealed, repository/defect-family-disjoint holdout.
- `actionable_precision >= 0.80` under blinded human adjudication.
- `reviewer_burden` at least 20% lower at non-inferior defect recall.

To establish a meaningful industry improvement, require paired, same-task comparisons against a strong reviewer and human baseline:
- At matched compute + tool + judge + human cost, target at least **+5 percentage points** Recall@5 or actionable precision without degradation of the other primary metric.
- Alternatively, show **at least 20% lower total review cost** at non-inferior Recall@5 and actionable precision.
- On a matched SecPRBench-style security subset, target **≥85% causal target diagnosis recall**, with no critical-severity false accept in the evaluated sample; this is an application-specific prospective research gate, not a new CEB primary metric.
- On a CR-Bench-v2-style known-target subset, target **≥45% KDDR** while maintaining **≥90% comment usefulness** under the same frozen judge protocol and separately human-audited matcher; report cost/latency.
- On ReviewBench, any matched replication must beat the contemporaneous high-severity grounded-F1 leader captured at evaluation time. The latest 20:14 SAST capture on November 7 lists intent-review-agent at **82.2 F1 / 92.0 precision / 74.4 recall**; the earlier 83.0 OhMyBug row remains a prior same-day observation. Re-capture the board immediately before comparison and freeze the exact filter, run, dataset, judge and matcher versions. This external target is meaningful only on that benchmark's own slice and is not itself the CEB target.

These are prospective aspirations, not achievements. Do not pick thresholds after inspecting a holdout. The ≥85% security-diagnosis, ≥45% KDDR and ≥5-point gains require preregistration, power analysis, paired confidence intervals and an independent evaluation sample.

## 7. How to improve beyond the current industry frontier

1. **Mechanism-first explanations:** Require a minimal causal chain, with the source and counterevidence for each link. The aim is not chain-of-thought disclosure; it is a verifiable, concise diagnosis artifact.
2. **Causal counterexample search:** For a proposed diagnosis, search for one neighboring mechanism, one counterexample and one changed precondition that would invalidate the claim. Use bounded source-level reasoning until execution is safe.
3. **Evidence-directed tool selection:** Choose a compiler/test/reproduction only if it discriminates a named competing hypothesis. Record no-op and irrelevant checks as cost, not quality.
4. **Cross-file reasoning:** Stress test data/control-flow edges spanning modules, dependencies, platform/feature configurations and security boundaries. SecPRBench's multi-root-cause performance gap makes this a priority.
5. **Dual-view gold truth:** Preserve the frozen target defect as the strict comparison, but separately adjudicate genuine new findings against an independent multi-source process. Never let a candidate generate its own gold labels.
6. **Evaluator independence:** Separate defect provenance, finding-to-defect matcher, comment-utility judge and independent human audit. Track model-family/prompt/data dependence; keep per-judge scores and unresolved adjudications.
7. **Human utility and cost:** Measure comments read, time to adjudicate, accepted/rejected findings, edits prompted, false alarms, and whether the review changes a sound engineering decision. Report model, tool, judge and human effort on the same ledger.
8. **Robustness over time:** Bind every finding to a source revision and re-evaluate the claim after relevant source, dependency, policy and configuration changes using the existing continuity lane #10630/#10631.

### The core paradigm shift

Move from **“Did the model point at the bug?”** to **“Can the reviewer discover, explain, evidence, challenge and communicate the defect accurately at the right cost?”** This is a more meaningful engineering standard than optimizing comment count, lexical overlap, pass rate or a single composite leaderboard score.

A further step is to ask **“Which causal conclusions are identifiable from the evidence actually available?”** A third extension asks **“Can every claim be selectively retained only when its full support closure remains valid?”** Bind each claim to supporting and counter-evidence, assumptions, contract/policy, source/configuration, method/oracle and authority dependencies. A retain decision needs an unaffectedness witness; missing edges, shared dependencies or circular support cannot be silently treated as proof of non-impact. Require a compatible-model set, claim-specific identifiability disposition and minimum discriminating evidence request. This prevents a coherent but underdetermined graph from being rewarded as a proven cause. The three development levels are I0 unstructured diagnosis, I1 causal topology, and I2 topology plus identifiability. I2 is exploratory and does not change the registered Phase A/B/C hypotheses. Also record the claim kind: source structural possibility, build-world selection, execution realization, actual-cause attribution, interventional-effect identifiability, or incident attribution. These distinctions prevent one evidence type from silently proving a stronger claim. A further extension asks whether the proposed repair preserves canonical behavior. Require both a target witness and a preservation witness, and compare the original and proposed patch against an independently frozen requirement/oracle. This addresses documented over-rejection of correct code under richer prompts; do not treat the candidate's own generated tests as an independent oracle.

### Preservation-aware repair verification

A review finding must not only identify a target violation; a proposed correction must preserve the nearest valid neighboring behavior. Require a **target witness** and a **preservation witness**, both grounded in the same frozen requirement, plus independent target and preservation oracle plans. Compare three development conditions: P0 verdict only, P1 verdict plus rationale/patch, and P2 dual-witness review. The public six-case preservation challenge is design-only; no candidate code is executed. A patch that passes only the target oracle is not a verified repair, and a correct canonical implementation must not be rejected simply because an agent can imagine a stricter alternative contract.

### Calibrated acceptance and abstention

Use disjoint calibration material to estimate selective risk among accepted claims, then measure risk versus coverage by claim type and severity. Calibrate repository-/defect-family groups separately or report when the exchangeability assumption is implausible. An uncalibrated confidence number, agent consensus, or persuasive explanation is not a guarantee. The SCOPE results are a methodological reference for selective judging, not evidence that its FDR bounds transfer directly to code review.

### Causal topology as the next research extension

A five-link chain remains the minimum explanation. For multi-path, stateful and concurrent defects, compare that baseline with a typed graph that represents AND-gated interactions, OR alternative routes, temporal partial orders and residual path reachability after intervention. The graph is a falsifiable mechanism contract, not a decorative dependency map. See [CEB Review Causal-Topology Protocol V0](CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md) and its six public synthetic development cases.


### Prospective CEB benchmark targets — not achieved results

Use these as a staged target ladder. They are aspirations for future preregistered, matched studies, not present Symthaea performance or safety claims.

| Capability | Prospective target | Correct industry comparison | Qualification condition |
|---|---|---|---|
| **CEB-REVIEW core** | defect Recall@5 ≥ 0.85; actionable precision ≥ 0.80; reviewer burden ≥ 20% lower at non-inferior recall | ReviewBench grounded high-severity F1/precision/recall, plus CR-Bench-v2 KDDR and Usefulness Rate on separate matched subsets | Preserve the three existing CEB primary measures; common tasks, denominators, model/harness/tool budgets and independent adjudication |
| **Matched industry improvement** | +5 percentage points on a compatible quality measure at equal all-in cost, or ≥20% lower all-in cost at non-inferior quality | ReviewBench/CR-Bench-v2/SecPRBench, only on genuinely compatible task slices | Paired repository-clustered inference; do not translate unmatched benchmark scores |
| **Security causal diagnosis** | ≥85% correct target-mechanism diagnosis on a matched security slice | SecPRBench direct-review top reported recall 78.1% (265 tasks); its 68% wrong-mechanism/file-localization result is the diagnostic target | Same introduction-state, source evidence, judge protocol, failure denominator and budget; 85% is an aspiration, not a claimed result |
| **Known-defect detection + comment utility** | KDDR ≥45% while usefulness ≥90% on a matched CR-Bench-style slice | CR-Bench-v2 verified slice: best reported KDDR ≈33.7%, usefulness ≈72.6% for GPT-5-mini Reflexion in the cited comparison | Preserve the single-known-defect denominator and use independent judge/human audit; KDDR is not defect Recall@5 |
| **Preservation-aware repair** | ≥95% canonical acceptance; ≥95% target-mutant detection; ≥90% of repair-bearing findings carry valid target and preservation witnesses; zero critical over-repair events in the adjudicated critical slice | Jin & Chen's paired canonical/buggy conformance study and Fix-guided Verification Filter | Diagnostic targets only, not new CEB primary metrics; independent contract/oracles and repository-/defect-family-disjoint cases |
| **Risk-controlled acceptance** | one-sided 95% upper FDR bound ≤5% for ordinary accepted claims and ≤1% for critical claims; ≥70% coverage on eligible non-critical claims after risk qualification | SCOPE as selective-judging methodology only; no direct code-review score equivalence | Disjoint development/calibration/confirmatory sets, cluster/exchangeability checks, absolute counts; at zero observed errors the planning examples require 59 and 299 independent accepted calibration claims respectively |
| **CEB-CONTINUITY** | ≥20% lower revalidation cost than reset-and-recheck at non-inferior sequence correctness; zero stale-evidence use on the predeclared high-severity controls | SWE-Milestone, SWE-CI, ASE incremental program analysis, DiffCI change-aware test selection and Bazel/Skyframe | Use the existing seven CEB-CONTINUITY primary measures; full-cost accounting, clean-recomputation oracle and sequence/repository clustering |
| **Claim-support closure** | C2 (typed support closure) beats both C0 path overlap and C1 existing continuity dispositions on false-retain/stale-evidence safety without increasing false invalidation; incremental updates agree with clean recomputation on sampled sequences | Bazel reverse-transitive invalidation/change pruning, ASE IPA and DiffCI as contextual analogues | Do not add a metric lane; score with existing CEB-CONTINUITY metrics and secondary closure/witness annotations |

**Interpretation:** CEB should aim to beat the strongest compatible baseline, not the largest number visible on any nearby leaderboard. A 90% source-only diagnosis rate cannot be compared with 78.1% execution-validated security review; 95% canonical acceptance cannot be averaged with KDDR; and FDR/coverage cannot be folded into reviewer precision. Publish a Pareto view across quality, safety, coverage, cost, latency and human burden, plus every failed/unresolved denominator.


## 8. Reuse and integration boundaries

- #10419 continues to own the CEB-REVIEW capability/task profile and primary metric contract.
- #10723 owns disconfirm-first, human-agent complementarity, and the review comparison experiment.
- #10367 owns the causal debugging ledger and broader causal-diagnosis integration.
- Draft #10987 operationalizes dual-reviewer annotation; the causal-topology development tranche is stacked on that reviewer-calibration work and does not introduce a second annotation/receipt system.
- #9431 owns dependence-aware judge ensembles/correlated error.
- #10680/#10681 own verifier-of-verifiers qualification.
- #10619/#10624 own statistical claim qualification.
- #10630/#10631 own evidence continuity across source/repository evolution.
- #10064 remains the candidate-controlled execution-isolation gate.
- #10369 remains the independent signer trust and pristine-regrade gate.

No new metric lane, task schema, receipt, oracle authority, generic scheduler, judge ensemble, admission registry or hidden-test store is introduced.

## 9. Runner-independent work now permitted

1. Add the diagnosis ladder and failure taxonomy to reviewer annotation guidance.
2. Extend the public synthetic evidence challenge with same-file/wrong-mechanism, fixed-companion, and valid-non-target finding cases.
3. Create a comparator correspondence note with exact denominators, task scope, model+harness, judge version, budgets, and release dates.
4. Write a blinded audit sampling plan for accepted/rejected/unresolved matcher cases.
5. Draft matched-budget experiment arms and human-time accounting without running candidates.
6. Audit public dataset reuse/licensing and contamination risk before any import.
7. Reconcile the prospective target language with the CEB matrix owner before registering any new metric.

No public scenario here is a sealed holdout task. No candidate-controlled code is run. No hidden labels, reference patches, executable graders or SOTA result are created. The new identifiability protocol and six public synthetic cases are development-only; they do not demonstrate calibrated abstention or causal identification.

## 10. Sources — current snapshot 2026-11-07

- GitHub, [ReviewBench: An open benchmark for AI code review](https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/), October 5, 2026; [live high-severity grounded-F1 leaderboard](https://review-bench.ai/), viewed November 7, 2026.
- Yang et al., [Can Agentic Code Review Catch Vulnerabilities Before They Are Merged? (SecPRBench)](https://arxiv.org/abs/2610.33509), submitted October 24, 2026.
- Pereira et al., [CR-Bench-v2: Evaluating the Real-World Utility of AI Code Review Agents](https://arxiv.org/abs/2611.02566), submitted November 2, revised November 3, 2026.
- Lieret et al., [SWE-sweep: Can Agents Autonomously Find and Fix Bugs?](https://swesweep.com/paper), October 1, 2026; [live leaderboard](https://swesweep.com/) viewed November 7, 2026.
- Townsend, [LoopControlBench](https://github.com/kltownsend/loopcontrolbench), live repository/readme viewed November 7, 2026.
- Garg, Steenhoek & Huang, [Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation](https://arxiv.org/abs/2510.08996), submitted October 10, 2025, revised January 23, 2026; accepted to CAIN 2026.
- Yang et al., [CRBugBench](https://arxiv.org/abs/2610.13668), October 9, 2026; [Microsoft Research overview](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/).
- Arsentev, [PREreview of CRBugBench](https://prereview.org/reviews/23335998), October 13, 2026.
- Kodus, [CodeReviewBench leaderboard](https://www.codereviewbench.com/leaderboard), [DeepSeek V4 Pro scorecard](https://www.codereviewbench.com/model/deepseek-v4-pro), and [method/limitations post](https://www.codereviewbench.com/blog/ai-code-review-recall-2026), data generated 2026-09-11; accessed November 7, 2026.
- Martian, [Code Review Benchmark production-adoption dashboard](https://www.codereviewbenchmark.com/), accessed November 7, 2026.
- Kodus, [State of AI Code Review 2026](https://kodus.io/data), production pipeline snapshot through June 8, 2026; accessed November 7, 2026.

- Sun et al., [SDRCA: software defect root-cause analysis with causal graphs and interventions](https://doi.org/10.1016/j.jss.2026.112931), Journal of Systems and Software, 2026.
- UniVal authors, [Improving Fault Localization by Integrating Value- and Predicate-Based Causal Inference Techniques](https://arxiv.org/abs/2102.06292), ICSE 2021 Distinguished Paper.
- Alibaba, [AACR-Bench](https://arxiv.org/abs/2601.19494) and [benchmark artifacts](https://github.com/alibaba/aacr-bench), 2026.
- Navaratnarajah et al., [Causality-Based Fault Localization / SoftReX](https://arxiv.org/abs/2611.01292), submitted October 31, 2026.
- Wu et al., [CausalRepair: Bridging the Causality Gap in LLM-Based APR via Dual-Slicing](https://arxiv.org/abs/2608.10613), submitted August 11, 2026; [ISSTA 2026 research-program entry](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/134/CausalRepair-Bridging-the-Causality-Gap-in-Large-Language-Model-based-Automated-Prog).
- Orujlu et al., [Partially Observed Structural Causal Models](https://proceedings.mlr.press/v337/orujlu26a.html), UAI 2026, PMLR 337.
- Yuan et al., [CGARF: a causality-guided framework for reliable automated program repair](https://doi.org/10.1007/s10664-026-10940-1), published August 22, 2026.
- Shen et al., [SecRepoBench](https://research.google/pubs/secrepobench-benchmarking-code-agents-for-secure-code-completion-in-real-world-repositories/), 2025; [leaderboard and methodology](https://secrepobench.github.io/), accessed November 7, 2026.

- Jin and Chen, [Are LLMs reliable code reviewers? systematic overcorrection in requirement conformance judgement](https://doi.org/10.1007/s10515-026-00638-5), Automated Software Engineering 33, article 90 (June 26, 2026).
- Li et al., [ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents](https://aclanthology.org/2026.acl-long.1477/), ACL 2026.
- Badshah, Emami and Sajjad, [SCOPE: Selective Conformal Optimized Pairwise LLM Judging](https://hypermatrix.cs.dal.ca/selective-prediction/), ACL 2026; research summary accessed November 8, 2026.
- Ashby, Nguyen and Luo, [Uncertainty-aware large language models: a scoping review of conformal prediction methods](https://doi.org/10.1098/rsta.2026.0061), Philosophical Transactions of the Royal Society A (August 27, 2026).

- Bazel team, [Skyframe: Incremental build evaluation](https://bazel.googlesource.com/bazel/+/refs/heads/master/site/en/reference/skyframe.md), accessed November 8, 2026.
- SLSA, [v1.1 specification](https://slsa.dev/spec/v1.1/), including [provenance](https://slsa.dev/spec/v1.1/provenance) and [artifact verification](https://slsa.dev/spec/v1.1/verifying-artifacts), accessed November 8, 2026.
- in-toto, [Attestation Framework Specification v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md), accessed November 8, 2026.

- Wang et al., [Incremental Program Analysis in the Wild: An Empirical Study on Real-World Program Changes](https://doi.org/10.1109/ASE63991.2025.00194), ASE 2025.
- DiffCI, [DiffCI Open Evidence Study 2026](https://diffci.com/research/diffci-open-evidence-2026), published September 3, updated September 25, 2026; self-published open study, accessed November 8, 2026.

**Cross-benchmark claim rule:** never rank a system by combining these numbers. Preserve each benchmark's task denominator, single-/multi-defect design, language/repository mix, review budget, tool/harness, judge/matcher, severity slice and release version. A CEB-versus-industry superiority claim requires a paired, protocol-matched evaluation and independent audit.
