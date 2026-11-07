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
- On ReviewBench, any matched replication must beat the contemporaneous **83.0 grounded-F1 / 92.0 precision / 75.6 recall** top row on its exact high-severity slice, or demonstrate a favorable Pareto trade-off. This external target is meaningful only under its own dataset/judge/matcher version and is not itself the CEB target.

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

## 8. Reuse and integration boundaries

- #10419 continues to own the CEB-REVIEW capability/task profile and primary metric contract.
- #10723 owns disconfirm-first, human-agent complementarity, and the review comparison experiment.
- #10367 owns the causal debugging ledger and broader causal-diagnosis integration.
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

No public scenario here is a sealed holdout task. No candidate-controlled code is run. No hidden labels, reference patches, executable graders or SOTA result are created.

## 10. Sources — current snapshot 2026-11-07

- GitHub, [ReviewBench: An open benchmark for AI code review](https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/), October 5, 2026; [live high-severity grounded-F1 leaderboard](https://review-bench.ai/), viewed November 7, 2026.
- Yang et al., [Can Agentic Code Review Catch Vulnerabilities Before They Are Merged? (SecPRBench)](https://arxiv.org/abs/2610.33509), submitted October 24, 2026.
- Pereira et al., [CR-Bench-v2: Evaluating the Real-World Utility of AI Code Review Agents](https://arxiv.org/abs/2611.02566), submitted November 2, revised November 3, 2026.
- Yang et al., [CRBugBench](https://arxiv.org/abs/2610.13668), October 9, 2026; [Microsoft Research overview](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/).
- Arsentev, [PREreview of CRBugBench](https://prereview.org/reviews/23335998), October 13, 2026.
- Kodus, [CodeReviewBench leaderboard](https://www.codereviewbench.com/leaderboard), [DeepSeek V4 Pro scorecard](https://www.codereviewbench.com/model/deepseek-v4-pro), and [method/limitations post](https://www.codereviewbench.com/blog/ai-code-review-recall-2026), data generated 2026-09-11; accessed November 7, 2026.
- Martian, [Code Review Benchmark production-adoption dashboard](https://www.codereviewbenchmark.com/), accessed November 7, 2026.

**Cross-benchmark claim rule:** never rank a system by combining these numbers. Preserve each benchmark's task denominator, single-/multi-defect design, language/repository mix, review budget, tool/harness, judge/matcher, severity slice and release version. A CEB-versus-industry superiority claim requires a paired, protocol-matched evaluation and independent audit.
