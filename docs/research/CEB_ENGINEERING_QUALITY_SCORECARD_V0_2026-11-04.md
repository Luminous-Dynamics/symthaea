# CEB Engineering Quality Scorecard V0 — 2026-11-04

**Status:** research addendum / design-only  
**Related:** [CEB-CAUSE-001 #10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366), [CEB-DEBUG-QUALITY-001 #10385](https://github.com/Luminous-Dynamics/symthaea/issues/10385), [CEB-CAUSE-INTEGRATION-001 #10387](https://github.com/Luminous-Dynamics/symthaea/issues/10387), [protocol PR #10367](https://github.com/Luminous-Dynamics/symthaea/pull/10367)  
**Claim ceiling:** this is a benchmark portfolio and experimental design. It is not a CEB score, an implementation claim, or evidence that Symthaea outperforms any published system.

## 1. Strategic correction: do not optimize one coding score

The current frontier makes a single "coding success rate" an inadequate research target.

- SWE-Bench Pro V2 full is near saturation on its refreshed 642-task public split: Scale reports Opus 5 + Claude Code at 99.40 ± 0.40, Gemini 4 Argon + mini-SWE-agent at 99.22 ± 0.45, Fable 5.1 at 99.10 ± 0.50, and GPT-6 Astra + Codex at 96.90 ± 1.10. The September 22 refresh removed 89 invalid tasks from 731 and added a pristine-image regrade plus a reference-pass / empty-patch-fail admission gate. It still has residual trusted-relay and verifier-execution risks. Public leaderboard success does not establish private-repository generalization.
- SWE-Bench Pro V2 HARD is a separate 51-task subset with a much wider spread: the November 2 snapshot reports Opus 5 at 98.0, Fable 5.1 at 92.2, Gemini 4 Argon at 92.16 and GPT-6 Astra at 90.2. Keep its sample size and uncertainty visible; it is a useful discriminative slice, not a replacement for private holdouts.
- SWE Atlas - Codebase QnA measures deep repository comprehension before editing: 124 tasks across 11 production repositories and four languages. Its published leader table reports Opus 5 at 63.17 ± 5.01, Fable 5.1 at 59.95 ± 4.85 and GPT-6 Astra at 59.14 ± 4.88. The page also notes only about 30% pass at its strictest rubric threshold. This is a separate scorecard from patch resolution.
- SWE Atlas - Test Writing evaluates production-grade test authoring as a distinct capability. The Scale agentic dashboard snapshot accessed November 4, 2026 lists Fable 5.1 + Claude Code at 67.04 ± 5.33, Opus 5 + Claude Code at 62.22 ± 5.58 and Fable 5 at 55.60 ± 5.80. CEB should not reward test count alone: score mutation sensitivity, assertion specificity, false-positive resistance, test independence and whether tests detect the intended fault without being tailored to the reference patch.
- SWE Atlas - Refactoring uses 70 tasks across 10 production repositories and six languages. It covers decomposition, interface evolution, extraction and relocation. The current Scale agentic dashboard snapshot lists GPT-6 Astra (Codex) at 59.05 ± 6.43, Fable 5.1 (Claude Code) at 56.67 ± 6.52, and Fable 5 at 54.76 ± 6.76. The detailed benchmark page still contains older narrative text reporting Opus 4.7 at 48.57 and saying all leaders are below 50%; treat the dashboard as the newer displayed score snapshot, but require exact evaluator/harness revision before a direct numerical comparison. The task design remains larger than SWE-Bench Pro (about 2x changed lines and 1.7x files) and grades preserved behavior plus maintainability, cleanup and anti-pattern rubrics.
- SWE-fficiency has 498 repository-level performance tasks across nine data-science, ML and HPC repositories. Published leading agents average less than 0.23x validated expert speedup. This exposes a large optimization gap even when code generation and issue resolution are strong.
- Precise Debugging Benchmark (PDB) separates bug-level recall from edit-level precision. Its April 2026 paper's initial evaluated systems included >76% unit-test pass rates but <45% edit precision; that is a historical finding, not today's best published PDB result. The later public PDB-Results table reports Claude Opus 4.7 at 83.3 ± 0.8 precision / 89.3 ± 0.6 recall on PDB-Single Hard (5,751 tasks), and 77.8 ± 3.1 / 83.4 ± 2.8 on PDB-Wild (484 mixed tasks). Gemini 3.1 Pro reports 82.5 ± 0.8 / 89.4 ± 0.7 and 77.8 ± 3.0 / 85.6 ± 2.8 respectively. Keep the sets separate; PDB uses epsilon-relaxed edit matching. Source: [PDB-Results](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results). Passing tests can still conceal broad, unnecessary or collateral edits.
- HiL-Bench contains 300 tasks across SWE and text-to-SQL, with 1,131 blockers and a 200-public / 100-held-out split. It reports ASK-F1 (question precision × blocker recall harmonic mean) and Pass@3. The explicit takeaway still cites an older 24% combined Pass@3 and highest ASK-F1 of 44%, compared with 75–91% Pass@3 when all information is provided upfront. The live leaderboard now displays Muse Wonder 1.0 at 79.67±4.17 but does not label the numeric field in the table text; do not call it ASK-F1 or Pass@3 until the benchmark owner clarifies. The gap between defined help-seeking metrics and actual task completion isolates selective-escalation judgment rather than raw task capability.
- AgentAbstain contains 263 near-identical act/abstain pairs in 42 environments and eight abstention scenarios. Its best reported paired accuracy is 59.5%. Pairing prevents always-act or always-refuse strategies from gaming the benchmark.
- SRE-Bench is a distinct source-free reverse-engineering track: 262 binary instances from 19 private programs and 1,572 deterministic task grades. The benchmark page updated October 2026 reports GPT-6-Astra fully solving 82.3% of graded instances in the public-facing setting, but only 181/262 instances were graded; in the unconstrained setting it reports 88.0% pass@1 and 99.2% pass@4. The strong conditional result must be reported beside its coverage and refusal denominator, not as an unconditional 262-task solve rate.
- Terminal-Bench 4.0 remains diagnostic only: Epoch AI labels the release flawed and reports publicly noted scoring defects in 30/66 tasks. Vals AI's October 29 Mini-SWE-agent results are third-party measurements, not official maintainer leaderboard results.
- CursorBench 4.0 measures ambiguous multi-file tasks from actual Cursor sessions. Its current table reports Opus 5.5 Max at 57.8%, $13.43, 218,363 tokens and 185 steps per task. Retain cost and trajectory measures, not just the score.
- CRBugBench (October 2026) measures human-verified bugs in incoming PRs: 1,000 tasks across 98 OSS repositories and 800 private-industry tasks. Claude Opus 5 + Copilot CLI catches 71.9% of bugs within its top five comments. Detection falls on complex changes and non-local bugs, and drops sharply on proprietary code; prompt nudging improves up to 5.8% at 10% extra cost. Add review quality as a distinct capability, not a proxy for patch success.
- GitHub, ReviewBench: An open benchmark for AI code review, October 5, 2026: https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/; live leaderboard: https://review-bench.ai/ (viewed November 7, 2026).
- Yang et al., SecPRBench, October 24, 2026: https://arxiv.org/abs/2610.33509.
- Kodus, CodeReviewBench leaderboard/DeepSeek V4 Pro scorecard (30 PRs / 95 human-authored bugs; data generated 2026-09-11): https://www.codereviewbench.com/leaderboard; https://www.codereviewbench.com/model/deepseek-v4-pro.
- Pereira et al., CR-Bench-v2, November 2, revised November 3, 2026: https://arxiv.org/abs/2611.02566.
- Agent Security League measures both functional and security correctness on 200 tasks from 108 Python projects across 77 CWE classes, with 34 independently evaluated agent/model combinations. The live table shows Claude Code + Fable 5.1 at 86.6% Functional / 36.9% Secure, Codex + GPT-6 Astra at 82.1% / 34.6%, and Codex + GPT-6.1 Sol at 77.7% / 34.1%. Muse Code + Muse Wonder leads Functional at 91.6% but has 25.1% Secure. A task that works but is insecure must not be accepted as a successful engineering outcome.
- Rust-SWE-bench is a dedicated Rust repository-level repair benchmark: 500 tasks from 34 repositories. Its February 2026 paper reports RUSTFORGER at 28.6% using Claude Sonnet 3.7, versus up to 21.2% for the studied ReAct baselines. The result is a valuable Rust-specific reference but predates current frontier models/harnesses and requires matched replication.
- ContextBench (leaderboard updated October 23, 2026) includes 1,136 tasks across 66 repositories and eight languages, including 63 Rust tasks. CodeNib leads Recall at 0.779, Pass@1 77.6%, Context F1 0.281 and $0.68 average cost; openJiuwen-JiuwenSwarm + DeepSeek-V4-Pro has the best Context F1 at 0.389, with Recall 0.411 and Pass@1 60.8%. Best recall and best F1 are not the same row. This creates a measurable retrieval-quality lane for the repository twin.

**Decision:** maintain a vector of independent capabilities. Do not average these benchmarks into an omnibus "engineering intelligence" score. They measure different constructs, use different task populations, harnesses, graders and resource budgets.

## 2. CEB capability portfolio

| Capability | External reference | CEB's independent measurement |
|---|---|---|
| Issue resolution | SWE-Bench Pro V2 Full / HARD; SWE-rebench | Independently accepted repair on fresh, sealed repositories |
| Repository comprehension | SWE Atlas Codebase QnA | Typed graph, runtime trace explanation, fact provenance, uncertainty and retrieval |
| Refactoring quality | SWE Atlas Refactoring | Behavior-preserving decomposition, extraction, relocation and interface evolution |
| Test-oracle quality | SWE Atlas Test Writing; PDB | Mutation sensitivity, independent oracle validity, bug recall and edit precision |
| Causal debugging | PDB + CEB-CAUSE | Hypothesis calibration, predicted intervention outcomes, experiment utility and verified repair |
| Performance engineering | SWE-fficiency | Candidate/expert speedup, semantic equivalence, resource/latency trade-offs |
| Help-seeking | HiL-Bench | Question precision, blocker recall, ask timing and successful conversion of answers into a solution |
| Safe abstention | AgentAbstain | Paired act/abstain accuracy, selective risk and false-action rate |
| Reverse engineering | SRE-Bench | Source-free semantic recovery and behavior prediction with explicit task coverage |
| Evaluator integrity | Epoch AI benchmark review / CEB-VER | False accepts, false rejects, leakage, mutation resistance and independent regrading |
| Pre-merge review | CRBugBench | Human-verified bug Recall@1/3/5, actionable precision, severity and reviewer burden |
| Secure code changes | Agent Security League | Functional and secure pass rates, CWE strata and high-severity false accepts |
| Rust-specific engineering | Rust-SWE-bench | Reproduction, trait/type correctness, compile/repair and accepted issue resolution |
| Context retrieval | ContextBench | Context Recall@k, Precision@k, F1, sufficiency, tokens, cost and downstream solve |

Use released datasets and adapters only after license, terms, runner contracts, hidden-label custody and evaluator semantics are reviewed. A local synthetic pilot is not a reproduction of the published benchmark.

## 3. Paradigm shifts

### 3.1 From "tests pass" to a repair-quality envelope

Require every repair to satisfy all of:
- behavior contract / hidden acceptance;
- protected invariant preservation;
- relevant mutation controls;
- no new regressions;
- edit-scope justification;
- independently reproduced result.

For controlled injected bugs, report PDB-style bug recall and edit precision against an independently curated *admissible repair set*. For natural repository tasks, do not equate textual overlap with a reference patch to correctness; judge behavior, invariants and collateral effects.

A tiny patch that leaves the defect unresolved must not score well. A semantically necessary refactor must not be penalized just because it differs from one reference diff.

### 3.2 From "ask human" as a fallback to active information acquisition

A computational engineer should distinguish:
- information discoverable from source, tools, traces or controlled experiments;
- information that is missing but inferable only after a cheaper probe;
- information that exists only with a human or authority-holder;
- information that is contradictory or not safely resolvable.

Reuse the HiL-Bench and AgentAbstain pair design principles. A good question must target a real blocker; question spam must be penalized. A good abstention must withhold the side effect and communicate the missing information / approval.

### 3.3 From optimization by intuition to expert-normalized improvement

Compare candidate speedup to a validated expert baseline under the same workload and environment:

candidate_speedup / expert_speedup

Keep absolute performance, baseline variance, semantic equivalence, resource consumption and tail latency visible. Use paired randomized runs, fixed warm-up, stable hardware and noise thresholds. Do not claim an optimization from a single run.

### 3.4 From repository index to executable system comprehension

SWE Atlas QnA shows that repository understanding is a separate bottleneck from patch generation. Symthaea's repository twin should therefore answer questions with source/compiler/runtime evidence, trace the executed path, predict consequences and identify unknowns. A correct answer without evidence provenance is not sufficient for the internal CEB scorecard.

### 3.6 From patch authoring to independent pre-merge review

The review problem is not the inverse of coding. A reviewer must identify a real defect, localize it, establish why it matters, and avoid noisy speculative comments. CRBugBench's top-5 bug coverage at 71.9% shows why comment volume is not enough. CEB-REVIEW should report Recall@1/3/5, actionable precision, severity-weighted recall, non-local dependency coverage and human review minutes per confirmed defect. A finding without a reproduced or source-grounded violation should not be counted as a true positive.

**Human–agent complementarity experiment:** do not assume that the winning architecture is an autonomous “lights-off” reviewer. Compare human-only, agent-only and human+agent review under blinded, paired tasks and a fixed total effort budget. The assisted arm should measure incremental confirmed-defect recall, severity-weighted recall, false-positive review minutes, reviewer override rate, time-to-detection and total cost. A prospective success condition is either at least +5 percentage points of confirmed-defect recall over the better single-agent arm at equivalent total effort, or non-inferior recall with at least 20% lower human review time; both require a separate high-severity false-negative ceiling. These are CEB goals, not CRBugBench-reported outcomes.

**Disconfirm-first review extension (2026-11-06):** add the experiment in [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723), building on CRBugBench and its October 13 PREreview. Keep three second-pass strategies distinct: (1) ordinary self-refinement, (2) a blind independent second reviewer who does not see the first review, and (3) a disconfirm-first reviewer who sees the first findings but must challenge their evidence, try to refute each claim, and search independently for high-severity omissions. A blind reviewer tests unanchored independent discovery; a disconfirm-first reviewer tests adversarial critique. Do not call them equivalent independent votes. Compare human-only, single-agent, human+agent, self-refinement, multi-agent consolidation and both reviewer variants under equal total effort/cost. Include clean/no-known-defect PR controls. Distinguish defect-level recall (one true defect counted once) from finding-level actionable precision, location quality and false-positive comments per 1,000 changed lines. CRBugBench's reported human/agent/combined values of 50.0% / 71.9% / 82.1% are summarized by PREreview, not a CEB measurement; the proposed disconfirm-first method is a reviewer hypothesis, not a reported result. The PREreview also describes three LLM judges matching submitted findings to known defects; that match layer needs its own audited sample against blinded expert adjudication. Reuse #9431 for correlated judge errors and #10680/#10681 for verifier qualification rather than treating majority agreement as truth.

### 3.7 From functional correctness to secure correctness

The Agent Security League's different functional and secure maxima (91.6% versus 36.9%) expose a major evaluation gap. CEB must never combine “works” and “is safe” into one undifferentiated pass. Require security-negative controls, CWE-stratified outcomes, authorization/data-flow invariants and a separate high-severity false-accept ceiling. Preserve functional pass as a separate outcome so a defensive patch that simply disables functionality cannot win.

### 3.8 From generic coding competence to language-specific semantic competence

Rust-SWE-bench shows that Rust repository issue resolution is its own problem: understanding repo-wide structure, type/trait constraints and reproducing the issue are critical. CEB-RUST should evaluate these separately, then assess whether repository-twin context and specification recovery improve the same model/harness on the same pinned tasks. The paper's February 2026 RUSTFORGER result is a historical anchor, not a direct November frontier comparison.

### 3.9 From retrieving more context to retrieving the *right* context

ContextBench reports retrieval recall and task success separately; its highest recall configuration does not also have the highest context F1. The repository twin should therefore be measured on Precision@k, Recall@k, F1, context sufficiency, irrelevant-token ratio, latency and cost. More context is not inherently better. The key experiment is whether typed graph and provenance edges improve useful-context F1 and task success at a fixed token budget.

### 3.5 From one leaderboard to a quality/cost/risk frontier

For each capability, publish:
- correctness / quality;
- coverage and abstention;
- regression and false-accept rates;
- resource cost and latency;
- human review and escalation burden;
- transfer to sealed repositories/fault families;
- assurance tier and limitations of the evaluator.

Do not collapse these into one number. Report Pareto-dominant configurations and the cost of achieving a given quality/risk point.

## 3.10 From code location to causal mechanism diagnosis

The November 2026 review frontier demonstrates that a reviewer can point at the correct file yet describe the wrong defect mechanism. SecPRBench reports that 68% of analyzed misses include a finding in a root-cause file but a rejected explanation of a neighboring mechanism; multi-file causal mechanisms are substantially harder than single-file cases. ReviewBench adds multi-source golden findings and separate grounded versus augmented scoring; CR-Bench-v2 demonstrates that useful comments and known-defect detection can diverge sharply. These are complementary measurements, not interchangeable scores.

**New research requirement:** CEB-REVIEW adjudicates discovery, localization, mechanism, evidence and decision utility separately. A target-defect hit requires the correct causal mechanism, not merely the right file, function, line or vulnerability family. The concise evidence artifact should identify: (1) the triggering precondition/event, (2) the relevant caller/data/control path and state, (3) the unsafe or incorrect operation, (4) the violated requirement/invariant, and (5) the source revision and strongest evidence authority supporting the conclusion. This is a verifiable diagnosis summary, not a request to expose model chain-of-thought.

Add paired controls for same-file/wrong-mechanism, neighboring defects, non-local multi-file causes, stale/repaired companions, valid non-target findings and clean PRs. Preserve unresolved status when the causal chain is not established. A fixed companion is a target-specific negative control, not a proof that the whole patch is safe.

**Do not add a metric lane.** Keep the existing CEB-REVIEW primaries unchanged:
- `defect_recall_at_5`: a finding is credited only when it identifies the target defect's causal mechanism.
- `actionable_precision`: independently adjudicated actionable findings; correct location alone does not qualify.
- `reviewer_burden`: count the full effort needed to understand and disposition findings, including evidence reconstruction and human adjudication.

Supplementary failure labels (discovery miss, localization miss, mechanism misdiagnosis, revision/scope error, authority overclaim, useful non-target finding, unsupported/noisy finding, unresolved adjudication and evaluator mismatch) are annotation aids, not new registered primary metrics.

**Comparator distinctions as of November 7, 2026:**
- ReviewBench high-severity grounded leaderboard: leading public row OhMyBug 83.0 F1 / 92.0 precision / 75.6 recall; it also separates fixed-gold grounded outcomes from augmented discovery findings. Its corpus has 219 PRs / 187 repositories / 19 languages, and GitHub reports 96.6% expert agreement on its gold true/false-positive labels. These are ReviewBench-specific measures.
- SecPRBench: 265 real vulnerability-introducing changes across nine C/C++/Rust systems projects; 217 mechanically validated, 48 human-adjudicated; direct-review target recall 32.5–78.1%. The highest raw recall is not necessarily the best cost point: Grok 4.6 reports 78.1% at estimated $3.20 / 12.6 minutes, while Opus 5.5 reports 74.0% at $0.44 / 1.4 minutes. Its fixed companions do not measure general report precision, and the judge is itself an evaluated model.
- CR-Bench-v2: verified slice 174 Python tasks/11 repositories; best reported known-defect detection is about 33%, with usefulness around 90% for GPT-5.2 Reflexion. This is a single-known-defect-per-instance, review-only comparator.
- CRBugBench: scale comparator (1,000 public tasks/98 repositories plus 800 proprietary); best reported detection is 71.9% in top five findings; not a precision measure.

Sources and full correspondence are recorded in [CEB Review Benchmark Triangulation V0](CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md). For a matched CEB superiority claim, compare the same frozen tasks, language/repository mix, issue/PR snapshots, model+harness, tool permissions, cost budget, grader and metric definition. A 0.85 CEB Recall@5 target is an ambitious CEB-specific sealed-holdout gate, not a conversion from SecPRBench or ReviewBench.

## 3.11 From human-found gold to independently audited open-set discovery

A multi-source gold set improves on a single-source oracle, but it is not exhaustive by declaration. CodeReviewBench measures recovery of 95 human-authored bugs on 30 merged PRs and explicitly excludes defects missed by all original human reviewers. ReviewBench likewise states that a finite producer union cannot reveal findings missed by every producer. Therefore fixed-gold recall is a necessary common comparison, but it does not measure the entire latent defect population.

CEB must separate:
1. **Frozen target set** — independently established defects used for the fixed Recall@5 denominator.
2. **Adjudicated open-set discoveries** — candidate findings independently confirmed after blinded review; report separately and never retroactively add them to the current denominator.
3. **Unobserved remainder** — defects not found by any producer/audit; keep unknown beyond the declared audit sample and uncertainty.

Blindly audit zero-finding PRs and multi-producer "clean" consensus strata. Track producer model/provider/prompt/context dependence, and run leave-one-producer-family-out sensitivity. A candidate's unmatched claim is not gold before independent adjudication. Do not use naïve capture–recapture to estimate the total defect count when heterogeneous producers have shared information and correlated capture probabilities.

This is an extension of the existing CEB-REVIEW / #8311 oracle-quality / #9431 correlated-judge / #10680–#10681 verifier-qualification contracts. It creates no new metric lane, task schema or gold-label authority. See [CEB Review Gold-Completeness and Open-Set Discovery V0](CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md).

## 3.12 From tool use to sequential evidence acquisition

Tool volume is not engineering intelligence. The reviewer should name its current proposition, nearest competing mechanisms, and the smallest safe observation that would discriminate them. Before an action, record applicability to the exact source revision, expected result for each hypothesis, evidence authority, safety/permissions and total expected cost. After the action, bind the result to source/configuration/environment and update the decision.

Compare static-first, unconstrained model-selected tools, a transparent evidence-directed heuristic, a calibrated sequential value-of-information policy and explicit ask/abstain/escalate. Stop when evidence is sufficient, no useful safe action remains, human authority is needed or budget is exhausted. Preserve unresolved, skipped, timeout and infrastructure-error states; do not award points for tool/test count.

Compare quality at matched all-in budget and cost at non-inferior quality, including model, retrieval, compiler/test, judge, human review and retry costs. This is not a new CEB metric lane or a claim that expected information gain is calibrated. See [CEB Review Sequential Evidence Acquisition V0](CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md).
 
## 3.13 From uptake to verified engineering impact

Code-review evaluation now has a useful split between controlled replay and production workflow evidence.

**Kodus CodeReviewBench** uses 30 merged PRs and 95 human-authored bugs from five production repositories. All systems use one frozen-replay Kodus harness and one Claude Haiku 4.5 judge. DeepSeek V4 Pro reports 43.9 F1 / 43.6% precision / 44.2% recall at approximately $0.30 per PR. The 95% PR-bootstrap interval on recall is 32.3–56.3 over 2,000 resamples. Every model receives one agent pass; model run-to-run variance is not measured. Its gold bugs were already surfaced by human reviewers, so this is a recovery-of-human-found-bugs benchmark, not exhaustive defect discovery. Small-sample ranks should be treated as uncertain. Sources: [leaderboard](https://www.codereviewbench.com/leaderboard), [scorecard](https://www.codereviewbench.com/model/deepseek-v4-pro), [method and limitations](https://www.codereviewbench.com/blog/ai-code-review-recall-2026).

**Martian Code Review Benchmark** analyzes production bot comments and developer response/change timelines. Its dashboard reports a recent-period view across 17,768 scored PRs, but system-specific recent samples differ (e.g., Greptile 1,600; Cubic Dev AI 1,525; Qodo 2,435). Its reported F1/precision/recall reflect an LLM-derived production-review adoption analysis, not fixed-gold causal diagnosis. Sources: [dashboard](https://www.codereviewbenchmark.com/), [alternate dashboard](https://codereview.withmartian.com/?fscore_group=day).

**Do not collapse four states:**
1. **Truth:** independently verified defect and causal mechanism on the exact reviewed revision.
2. **Utility:** whether the finding is actionable and useful under the frozen review scope.
3. **Uptake:** whether the developer accepted, rejected, deferred, discussed or independently reproduced the suggestion.
4. **Outcome:** whether a later change prevents recurrence, reduces rework, causes regressions or survives subsequent evolution.

Acceptance is not proof of correctness. Rejection is not proof of a false finding. A later code change may be independently motivated or may respond to several comments. Use adoption as observed process evidence, not an oracle. When attributing an outcome to a review comment, record the temporal link, alternative causes and the comparison/assignment design; do not infer causality from a post-review diff alone.

**CEB measurement mapping:** keep the existing CEB-REVIEW primary measures unchanged—`defect_recall_at_5`, `actionable_precision`, and `reviewer_burden`. Human uptake and downstream impact can be recorded as contextual observations under existing CEB-HUMAN / continuity research and human-review protocols; no new primary metric lane or composite score is introduced. Keep non-response, defer, explicit rejection, accepted suggestion and independently reproduced finding as distinct events. Do not infer preferences or correctness from silence.

**Research requirement:** build the human review calibration exercise before using production uptake to compare systems. Independently label truth/actionability first; then analyze uptake. Preserve all four outcomes separately, including false-but-accepted suggestions and correct-but-rejected findings. The current [CEB Review Human Calibration Pack V0](CEB_REVIEW_HUMAN_CALIBRATION_PACK_V0_2026-11-07.md) provides three source-only causal cases and eighteen shuffled variants for dry-run adjudication; it is public development material, not a sealed benchmark.

## 3.14 From historical fix commits to qualified counterfactual controls

A parent/introduction/fix revision triad is not automatically a controlled causal experiment. Bug-fix commits are frequently tangled: Herbold et al.'s fine-grained study reports that only 17–32% of all changed lines in sampled bug-fix commits directly fix the underlying problem, and estimates 3–47% dataset noise without manual untangling depending on use case. These are study-specific estimates, but they make commit-level causal claims unsafe without change-contribution review. Schulte et al. (2026) further show how process variables and confounders complicate claims about bug-introducing changes, while explicitly distinguishing observed relations from causal inference. [Herbold et al.](https://link.springer.com/article/10.1007/s10664-021-10083-5) · [Schulte et al.](https://link.springer.com/article/10.1007/s10664-026-10822-6).

**CEB shift:** qualify each control for the specific claim. Independently disentangle target-causal hunks from validation, contract context, enabling/follow-on changes, unrelated edits and disputed changes. Check every in-scope alternate path, residual path, causal parent, configuration and environment. “Minimal” means minimal in the causal model, not fewest changed lines. “Irrelevant” must be demonstrated relative to the target mechanism, not inferred from code distance. Historical P/I/F states are observational source snapshots; M/IRR are interventions only to the extent their treatment fidelity is established.

Use six separate support dimensions: exact lineage; change disentanglement; mechanism/reachability; intervention fidelity; environment/applicability; outcome/oracle validity. Do not average them into one qualification score. Source-only evidence supports bounded structural claims; runtime causal claims require qualified execution and exact-bound outcomes. A fixed companion is target-specific, not whole-program safety.

The detailed [CEB Review Counterfactual-Control Qualification V0](CEB_REVIEW_COUNTERFACTUAL_CONTROL_QUALIFICATION_V0_2026-11-07.md) defines the annotations, confounder inventory, mutation-equivalence caveats, stop/go rules and human worksheet. It adds no metric lane, schema or receipt.

## 4. Matched industry comparison design

For every CEB slice, compare:
1. deterministic or conventional tool baseline where applicable;
2. foundation model with minimal fixed harness;
3. the same model with a strong coding-agent harness;
4. the same model plus Symthaea repository twin / causal reasoning / evidence workflow;
5. Symthaea-native generation path where the capability is genuinely implemented without external model fallback.

Pin model revision, harness/scaffold, prompt, tools, task revision, execution image, toolchain, network, budgets and grader. Randomize task order. Test model × scaffold interactions when feasible. Attribute any improvement to the component actually changed.

The unit of efficiency should be **cost per independently accepted engineering outcome**, not cost per attempted patch. Also report human review minutes, experiment count and time-to-correct-result.

## 5. Statistical and benchmark-maintenance protocol

- Use paired task assignments and repeated runs for stochastic configurations. Three independent seeds are the minimum proposal; five are preferred when affordable.
- Report pass@k separately from pass-all-k / reliability. Pass@k measures whether a system can succeed at least once; it is not a reliability claim.
- Use paired confidence intervals and cluster bootstrap by repository and fault family. Variants derived from the same template are correlated; do not count them as independent observations.
- Pre-register primary outcomes, practical margins, exclusions, sample-size reasoning, missingness, stop rules and sealed confirmation data.
- Include positive and negative admission checks: reference solution must pass; empty patch must fail; known-bad, regression-inducing and evaluator-mutating controls must fail.
- Preserve all task dispositions in published denominators: pass, fail, refusal, abstention, unsupported, tool failure, timeout, ungraded and not attempted.
- Review rubric quality for prescriptiveness and inter-rater agreement. A rubric must describe observable behavior, not force an implementation identical to the reference patch.
- Retire saturated tasks and version the evaluator. Preserve old task/grader snapshots for longitudinal comparison; never silently replace them.
- Use human review for ambiguous task validity and grader disputes; do not use an uncalibrated LLM judge as the sole correctness oracle.

## 6. Proposed qualification gates

Prospective internal targets only; not achieved results or industry standards.

| Slice | Initial target | Sealed/transfer target |
|---|---:|---:|
| Natural issue resolution | +5 percentage points over strongest matched baseline | +10 pp over matched baseline |
| Controlled seeded-bug recall | >=0.90 | >=0.95 |
| Controlled edit precision | >=0.80 | >=0.90 |
| Repository graph syntax P/R | >=0.95 | >=0.97 |
| Supported semantic relation P/R | >=0.90 | >=0.95 |
| Causal experiment outcome calibration | Brier/log score baseline published | significant paired gain vs baseline |
| Performance candidate/expert speedup ratio | >=0.50 | >=0.80 |
| Selective escalation ASK-F1 | >=0.65 | >=0.75 |
| Act/abstain paired accuracy | >=0.75 | >=0.85 |
| High-impact false-action rate | <=1% | <=0.5% |
| Calibration ECE | <=0.05 with adequate holdout | <=0.03 |
| Cost per accepted outcome | >=20% reduction | >=30% reduction |
| Published evidence receipts | 100% | 100% |
| Pristine independent regrade | 100% of scored tasks | 100% of scored tasks |
| Code review defect recall@5 | >=0.80 | >=0.85 |
| Human+agent review complementarity | +5 pp confirmed-defect recall or non-inferior recall with >=20% less human time | replicate on sealed reviewers/repos without increased high-severity false negatives |
| Secure task pass | >=0.55 with functional non-inferiority | >=0.65 with functional non-inferiority |
| Context retrieval Recall@10 | >=0.85 | >=0.90 |
| Context retrieval F1 | >=0.45 | >=0.50 |
| Rust-SWE-bench resolution | >30% matched pilot | >35% matched holdout |

These targets must be accompanied by confidence intervals, task counts, scenario-level floors and false-accept / false-reject analysis. No zero-risk claim follows from observing zero failures in a finite control set.

## 7. Immediate work plan

### No-runner tranche
- Update #10366/#10367 with this scorecard and explicit benchmark adapters.
- Define a compact non-executable pair/task authoring template for causal intervention and act/abstain twins.
- Define an admissible-repair annotation guide for controlled PDB-style seeded tasks.
- Define an expert-workload measurement template and metrology checklist for the performance slice.
- Write a preregistration template with paired/clustered analysis and missingness accounting.
- Audit the CEB task catalog so existing Rust/performance/system lanes are extended, not duplicated.

### Isolation-dependent tranche
- Execute candidate code, build scripts, tests and real workloads only after #10064 qualifies the isolation boundary.
- Establish independent trust roots and regrade failure domains under #10369.
- Reconcile generic accounting, task profile and DSSE attestation under #10370 before publishing signed results.

## 8. Sources and dated snapshots

- Scale SWE-Bench Pro V2 (refreshed September 22, 2026): https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full
- SWE-Bench Pro V2 HARD (November 2 snapshot mirror): https://isitsota.com/benchmarks/swe-bench-pro-hard
- Scale SWE Atlas Codebase QnA (updated July 28, 2026): https://labs.scale.com/leaderboard/sweatlas-qna
- Scale Agentic Leaderboard snapshot (accessed November 4, 2026; current SWE Atlas Refactoring / Test Writing / QnA and SWE-Bench Pro V2 rankings): https://labs.scale.com/leaderboard?category=agentic
- Scale SWE Atlas Refactoring detail page (contains older narrative snapshot; retain for task/rubric methodology): https://labs.scale.com/leaderboard/sweatlas-refactoring
- Scale HiL-Bench: https://labs.scale.com/leaderboard/hil
- SWE-fficiency (ICML 2026): https://proceedings.mlr.press/v306/ma26l.html
- Precise Debugging Benchmark (April 2026): https://www.microsoft.com/en-us/research/publication/precise-debugging-benchmark-is-your-model-debugging-or-regenerating/
- AgentAbstain (NeurIPS 2026): https://agentabstain.github.io/
- Columbia DAPLab SRE-Bench (updated October 2026): https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- CRBugBench: Yang et al., October 2026, Microsoft Research: https://arxiv.org/abs/2610.13668
- Agent Security League: Endor Labs: https://www.endorlabs.com/research/ai-code-security-benchmark
- Rust-SWE-bench / RUSTFORGER: Xiang et al., ICSE 2026: https://arxiv.org/abs/2602.22764
- ContextBench: Li et al., 2026: https://contextbench.github.io/
- Terminal-Bench 4.0 Epoch AI review (September 4, 2026): https://epoch.ai/benchmarks/terminal-bench-4/review
- Vals AI Terminal-Bench 4.0 (updated October 29, 2026): https://www.vals.ai/benchmarks/terminal-bench-4
- CursorBench 4.0: https://cursor.com/cursorbench
- NeurIPS Agentic Benchmark Checklist: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f316275b44ee2de533102913828a8107-Abstract-Datasets_and_Benchmarks_Track.html

All leaderboard values are snapshots of specific tasks, dates, model configurations and harnesses. Re-fetch before public comparisons. CEB claims must never be presented as directly comparable unless task contracts, budgets, graders and coverage are matched.
