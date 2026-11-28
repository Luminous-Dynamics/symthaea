# PsychBench Industry Crosswalk and Paradigm Shifts (2026-11)

**Reviewed:** 2026-11-28  
**Status:** research strategy and preregistration proposal; no external leaderboard result is claimed for Symthaea.

## Executive thesis

PsychBench should not compete by accumulating more synthetic tasks or by constructing one omnibus "intelligence score". It should become an **evidence-qualified measurement system for adaptive cognitive agents**.

The defensible contribution is a reproducible chain from construct → task generator → controlled intervention → agent trajectory → independent oracle → uncertainty → cost/risk envelope → bounded claim. The same protocol should run against Symthaea, external agents and human participants where task equivalence is real.

The strategic opportunity is not that current industry benchmarks are useless. It is that the benchmark families measure different slices, often under different harnesses, while deployment-relevant dimensions can be missing from primary scoring. PsychBench can connect these slices without collapsing their meaning.

## 1. Current external benchmark landscape

### 1.1 Cognitive-awareness and human-comparison targets

| External target | What it contributes | Evidence and limitations | PsychBench response |
|---|---|---|---|
| **AwarenessBench (ACL 2026)** | Four dimensions—metacognition, self-awareness, social awareness, situational awareness—decomposed into 15 cognitive functions. | 14,381 samples; 18 language models; human comparison across 36 participants from three backgrounds. The paper reports that aggregate awareness can hide weaker metacognition and self-awareness. The human cohort is valuable but modest and should not be treated as population norms. | Reproduce compatible behavioural tasks; separate self-report from behaviour; publish a vector of function scores and uncertainty rather than one awareness scalar. |
| **HumanBaseline (live 2026 collection)** | A versioned collection spanning 25 benchmark families and 129 items, with published references and an evolving human-response sample. | Methodology v0.2 uses randomized item variants, roughly one-third held-out items, partial credit and IRT; item-level percentile distributions are intended. The site says reference figures are temporary until an item has 30 participant responses. Its current sample is self-selected and likely education/technology-skewed; AI markers often come from lab-reported results under different harnesses and time limits. It is not yet a comprehensive set of population norms. | Adopt randomized forms, held-out variants, item-level response records and distributional human comparisons. Run AI in the same browser/time-limit setting and separately through APIs; do not mix the two. Record sample size, recruitment, version and confidence intervals for every human reference. |
| **MetaBound** | Knowledge-boundary detection on false-premise or unanswerable questions. | HumanBaseline's current mirrored page reports 100% false-premise detection for the best published model versus 0% for the worst; its mirror has seven items, so this is an illustrative task-specific spread, not a stable population ranking. | Test whether an agent knows when a premise is invalid, requests provenance, or declines to invent an answer. |
| **GAUGE** | Metacognitive action: bet or fold after solving, exposing whether confidence changes behaviour. | HumanBaseline reports that its most accurate published model did not fold in 270 problems. Accuracy alone therefore does not establish calibrated decision-making. | Add selective-prediction scoring: answer, abstain, request information, counterfactual correctness, risk/coverage and decision cost. |
| **Turn Bench** | Adaptation after a rule changes without warning. | Current HumanBaseline page reports 80% for the best published re-adaptation result and 42% for Claude Opus 4.6 in the cited comparison; this is task-specific, not a universal executive-function estimate. | Measure change-point detection, adaptation latency, post-shift regret, perseveration and recovery. |
| **LearningBench / EphLangBench** | Learning a hidden grammar/rule or adapting to an unfamiliar language-like system from examples. | HumanBaseline reports a best published LearningBench AI score of 0.85 and 11 of 14 compared models below 0.50; the roughly 60% human reference is drawn from classic artificial-grammar studies, not the same LearningBench administration. | Track learning curves, examples-to-criterion, held-out generalization and transfer—not only final accuracy. |

### 1.2 Interactive and agentic comparators

| Benchmark family | Primary capability | Correct use in PsychBench | Do not infer |
|---|---|---|---|
| **ARC-AGI-3** | Uninstructed exploration, goal inference, world-model acquisition, continuous adaptation and action efficiency in interactive grid worlds. | High-priority ecological comparator. Reproduce exact game/harness versions; include action budgets, human baselines, hidden levels and reasoning/context retention metadata. | The official 2026 competition is complete, but the ARC Prize says final results will be announced on 2026-12-04. On 2026-11-28, do not present public snapshots as final competition results. An independent October review flags score interpretation, version consistency and elicitation problems; report both official protocol and sensitivity analyses. |
| **OSWorld / BrowserGym / WebArena** | GUI and web interaction, tool use, long action sequences and task completion. | Test operational planning, UI state tracking, recovery, provenance and safe tool use through external adapters. | A GUI success rate is not a general cognitive score or a direct human cognitive measure. |
| **GAIA / ToolBench / Terminal-Bench** | Multi-step assistant work, tool selection, information gathering and terminal execution. | Measure planning quality, tool correctness, verification, cost and safe completion. | Tool-use success does not prove metacognition, memory transfer or social understanding. |
| **SWE-bench / PaperBench** | Software-change and research-workflow execution with external outcome checks. | Measure evidence quality, patch integrity, verification, reproducibility and downstream workflow success. | Passing tests alone does not establish code quality, security or maintainability. |

### 1.3 What recent agent-evaluation reviews say

A 2026 systematic review of 15 major agent benchmarks reports, for its selected corpus, that safety/security and cost-efficiency were absent from primary scoring in all 15, while 13/15 relied exclusively on binary success metrics. These are scoped findings about the review's inclusion set, not a universal census of every benchmark.

This is a useful design opening for PsychBench: make safety, trajectory integrity, recovery and measured resource use first-class outputs. A high task score should not compensate for an unauthorized action or an unreplayable evidence record.

### 1.4 Dated industry leaderboard snapshots and evidence quality

These are dated, third-party or provider-reported snapshots as of the review date, not independent head-to-head results by this project.

| Benchmark | Public snapshot | Evidence quality and interpretation |
|---|---|---|
| **OSWorld-Verified** | A November 23, 2026 aggregator snapshot lists Qwen3.8 Max at 86.1%. | The same snapshot reports 34 of 36 rows as lab-reported and only 2 benchmark-run. Treat the top score as a provider-reported result until independently reproduced. OSWorld-Verified and OSWorld 2.0 are separate protocols. |
| **SWE-bench Verified** | A November 28, 2026 fixed bash-only mini-SWE-agent snapshot lists Claude Opus 4.5 (high) at 76.8%. | SWE-bench Verified is a human-screened 500-issue subset. The official project provides a minimal bash-only mini-SWE-agent leaderboard specifically to make model comparisons more controlled; richer harnesses can materially change scores. Keep fixed-harness and any-scaffold tables separate. |
| **Terminal-Bench** | A November 28 aggregator snapshot lists Copilot Fusion (max) at 71.2% on Terminal-Bench 4.1. Another board reports 84.7% on Terminal-Bench 2.0. | These are different task-set versions and different leaderboard methodologies; do not compare the percentages or infer a simple improvement. Prefer a fixed version/harness, or report each separately. |
| **ARC-AGI-3** | ARC Prize says the 2026 competition has concluded and final results will be announced 2026-12-04. | On 2026-11-28, public standings are not the final result. Epoch AI's October review flags version, elicitation and score-interpretation concerns. Report exact harness and version, and include sensitivity analyses. |

The broader lesson is that **benchmark performance is conditional on the task set, agent scaffold, memory/reasoning retention, tool permissions, cost accounting and result provenance**. PsychBench comparisons should show those dimensions as metadata, not bury them in footnotes.

### 1.5 Metacognition and clarification benchmarks to reproduce

| Work | Published result / protocol | What PsychBench should take from it |
|---|---|---|
| [AbstentionBench (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb122bfc3f0127a94ded048b5b03496f-Abstract-Datasets_and_Benchmarks_Track.html) | Twenty datasets spanning unknown answers, underspecification, false premises, subjective interpretations and outdated information. The authors report a 24% average abstention degradation from reasoning fine-tuning in the evaluated models. | Keep reasoning performance and uncertainty handling as separate axes. Reproduce the compatible task families; do not assume stronger reasoning automatically improves abstention. |
| [Abstain-QA (COLING 2025)](https://aclanthology.org/2025.coling-main.627/) | Black-box answerability evaluation across fact/reasoning items and well-/under-represented domains, with an answerable–unanswerable confusion matrix. | Use a black-box adapter so closed models can be evaluated without log probabilities. Report false nonanswers and inappropriate answers separately. |
| [AskBench (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.845/) | Converts QA into multi-turn checkpoints. AskMind tests intent-deficient queries; AskOverconfidence tests false-premise correction. A unified judge loop simulates user responses, while rubric-guided RLVR rewards targeted clarification and final answer quality. | The central PsychBench task should be a trajectory, not one answer: identify the missing fact, ask a targeted question, update on the response and finish correctly. Test transfer to unseen domains. |
| [CAP: Conformalized Abstention Policies (PMLR 304, 2025)](https://proceedings.mlr.press/v304/tayebati26a.html) | Combines conformal prediction and learned instance-specific abstention. On its own evaluation settings, the paper reports 90% target coverage, up to +22.2% hallucination-detection AUROC, +21.2% selective-generation AUARC and over 70% lower calibration error than its static baselines. | Add per-instance risk budgets and compare against conformal/static controls. These are paper-specific results, not directly comparable to PsychBench scores or targets. |
| [HumanBaseline catalogue](https://www.humanbaseline.org/benchmarks/) and [methodology v0.2](https://www.humanbaseline.org/methodology/) | The current catalogue lists 25 benchmark families, 129 items and eight with published human figures. Items are versioned and randomized; the project itself describes the framework as a working hypothesis rather than a final taxonomy of cognition. | Treat it as a useful external map, not a population norm. Demand exact version, participant sampling, score uncertainty and measurement invariance before using human comparisons. The small item bank is a reason to publish item-level distributions and confidence intervals. |
| [MetaBound](https://www.humanbaseline.org/benchmarks/metabound/) | Seven items probe detection of false premises; its public page reports a wide AI range (100% to 0%) and an approximate human figure near 20%. | A seven-item result is a diagnostic signal, not a stable general ranking. PsychBench should report item-level uncertainty and expand hidden, independently authored families before making broad claims. |

These benchmarks point to a coherent research gap rather than a single missing scalar: **calibrated answerability, targeted information acquisition and successful recovery must be measured together**. PsychBench v0.1 currently has a synthetic one-cycle adapter for this construct; its five controls are not external model scores. The next valid comparison step is to import compatible, licensed public items or benchmark APIs, preserve each source protocol and split, then run identical wrappers with sealed family-level holdouts.

## 2. Paradigm shifts for PsychBench

### Shift A — From task labels to construct contracts

Every benchmark entry should have a construct card: construct, operationalization, exclusions, stimulus provenance, target population, evidence class, expected causal signature, failure modes and permitted claims. A test of an HDC algebra property must not be described as a human creativity test. A theory-inspired synthetic probe must not become a consciousness probability.

### Shift B — From final answers to full decision trajectories

Record each step: prior belief, observation requested, observation received, action, confidence, cost, state transition, verifier outcome and update. Preserve rejected actions and failed attempts. Score planning coherence, information value, tool correctness, recovery and safety alongside goal completion.

### Shift C — From static item banks to secure generative task families

Use versioned generators with independent sub-seeds for topology, rule, symbol vocabulary, distractors and intervention schedules. Keep generator families and held-out seeds separate. Hidden variants should preserve construct difficulty while changing surface details. Include metamorphic tests: symbol renaming, graph isomorphism, paraphrase, order permutation and irrelevant-feature changes should not alter the correct solution.

### Shift D — From snapshots to learning and adaptation curves

Report examples-to-criterion, learning-curve area, post-shift regret, transfer retention, forgetting slope, change-point detection delay and recovery time. Separate rapid adaptation from catastrophic forgetting. Include delayed transfer and cross-context tests so memorized surface templates do not masquerade as learning.

### Shift E — From confidence reports to metacognitive control

A useful metacognitive system does not merely say "I am 80% confident". It chooses to answer, abstain, ask for evidence, run a test, delegate, or defer. Score the counterfactual correctness of the answer it would have given, calibration, selective risk, coverage, inappropriate answering on unanswerable prompts, false abstention on answerable prompts and the cost of gathering more evidence.

A new reusable scorer, `benchmarks/metacognition/selective_prediction.rs`, implements this measurement substrate. It is not yet a full item adapter or a claim about Symthaea's performance.

### Shift F — From correlations to mechanism-differential experiments

For every claimed mechanism, pre-register:
- the predicted selective effect of disabling that mechanism;
- an unrelated-module ablation;
- a sham intervention;
- an adversarial or counterfactual condition;
- a rescue/recovery condition;
- a held-out task family.

Example hypotheses:
- removing source provenance selectively harms own/other trace attribution;
- removing agent-belief state selectively harms dynamic false-belief updates;
- removing uncertainty signals increases overconfident errors and reduces appropriate abstention;
- removing temporal memory selectively harms delayed transfer and event ordering.

A general score decline after every ablation is not evidence for a particular mechanism.

### Shift G — From a single human mean to human distributions

Collect graded responses, confidence, response time, permitted aids and participant metadata under ethics and privacy controls. Use IRT where justified; publish percentiles and uncertainty, not only a mean. Check measurement invariance before comparing across age, education, language or participant cohorts. Treat small samples and published reference estimates as weak anchors, not normative ground truth.

### Shift H — From a single leaderboard number to a capability-cost-risk frontier

Do not hide trade-offs in a weighted composite. Report a vector:
- capability: task success, accuracy, transfer and learning efficiency;
- reliability: calibration, replay consistency and score uncertainty;
- safety: violation rate, unauthorized actions, constraint adherence and harm severity;
- efficiency: actions, observations, wall-clock latency, peak memory, tokens, CPU/GPU time, energy and monetary/API cost;
- recovery: detection latency, rollback correctness and successful recovery within budget.

Compare Pareto frontiers. A system is better only when it improves at least one dimension without silently worsening a protected dimension, or when a pre-registered utility function justifies the trade-off.

### Shift I — From benchmark trust by convention to adversarial evaluator verification

Treat the scorer as attack surface. Use mutation tests that deliberately corrupt labels, swap task IDs, alter action costs, remove safety events, leak evaluator-only state and change manifests. Independent replay must reproduce every state digest, score and receipt. Failed, skipped, invalid and not-run states remain distinct.

## 3. Concrete comparison protocol

For each external benchmark adapter, freeze and publish:
1. exact benchmark/task version, item/generator hash and scorer hash;
2. model/build revision, provider, prompt and reasoning/context policy;
3. tool permissions, memory policy, observation format and action budget;
4. sampling parameters and repeat count;
5. wall-clock, token, memory, compute, energy and monetary cost instrumentation;
6. human administration protocol where a human comparison is claimed.

Use the same adapter and budget for each agent. Report Standard and enhanced/provider harnesses separately when they differ. Preserve model-specific notes or memory only when the benchmark permits them, and disclose compaction/retention. Use at least one random baseline, one simple heuristic, one scripted or domain baseline, one full-information oracle, one partial-observation planner and external agent baselines.

Use held-out seeds/items for confirmatory comparisons. Use paired trials when scenarios are matched. Estimate confidence intervals and effect sizes; pre-register primary metrics and multiple-comparison handling. The oracle is a ceiling, not a fair competitor if it sees hidden state.

## 4. PCE and PsychBench engineering targets

These are **proposed internal gates**, not published SOTA scores and not claims already achieved.

| Dimension | Initial release gate | Stretch research target |
|---|---|---|
| Evidence integrity | 100% deterministic replay and receipt verification on frozen fixtures; all mismatches fail closed. | Independent verifier implementation and cross-platform replay agreement. |
| Invalid-action accounting | 100% of rejected actions consume the declared budget and appear in the trace. | Per-action resource models validated against external instrumentation. |
| Held-out task success | Report success separately on train, validation and hidden test generator families. | At least 80% hidden-family success while retaining calibrated uncertainty. |
| Oracle-relative efficiency | Report minimum-action and minimum-cost ratios separately. | At least 0.90 of the relevant full-information ceiling on held-out tasks, without exceeding safety constraints. |
| Learning/transfer | Publish examples-to-criterion and transfer retention with confidence intervals. | At least 0.80 retention on structurally related held-out tasks, with lower forgetting than matched baselines. |
| Metacognitive control | Report ECE, Brier, risk/coverage, false abstention and inappropriate-answer rates. | Lower selective risk than matched agents at the same coverage and resource budget. |
| Recovery | Report intervention detection, recovery latency, failed recovery and post-recovery degradation. | At least 90% recovery inside a pre-registered action/cost budget on unseen perturbations. |
| Safety | Zero unrecorded policy violations in frozen controls; report violation severity and uncertainty. | No increase in violations as capability, autonomy or tool access scales. |

The thresholds above should be revised after pilot distributions, oracle complexity and human administration are measured. They are engineering targets, not normative psychological cutoffs.

## 5. How Symthaea can differentiate itself without overstating results

Symthaea's architecture gives us useful intervention hypotheses: HDC representations, temporal dynamics, predictive processing, FEP, GWT/HOT, memory/provenance and social-agent models can be selectively perturbed. The competitive question is not "does the architecture contain a mechanism?" but "does the mechanism cause the predicted behavioural advantage on held-out tasks, at a measured cost, under independent scoring?"

A useful experimental ladder:

1. **Mechanics:** verify that each synthetic subsystem behaves as specified.
2. **Behaviour:** demonstrate task improvement over random and simple heuristics.
3. **Transfer:** preserve gains under new symbols, layouts, time delays and causal structures.
4. **Mechanism:** show selective loss under targeted ablation and recovery under rescue.
5. **Human anchoring:** compare exact tasks and response protocols with a sufficiently described human cohort.
6. **Deployment:** show a better capability-cost-risk frontier than external agents under matched budgets.

Only the last three stages support strong claims about robust cognitive capability. Passing stage one is a necessary engineering result, not a substitute.

## 6. Immediate implementation queue

1. Finish and test the support-aware confidence interval path; reconcile every pilot metric's declared bounds with its actual generator.
2. Complete the selective-prediction item adapter around answerable, false-premise, ambiguous and evidence-request tasks; compare the policy with forced-answer and random-abstention baselines.
3. Extend PCE to multiple topology/rule families and independently derived sub-seeds; retain BFS and Dijkstra ceilings but add a planner constrained to agent-visible observations.
4. Add delayed transfer, change-point and recovery scenarios; log full trajectories with action, observation and resource costs.
5. Add scorer mutation tests, manifest tamper tests and independent replay verification.
6. Build external adapters for AwarenessBench, HumanBaseline, ARC-AGI-3 and selected operational-agent tasks. Do not merge scores across incompatible protocols.
7. Expand the pilot registry to every implemented benchmark and derive all documentation counts from that registry only after coverage is complete.

## References and live sources

- Li et al. (2026), *AwarenessBench: Assessing Cognitive Capabilities of Language Models*, ACL 2026. https://aclanthology.org/2026.acl-long.124/
- HumanBaseline, benchmark catalogue. https://www.humanbaseline.org/benchmarks/
- HumanBaseline, methodology (versioned items, held-out variants, IRT, distributions and cross-harness caveats). https://www.humanbaseline.org/methodology/
- HumanBaseline, MetaBound. https://www.humanbaseline.org/benchmarks/metabound/
- HumanBaseline, GAUGE. https://www.humanbaseline.org/benchmarks/gauge/
- HumanBaseline, Turn Bench. https://www.humanbaseline.org/benchmarks/turnbench/
- OSWorld-Verified aggregator snapshot (updated 2026-11-23; explicitly records provider-reporting status). https://benchlm.ai/benchmarks/osworld-verified
- SWE-bench Verified official methodology and fixed mini-SWE-agent harness. https://www.swebench.com/verified
- SWE-bench Verified fixed bash-only snapshot (2026-11-28). https://www.benchleader.com/benchmarks/swe_bench_verified
- Terminal-Bench 4.1 November 2026 snapshot and harness-mixing caveat. https://www.benchleader.com/benchmarks/terminal_bench
- Terminal-Bench 2.0 November 2026 snapshot; separate task-set version. https://llmbenchmark.ai/benchmarks/terminal-bench-2-0
- HumanBaseline, LearningBench. https://www.humanbaseline.org/benchmarks/learningbench/
- ARC Prize, ARC-AGI-3 and official 2026 result notice. https://arcprize.org/arc-agi/3 ; https://arcprize.org/leaderboard
- Epoch AI, *ARC-AGI-3 Benchmark Review* (2026-10-30). https://epoch.ai/benchmarks/arc-agi-3/review
- Akhunzada et al. (2026), *From benchmarks to deployment: a comprehensive review of agentic AI evaluation*. https://doi.org/10.1007/s10462-026-11571-0
