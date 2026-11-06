# CEB Industry Frontier and Paradigm Targets V0 — 2026-11-06

**Status:** Research synthesis and prospective target proposal. Not a benchmark result, validated standard, or superiority claim.  
**Parent:** [#10663 CEB pilot authoring pack](https://github.com/Luminous-Dynamics/symthaea/issues/10663)  
**Companion:** [CEB Pilot Authoring Pack V0](CEB_PILOT_AUTHORING_PACK_V0_2026-11-06.md)

## 1. Executive thesis

The next step in software-engineering-agent evaluation is not another isolated patch leaderboard. CEB should qualify whether an engineering system can make **correct, authorized, evidence-backed decisions over the lifetime of a changing codebase**.

The field is moving beyond one-shot bug fixing already:
- SWE-Bench Pro V2 has pushed single-issue resolution near a public ceiling on its full split.
- SWE-Milestone and SWE-CI measure continuous repository evolution and maintenance.
- SWE Atlas broadens coverage to repository understanding, test writing and refactoring.
- Artificial Analysis's Coding Agent Index combines software engineering, terminal work and repository Q&A, while reporting cost and time.
- HiL-Bench makes selective escalation visible rather than rewarding confident guessing through missing requirements.

CEB should integrate these benchmarks as context and use common tasks where feasible, but it should not pretend their scores are interchangeable. Its differentiator is the validity and cost of engineering evidence as the source, requirements, environment, policy, evaluator and workload evolve.

## 2. Current industry landscape — verified 2026-11-06

### 2.1 SWE-Bench Pro V2: patch resolution is approaching saturation

Scale AI's refreshed public split contains 642 tasks across 11 repositories. The release removed 89 tasks judged invalid; 69 task descriptions were corrected after instruction/test contradictions were found and blind-solved by an expert. Agent network access is restricted to the model endpoint, patches are regraded on pristine images, and every task must pass with the reference patch and fail with the empty patch. The public methodology also discloses residual concerns: the model endpoint is a trusted relay, and code introduced by a patch can still execute in the verifier. [Scale AI methodology and release notes](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)

The currently visible full-split leaderboard shows Opus 5 / Claude Code at **99.40 ± 0.40**, Gemini 4 Argon at 99.22 ± 0.45 and Fable 5.1 at 99.10 ± 0.50. The companion 51-task Hard split ranges from 52.9% to 98.0% across frontier systems in the V2 paper. These numbers demonstrate both saturation on the full split and the value of difficulty stratification. [Live leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full) · [V2 paper, 16 October 2026](https://arxiv.org/abs/2610.24206)

**CEB implication:** use patch correctness as a necessary baseline and safety gate, not the sole north-star. Do not claim superiority by comparing CEB's sequence or evidence metrics to SWE-Bench resolve rate.

### 2.2 SWE-Milestone: continuous evolution is the closest adjacent construct

SWE-Milestone reconstructs real repository histories into milestone dependency DAGs. An agent works through consecutive milestones without a reset; a milestone resolves only if all fail-to-pass tests pass and no previously-passing test regresses. The paper evaluates 12 frontier models across four agent frameworks and reports an isolated-to-continuous decline from over 80% to at most 38%. [ICML 2026 paper](https://proceedings.mlr.press/v306/deng26d.html) · [v1.0.2 project and leaderboard](https://swe-milestone.com/)

This is directly relevant to CEB but is not identical. SWE-Milestone measures continued implementation and regression control. CEB continuity adds a distinct question: does the agent correctly retain, revalidate, invalidate, supersede or escalate claims and their evidence as the system's governing conditions change?

**CEB implication:** build paired experiments on compatible milestone streams where possible. Compare a full evidence-continuity arm against stateless reset, persistent-history and repository-twin-only ablations under matched budgets.

### 2.3 SWE-CI: long histories and maintainability

SWE-CI contains 100 repository-maintenance tasks. Each task's evolution history averages 233 days and 71 consecutive commits. Its stated objective is to move beyond static functional correctness toward dynamic, long-term maintainability. [SWE-CI paper](https://arxiv.org/abs/2603.03823)

**CEB implication:** sequence depth and real repository diversity matter. A 24-blueprint authoring bank is only a feasibility inventory; confirmatory claims require many more independent repository clusters and real chronological history.

### 2.4 SWE Atlas: breadth outside issue resolution

SWE Atlas v1.1 covers 284 tasks from 15 production repositories in six languages and three categories: codebase comprehension, test writing and refactoring. Its tasks are single-turn: a prompt states the goal and leaves the approach to the agent. V1.1 changed its judge and grading criteria, so scores from v1.0 are not directly comparable. [Scale AI SWE Atlas methodology/results](https://labs.scale.com/leaderboard/swe-atlas)

**CEB implication:** use repository Q&A, tests and refactoring as component-level probes, but measure whether their outputs remain valid across later source, configuration and policy events.

### 2.5 Artificial Analysis Coding Agent Index V2: composite breadth and economics

The October 2026 Coding Agent Index V2 is an equal-weight average of FrontierCode v1.1 (100 tasks), Terminal-Bench 4.0 (66 tasks) and SWE-Atlas-QnA-AA (124 tasks), for 290 total tasks. Each component is attempted three times. FrontierCode is a private 100-task sample graded with a maintainer-oriented blocking/non-blocking rubric; Terminal-Bench covers terminal workflows; SWE-Atlas-QnA-AA covers repository questions. The methodology also publishes cost, token use and execution time and explicitly tracks reward hacking. [Artificial Analysis V2 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking)

**CEB implication:** cost and time must accompany correctness. A single average conceals whether a result came from higher compute, more attempts, extra human labor, or genuinely better evidence decisions.

### 2.6 HiL-Bench: selective escalation is measurable

HiL-Bench contains 300 tasks across software engineering and text-to-SQL, with 1,131 blockers (3.77 per task average). Its design hides missing, ambiguous or contradictory information and measures both ASK-F1 and task outcome, because asking well and integrating answers are distinct capabilities. [HiL-Bench overview and methodology](https://labs.scale.com/leaderboard/hil)

**CEB implication:** reuse #10612/#10613 for ASK-F1 and help-enabled outcome. Continuity and collaboration should test whether an escalation is correctly tied to a concrete claim, policy source, or evidence gap—not create a competing ASK metric.

## 3. What the industry does well — and what remains open

| Industry benchmark | What it measures well | Remaining CEB research opportunity |
|---|---|---|
| SWE-Bench Pro V2 | Real-issue patch resolution, task repair, reference/empty controls, pristine-image regrading | Ongoing evidence validity, policy authority, dependency-aware reuse and long-term decision traceability |
| SWE-Milestone | Multi-milestone implementation, dependency DAGs, F2P/P2P regressions, compounding errors | Typed claim/evidence disposition and selective proof refresh across milestones |
| SWE-CI | Long repository histories and maintainability under repeated evolution | Separating product evolution from evaluator/metrology drift and claim-admissibility changes |
| SWE Atlas | Comprehension, test writing, refactoring beyond issue resolution | Whether generated understanding and tests stay valid after subsequent changes |
| Coding Agent Index V2 | Broad capability coverage plus cost, tokens and time | Human effort, reviewer reconstruction burden, durable acceptance and evidence lifecycle economics |
| HiL-Bench | Progressive blocker discovery and targeted help seeking | Authority-bounded collaboration and evidence-aware escalation across a changing repository |

The V2 task audit is itself instructive: 480 of the original 731 tasks were repaired or retired, according to the paper. A benchmark can become invalid as agents improve or exploit its harness; benchmark maintenance and evaluator integrity are therefore ongoing engineering disciplines, not one-time dataset preparation. [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206)

## 4. Paradigm shifts CEB should pursue

### Shift 1 — From patch-as-output to claim-as-unit

The core object is not only a patch or task result. It is a consequential engineering claim with:
- a precise proposition;
- an applicability domain (source revision, configuration, version, target, workload, policy and time);
- evidence provenance and authority;
- dependencies that can change;
- a current disposition and a decision-use boundary.

This makes stale evidence a first-class failure rather than a documentation nuisance. #9112 owns admissibility semantics; #10173 owns repository graph and change-impact facts; #10630 scores the continuity behavior.

### Shift 2 — From chronological memory to selective proof maintenance

Persistent memory is not enough. A long transcript can faithfully preserve an obsolete conclusion. The system must decide which evidence to retain, revalidate, invalidate, supersede or mark unresolved, and justify the minimum sufficient check.

Three baselines are essential: reset-and-recheck, remember-everything, and invalidate-everything. CEB should beat all three on a Pareto surface, not merely claim that memory helps.

### Shift 3 — From static tests to contrastive causal sequences

For every consequential change, author a matched contrast:
- one version changes a governing dependency;
- the other changes an irrelevant surface;
- a third changes only one branch of a conditional applicability domain.

The agent's change in disposition and action should match the causal difference. This is stronger than merely measuring whether it can describe a diff. #10366/#10387 supply causal hypothesis and intervention discipline; no second causal scheduler is needed.

### Shift 4 — From “ask or act” to risk-sensitive value of information

Use #10612 for selective escalation and #10366/#10387 for prospective diagnosis. The novel test is whether the agent compares the risk of acting on uncertain evidence with the cost and expected value of a targeted check or human question.

A good system should not ask for everything, rerun everything, or act confidently on nothing. It should seek the smallest sufficient information that changes a real decision. Keep the information-gathering action, the engineering action, and the human authority decision distinct.

### Shift 5 — From trustworthy artifacts to current authority

Cryptographic integrity answers whether an artifact has changed; it does not by itself prove that the signer is still trusted, the policy is still current, or the evidence applies to the present release. CEB should test authenticity, custody, policy validity and decision-use scope as separate conditions through #10314/#10370/#10369/#9112.

### Shift 6 — From code correctness to qualified collaboration

A patch that passes tests can still be unsafe to merge or deploy if the agent bypassed approval, overstated a skipped check, ignored current policy, or made the review impossible. Human-agent collaboration is a distinct socio-technical outcome, not a proxy for patch pass rate.

#10637/#10640 adds this construct without duplicating #10612's question metrics or #10385's debugging/human-complementarity outcomes.

### Shift 7 — From fixed benchmark to continuously audited measurement system

Treat task provenance, contamination, oracle validity, denominators, environment digests, evaluator versions, regrading and external observations as a maintained evidence base. A benchmark result should have an explicit claim class and a reproducible path back to task, system, run and adjudication. Reuse #10610, #10619, #10626 and #10589/#10593 rather than adding another registry.

## 5. Recommended CEB scorecard — independent dimensions

Do not publish one opaque weighted number. Maintain hard gates and a Pareto surface across:

1. **Engineering outcome:** accepted behavior, regression preservation, repair survival.
2. **Evidence validity:** disposition correctness, stale-evidence use, false invalidation, lineage completeness.
3. **Decision quality:** correct action/ask/abstain/escalate, prospective diagnosis, minimum sufficient revalidation.
4. **Security and authority:** critical policy compliance, unauthorized consequential actions, custody and trust validity.
5. **Collaboration:** factual handoff, correction integration, reviewer reconstruction time, human effort.
6. **Economics:** compute cost, wall-clock, tool use, human labor and revalidation cost per accepted sequence.
7. **Measurement integrity:** contamination, oracle quality, verifier trust, denominator fidelity, regrade reproducibility.

A result can be strong on one dimension and unacceptable on another. Critical unauthorized actions, critical stale-evidence use, verifier escape or invalid holdout custody must not be averaged away.

## 6. Prospective benchmark targets

These are recommended **research targets**, not measured results, industry standards or a guarantee that the first pilot can meet them. #10619 must determine confirmatory margins and statistical adequacy before scoring begins.

| Capability | Feasibility target | Hard guardrail |
|---|---:|---|
| Continuity sequence completion | ≥0.90 | Correct at every milestone; no skipping difficult events |
| Evidence disposition macro-F1 | ≥0.95 | Publish precision/recall for each disposition |
| Stale-evidence use | ≤1% overall | Zero critical use of stale, revoked or out-of-scope evidence |
| False invalidation | ≤5% | No “invalidate everything” strategy |
| Revalidation economics | ≤80% of reset-and-recheck cost | No increase in stale use or false acceptance |
| Repair survival | ≥0.90 through three successor milestones | Verify on the exact successor revision |
| Evidence lineage completeness | ≥0.99 | Missing provenance is never inferred |
| Policy/process compliance | ≥0.95 | Zero unauthorized high-impact actions |
| Handoff factual precision | ≥0.95 | Claims must bind to exact revision and actual check state |
| Correction integration accuracy | ≥0.90 | No scope drift or regression accepted as a correction |
| Reviewer reconstruction time | ≥20% lower than matched baseline | Independently accepted-work correctness non-inferior |
| Durable outcome economics | Improve accepted engineering output per total human + agent effort | Report compute, review, interruption and revalidation costs separately |

**Interpretation:** these targets are calibration bands. Do not turn them into a success criterion by choosing a convenient denominator after the run. Report per-repository, per-policy and per-event strata with uncertainty and missingness. A 24-sequence / 24-episode authoring bank is not statistically sufficient for superiority claims.

## 7. How to compare with and outperform the state of the art

### 7.1 Same agent, same task, only the CEB mechanism changes

The primary causal experiment should keep the base model and agent scaffold constant, then compare:
- no explicit evidence continuity;
- persistent history only;
- repository-twin only;
- decision-use validity only;
- causal investigation only;
- full continuity composition.

For collaboration, compare human-only, agent-only, conventional agent, technical CEB without collaboration protocol, selective escalation only, and full CEB collaboration.

Match prompts, tools, context, time, tokens, compute, execution environment, task order and human effort. Use paired tasks and repository/policy cluster-aware inference. A stronger model or bigger budget is not evidence that CEB caused the gain.

### 7.2 Benchmark on a common anchor, but report distinct constructs

Where license, task access and protocol permit, re-run the same agent configurations on selected SWE-Milestone and SWE-CI tasks to establish functional-performance context. Then measure CEB-specific dispositions and lifecycle outcomes on the same event stream.

Do not paste a published third-party leaderboard score beside a CEB metric and call the difference a win. Published model, harness, prompt, budget, version and task split frequently differ.

### 7.3 Demonstrate value with a Pareto frontier

The desired frontier is not maximum accuracy at unlimited cost. It is:
- more correct, durable engineering decisions;
- fewer consequential stale-evidence uses;
- lower unnecessary revalidation and reviewer reconstruction cost;
- zero critical unauthorized action;
- no regression in engineering correctness;
- matched or lower total human-plus-agent effort.

An agent that saves 30% of test cost but increases critical stale acceptance is not an improvement. An agent that asks fewer questions but silently guesses is not an improvement. A system that retains every artifact but cannot explain current applicability is not an improvement.

### 7.4 Pre-register before confirmation

Use the ownership already established:
- #10614/#10617 for balanced repository and fault-family sampling;
- #10619/#10624 for paired analysis, clustering, multiplicity and claim class;
- #10610/#10611 for rerun/regrade/reuse;
- #10626/#10634 for contamination;
- #10612/#10613 for escalation;
- #10630/#10631 for continuity;
- #10637/#10640 for collaboration.

The starter pilot is exploratory. The existing research plan requires a much larger confirmatory repository-cluster set; a six-repository cohort is not a defensible basis for general superiority.

## 8. Runner-independent work that can proceed now

The following are safe and useful before #10064 isolation and #10369 trust qualification:

1. Complete the 24 + 24 public development blueprint pack and reviewer rubrics (companion document).
2. Calibrate reviewers on non-evaluative synthetic vignettes and revise ambiguous definitions.
3. Build an event-to-claim authoring worksheet outside the task-admission schema.
4. Audit every planned sequence for paired decisive/irrelevant and mixed-impact controls.
5. Create a preregistration draft with outcomes, denominators, cluster unit, missingness and stop/go gates; do not lock margins from post-hoc observations.
6. Audit the external observation registry for source revision, denominator, uncertainty, environment, cost and benchmark version completeness.
7. Map the current CEB portfolio to ownership boundaries and block overlapping registries, metrics and receipts.
8. Continue static negative-control design for admission, contamination, lifecycle and claim licensing.

Do not execute candidate-controlled build scripts, tests, macros, graders or attack payloads on the current host-opt-in runner. Do not claim independent regrading, sealed holdout, candidate execution or production isolation.

## 9. Current CEB maturity and limits

The November 5 work added or advanced:
- task admission and two-sided oracle qualification (#10609);
- release lifecycle (#10611);
- selective escalation (#10613);
- balanced sampling (#10617);
- comparative statistical claim qualification (#10624);
- evidence continuity protocol and measurement lane (#10631/#10632);
- T1–T4 contamination transparency and task/card binding (#10634);
- collaborative engineering protocol (#10640).

These remain open draft PRs, not merged releases. Their CEB Contract Validation V0 runs passed at their recorded exact heads, but that is contract validation only; general CI, execution isolation and independently trusted regrading are not implied. The task and contamination registries still contain zero admitted real tasks/cards, and no continuity or collaboration episodes have been executed. P0 #10064 and #10369 remain prerequisites for candidate execution and independent-regrade claims.

**The honest present claim:** CEB has a more developed design for evaluating engineering evidence, continuity and collaboration. It has not yet demonstrated that Symthaea outperforms any industry system on these dimensions.

## 10. Sources and comparison notes

- Scale AI, [SWE-Bench Pro V2 release/methodology and live leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full), updated September 22, 2026; leaderboard viewed November 6, 2026.
- Dan et al., [SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark](https://arxiv.org/abs/2610.24206), 16 October 2026.
- Deng et al., [SWE-Milestone: Evaluating AI Agents on Continuous Software Evolution](https://proceedings.mlr.press/v306/deng26d.html), ICML 2026, PMLR 306.
- Chen et al., [SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration](https://arxiv.org/abs/2603.03823), 4 March 2026.
- Scale AI, [SWE Atlas v1.1](https://labs.scale.com/leaderboard/swe-atlas), updated September 22, 2026.
- Artificial Analysis, [Coding Agent Index V2 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking), October 2026.
- Scale AI, [HiL-Bench overview and methodology](https://labs.scale.com/leaderboard/hil), 2026.

**Cross-benchmark caveat:** every metric in this synthesis has its own task population, unit, harness, agent configuration, grader and release version. The numbers establish context and research motivation; they do not provide a direct CEB score conversion or a matched CEB-vs-industry comparison.