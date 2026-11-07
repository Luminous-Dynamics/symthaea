# CEB Review Causal-Diagnosis Preregistration V0 — 2026-11-07

**Status:** Public research design and preregistration template. No candidate evaluation, task admission, sealed holdout or model-performance result is claimed.  
**Program owner:** Existing CEB-REVIEW / CEB-HUMAN / causal-diagnosis work; [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723), [#10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419), [#10367](https://github.com/Luminous-Dynamics/symthaea/pull/10367).  
**Primary CEB contract:** 14 measurement lanes / 56 primary metrics. CEB-REVIEW remains exactly three primary measures: `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`.  
**Related artifacts:** [benchmark triangulation](CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md), [20-scenario evidence challenge](CEB_REVIEW_EVIDENCE_GROUNDING_CHALLENGE_V0_2026-11-06.md), [finding adjudication card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md).

## 1. Research question

Can an engineering reviewer move from plausible suspicion to **counterfactually testable causal diagnosis**—identifying the correct defect mechanism, grounding each step in the reviewed revision, predicting which controlled source changes should remove or preserve the mechanism, and communicating a useful finding without imposing excessive false-positive or human-review cost?

The research does not ask whether an agent can narrate a persuasive chain of thought. It asks whether the review's concise, externally checkable diagnosis predicts the correct behavior across independently established source states and counterfactual controls.

### Central hypothesis

A reviewer constrained to produce a short causal diagnosis and explicit counterfactual predictions will outperform an otherwise matched reviewer on correct target-defect Recall@5 and actionable precision, or reach non-inferior primary quality with lower human adjudication burden.

The intervention is a **diagnosis contract**, not a larger prompt asking the model to “think harder.” It makes the claim falsifiable:

1. identify the trigger/precondition;
2. trace the relevant path/state transition;
3. identify the unsafe/incorrect operation;
4. name the violated requirement/invariant;
5. state the consequence under the claimed conditions;
6. predict what happens in the clean parent, introducing revision, repaired revision and a neighboring-mechanism control;
7. specify the exact evidence authority and source revision for every material claim.

## 2. Why the field needs this

The current comparator landscape shows that code review needs more than location overlap or a binary “vulnerable” verdict.

- **SecPRBench:** its October 24, 2026 preprint contains 265 real vulnerability-introducing tasks from nine systems projects in C, C++ and Rust. Its direct-review systems report 32.5–78.1% target recall. The paper reports that 68% of pooled misses locate a root-cause file but describe a neighboring or incorrect mechanism. Its cases pair the introducing revision with a repaired companion, using parent/introducing/repaired PoC replay where available; 217 cases pass mechanical checks and 48 require human adjudication. Fixed companions check only the target and do not establish report precision. The judge is itself one of the evaluated models, and each agent has one run per task without matched token budgets. [Paper](https://arxiv.org/abs/2610.33509), submitted October 24, 2026.
- **ReviewBench:** the current high-severity grounded-F1 board lists OhMyBug at 83.0 F1 / 92.0 precision / 75.6 recall, dated October 25. The methodology's grounded outcomes compare against a fixed gold set; unmatched candidate findings are separately classified. Its augmented recall is agent-dependent because novel true positives increase both that agent's numerator and denominator; it is a per-agent discovery diagnostic, not a clean cross-agent ranking metric. Its methodology uses a classifier for gold and unmatched finding labels, with an independent expert re-label audit reported by GitHub. [Leaderboard](https://review-bench.ai/) · [methodology](https://review-bench.ai/methodology/).
- **CR-Bench-v2:** its 174-instance verified slice has one labeled defect per task; KDDR and comment usefulness diverge. A valid suggestion is not automatically a hit on the known target. The verified slice is Python-only across 11 repositories, and reference/judge lineage is not wholly independent. [Paper](https://arxiv.org/abs/2611.02566).
- **CRBugBench:** its best reported system detects 71.9% of human-verified bugs within its top five findings across its public benchmark setup. This is a top-five detection result, not precision or causal-diagnosis accuracy. [Paper](https://arxiv.org/abs/2610.13668).

The comparators motivate the design; their percentages are not directly comparable. CEB uses its own unchanged metric definitions and must demonstrate superiority on a matched task subset before claiming it.

## 3. The key paradigm shift: diagnosis as a counterfactual contract

A natural historical example often provides three source states:

- **P — Parent / pre-introduction:** the target mechanism is absent.
- **I — Introducing:** the real historical change makes the target mechanism live.
- **F — Fixed companion:** a later repair removes the target mechanism.

Those three states are powerful but insufficient on their own. A repair can bundle unrelated commits, alter surrounding code, or leave another path exposed. Therefore the design adds two controlled counterfactuals when an independently reviewable construction is feasible:

- **N — Neighboring-mechanism control:** a nearby change produces a similar symptom or shares the same file/function, but the alleged target mechanism is absent or different.
- **M — Minimal causal intervention:** change only the smallest necessary causal condition/operation while preserving other relevant context; the target mechanism should cease to hold. A separate negative intervention can alter an irrelevant line or nearby behavior while leaving the target mechanism intact.

The expected diagnosis signature is recorded *before* candidate runs:

| State | Target mechanism expected | What a target diagnosis should predict |
|---|---|---|
| P — parent | absent | Do not claim the introduced target is live |
| I — introducing | present | Identify the target's actual mechanism and conditions |
| F — repaired | removed by the fixing change | Do not reassert the removed target; do not infer whole-PR safety |
| N — neighboring mechanism | target absent; a different mechanism may be present | Reject the target mechanism and explain the distinction |
| M — minimal causal intervention | target disabled by the necessary intervention | Predict target absence; preserve unrelated findings |
| IRR — irrelevant intervention | target unchanged | Predict persistence of the target mechanism |

These states are **paired variants of one causal task family**, not independent task samples. Statistical analysis clusters by original defect/PR and repository. Public counterfactual examples are development material, not hidden or sealed holdout tasks.

A reviewer earns a target hit only if it identifies the right causal mechanism in I and does not assert the same target as live in P/F/N/M when the applicable evidence says otherwise. The target diagnosis must identify the necessary precondition/path/operation, not merely the line or file.

### Counterfactual predictions are not executable proof

Before #10064 is qualified, the candidate may only make source-bounded predictions from allowed material. No candidate-controlled build, test, macro, grader, PoC or attack payload may be executed. Existing independently generated historical validation records can inform human gold adjudication, but they must not be represented as candidate-run evidence. A prediction is not a runtime observation.

## 4. Experimental phases and hypotheses

Use a staged design to avoid a wasteful full factorial across tools, reviewers, models and human review.

### Phase A — Source-only diagnosis contract (primary research contrast)

Four matched policy arms, same model/harness, same context, same task snapshot, same output limits and same total inference budget:

| Arm | Diagnosis contract | Counterfactual prompts |
|---|---|---|
| A0 — Baseline | Ordinary code review | None |
| A1 — Evidence contract | Claim/source revision/evidence authority must be scoped | None |
| A2 — Causal diagnosis | Five-link trigger/path/operation/invariant/consequence summary | None |
| A3 — Counterfactual diagnosis | Evidence contract + causal diagnosis + P/I/F/N/M/IRR predictions | Required |

Primary hypothesis H1: A3 improves target defect Recall@5 and actionable precision relative to A0 at matched cost.  
H2: A3 reduces mechanism misdiagnosis relative to A0/A1/A2. Mechanism misdiagnosis is an adjudication attribute, not a new primary metric.  
H3: A3 does not obtain apparent gains by over-reporting, suppressing difficult findings or abstaining on high-value cases; clean controls and actionable precision must remain non-inferior.

### Phase B — Evidence acquisition value (isolation-dependent)

Only after #10064 qualifies candidate-controlled execution:

- B0: source-only A3.
- B1: A3 may request one or more predeclared, safe, target-relevant checks.
- B2: A3 may use an unrestricted but policy-authorized tool budget, with every command/run bound to exact source, config, environment and output.

The check selection must name the competing hypotheses it discriminates *before* execution. A command that does not distinguish them is cost, not evidence quality. Keep compiler, test, static analysis, reproduction and infrastructure failure dispositions distinct. No credit for execution count.

Primary question: does targeted evidence acquisition improve the three existing CEB-REVIEW measures enough to justify its additional total cost? Include tool, model, verifier, human audit and setup costs.

### Phase C — Human complementarity and second-review policy

Matched arms:

1. human-only;
2. single agent;
3. human + single agent;
4. agent + ordinary self-refinement;
5. blind independent second reviewer (does not see first findings);
6. disconfirm-first reviewer (sees first findings and must attempt to refute each claim and search for high-severity omissions);
7. consolidated multi-agent reviewer, with dependence-aware adjudication.

Blind second review measures independent discovery; disconfirm-first review measures adversarial critique. They are not equivalent votes. Equalize total compute and human time where feasible. Reuse #9431 for correlated judge error and #10637/#10640 for collaboration process.

## 5. Task populations and controls

No tasks are admitted by this preregistration. The task-authoring and admission owners must later construct independent samples under #10607/#10609 and contamination/custody rules.

### Target population

The desired study population is real, reviewable changes with independently established defect provenance and frozen source revisions. It should deliberately include:

- single-file and non-local multi-file defects;
- same-file neighboring mechanisms and similar-looking symptoms;
- security boundary / authorization / data-flow cases;
- Rust type/trait/async/lifetime and multi-crate cases;
- API/configuration/feature/target interactions;
- stateful, event-ordering and temporal regressions;
- positive clean/no-target PRs and valid non-target discoveries.

Avoid allowing one language, project, defect family, or benchmark source to dominate. Report each stratum and the actual denominator; do not invent denominators for leaderboard rows that omit them.

### Required control classes

1. **Clean/no-known-defect PRs:** measure noise and false alarms.
2. **Introducing revisions:** known target mechanism is live.
3. **Pre-introduction parents:** target not yet introduced.
4. **Repaired companions:** target removed, with intervening changes disclosed.
5. **Neighboring-mechanism controls:** same file/function or similar symptom, different causal mechanism.
6. **Minimal causal interventions:** disable the necessary target condition without unrelated edits.
7. **Irrelevant interventions:** alter non-causal context while preserving the target.
8. **Valid non-target defects/suggestions:** independently adjudicate discoveries outside the original target.
9. **Evidence integrity controls:** stale revision, mismatched run receipt, missing check, skipped test, timeout and infrastructure failure.
10. **No-authority controls:** execution is prohibited or required evidence is unavailable; correct abstention/escalation must remain possible.

Do not treat a repaired companion as a clean whole-PR negative. It is negative only for the specific target mechanism.

## 6. Gold truth, independent adjudication and matcher qualification

The candidate is never the source of its own gold labels.

### Defect provenance

For each target, record the originating change, parent source state, introducing source tree, fixing change, exact tree/diff digests, specification/contract and original evidence. Distinguish:
- mechanical validation;
- source-based expert adjudication;
- mixed validation;
- unresolved labels.

PoC or test replay is useful only if the same declared input/invocation and relevant environment establish absence in P, presence in I and absence in F. A build error, unrelated crash or timeout is not proof. If a vulnerability is not mechanically observable, explicitly record human evidence and its limits.

### Finding adjudication

- Two independent blinded reviewers adjudicate each consequential finding against the frozen requirement and exact source revision.
- A third expert resolves disagreement using evidence and a recorded rationale, not an unqualified majority vote.
- Reviewers must distinguish target hit, valid non-target discovery, useful suggestion, unsupported/noisy finding, wrong mechanism, stale evidence and unresolved.
- Human adjudicators are blind to candidate arm/model where practical. If blinding is broken, record it.
- Preserve reviewer IDs/roles, rubric revision, time and any conflict of interest.

### Evaluator audit

ReviewBench's methodology has an important mathematical limitation: augmented recall's denominator includes candidate-specific unmatched true positives, so it is useful as a per-agent discovery diagnostic but not a simple common-denominator leaderboard rank. CEB's strict primary comparison remains fixed-gold Recall@5. Unmatched discoveries require separate blinded adjudication; do not automatically add them to gold or call them noise.

The judge stack must be decomposed:
1. target provenance and reference mechanism;
2. finding-to-defect semantic matcher;
3. comment usefulness/actionability adjudicator;
4. independent human audit.

Record each judge's model/provider/version/prompt/rubric, lineage with reference generation, per-judge output and disagreements. Audit accepted, rejected and unresolved matches using a stratified, blinded human sample. Reuse #9431/#9022/#9130 and #10680/#10681. Shared-model majority agreement is not independent truth.

## 7. Output contract for candidate findings

This is a prompt/output contract, not a new CEB task or run-receipt schema. Each finding must contain:

- exact location and reviewed source revision;
- concise observable claim and severity rationale;
- trigger/precondition;
- relevant call/data/control path and state;
- unsafe/incorrect operation;
- violated requirement/invariant;
- expected consequence under the stated conditions;
- evidence class and anchors;
- one nearest alternative mechanism;
- P/I/F/N/M/IRR counterfactual prediction with uncertainty;
- smallest next evidence request if unresolved.

Do not request or score hidden chain-of-thought. Score the correctness, completeness, evidence and falsifiability of the final structured diagnosis.

## 8. Registered outcomes and analysis

### Existing primary measures only

- **`defect_recall_at_5`:** fraction of human-verified introduced target defects correctly identified among the top five findings. A location-only or wrong-mechanism finding is not a hit.
- **`actionable_precision`:** fraction of submitted findings independently confirmed actionable under the frozen scope/rubric. Track target and non-target actionable findings separately in adjudication so a valid discovery cannot be misreported as a target hit.
- **`reviewer_burden`:** human minutes to adjudicate findings per reviewed change, including reconstruction of evidence and second-review effort.

The primary task-level unit remains the frozen review change. Paired revision variants, counterfactual siblings, multiple findings and repeated model runs are clustered to their originating task/repository and must not inflate n. No new primary metric lane, composite score or CEB metric ID is introduced.

### Diagnostic annotations (not new primary metrics)

Report mechanism-misdiagnosis rate, location-only-hit rate, counterfactual prediction consistency, stale-evidence use, high-severity false negatives, duplicate findings, unsupported comments, unresolved fraction, correct abstention, human override and cost/latency by family and severity. These explain the three primary outcomes; they do not replace or combine them.

### Primary estimands

For Phase A:
- paired difference in Recall@5, A3 minus A0;
- paired difference in actionable precision, A3 minus A0;
- ratio/difference in reviewer burden, A3 relative to A0.

H2 mechanism-misdiagnosis difference is a preregistered diagnostic secondary contrast. A1/A2 comparisons isolate evidence-contract and causal-diagnosis contributions; A3 vs A2 isolates the incremental counterfactual-prediction contract.

### Statistical plan

- Use paired task-level analysis because each arm sees the same frozen task.
- Stratify by repository and defect family; do not assume variants from one defect are independent samples.
- Report macro-average over PRs and micro-average over findings, preserving denominators.
- For proportions, report paired confidence intervals and exact counts; use a predeclared paired test such as McNemar for binary task outcomes where appropriate.
- Use repository-cluster bootstrap or a preregistered hierarchical model to preserve within-repository dependence.
- Adjust confirmatory multiple comparisons with Holm or a predeclared gatekeeping sequence.
- Report effect sizes and intervals, not only p-values. Retain per-task and per-run results.
- Estimate sample size from a development-only pilot's paired discordance and repository clustering, targeting at least 80% power for the predeclared +5 percentage-point primary improvement at family-wise alpha 0.05. Freeze the final sample size, analysis code and stopping rule before holdout access. Do not use holdout outcomes to tune sample size or thresholds.
- If the independent repository/family count is too small for the planned inference, report the work as exploratory rather than claiming confirmatory generalization.

No arbitrary task count is declared here: the pilot data and design effect must determine n before the sealed sample is accessed.

## 9. Budget, repeatability and contamination

- Use the same model version, system prompt, harness, reasoning/tool limits and output cap across each paired arm unless that is the independent variable.
- Three independent runs per task/arm are a proposed default for nondeterministic configurations; preserve all run-level outputs and do not pool retries as extra tasks.
- Record input/output/reasoning/cache tokens where available, provider cost, local compute, wall-clock, context reconstruction, judge calls, human annotation and second-review minutes.
- Compare at matched total cost, or report the entire quality/cost Pareto frontier. Do not compare model-only token cost with another system's all-in cost.
- Preserve network policy. General documentation/library access may be allowed where consistent with task goals, but reference fixes, target commits, hidden labels, benchmark solutions and answer-bearing mirrors remain prohibited for blind evaluation.
- Audit trajectories for solution retrieval, verifier/test edits, grader access and fabricated evidence. Human audit all passing attempts selected by the integrity protocol; do not rely on a single same-family reward-hacking judge.
- Freeze source tree, diff, task statement, dependency state, toolchain, environment image, rubric and evaluator versions. Re-run/regrade only under the existing release-lifecycle and trust contracts.

## 10. Stop/go gates

**Design-ready:** source/requirement authority, target identity, control-pair design, reviewer output contract and adjudication rubric are independently reviewed.

**Admission-ready:** task passes #10609's admission/oracle rules, contamination disclosure and balanced sampling. Public design examples are permanently ineligible as sealed holdouts.

**Execution-ready:** #10064 qualifies OS-level isolation, with the applicable runner threat model and artifact custody.

**Independent-regrade-ready:** #10369 qualifies signer trust, runner attestation, pristine image and evidence convergence.

**Analysis-ready:** frozen model/harness, paired cost budget, randomization, sample-size calculation, adjudicator agreement protocol, judge audit and statistical analysis are preregistered before holdout access.

Fail closed. Missing evidence, insufficient oracle authority, unresolved target identity or incomplete runner qualification remains unresolved; it does not become a pass by default.

## 11. Prospective targets and decision rules

The existing CEB-REVIEW matrix continues to govern:
- Recall@5 **≥ 0.85** on a sealed repository-/defect-family-disjoint holdout;
- actionable precision **≥ 0.80**;
- reviewer burden **≥ 20% lower** at non-inferior Recall@5.

The research program's matched-improvement aspiration is **+5 percentage points** on a primary quality measure at equal total cost, without meaningful degradation of the other primary measure, or **≥20% lower all-in cost** at non-inferior Recall@5 and precision.

The diagnosis-contract hypothesis is supported only if:
1. A3 improves Recall@5 by the preregistered margin or achieves a predeclared non-inferiority/superiority trade-off;
2. actionable precision remains non-inferior;
3. reviewer burden does not increase beyond the predeclared margin;
4. mechanism-misdiagnosis and wrong-revision overclaims fall in adjudication;
5. clean/repaired/neighbor controls show no offsetting increase in target false alarms;
6. gains persist across repository and fault-family strata, not just the easy subset.

These are prospective targets, not established performance and not new registered CEB primary metrics.

## 12. Runner-independent authoring sequence

1. Two independent authors apply the adjudication card to a sample of the 20 synthetic evidence scenarios; resolve ambiguous wording before any task construction.
2. Author paired source-level counterfactual vignettes for parent/introducing/repaired/neighbor/intervention states. Do not execute them.
3. Build a development-only example of a finding that is in the right file but has the wrong mechanism, and an example where a correct target diagnosis must be withdrawn after the fix.
4. Create a blinded reviewer calibration pack with clean, target-positive and non-target positives, clearly marked public development.
5. Draft the exact prompt/output schemas and analysis notebook/specification (without running candidates).
6. Obtain independent task-admission, security, statistical and human-review sign-off.
7. Only then consider a sealed sample and execution after #10064/#10369.

## 12.1 Gold-completeness protocol and sequential evidence-policy composition

The six-state counterfactual design must be evaluated within a defect-provenance process that does not assume human-found gold is exhaustive. Use [CEB Review Gold-Completeness and Open-Set Discovery V0](CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md) to keep frozen targets separate from independently adjudicated novel findings and the unknown remainder. Blind audits of zero-finding PRs and multi-producer clean consensus are required before making a gold-completeness claim. Do not use naïve capture–recapture under correlated producer populations.

For Phase B, use [CEB Review Sequential Evidence Acquisition V0](CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md) as the action-policy contract: name the claim and competing mechanisms, select the smallest safe discriminating observation, bind it to exact revision/scope/authority, and account for all-in cost and stopping behavior. It expands the Phase B policy contrast; it does not add a metric or authorize execution before #10064.

## 12.2 Counterfactual-control qualification

Use [CEB Review Counterfactual-Control Qualification V0](CEB_REVIEW_COUNTERFACTUAL_CONTROL_QUALIFICATION_V0_2026-11-07.md) before treating any P/I/F/N/M/IRR source state as a causal control. In particular:

- Historical parent/introducing/fix revisions are source-state observations, not randomized interventions.
- A repair commit must be disentangled from tests, docs, refactors, dependencies, environment changes and unrelated fixes.
- A fixed companion may be partial, symptom-masking or tangled; verify every in-scope alternate path before claiming the target is absent.
- Minimality is semantic in the causal model, not a count of changed lines.
- An irrelevant intervention must be shown not to alter target causal parents or paths; distance or formatting alone is insufficient.
- Source-only controls can support bounded structural claims, not runtime outcome claims.
- Each control is qualified per claim across identity/lineage, change disentanglement, mechanism/reachability, intervention fidelity, environment/applicability and outcome/oracle validity. Unresolved dimensions remain unresolved; do not average them into one score.

The control protocol is a methodological companion only. It adds no task schema, receipt, authority class, metric lane or execution permission.

## 13. Claim boundary

This document specifies an experiment; it reports no experiment. The historical benchmark findings above are contextual external claims with their own limitations. The counterfactual ladder is a proposed CEB design, not an established state-of-the-art method. No model has been scored, no task admitted, no holdout constructed, no hidden label created, and no candidate-controlled code, tests, PoCs, graders or attack payloads executed.
