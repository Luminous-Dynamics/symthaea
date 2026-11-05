# CEB Evidence Continuity Protocol V0 — 2026-11-05

**Status:** Design-only research protocol. No candidate task or repository sequence is admitted or executed by this document.  
**Parent:** [CEB-CONTINUITY-001 #10630](https://github.com/Luminous-Dynamics/symthaea/issues/10630)  
**Composition:** #10173 repository twin; #9112 decision-use validity; #10366/#10387 causal investigation; #10314 canonical benchmark receipts; #10436 task custody; #10064 execution isolation; #10369 trust roots.

## 1. Mission and claim boundary

CEB must evaluate whether an engineering system can preserve the validity of engineering decisions as the repository and its operating context evolve.

A one-shot repair benchmark asks whether a patch passes at revision R. Evidence continuity asks whether the system can carry justified conclusions from R0 through R1…Rn without either:

- relying on evidence invalidated by later changes; or
- discarding still-valid evidence and paying to reconstruct it unnecessarily.

The target capability is **selective, justified evidence revalidation**. For every consequential engineering claim, the system should explain whether it remains valid, was superseded, must be revalidated, has been invalidated, or is unresolved.

This protocol does not add a repository graph, decision-use validity engine, causal scheduler, task manifest, run receipt, DSSE envelope, or trust system. It defines an evaluation adapter over existing work. #10173 remains authoritative for repository graph and change-impact facts; #9112 remains authoritative for decision-use admissibility semantics; #10366/#10387 remain authoritative for causal forecasts and interventions; #10314 remains the canonical run-accounting receipt.

Passing a design validator is not evidence that an agent performs this capability. No execution or performance claim is enabled here.

## 2. Research basis and distinction from existing benchmarks

Continuous software evolution is an emerging benchmark direction:

- **SWE-Milestone (ICML 2026)** reconstructs milestone streams from repository history and evaluates maintenance without resetting the agent between milestones. Its paper reports a fall from greater than 80% on isolated tasks to at most 38% in continuous settings across 12 models and four agent frameworks. Source: https://proceedings.mlr.press/v306/deng26d.html
- **SWE-CI** uses 100 repository-maintenance tasks with histories averaging 233 days and 71 commits. Source: https://arxiv.org/abs/2603.03823
- **Terminal-Bench 4.0** explicitly distinguishes task/environment changes requiring reruns, verifier changes requiring regrades, and metadata-only changes that may reuse results. Source: https://www.tbench.ai/news/terminal-bench-4-0
- **SWE-Bench Pro V2** uses a refreshed 642-task public split, task repair/review, network restrictions, pristine-image regrading and two-sided reference-patch/empty-patch controls. Source: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2

CEB should learn from these designs, not claim direct score equivalence. Its distinct research question is the continuity of **evidence validity and assurance** across changes. A sequence can preserve functional behavior but invalidate a performance claim because the workload changed; an API refactor can invalidate a source anchor while preserving a behavioral contract; a security-policy change can revoke prior authorization even when all tests still pass.

## 3. Evaluation unit and event model

The evaluation unit is a frozen sequence:

- an initial repository and operating-context state;
- an ordered event stream;
- one agent-visible state at each decision point;
- a set of claims and evidence records available at that point;
- a grader-only expected validity transition for each affected claim;
- an independently scored sequence outcome.

Each event must bind to an exact source/configuration identity and an event digest. An event is not presumed to invalidate all evidence merely because it is later in time.

### 3.1 Event strata

The authoring bank should cover at least these six strata:

1. **Source/API events:** symbol rename, module move, trait-bound change, public API revision, generated-code or macro expansion change.
2. **Configuration events:** Cargo feature, target triple, compiler/toolchain, dependency, build-script input or environment change.
3. **Behavioral-contract events:** requirement clarification, deprecation, compatibility change, state-machine invariant or error-semantics revision.
4. **Security/authority events:** permission revocation, trust-root rotation, policy change, threat-model change, secret-handling rule or signer revocation.
5. **Evaluator events:** test addition/removal, oracle correction, grader bug fix, task prompt correction or benchmark rubric revision.
6. **Workload/metrology events:** workload change, hardware change, measurement method change, sampling change or expert-baseline revision.

Product changes and evaluator changes must remain distinct. A grader correction can require regrading old artifacts without implying the candidate product changed. A model, prompt, tool, runner image, resource budget or task/environment change generally requires a new run rather than a retrospective regrade.

### 3.2 Event relationship to claims

Each claim has a declared applicability domain and typed dependency references. Relevant dependencies may include:

- exact source paths, spans, symbols and graph edges;
- crate, feature, target and toolchain conditions;
- external contract and policy versions;
- tests, workload, hardware and measurement procedure;
- signer, trust policy, runner image and isolation policy;
- benchmark task, oracle and grader revisions.

Use #10173's graph and invalidation labels where they exist. Do not infer a semantic dependency from text similarity alone. If the relationship is not established, record it as uncertain rather than fabricating a precise edge.

## 4. Claim and evidence dispositions

The agent proposes a disposition; an independent scorer adjudicates it against the hidden event/evidence gold set.

| Disposition | Meaning | Required behavior |
|---|---|---|
| **Retain** | The event does not invalidate the claim within its declared scope | Preserve the claim and its original evidence lineage |
| **Revalidate** | The claim may remain true, but a changed dependency requires fresh evidence | Identify the minimum sufficient check and obtain it before consequential use |
| **Invalidate** | A governing premise or dependency is no longer true | Withdraw the claim from current decision use; do not silently reuse it |
| **Supersede** | A newer, scoped claim replaces the previous claim | Preserve both versions and their temporal applicability |
| **Unresolved** | Available evidence cannot establish the correct transition | Abstain, ask, or escalate; do not guess a validity state |

The following source/evidence states remain distinct: not run, blocked, failed, timed out, unavailable, censored, stale, revoked, contradicted, and confirmed zero. Missing evidence is not a negative result; a failed attempt is not a confirmed absence.

A fact may remain historically true while becoming inadmissible for a current decision. For example, “the service accepted token format V1 at release R0” may remain historically true but cannot establish compatibility at R2 after a breaking contract change. This is a decision-use question, not a rewrite of history.

## 5. Required agent behavior

At every milestone, the agent should:

1. Identify which prior claims it intends to use for the current decision.
2. Resolve each claim's applicability domain, provenance, time/scope and governing policy.
3. Inspect intervening events and relevant change-impact evidence.
4. Classify the claim using the dispositions above and cite the decisive dependencies.
5. For revalidation, propose the lowest-cost adequate check, forecast its possible outcomes before execution, and record risk/reversibility.
6. Obtain evidence at an authority level sufficient for the claim; do not promote static text or correlation into compiler/runtime/intervention evidence.
7. Update the decision only after observing the result; preserve missing or non-identifiable cases.
8. Bind any repair or operational action to the exact current evidence state and authorization.
9. Preserve prior records; append a new state rather than rewriting a historical decision.

The system must not “solve” evidence continuity by invalidating everything, rerunning every test, or abstaining from every consequential decision. These are explicit failure modes.

## 6. Paired and adversarial sequence design

Author 20–24 design-only sequence blueprints across at least five event strata for the first feasibility tranche. The blueprint count is not an admitted task count.

Every sequence should include:

- **Decisive-change pair:** change one governing condition; the affected claim must no longer be used unchanged.
- **Irrelevant-change pair:** change unrelated code or documentation; unaffected claims should remain valid.
- **Mixed-impact case:** invalidate one claim while preserving adjacent claims.
- **Conditional-world case:** a claim remains valid for one feature/target/version but not another.
- **Unresolved case:** event history is insufficient to establish the dependency; the correct action is escalation or abstention.
- **Missingness case:** distinguish not-run, blocked, failed, censored, unavailable and confirmed zero.
- **Common-origin case:** copied documentation, tests or logs share a single origin and cannot count as independent corroboration.
- **Revocation case:** an otherwise authentic historical assertion is no longer authorized for current use.

Include multi-event sequences where an early repair is correct, a later change silently invalidates one of its assumptions, and a subsequent unrelated change must not cause indiscriminate rollback. Include evaluator updates that require regrading without rerunning candidate code, and source/environment updates that do require reruns.

Keep future events, expected invalidation sets, answer keys, hidden tests and adjudication labels outside the agent-visible view. Publicly committed blueprints must not disclose a future event's gold disposition in a way that leaks the holdout.

## 7. Baselines and causal attribution

Use the same sequence instances and budgets for each treatment:

1. **Stateless reset:** retrieve context afresh at every milestone, with no persistent evidence state.
2. **Persistent history:** ordinary conversation/repository history without explicit validity transitions.
3. **Repository-twin only:** change-impact graph from #10173, without decision-use validity.
4. **Decision-use validity only:** #9112-style admissibility state without repository dependency propagation.
5. **Causal investigation only:** #10366/#10387 forecasts and interventions for selected fault cases.
6. **Full continuity pipeline:** repository change-impact, evidence lineage, decision-use validity and selective revalidation.
7. **Expert reference:** maintainer-adjudicated evidence transitions; human-plus-agent as a separate assistance arm where feasible.

Ablate graph dependencies, source provenance, temporal/scope rules, event classification, prospective experiment forecasts and causal interventions separately. Keep model, harness, prompt, tools, task state, event stream, context budget, tool budget, runner and grader fixed inside each matched comparison.

Ablation is necessary to support the claim that the continuity mechanism caused an improvement. A better underlying model, larger context, extra tools or more execution budget must not be attributed to the evidence-continuity component.

## 8. Metrics — do not collapse into one score

Report each metric with its exact denominator, sequence depth, event stratum and uncertainty:

- **Sequence-complete rate:** fraction of sequences for which every required milestone decision is correct and protected invariants remain satisfied.
- **Evidence disposition macro-F1:** retain/revalidate/invalidate/supersede/unresolved, with per-class precision and recall.
- **Stale-evidence use rate:** consequential decisions that rely on evidence already invalidated or outside its validity scope.
- **False-invalidation rate:** valid evidence discarded without a governing change.
- **Selective revalidation efficiency:** required checks performed divided by all checks performed; pair with false-revalidation and missed-revalidation rates.
- **Revalidation burden:** tool calls, execution time, compute, tokens, human review minutes and monetary cost per accepted sequence.
- **Evidence lineage completeness:** proportion of decision-governing claims with exact source, configuration, authority and freshness references.
- **Repair survival:** accepted repairs that remain valid through later milestones without recurrence or protected-invariant regression.
- **Escalation quality:** precision and recall for genuinely unresolved/non-identifiable transitions; report question precision and blocker recall separately where humans supply missing information.
- **Human–agent complementarity:** review time, accepted changes, overrides, defect detection and severe false negatives at equal effort.

Do not report only micro-averaged F1: a large number of easy retain cases could conceal a dangerous miss on security revocation. Report macro and micro views, severity-stratified results, worst-family performance, and all task dispositions. Always-abstain and invalidate-everything baselines must be reported and must fail the utility/retention requirements.

### 8.1 Economic outcome

Report full lifecycle cost per accepted sequence:

`(inference + retrieval + tool + execution + revalidation + human review + rework cost) / independently accepted sequences`

Do not count only agent token costs. Report fixed costs and marginal costs separately. A reduction in checks is not a win if it increases stale-evidence use or human rework.

## 9. Versioning, rerun, regrade and evidence reuse

Use the canonical #10314 receipt and its task, arm, budget and evidence identities. This protocol defines no parallel receipt.

Classify a benchmark revision before using historical results:

| Change | Default treatment |
|---|---|
| Candidate model, prompt, harness, tool access, budget, task source state, runner image or execution policy changes | New run |
| Task sequence, event stream or agent-visible context changes | New run with a new task/protocol identity |
| Grader/verifier logic changes while preserved candidate artifacts and inputs are byte-identical | Regrade retained artifacts under a new grader version; preserve the old grade |
| Metric formula/reporting presentation changes without changing task, evidence or grading semantics | Recompute from retained raw records; preserve the original metric definition |
| Metadata-only correction with no semantic or byte change to evaluation inputs | Append a correction record; reuse only where the receipt permits it |
| Trust root, signer validity, runner attestation or isolation policy changes | Re-evaluate the relevant assurance claim; do not silently preserve an independence/isolation assertion |

A rerun and a regrade answer different questions. Do not rerun an agent merely because a report label changed; do not reuse a historical execution when the task or environment materially changed. Keep prior grades immutable and expose supersession/withdrawal.

## 10. Statistical plan

The 20–24 blueprint tranche is for feasibility, annotation reliability and variance estimation, not a powered SOTA claim.

Before any confirmatory run:

- Freeze the primary sequence-level outcome, practical effect margin, exclusions, missingness policy and analysis code.
- Pair every treatment on the same sequence and budget.
- Use multiple independent seeds; estimate variance in the pilot rather than asserting a fixed seed count is sufficient.
- Cluster uncertainty by repository template/root-cause family and sequence, not by treating each event within a sequence as independent.
- Report paired effect sizes and confidence intervals, along with per-sequence disagreement tables.
- Use a sealed holdout separated by repository family and event-generation template, not merely random events drawn from familiar templates.
- Keep exploratory metrics and post-hoc subgroups explicitly exploratory.
- Preserve timeouts, refusals, abstentions, escalations and ungraded outcomes in task accounting.

Prospective research hypotheses, to be calibrated and frozen after the feasibility pilot:

- The full pipeline improves sequence-complete rate over the strongest matched ablation with a paired confidence interval excluding zero and a preregistered practical margin.
- It reduces lifecycle revalidation cost by at least 20% versus reset-and-recheck **without increasing stale-evidence use or false acceptance on the declared critical controls**.
- It has materially better invalidation precision/recall than persistent-history and graph-only baselines, without an unacceptable false-invalidation burden.
- It transfers to held-out repository families and unseen event compositions.

These are proposed goals, not standards, measured outcomes or guarantees. Do not publish “beats SWE-Milestone” based on unrelated task units or scores. External comparisons are contextual until a common matched protocol exists.

## 11. Runner-independent implementation tranche

Before #10064 and #10369 are qualified, permissible work includes:

- versioned event/disposition vocabulary and this composition protocol;
- synthetic, non-executable repository and evidence-event fixtures;
- deterministic scoring over inert recorded ledgers;
- gold-label authoring guidance stored outside agent-visible data;
- paired control inventories and baseline/preregistration definitions;
- static schema/manifest validation and adversarial self-tests;
- documentation of which changes require rerun, regrade, reuse or assurance re-evaluation.

Do not execute candidate-controlled code, build scripts, proc macros, tests, grader code or attack payloads on the current host-opt-in runner. Do not label a public task/fix pair as a sealed holdout. Do not claim isolated execution or independent regrading until #10064 and #10369 are qualified.

## 12. Acceptance checklist for V0 design

- [ ] Protocol is explicitly linked to #10630.
- [ ] All five dispositions have operational definitions and scoring guidance.
- [ ] Product events and evaluator events are distinct.
- [ ] At least 20 sequence blueprints span five or more event strata.
- [ ] Every blueprint includes decisive, irrelevant and unresolved controls.
- [ ] Missingness states remain distinct.
- [ ] Agent-visible and grader-only material are separated by custody design.
- [ ] All metrics are recomputed from raw per-event records and reported separately.
- [ ] Rerun/regrade/reuse policy is mapped to canonical receipt identities.
- [ ] No new task schema, receipt, DSSE envelope, repository graph or generic scheduler is introduced.
- [ ] Isolation, trust, task admission and performance claim ceilings remain explicit.

## 13. Claim boundary

This is a design and research protocol. It does not establish that Symthaea currently maintains evidence correctly across repository evolution, that any sequence is admitted, that the scorer is independently trusted, or that a runner is isolated. It does not establish superiority over SWE-Milestone, SWE-CI, Terminal-Bench or any coding-agent leaderboard. Those claims require matched tasks, exact protocol versions, qualified execution and independently adjudicated evidence.


## 14. CEB portfolio integration and ownership boundaries

The continuity track is deliberately downstream of, and complementary to, several CEB workstreams opened during the November 5 research expansion. The protocol must compose with these owners instead of duplicating their schemas or validators.

| Existing work | Owns | Continuity integration |
|---|---|---|
| [#10607 / #10609](https://github.com/Luminous-Dynamics/symthaea/issues/10607) task admission | Task provenance, license, human oracle, two-sided controls and admission states | Every sequence milestone must reference admitted task identity; a continuity sequence is not itself an admission bypass |
| [#10610 / #10611](https://github.com/Luminous-Dynamics/symthaea/issues/10610) release lifecycle | Semantic versioning, saturation, resource calibration, rerun/regrade/reuse policy | Reuse its change-impact classification; continuity records the engineering-claim consequences, not another release manifest |
| [#10612 / #10613](https://github.com/Luminous-Dynamics/symthaea/issues/10612) selective escalation | Question precision, blocker recall, ASK-F1 and help-enabled task success | Use escalation as an explicit disposition when evidence dependencies are missing or non-identifiable; preserve ASK-F1 and Pass@3 separately |
| [#10614 / #10617](https://github.com/Luminous-Dynamics/symthaea/issues/10614) balanced sampling | Repository × fault-family allocation, holdout balance and pre-score difficulty | Sample continuity sequences by repository family and event/fault stratum; do not reuse the six-repository pilot as confirmatory evidence |
| [#10619 / #10624](https://github.com/Luminous-Dynamics/symthaea/issues/10619) comparative statistics | Paired inference, cluster structure, primary/secondary claim rules and Pareto reporting | Use its claim classes, power and inference gates; do not invent continuity-only superiority thresholds after seeing outcomes |
| [#10626](https://github.com/Luminous-Dynamics/symthaea/issues/10626) contamination transparency | T1–T4 leakage assessment, search-surface records, mitigation and residual uncertainty | Require task/sequence contamination cards and distinguish searched-no-match from verified-absent; never claim a public task is sealed |
| [#10571 / #10589](https://github.com/Luminous-Dynamics/symthaea/pull/10589) observation registry | External observation provenance, configuration and denominator lineage | Keep any external continuous-benchmark observation attached to its original task/version/system definition |
| [#10592 / #10593](https://github.com/Luminous-Dynamics/symthaea/issues/10592) metric crosswalk | Semantic correspondence and SOTA claim guard | A continuity metric can be a new construct; do not map it to generic milestone resolve rate without a matched operational definition |
| [#10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173) repository twin | Typed graph, change impact and invalidation labels | Supplies dependency/evidence-impact facts; continuity scores whether the resulting claim disposition is correct |
| [#9112](https://github.com/Luminous-Dynamics/symthaea/issues/9112) decision-use validity | Scope, temporal validity, source-linked governing state and admissibility | Supplies the validity vocabulary and decision-use semantics; continuity measures them through repository evolution |
| [#10366 / #10387](https://github.com/Luminous-Dynamics/symthaea/issues/10366) causal debugging | Hypotheses, prospective forecasts, intervention and update | Used for ambiguous changes and targeted revalidation; no second scheduler is introduced |
| [#10314 / #10370](https://github.com/Luminous-Dynamics/symthaea/issues/10370) evidence and receipts | Canonical run accounting, DSSE and primary/regrade convergence | Binds task/arm/run identity and assurance evidence; no continuity-specific receipt format is allowed |
| [#10436, #10064, #10369](https://github.com/Luminous-Dynamics/symthaea/issues/10436) custody/isolation/trust | Agent-visible release, execution isolation and signer custody | Remain hard gates for executable tasks, sealed holds and independent-regrade assertions |

### 14.1 Four distinct change questions

Do not conflate these questions:

1. **Did the engineering system change?** #10173 and the source/configuration event record establish candidate impact.
2. **Did the claim's validity change?** #9112 semantics determine scope, freshness and admissibility; the continuity grader assesses retain/revalidate/invalidate/supersede/unresolved.
3. **Did the benchmark release or evaluator change?** #10610 determines rerun/regrade/reuse treatment and preserves historical denominators.
4. **Did the comparative claim remain statistically licensed?** #10619 governs paired analysis, clustering, multiplicity, power and claim class.

A single change may affect one, several, or none of these dimensions. For example, a grader-only correction may require a regrade under #10610, update a previously published comparison under #10619, and leave the engineering source claim unchanged. A dependency update may invalidate a compiler-resolved API claim and require a new run, without changing the benchmark's official grader.

### 14.2 Required integration tests for a later executable release

The static design should eventually be represented by paired cases where:

- a product dependency change invalidates an affected claim but preserves unrelated claims;
- a verifier-only update triggers regrade without falsely claiming a product change;
- a metadata-only edit permits reuse only after evaluation-input digest equality;
- public-fix exposure downgrades a task's contamination status without silently changing the task denominator;
- holdout difficulty and repository-family allocation are fixed before candidate scores;
- an unresolved evidence gap generates a precise question/escalation and is not counted as a failed repair;
- a task/event sequence change triggers rerun and cannot inherit a prior score;
- a statistical claim is downgraded when its cluster count, power evidence, or matched-input conditions fail.

These are future qualification cases, not executed tests or admitted task records.
