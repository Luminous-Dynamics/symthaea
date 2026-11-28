# Symthaea adaptive research orchestration: fixed workflows versus evidence-contingent control

**Version:** v0 · **Date:** 2026-11-07  
**Owner:** #11012  
**Composes:** #10877, #9230, #8721, #10921, #9332, #10756, #10717, #10630, #10497

## Executive position

A multi-agent system is not necessarily an adaptive research system.

A workflow may have separate literature, hypothesis, experiment, implementation and analysis agents yet call them in the same order regardless of what the evidence says. In that case, role decomposition has increased system complexity without demonstrating better research judgment.

The central experiment is:

> Holding the available roles, tools, model backend and total budget constant, does choosing the next research action from the current evidence state improve independently verified scientific outcomes?

The unit of comparison is the orchestration policy, not the number of agents.

## External research motivation

### Robin — evidence loop with a largely fixed sequence

Ghareeb et al., *A multi-agent system for automating scientific discovery*, *Nature* 655, 497–505 (published 19 May 2026), integrates literature search agents (Crow and Falcon), hypothesis generation, scientist-executed experiments and data analysis (Finch). The paper reports Robin identifying ripasudil and KL001 as in-vitro candidates for dry AMD retinal pigment epithelium phagocytosis enhancement; follow-up RNA-seq analysis suggested ABCA1 upregulation.

This is a meaningful research-workflow comparator, but the claim boundary matters:

- Scientists performed the experiments and supplied the data; Robin is lab-in-the-loop, not autonomous physical laboratory execution.
- Its proposed assay outlines required human scientists to produce executable laboratory protocols.
- Finch analysis required domain-expert prompt engineering.
- The authors observed that tools were called in nearly the same order and translated the workflow into a streamlined Jupyter notebook for stability.

The opportunity for Symthaea is not merely to reproduce Crow/Falcon/Finch roles. It is to test whether evidence-contingent method selection, prospective EVSI, counterfactual branch responsiveness, adaptive stopping and claim/evidence checks improve the research loop.

Source: https://www.nature.com/articles/s41586-026-10652-y

### Broader methodological framing

The *Nature Reviews Methods Primers* article “Agentic AI systems for scientific discovery” (published 22 October 2026) describes traceable workflows coordinating literature, data, theory, simulation and experiment. It identifies hallucination, tool misuse, interpretability, safety and reproducibility as practical bottlenecks. This supports evaluating the decisions and evidence behind orchestration, not counting agent roles.

Source: https://www.nature.com/articles/s43586-026-00534-y

## Paradigm shift: from workflow graphs to evidence-contingent policies

Treat orchestration as a policy:

`next_action = pi(current_research_state, available_actions, budget, risk_policy)`

The research state must be a versioned, evidence-prefix-bound snapshot. A policy decision should include:

- state and available evidence-prefix identity;
- admissible next actions and resource costs;
- hypothesis, assumption or uncertainty the action addresses;
- predicted outcome distribution and what each outcome would change;
- gross EVSI, direct cost, opportunity cost and net decision value;
- downside-loss model, CVaR confidence and risk ceiling;
- stop/continue/pivot/replicate/escalate criterion;
- chosen action, rejected alternatives and reason.

A tool call is not a reward. An agent handoff is not an improvement. More exploration is not automatically better. The policy must justify why the next action is worth its cost and why the evidence available now supports that branch.

## Synthetic counterfactual branch controls

The initial machine-readable pack is:

- `SYM_RSI_ORCHESTRATION_COUNTERFACTUAL_FIXTURES_V0_2026-11-07.json`
- Companion schema: `SYM_RSI_ORCHESTRATION_COUNTERFACTUAL_SCHEMA_V0_2026-11-07.json`

It contains six synthetic, known-policy cases:

1. **Decision-relevant signal:** discriminating evidence crosses a frozen decision threshold; action should change.
2. **Irrelevant evidence:** formatting-only metadata changes; preferred action should remain invariant.
3. **Contradictory evidence:** a negative control contradicts an intermediate mechanism prediction; pivot or replicate based on control validity.
4. **Non-identifiability:** allowed actions cannot distinguish competing hypotheses; abstain or request a genuinely discriminating intervention.
5. **Negative net value:** positive gross EVSI is outweighed by direct plus opportunity cost; do not run the experiment.
6. **Downside-risk veto:** positive net value does not override a failed CVaR ceiling; redesign or reject.

These controls jointly test **responsiveness** and **invariance**:

- A fixed script fails when decision-relevant evidence should change the next action.
- A random controller may change actions frequently but fails when irrelevant evidence should leave the action unchanged.
- A brittle threshold policy may pass simple cases but fail when the evidence is contradictory, non-identifying or risk-constrained.

Passing synthetic controls is policy-contract conformance only. It is not proof of scientific reasoning, model calibration or scientific discovery.

## Treatment arms

On identical tasks, seeds, starting observations and inclusive budgets, compare:

1. **Fixed script:** predeclared role/tool order, no outcome-dependent branch except error recovery.
2. **Fixed sequential best-of-N:** same stage order, multiple candidate outputs or analysis trajectories within the same total budget.
3. **Static threshold controller:** same workflow with frozen uncertainty/disagreement thresholds for optional retrieval or verification.
4. **State-contingent adaptive controller:** chooses the next role/tool/experiment using the typed research state and prospective decision-value forecast.
5. **Matched-cost random branch controller:** selects among permitted next actions randomly while matching calls and total resource use.
6. **No-optional-tool control:** core workflow with optional retrieval/verification/analysis disabled where valid.
7. **Human-designed workflow:** experts specify the sequence and prospective predictions; record their time and disagreement.
8. **Oracle diagnostic:** only in tiny synthetic worlds where all branches can be enumerated. Never present this as a feasible agent or competitive model.

All treatments use the same role/tool/model inventory. The only intended treatment is orchestration policy. Adding more agents, tools, tokens, context, model strength or human intervention to the adaptive arm invalidates a matched-budget comparison.

## First execution substrate

Compose with AS-Bench from #10921:

- **Nuclease replay:** multiple experimental rounds, discovery hits and family diversity, followed by held-out ranking.
- **Propylene digital twin:** adaptive catalyst design selection followed by held-out ranking; preserve the measured-versus-surrogate provenance distinction.

Keep the upstream discovery and learning gates unchanged. The AS-Bench adapter is the interface; this issue owns only the orchestration-policy comparison.

Start with offline synthetic branch tests and replay. Move to the digital twin after replay API/ledger conformance is established. No live-lab execution is required for this work.

## Causal design and anti-confounding controls

- Freeze model/backend, model version, reasoning mode, role set, tool registry, permissions, prompts, memory/context policy and task inputs.
- Pair arms on the same task realization, seed and initial evidence where possible.
- Freeze task-family splits, scorer version, outcome definitions, budget envelope, stopping rules and exclusions before confirmation.
- Keep best-of-N and parallel-agent comparisons separate from adaptive orchestration. Their added compute/candidate count must be charged to the same inclusive budget.
- Use intact, disabled-consumption, sham and rescue interventions for a proposed controller mechanism where technically valid.
- A telemetry predictor that cannot change action selection is not evidence of adaptive control.
- Keep a task's hidden truth and held-out evaluator materials outside the controller and all research roles.
- Preserve all runs: accepted, rejected, failed, invalid, no-op, incomplete, timed-out and not-run.

## Metrics

Report a vector:

### Scientific outcomes
- AS-Bench discovery score and gate;
- held-out learning score and gate;
- joint campaign success;
- per-task/per-family and worst-family performance.

### Decision quality
- prospective action prediction and calibration;
- realized decision regret and validated value of information;
- correct continue/pivot/replicate/stop/escalate rate;
- premature commitment and correct unresolved rate;
- EVSI forecast quality and utility/risk sensitivity.

### Counterfactual policy behavior
- relevant-evidence action accuracy;
- irrelevant-evidence action invariance;
- contradiction response correctness;
- non-identifiability abstention correctness;
- cost and risk gate compliance;
- branch-policy violations and unexplained action changes.

### Integrity and cost
- claim/evidence integrity and unsupported-claim rate;
- model/API tokens and costs;
- tool calls by type;
- CPU/GPU/energy where measurable;
- experiment/query budget;
- evaluator calls, retries, failed candidates and invalid runs;
- wall-clock and human review/rework.

Do not reward call count, role count, output tokens, trajectory length or agent complexity. If the fixed workflow is Pareto-equivalent, prefer it.

## Confirmatory design

Use a debugging pilot to check state-prefix binding, action traces, job lifecycle, budget accounting and counterfactual controls. It may use five paired seeds where tasks are stochastic, but it is not a superiority result.

Set confirmatory repetitions through a preregistered precision/power analysis that accounts for task-family clustering, multiple comparisons and any sequential looks. Separate development, calibration, sealed confirmation and fresh-family transfer. Do not tune the branch policy on sealed confirmation outcomes.

Require a positive effect on a preregistered primary endpoint against the strongest simple controller, with no integrity or protected-capability regression. A gain on one campaign does not establish general orchestration intelligence.

## Limitation-to-method-response obligations (v0 extension, 2026-11-26)

The DISCERN L2 results expose a failure that ordinary issue recognition does not catch: a researcher may name a confound or limitation yet leave the contaminated analysis and conclusion unchanged. An L3 agent may accept an adversarial critique without producing a more evidence-faithful hypothesis. This protocol therefore requires every consequential limitation to become an explicit **response obligation** in the existing prospective action trace.

Schema fragment:
- `SYM_RSI_LIMITATION_RESPONSE_TRACE_SCHEMA_V0_2026-11-26.json`

It references existing authority records only. It must not duplicate the ResearchState, evidence graph, action-offer list, executor receipt store, auditor identity registry or adjudication authority.

### Required binding

Each limitation obligation binds:
- the exact serialized ResearchState SHA-256;
- the exact evidence-prefix SHA-256;
- the active decision ID and, where applicable, scoped-claim digest;
- origin of the observation: policy self-detected, independent auditor, domain expert, executor observation or benchmark control;
- typed limitation class, affected claim/evidence IDs, severity and consequence;
- required response and the exact currently permitted upstream action-offer ID (or null when no action is permitted);
- prospective predicted outcome, decision consequence, success condition and reversal condition;
- after execution, the upstream action receipt and the exact post-action ResearchState/evidence-prefix digests;
- independent adjudication disposition and authority reference when the obligation is marked adjudicated or waived.

### Response semantics

The required responses are action intents, not executable commands or permissions:
- `REANALYZE`, `ADD_CONTROL`, `REPLICATE`, `SENSITIVITY_ANALYSIS`, `REVISE_SCOPE`, `SEEK_INDEPENDENT_ADJUDICATION`, `ABSTAIN`, `JUSTIFIED_NO_CHANGE`.

A non-material limitation may justify no method change, but that decision must explain why the limitation does not change the target estimate/claim and must have a prospective success/reversal condition. It is not acceptable to silently discard a detected limitation.

A critical limitation that invalidates the analysis cannot be marked resolved by prose alone. It needs an executed permitted response and the resulting state/evidence receipt, or a separately authorized abstention/stop decision. A model's own confidence cannot adjudicate its response. A reviewer concession is not proof that the revised claim is more accurate.

### Lifecycle and non-promotion

1. **OPEN** — limitation identified and bound to the current state/prefix.
2. **RESPONSE_PLANNED** — response intent and prospective predictions recorded against an existing permitted action offer or a justified no-change/abstention.
3. **EXECUTED** — actual action receipt and post-state/evidence-prefix digests recorded; failures remain visible.
4. **ADJUDICATED** — an independent authority marks the consequence resolved, not resolved, unresolved or waived.
5. **WAIVED_WITH_REASON** — only for a justified no-change response with an explicit independent authority and reason.

Creating an obligation never mutates ResearchState, grants a new offer or schedules a tool. It is consumed by #11015's existing policy trace and action owners. After the executor returns, the policy must read the new typed ResearchState; the obligation cannot be closed against the stale pre-action state.

### Required controls

Add cases for:
- issue recognized, but analysis/result unchanged;
- confound identified and appropriate reanalysis or control executed;
- scope mismatch handled by narrowing the claim rather than inventing a method;
- low power handled by replication or a justified uncertainty/abstention;
- non-identifiable alternatives handled by a genuinely discriminating experiment or explicit unresolved set;
- non-material limitation with justified `JUSTIFIED_NO_CHANGE`;
- valid auditor objection accepted and method revised;
- invalid auditor objection rejected with exact evidence-backed rationale;
- accepted critique that worsens evidence fidelity (must not count as success);
- action execution fails or times out (obligation remains unresolved and the failed receipt stays visible);
- state/evidence-prefix changes between planning and execution (stale plan is rejected/replanned);
- candidate attempts to self-adjudicate or mark its own obligation resolved.

### Metrics

Report separately:
- limitation detection precision/recall and severity-weighted recall;
- response appropriateness by limitation class;
- recognition-without-response rate;
- method-change / claim-change / justified-no-change rates;
- proportion of executed responses that satisfy the frozen prospective success condition;
- independently adjudicated resolution rate;
- post-response evidence fidelity and claim calibration;
- justified abstention, unjustified abstention and premature commitment;
- valid critique acceptance, invalid critique rejection and revision regression;
- action cost, wall time and human-review burden.

The primary endpoint is not “limitations mentioned” or “criticisms accepted.” It is **whether valid limitations are translated into the correct, evidence-bound methodological response, and whether that response improves or preserves independently adjudicated evidence fidelity**.

### External target and claim ceiling

The DISCERN adapter/evaluation is tracked in [#13852](https://github.com/Luminous-Dynamics/symthaea/issues/13852). This v0 schema fragment is a trace-contract proposal; it is not yet wired into the production policy or validated by a Rust implementation. Passing synthetic fixtures would establish schema/trace conformance only, not scientific judgment or general autonomous discovery.

## Ownership composition

- #10877 — research taste and interpretation;
- #9230 — direction allocation and diversity;
- #8721 — prospective tool/computation allocation;
- #10921 — AS-Bench replay/twin compatibility;
- #9332 — generated task quality and sealed oracle;
- #10756 — evaluator integrity;
- #10717 — causal attribution;
- #10630 — research-state continuity;
- #10497 — R4 successor/improver qualification.

Do not add another general scheduler, evaluator, receipt service, memory store, signer or promotion authority.

## Acceptance criteria

- Treatment arms use the same model, roles, tools, permissions and inclusive budgets.
- Adaptive decisions are bound to the evidence prefix and record prospective branch predictions, costs and risk.
- Synthetic tests show both decision-relevant responsiveness and irrelevant-evidence invariance.
- Fixed, static-threshold, random and adaptive policies are compared under matched budgets.
- AS-Bench discovery and held-out learning remain independent non-compensating gates.
- Every failure, invalid run, human intervention and resource cost is accounted for.
- Transfer to fresh task families and a fresh scientific domain precedes broad claims.
- If adaptation adds no validated value, retain the simpler fixed workflow and the null result.
- R4 remains governed exclusively by #10497.

## Explicit limits

Robin's laboratory work was performed by scientists; it is not an autonomous robot-laboratory baseline. AS-Bench replay and twin results are not live experiments. The counterfactual fixture pack is synthetic, not expert gold. This issue does not authorize unsupervised wet-lab execution and makes no SOTA, discovery or R4 claim.
