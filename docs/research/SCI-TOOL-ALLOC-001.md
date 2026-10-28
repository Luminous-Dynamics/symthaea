# SCI-TOOL-ALLOC-001: Prospective Tool Allocation for Scientific Law Discovery

**Status:** proposal / benchmark protocol  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8721  
**Version:** 0.1.0  
**Evidence class:** research plan; no measured results claimed

## 1. Research question

Does choosing *when and which scientific-reasoning tools to use* improve independently validated law discovery, or can additional tools push a discovery agent into premature exploitation, amplify early mistakes, and consume resources without improving the final scientific result?

The benchmark tests this rather than presuming that tools help or hurt. It specifically separates the effect of a tool being available from the effect of a policy choosing to use it. It also tests capability-conditioned tool effects: the same strategy may help a capable model and harm a weaker one, so pooled averages alone are insufficient.

## 2. Why this deserves a distinct track

NewtonBench evaluates interactive scientific law discovery over 324 tasks across 12 physics domains, including altered laws, different law/system complexity levels, and model probing. Its report discusses sensitivity to noise and complexity and cases where tool assistance can encourage premature exploitation. This is evidence that a tool-use assumption should be experimentally tested, not a universal law that tools hurt.

- NewtonBench paper: https://arxiv.org/abs/2510.07172
- NewtonBench public repository: https://github.com/HKUST-KnowComp/NewtonBench
- CoSR progressive law discovery, accepted 6 October 2026: https://doi.org/10.1103/f2kp-pp4q
- Science-Gym, data collection + experiment design + equation discovery: https://doi.org/10.1007/s10994-025-06914-x
- **AgentIdeaBench** (arXiv:2609.07611, submitted 7 September 2026) compares active versus static literature access across 33 matched models and 40 subfields. The authors report an active-minus-static effect correlated with static capability (r = +0.69, p < 10⁻⁴, n = 28), with the weakest quartile losing 0.18 and strongest gaining 0.76 on their scientific-ideation score. This is adjacent ideation evidence—not a law-discovery result or a direct NewtonBench comparison: https://arxiv.org/abs/2609.07611 ; https://github.com/HKUST-KnowComp/AgentIdeaBench
- 2026 symbolic-regression survey: https://doi.org/10.1007/s11831-026-10681-w

The contribution is not another scientific evaluator or tool-use framework. It is a controlled application of prospective metareasoning to a task where tool effects can be measured against independently known answers.

## 3. Compose with existing Symthaea work

- #8487: scientific discovery recovery protocol and baseline registry.
- #8671: progressive law discovery and the validity/negative-transfer of intermediate knowledge.
- #5476: prospective value-of-computation forecasts and realized-utility receipts.
- #5478: causal metareasoning ablations and architecture discrimination.
- #8286: active probe policy, intervention cost, and exact small-world oracle.
- #8705: cross-environment law transportability.

Keep these concerns distinct:
- **experiment selection** chooses what scientific observation to acquire;
- **tool allocation** chooses which computation or reasoning operation to run on current evidence;
- **hypothesis selection** chooses which candidate law to retain;
- **stopping** decides whether available evidence is sufficient.

A tool-use policy can influence all four, but its contribution must be measured separately. Do not duplicate the existing CognitiveBid / VoC contract, operation registry, PhysicsIR, or evaluator.

## 4. Tasks and splits

Prefer an adapter over NewtonBench's public tasks when license, runtime, and tool contracts permit. Preserve the original task split and published scorecard separately. Document any incompatibility; a custom subset is not a NewtonBench reproduction.

If the adapter is not yet runnable, start with a small deterministic, versioned synthetic pilot. It must not be presented as a reproduction of an external benchmark.

The initial panel should include:
1. simple laws where additional tool use is unnecessary;
2. complex target laws embedded in multi-equation systems;
3. low, moderate, and high observation noise;
4. misleading early observations where multiple candidates fit the current evidence;
5. cases where a specific verifier or symbolic operation is decisive;
6. cases where the available tool is irrelevant or its cost dominates;
7. non-identifiable worlds where no allowed tool call can distinguish the candidates within budget;
8. cases with hidden-context or environment shifts that expose an early overfit;
9. a null family in which adaptive scheduling should not outperform a simple static policy.

Stratify and report law complexity, system complexity, observation noise, and early-evidence ambiguity separately. Keep final holdout worlds hidden from candidate ranking, tool selection, policy tuning, and stopping.

### Pre-treatment model-capability blocking factor

Where several models or model versions can be tested, measure each one's capability on a separate **tool-free calibration set** before treatment outcomes are inspected. Freeze the calibration tasks, score and capability-tier cutpoints in advance. Do not define tiers from the same tasks used to estimate tool effects, and do not use post-treatment outcomes to decide which tier a model belongs to.

For each model, pair no-tool and tool-allocation policies on the same held-out task/seed cells. Report effects by preregistered capability tier as well as pooled effects. A pooled positive average must not hide a negative effect for a weaker tier. With only one model, make no cross-model capability-interaction claim.

## 5. Policy and tool information boundaries

The policy may read:
- public variable names/types/units and scenario constraints;
- observations already acquired through allowed experiments;
- outputs from tool calls it has actually made;
- public tool descriptions, prices/costs and resource caps;
- its own candidates, scores, forecasts and action history.

The policy must not receive the hidden generating equation, evaluator truth labels, hidden environment state, future observations, final-holdout scores, or oracle utility for unselected tool actions.

Separate evaluator-only truth and scoring from policy-visible input types. Test the boundary by injecting an evaluator-only field through a negative-test adapter and requiring rejection. A written assertion without a test is not a leakage control.

## 6. Tool action taxonomy

Use existing implementations and represent actions by explicit type, not a generic "tool call" count:

- **Scientific experiment/query:** request a model-system observation.
- **Symbolic simplification/normalization:** transform a candidate expression while preserving declared semantics.
- **Dimensional/type check:** check a candidate against PhysicsIR contracts.
- **Numerical fitting/solver execution:** fit parameters or evaluate a candidate numerically.
- **Retrieval/knowledge lookup:** access a permitted source or catalog.
- **Verification/counterexample search:** attempt to falsify a candidate over a declared domain.

Each type can change the state of knowledge in different ways. Log the pre-action evidence digest, declared expected benefit/cost, action and parameters, result digest, post-action state/evaluator-relevant change, and any failure. Reuse established receipt patterns; do not create a cryptographic format or claim that a digest authenticates execution.

## 7. Matched treatment arms

All comparable methods receive identical task worlds, model identity/version, prompt, starting observations, permitted action set, maximum budget, context constraints, and downstream evaluator.

Register:
1. **No optional tools** — baseline for tool-free behavior.
2. **Always-available tool contract** — all named tools are available with a frozen call policy.
3. **Static threshold** — use a specific tool once a frozen uncertainty/disagreement/residual threshold is crossed.
4. **Fixed schedule** — calls occur at preregistered discovery stages.
5. **Random matched-cost policy** — random feasible operation with matched resource limits.
6. **Declared heuristic VoC** — score usefulness using a non-calibrated, frozen heuristic.
7. **Prospective calibrated VoC** — predict utility and resource cost before outcome exposure; use #5476's forecast/outcome contract.
8. **Exploration-reserve policy** — reserve a frozen portion of the observation budget for candidate-discriminating experiments before commitment.
9. **Exact oracle upper bound** — on small enumerable synthetic worlds only, select the next action with the greatest true evaluator utility. Label this as an oracle diagnostic, never a feasible method or a fair winner.

"Always available" does not mean "always called"; record availability and invocation separately. If a published baseline includes a fixed tool strategy, preserve its original contract. Do not quietly improve or weaken it.

## 8. Resource accounting and causal attribution

Freeze and record separate caps for:
- scientific experiment/query count;
- optional tool calls by type;
- candidate evaluations;
- input/output tokens or another declared model-work proxy;
- wall-clock time;
- CPU/GPU or solver compute where reliable measurements exist;
- memory and other resource measurements when available.

Report two different comparisons:
- **equal maximum budget:** each policy may stop early but has the same allowed resources;
- **cost-to-target:** resources consumed to reach the same independently validated discovery criterion.

Where possible use paired worlds and common random seeds. Keep backend, prompt, reasoning mode, context capacity, operation registry, downstream evaluator and experiment budget fixed. An improvement due to a stronger model or a larger context is not evidence for scheduler quality.

Use causal ablations for individual scheduler components where safe and informative:
- intact scheduler;
- telemetry computed but not consumed (sham);
- named decision feature disabled;
- matched static or random policy;
- rescue restoring the exact decision edge.

Maintain identical upstream state until the intervention point when deterministic replay allows it. Missing counterfactual outcomes are not zero utility. Prefer exhaustive branch evaluation on tiny synthetic worlds, state-isolated shadow runs, or randomized allocation where safe.

## 9. Outcomes

### Primary outcomes
- **Validated recovery:** correct law/equivalence-class recovery confirmed on fresh, independent observations.
- **Cost to validated recovery:** report query, tool, compute and wall-clock costs separately.
- **Premature commitment rate:** the policy claims a uniquely supported law before the preregistered discriminating evidence has been obtained.
- **False confident-singleton rate:** a confident unique law is returned when evaluator truth is absent, unsupported or non-identifiable.
- **Paired tool effect:** change in recovery, predictive error, false confidence and cost relative to no-tool and strongest simple-policy arms.

### Secondary outcomes
- In-domain prediction and true-range extrapolation.
- Held-out environment/intervention performance when a task satisfies #8705.
- Candidate diversity and posterior/credible-set quality when the policy provides calibrated probabilities.
- Prospective VoC Brier/log score, reliability calibration, cost-prediction error and rank regret.
- Experiment selection quality, separately from compute/tool allocation.
- Number of tool calls by type, candidate attempts, context/token use, time, memory and failures.
- Performance by law complexity × system complexity × noise × early ambiguity.
- Tool-treatment effect by pre-treatment, tool-free capability tier, with per-tier denominators and paired uncertainty intervals.
- Model-capability × tool-policy interaction; report treatment-effect heterogeneity rather than assuming a universal benefit.
- Run-to-run variance, missing outputs, invalid actions, unresolved and not-evaluable rates.

Do not collapse these into one weighted score. Publish per-stratum results and the Pareto frontier for validated recovery, false confidence, unresolved behavior, and total cost.

## 10. Pilot and confirmatory programme

### Pilot
- Freeze a small subset before running.
- Use at least five paired seeds per task for harness development.
- Validate known-answer world generation, action-cost accounting, forecast-before-outcome ordering, policy/evaluator separation, deterministic replay, and independent scoring.
- Use pilot results only to repair harness semantics. They are not confirmatory performance evidence.

### Confirmatory run
- Freeze task/family splits, model/prompt/tool contract, resource caps, primary endpoint, scoring, exclusions and stopping rules before inspecting the outcomes.
- Target six or more structural families and 20 paired seeds per held-out world, harmonizing with the broader science-discovery plan. Increase sample size if a preregistered precision/power calculation demands it.
- Preserve the complete task/seed denominator and all terminal outcomes.
- Require calibrated VoC to improve a preregistered endpoint against the strongest simple policy under matched budgets, with no worsening of false-confidence or task-success guardrails, before calling it better.
- If a static threshold is Pareto-equivalent to adaptive VoC, prefer the simpler policy.
- If tools help only in certain regimes, declare an applicability envelope instead of claiming universal gains.
- If all advanced policies fail to beat a simple baseline, retain the null result and simplify the architecture.

## 11. External comparisons

### NewtonBench
Start with a compatible public subset if technically and legally feasible. Report its published metrics separately from this protocol's metrics. Pin repository revision, model version, prompts, tools, task manifests, environment config and resource settings.

### PySR / SymbolicRegression.jl and AI Feynman
Use these as symbolic-search comparators on tasks and interfaces they support. The question in this protocol is whether a tool-allocation policy improves the surrounding scientific workflow—not whether a metareasoning layer automatically makes a symbolic-regression package better.

### CoSR and other progressive methods
Use when implementations and training/evaluation contracts are accessible. Keep intermediate-unit quality and reuse outcomes (from #8671) separate from tool-scheduling quality.

### Science-Gym and ACDB
Use the closest compatible task/probe contract rather than claiming a direct score comparison across unrelated task semantics.

A cited paper's reported result is not a reproduced baseline. Do not rank methods whose task split, model/tool access or resource accounting cannot be aligned.

## 12. Reproducibility receipt

Each run should link:
- protocol and scoring revision;
- exact task/family/split/seed manifest;
- model, prompt, tool registry and configuration identities;
- allowed and consumed resource budgets;
- forecast-before-action records and action trace;
- output/model-candidate digest;
- sealed evaluator manifest reference and evaluator version;
- terminal result and score artifact digest.

Keep policy-visible receipts separate from truth-bearing evaluator receipts. A content digest proves byte-level commitment only; it does not prove execution authenticity, protect secrecy without separate custody, or certify scientific truth.

## 13. Proposed acceptance gates

1. The pilot's known-answer and leakage negative controls pass.
2. Tool-call cost and scientific query cost cannot be silently conflated.
3. Every forecast is demonstrably created before its outcome is visible.
4. No invalid, missing, failed or unresolved result is dropped from the denominator.
5. All baselines use the same allowed tool and experiment contracts where they are compared.
6. The confirmatory analysis shows paired estimates and uncertainty intervals, per-stratum outcomes and complete resource accounting.
7. No policy is called superior solely from self-reported confidence, training fit or a post-hoc threshold.
8. Exact-head implementation tests and independent evaluation are required before implementation qualification or SOTA claims.

## 14. Claim ceiling

This protocol can eventually support a bounded claim that a specific policy improves selected scientific-discovery outcomes under named task/model/tool/resource profiles.

It does not currently claim a reproduction of NewtonBench, a generally harmful effect of tools, a qualified Symthaea metareasoner, or superiority over any industry system. Those claims require executable, matched, independently evaluated evidence.
