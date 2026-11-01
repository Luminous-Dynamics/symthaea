# POD-Meta Proof Cost-to-Go and Hard-Budget Protocol V1

**Status:** research draft. Synthetic simulator only. No production scheduler integration and no external benchmark claim.

Issue: [#9652](https://github.com/Luminous-Dynamics/symthaea/issues/9652)

## 1. Research hypothesis

Operation-level success calibration is necessary but insufficient for proof-oriented compute allocation. A scheduler can correctly estimate that its next operation is promising and still waste the remaining budget because the complete proof frontier, mandatory verification, and recovery work are unaffordable.

The target of this experiment is therefore the **remaining cost to a sufficient, independently verified proof state**.

Let $G_t$ be the unresolved proof-obligation graph at decision time $t$, $x_t$ the execution and evidence context, and $C_t$ the random resource vector required to reach a sufficient verified proof. The forecaster emits a predictive distribution (or predeclared predictive quantiles) for $C_t$ and a completion-within-budget probability. It must bind that prediction to the exact state and frontier on which it was made.

The scheduler may use the forecast to decide whether to proceed, abstain, ask for missing evidence, or stop. It must not use the forecast as proof, as authorization, or as the mechanism that enforces a resource ceiling.

## 2. Proof-completion event

A successful event is not a tool response, planner assertion, partial obligation reduction, or a candidate evidence object. It is a verifier-confirmed state in which:

1. every mandatory obligation in the frozen task scope is discharged;
2. every accepted evidence item satisfies its authority, integrity, temporal, scope, and independence requirements;
3. the selected proof path is valid under the evaluator's proof rules;
4. all mandatory verification operations have completed; and
5. the final proof-state receipt binds to the authoritative post-action state.

The proof frontier is represented as a typed AND/OR obligation graph. AND edges encode mandatory dependencies. OR alternatives encode admissible ways to establish a claim. A candidate operation can only contribute to obligations it was explicitly authorized and selected to target.

## 3. Forecast receipt

Each forecast is immutable and pre-execution. It includes at minimum:

- run ID and monotonic decision sequence;
- pre-action state digest and proof-frontier digest;
- unresolved obligation IDs and mandatory verification reserve;
- forecast-model and calibration-cohort digests;
- task-family, obligation-family, operation-family, evidence-family, tool/runtime, and temporal cohort identifiers;
- probability of sufficient verified completion within the remaining budget;
- predictive quantiles (at least 50%, 90%, and 95%) for remaining work and each metered resource;
- calibration support count, uncertainty summary, shift state, and abstention reason if any.

The outcome is recorded in a separate evaluator / meter receipt. A forecast is never rewritten after execution. Corrections and revocations create new state and new forecast receipts.

## 4. Cost vector and hard-budget ledger

The first Rust slice uses integer resource dimensions:

- model tokens;
- tool calls;
- CPU milliseconds;
- GPU milliseconds;
- wall-clock milliseconds;
- spend in integer micro-currency units.

The resource ledger tracks a ceiling $B$, measured spend $S_t$, and outstanding reservations $R_t$. At every dispatch boundary, admission requires:

$$S_t^{(j)} + R_t^{(j)} + U_a^{(j)} \leq B^{(j)} \quad \forall j$$

where $U_a$ is the requested per-action upper reservation. The check is component-wise; there is no scalarized exchange rate between resources.

Lifecycle:

1. **Reserve:** atomically debit the available reservation before dispatch.
2. **Unknown:** after an ambiguous acknowledgement or unavailable read-back, retain the reservation. Do not retry a non-idempotent effect.
3. **Settle:** use authoritative metering to move the reservation into measured spend.
4. **Release:** release only after authoritative evidence that the action did not apply.
5. **Overrun:** record measured usage even when it exceeds its reservation or the ceiling. Never clamp, discard, or relabel the excess as zero.

A local ledger is not itself a physical enforcement mechanism. A hard-cap claim requires adapters that meter and cap actual calls before the provider or tool can exceed its limit. Adapters with unbounded, unmetered, or post-billed usage must be excluded from the corresponding hard-cap claim.

The simulator models serialized ledger transitions. A production concurrent implementation would need a linearizable reservation transaction (or equivalent single-writer journal) across all dispatchers.

## 5. Feasibility and admission

The initial feasibility rule is intentionally conservative:

$$P(\text{sufficient verified proof before remaining budget} \mid x_t) \geq 1-\alpha$$

and, for every resource dimension $j$,

$$Q_{0.90}(C_t^{(j)} \mid x_t) + V_t^{(j)} \leq F_t^{(j)}$$

where $V_t$ is the mandatory verification reserve and $F_t = B-S_t-R_t$ is available capacity.

A failed condition causes abstention, a request for evidence, or a calibrated infeasibility response. It does not authorize budget overruns. These marginal quantile checks are an initial decision rule, not a joint 90% coverage guarantee for a resource vector. The benchmark must report joint empirical coverage separately and evaluate a simultaneous vector-prediction method before claiming joint coverage.

No universal feasibility guarantee is claimed under arbitrary distribution shift.

## 6. Censoring and time-to-proof

A run stopped by its resource ceiling provides a lower bound on work-to-proof, not an observed exact completion cost. Budget exhaustion, timeout, interrupted infrastructure, permission loss, explicit abandonment, and ambiguous action outcomes are distinct terminal / censoring reasons.

The full V2 experiment should compare:
- complete-case estimates (diagnostic only; susceptible to selection bias);
- discrete-time hazard or survival baselines for time-to-sufficient-proof;
- censoring-aware baselines such as Kaplan–Meier and IPCW variants where independent-censoring and positivity assumptions are supportable;
- split-conformal / conformalized-quantile baselines on a separately declared uncensored exchangeable calibration subset.

Censoring caused by policy decisions can be informative. It must not be silently treated as independent. Competing terminal events must be reported separately. If the assumptions required by an estimator do not hold, report the limitation and do not claim its nominal coverage.

## 7. Shift and abstention

Evaluation partitions must separate at least:

- unseen task family;
- unseen obligation family;
- operation-family shift;
- evidence/source-family shift;
- temporal drift;
- tool or runtime version change;
- resource price or latency shift;
- label-conditional shift when observable.

Shift detection is a control input to admission, not a presentation-only metric. The conservative fallback is broad-prior prediction or abstention. Weighted conformal methods may be evaluated under explicit weighted-exchangeability / density-ratio and overlap assumptions. Arbitrary unknown shift has no blanket distribution-free coverage guarantee.

## 8. Benchmark arms

Use paired episodes, common random numbers where simulation permits, and the same evaluator outcomes for every policy.

1. Fixed catalogue order.
2. Budget-matched random.
3. Existing POD-Meta operation-success forecast selector.
4. Greedy expected verified progress per resource.
5. Remaining-cost-aware admission with mandatory-verification reserve.
6. Quantile / chance-constrained admission.
7. Hard-budget allocation with reflow of only settled or authoritatively released reservations.
8. Evaluator-only clairvoyant diagnostic; never describe it as deployable or as an exact optimum unless it actually solves the corresponding optimization problem.

No policy may bypass admissibility checks, proof-target binding, the hard ledger, or mandatory verification.

## 9. Metrics (no composite score)

### Forecast quality
- Brier score and log loss for completion-within-budget events;
- calibration intercept and slope, reliability plots, ECE with binning sensitivity;
- CRPS / interval score for remaining-cost distributions;
- 50%, 80%, 90%, and 95% quantile coverage, width, and quantile loss;
- marginal and joint resource-vector coverage;
- support counts, uncertainty, and drift-conditioned calibration.

### Decision quality
- verified obligations and fully verified completion;
- false-feasible rate and infeasibility detection delay;
- risk–coverage curve for abstention;
- cost to first sufficient proof;
- all-in expenditure per verified completion;
- unused capacity, reservation utilization, and reflow gain.

### Safety and systems
- actual hard-budget overruns;
- reservation overrun / adapter contract violations;
- false verified completion;
- missed mandatory verification;
- unauthorized or duplicated logical effects;
- unresolved unknown outcomes and concurrent reservation races;
- latency and energy only with a declared measurement method.

## 10. Statistical protocol

- Split by task family and time, not random adjacent rows alone.
- Keep fit, calibration, policy-selection, and final test partitions distinct.
- Freeze every forecast and policy decision before evaluator outcomes are revealed.
- Bootstrap at task / episode cluster level; report counts and confidence intervals.
- Include negative controls, data-leakage checks, forecast-label permutation controls, and deterministic replay.
- Keep synthetic abstract work units distinct from measured tokens, dollars, joules, GPU time, and wall-clock latency.
- Do not pool benchmark-native metrics into a weighted score.

## 11. Proposed research gates

These are project targets, not industry standards or demonstrated results:

- Zero actual hard-cap overruns in the enforceable adapter/fault suite.
- Zero false verified completions and zero omitted mandatory verification.
- Nominal 90% remaining-cost interval coverage near 90% on a predeclared in-domain held-out set, with uncertainty and interval width.
- Separate task-family and shift-partition coverage; no universal guarantee under arbitrary shift.
- At least 5% relative improvement in a proper scoring rule over the strongest simple prior on untouched families.
- A Pareto improvement in verified proof completion versus all-in expenditure at matched actor, tools, permissions, context, and ceilings.
- Exact-head Rust qualification, protected evaluator separation, and independent review before production integration.

## 12. External comparators and claim limits

- [BAGEN](https://arxiv.org/abs/2606.00198): progressive remaining-budget interval and infeasibility estimation for agents.
- [HARP](https://arxiv.org/abs/2610.07362): hard-budget allocation with adaptive reflow and finite-sample predictive bounds under its stated setup.
- [Proper Scoring Rules for Estimation and Forecast Evaluation](https://doi.org/10.1146/annurev-statistics-042424-050626): proper scoring rules and forecast evaluation.
- [Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019): weighted conformal under specified covariate-shift assumptions.

This lab is a research simulator and typed protocol. It is not a reproduction of those works, a production scheduler, a physical resource enforcer, or a state-of-the-art result.
