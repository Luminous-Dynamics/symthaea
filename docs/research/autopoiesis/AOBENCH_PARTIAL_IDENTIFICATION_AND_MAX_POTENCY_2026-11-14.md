# AOBench — Decision-Focused Partial Identification and Max-Potency Experiments

**Status:** Research specification and benchmark protocol; no implementation or benchmark score claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-013 #12206](https://github.com/Luminous-Dynamics/symthaea/issues/12206)  
**Research basis:** [Maringgele & Etesami, *Optimal Experiments for Partial Causal Effect Identification* (NeurIPS 2026)](https://arxiv.org/abs/2605.06993)

## 1. Why this changes the objective

Causal discovery benchmarks commonly ask whether the agent can recover the full directed graph. That is not always necessary for a justified decision, and sometimes it is not possible.

The partial-identification line studies what can be concluded about a target causal effect when the available data and assumptions do not identify one point value. The result is an identified set or interval. Experiments can be chosen to tighten that interval, even when full graph or mechanism identification remains impossible.

Maringgele and Etesami formalize a cost-constrained **max-potency** problem: choose a subset of experiments before observing their outcomes to maximize the guaranteed reduction in the target effect's bound width. They prove the optimization NP-hard by reduction from 0–1 knapsack. Their two graph-based pruning rules—a path-interception criterion using district structure and an identifiability check based on the ID algorithm—prune 50–88% of candidate experiments on average across Erdős–Rényi graphs and 11 bnlearn networks without solving polynomial programs. The paper includes an end-to-end NHANES demonstration for physical activity and diabetes.

This is a different objective from maximizing expected information gain or recovering every edge.

## 2. Core distinction: structural uncertainty versus decision uncertainty

AOBench should ask two separate questions:

1. What causal structures/mechanisms remain admissible?
2. Do those admissible worlds still support the same decision?

If every admissible world implies the target effect is above a decision threshold, the agent may be justified in acting without identifying the full graph. If the effect interval straddles the threshold, a precise-looking DAG is not sufficient to authorize a decisive action.

Conversely, if the identified interval is wide but entirely on one side of the threshold, additional graph recovery may have little decision value.

## 3. Formal task contract

Let the target estimand be \(\theta\), with current identified set \(I_0=[L_0,U_0]\). A candidate experiment set \(S\) has total cost \(c(S)\) and risk \(r(S)\). For possible joint outcomes \(y\), the post-experiment identified set is \(I(S,y)\).

Define worst-case residual width:

\[
W_{\max}(S)=\max_{y\in\mathcal{Y}(S)} \operatorname{width}(I(S,y)).
\]

Define guaranteed epistemic potency:

\[
P(S)=\operatorname{width}(I_0)-W_{\max}(S).
\]

The selection problem is to maximize \(P(S)\) subject to frozen cost, risk, intervention and trial budgets. Expected information gain remains a separate comparator; it is not interchangeable with guaranteed worst-case narrowing.

The evaluator must freeze:
- the estimand and target population;
- admissible causal graphs/mechanisms and identifying assumptions;
- the current identified set and method used to derive it;
- intervention semantics, interference/carryover and measurement contracts;
- the candidate experiment set and costs/risks;
- the outcome-to-bound update rule;
- the robust action threshold and loss function;
- the exact oracle for small enumerable instances.

## 4. Task ladder

### P1 — Bound validity

Given a finite admissible model class, verify the interval contains every evaluator-admissible true effect. A falsely narrow interval is a hard failure.

### P2 — One experiment, worst-case potency

Select one experiment that maximizes the worst-case reduction in interval width. Compare it against expected-information-gain and expected-width-reduction baselines.

### P3 — Budgeted experiment subset

Select a set of experiments before outcomes, subject to a cost/risk budget. Use exhaustive search as an oracle on small instances and report optimization gaps on larger instances.

### P4 — Graph-based zero-potency pruning

Identify experiments that cannot tighten the target bound under the declared graph/query conditions. Compare generic search with the path-interception and ID-based pruning ideas in the NeurIPS 2026 work.

### P5 — Sequential adaptive potency

After a real outcome arrives, update the identified set and choose the next probe. Score the realized interval contraction against the best safe remaining experiment.

### P6 — Robust decision

Given a threshold \(\tau\):
- choose action A only if every admissible effect is on the action-supporting side;
- choose action B only if every admissible effect is on the opposite side;
- otherwise defer or choose a preregistered minimax-regret action.

AOBench must penalize false decisive actions when the identified set straddles the threshold.

### P7 — Assumption stress and world transfer

Introduce latent confounding, selection bias, measurement error, interference, nonstationarity and model-class omissions. Require the agent to widen or invalidate bounds when assumptions fail, rather than silently preserving unjustified precision.

## 5. Baselines

- Random experiment subset.
- Greedy expected information gain.
- Greedy expected interval-width reduction.
- Greedy max-potency (worst-case contraction).
- Exact max-potency subset optimizer for enumerable small instances.
- Graph-pruned max-potency optimizer based on path-interception and ID criteria.
- Classical partial-identification estimators and bounds where their assumptions apply.
- AOBench sequential planner with independently qualified interventions, bound validation and robust-action rules.

All methods must share the same graph/query, candidate experiments, cost/risk budget, outcome semantics, data regime and evaluation oracle.

## 6. Metrics

Keep these separate:
- initial and final identified-set width;
- worst-case guaranteed contraction and normalized potency;
- realized versus guaranteed contraction;
- expected information gain as a comparator;
- experiment-set cost and risk;
- fraction of zero-potency experiments pruned;
- exact-oracle optimization gap;
- invalid-bound / undercoverage rate;
- calibration and interval coverage under stochastic extensions;
- robust action correctness across every admissible world;
- false decisive-action rate when the interval straddles the threshold;
- correct defer rate and minimax regret;
- intervention delivery, isolation and collateral-effect qualification.

No weighted composite may allow decision success to compensate for invalid bounds.

## 7. Proposed qualification targets (not achieved)

- 100% correct interval validity on deterministic finite fixtures.
- Zero accepted bounds that exclude an evaluator-admissible true effect.
- Exact max-potency optimum on all enumerable small cases.
- Zero false decisive actions when the admissible interval crosses the threshold.
- At least 95% correct robust-action/defer decisions on sealed deterministic tasks.
- Report graph-pruning yield, runtime and solution quality separately.

These thresholds must be frozen after pilot calibration and before sealed evaluation.

## 8. Graph-pruning implementation boundary

The current Rust increment implements the first published graphical criterion:

1. Build the ADMG reachability graph by representing each bidirected edge as a latent response-type node pointing to both observed endpoints.
2. Remove the query treatment and compute response types that can still reach the target outcome.
3. Take the district hull over the bidirected components to obtain the relevant latent response-type set.
4. For each candidate experiment, remove its intervention variables and test whether any path remains from that district hull to its measured-variable set (the published AllPathsBlocked criterion).
5. Cross-check every graph certificate against the exact finite-world oracle's singleton worst-case potency. A disagreement is surfaced and the experiment is not treated as prunable.

The current evaluator reports:
- path-interception candidate count;
- graph/finite-world confirmed zero-potency count;
- graph/finite-world disagreement count.

**Important limitation:** the graph certificate is currently an auditable candidate/pruning diagnostic. It is not yet removed from the exhaustive subset oracle, because this first finite-world representation provides arbitrary response tables rather than a complete SCM certificate proving every graphical assumption. The ID-algorithm identifiability test from the paper is also not implemented yet. Those are separate follow-on increments; do not claim the reported candidate count is a search-space reduction.

## 9. Implementation path

1. Define a finite exact identified-set representation and interval arithmetic.
2. Add a deterministic oracle over a small family of admissible worlds and intervention outcomes.
3. Implement the max-worst-case-width objective and exhaustive small-instance oracle.
4. Add cost/risk-constrained selection and solution-gap reporting.
5. Add independent robust-action qualification and false-decision controls.
6. Complete subset-inertness validation for graph certificates, port the published ID-algorithm identifiability test, and only then remove certified candidates from the exact subset search.
7. Extend to probabilistic/stochastic models only with coverage, calibration, multiplicity and stopping guarantees.
8. Connect to AOBENCH-006 intervention integrity, AOBENCH-011 sequential experiment selection, AOBENCH-012 model-class refinement and #12048 stochastic assay adequacy.

## 10. Claim boundary

This finite evaluator will not be a general solution to partial identification in arbitrary physical systems. The estimand, target population, admissible model class, causal assumptions and bound semantics must be frozen and independently reviewed. A stochastic interval requires a separate coverage guarantee. A correct point action in a synthetic task does not establish general causal truth.

## 11. References

- [Maringgele & Etesami (2026), *Optimal Experiments for Partial Causal Effect Identification*, NeurIPS 2026](https://arxiv.org/abs/2605.06993)
- [Causal-Discovery-Bench](https://arxiv.org/abs/2611.09934)
- [ACED-Bench](https://arxiv.org/abs/2610.14381)
- [ACDB](https://arxiv.org/abs/2609.31675)
- [CausalArena](https://arxiv.org/abs/2609.11897)
- [TCD-Arena](https://tcd-arena.github.io/)
