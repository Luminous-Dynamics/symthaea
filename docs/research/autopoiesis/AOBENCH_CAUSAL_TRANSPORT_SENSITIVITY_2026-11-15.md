# AOBench — Sensitivity-Bounded Causal Transport Across Populations

**Status:** Deterministic finite-distribution v0 implemented; Rust 1.96 exact-head qualification and broader sharpness/coverage study pending  
**Date:** 2026-11-15  
**Tracking:** [AOBENCH-019 #12278](https://github.com/Luminous-Dynamics/symthaea/issues/12278)  
**Implementation:** crates/domains/symthaea-organizational-closure/src/transport_bounds.rs

## 1. Research basis

Asiaee, Pal, Beck and Huling (CLeaR/PMLR 2026), *Sharp Bounds for Treatment Effect Generalization under Outcome Distribution Shift*, study target-population ATE bounds when conditional potential-outcome distributions differ between the source trial and target population.

For treatment arm a and measured covariates X, define the conditional outcome-density ratio:

    r_a(x,y) = f_target(y | A=a, X=x) / f_source(y | A=a, X=x)

Their sensitivity model constrains:

    1/Lambda <= r_a(x,y) <= Lambda
    E_source[r_a(X,Y) | X,A=a] = 1

Lambda = 1 recovers conditional outcome-distribution transportability. Larger Lambda permits larger departures. The extremal reweighting has threshold structure: put the largest allowed ratio on the highest outcomes for an upper-mean bound, or on the lowest outcomes for a lower-mean bound, with at most one partially allocated threshold atom. The paper gives an O(n log n) threshold algorithm and sharp target-effect intervals conditional on this sensitivity model.

Research:
- [Asiaee et al. (2026), PMLR 323, 1572–1603](https://proceedings.mlr.press/v323/asiaee26a.html)

A complementary 2026 line on externally controlled single-arm trials emphasizes that trial-to-external-population transport assumptions cannot be empirically proven from the external-control data alone. Under local violations, structural bias may be comparable to ordinary sampling fluctuation; discounting external borrowing creates a reliability–efficiency trade-off.
- [Lee (2026), Statistics in Medicine](https://doi.org/10.1002/sim.70748)

These are complementary approaches. The current Rust increment implements only the finite-distribution likelihood-ratio sensitivity model, not the external-control Bayesian borrowing procedure.

## 2. Paradigm shift

Do not turn a valid source-trial estimate into a target-population claim by silently assuming that populations are interchangeable.

AOBench should report one of:
- source effect identified;
- target effect identified under a registered Lambda = 1 transport contract;
- target effect partially identified under a declared Lambda sensitivity envelope;
- target effect unsupported because the target includes strata with no source support or the model assumptions are violated.

The sensitivity parameter is a registered assumption, not something the agent may tune after seeing target truth. A target effect being inside an interval supports the claim only conditional on the corresponding sensitivity model.

This is different from generic OOD accuracy. The evaluated object is the target causal estimand, its identifying assumptions, its sensitivity envelope and the decision justified by that envelope.

## 3. Finite-distribution v0 contract

The agent-facing scenario contains:
- a content identity for the source population;
- a content identity for the target population;
- a content identity for the frozen estimand;
- source randomized-trial rows with stratum, treatment assignment, known treatment propensity and factual outcome;
- target stratum masses;
- a preregistered Lambda;
- a decision threshold and a reported target-effect interval/decision.

The evaluator-only oracle contains:
- target-population potential outcomes for each target unit;
- target population weights;
- the expected source/target/estimand identities.

No target potential outcome or target effect is exposed in the agent-facing scenario.

### Source arm distribution

Within every measured stratum x and treatment arm a, v0 forms a normalized empirical outcome distribution from the source trial using inverse treatment-assignment propensity weights. This is a finite empirical conditional distribution; it does not include finite-sample uncertainty for estimating that distribution.

### Threshold reweighting

For a sorted finite source distribution with atoms y_i and probabilities p_i, solve:

    min/max sum_i p_i * r_i * y_i

subject to:

    1/Lambda <= r_i <= Lambda
    sum_i p_i * r_i = 1

The optimizer starts every ratio at its lower bound and allocates the remaining normalization mass to outcomes in ascending order for the minimum or descending order for the maximum. At most one atom receives a partial allocation.

The v0 implementation cross-checks a three-atom case against exhaustive enumeration of feasible polytope vertices.

### Aggregate target effect

For target stratum masses q_x, the conditional arm bounds aggregate as:

    lower_target = sum_x q_x * (lower_mu_1x - upper_mu_0x)
    upper_target = sum_x q_x * (upper_mu_1x - lower_mu_0x)

The result is an **identification/sensitivity interval conditional on the finite empirical source distributions and registered target stratum masses**. It is not yet a sampling confidence interval.

## 4. What the evaluator verifies

- Lambda is finite and at least one.
- Target stratum masses are finite, positive, unique and sum to one.
- Source treatment propensities are finite and strictly between zero and one.
- Source and target unit identifiers are non-empty/unique within their respective populations.
- Target oracle population weights match the declared target mass within each stratum.
- Source, target and estimand identities match the evaluator oracle.
- Every target stratum has observed source support in both treatment arms.
- Threshold-sharp bounds are recomputed independently of the agent's reported interval.
- The reported interval may not be narrower than the recomputed envelope.
- A decisive action is rejected if the reported interval crosses the decision threshold.
- The evaluator checks whether the actual finite target outcome distributions satisfy the registered Lambda likelihood-ratio envelope. A violation is NotQualified and separately reports whether the target truth is excluded.

The target truth, target outcome likelihood ratios and coverage flag are evaluator-only fields. They must never be exposed to the agent.

## 5. Deterministic controls implemented

Twelve tests currently cover:
1. Lambda = 2 threshold bounds on a binary-outcome distribution.
2. Lambda = 1 recovery when target and source conditional distributions match.
3. Monotonic interval widening as Lambda increases.
4. Target distribution outside the declared Lambda envelope is rejected and truth coverage is reported separately.
5. A target-only stratum without source support is Inconclusive, while full target truth remains evaluator-computable.
6. Invalid source treatment propensity is rejected.
7. A reported interval narrower than the sharp sensitivity envelope is rejected.
8. A decisive action is rejected when the target interval crosses the threshold.
9. Target stratum masses must sum to one.
10. Threshold reweighting matches exhaustive finite-vertex enumeration for a three-atom distribution.
11. Source/target/estimand identity mismatch is rejected.
12. Negative-valued outcome support is handled by the same threshold-ordering rule.

These are deterministic finite fixtures, not a Monte Carlo calibration study or a comparison against the complete published estimator implementation.

## 6. Baselines and next comparison work

- Lambda = 1 standard conditional transport.
- Source-only estimate naively reused as target effect (negative control).
- Inverse-odds-of-sampling weighting for measured effect modifiers.
- Outcome-regression target standardization.
- Doubly robust transported ATE/ATT.
- Exact discrete probability-mass redistribution oracle.
- Manski/worst-case bounds.
- The published Asiaee et al. implementation on matched finite distributions.
- Later, residualized/discounted external-control borrowing and its reliability–efficiency frontier.

Future evaluation should report target-weight concentration, overlap, effective sample size and unsupported target mass. It should compare endpoint error, width, coverage conditional on Lambda, false transport claims and correct UnsupportedTarget disposition.

## 7. Proposed qualification targets (not achieved)

- 100% exact sharp bounds on enumerable finite-distribution controls.
- Zero accepted target intervals excluding evaluator-admissible truth when the true conditional outcome shift is inside the registered Lambda envelope.
- Zero point-transport claims on target-only strata without an explicit extrapolation or sensitivity contract.
- At least 95% correct TransportSupported / SensitivityBounded / UnsupportedTarget disposition on sealed deterministic cases after pilot calibration.
- Full reliability–efficiency curves for external-control borrowing; do not select Lambda or borrowing discounts on the confirmatory target set.

These are proposed preregistration targets, not measured AOBench results.

## 8. Explicit limitations

The current v0 uses:
- finite discrete outcomes;
- empirical IPW source distributions within measured strata;
- fixed target stratum masses;
- one registered symmetric likelihood-ratio bound Lambda for each arm/stratum.

It does **not** yet include:
- source sampling uncertainty or confidence intervals for estimated conditional distributions;
- continuous-outcome density estimation;
- stochastic coverage calibration;
- unmeasured effect modifiers represented jointly with measured strata;
- an external-control power-prior/borrowing model;
- clustered source sampling, censoring or recurrent delayed outcomes.

Do not describe the v0 interval as a complete trial-to-target confidence interval. The first purpose is to make the sensitivity model and exact finite optimization executable and auditable.

## 9. Composition

- [AOBENCH-013 #12206: partial identification and max-potency](https://github.com/Luminous-Dynamics/symthaea/issues/12206)
- [AOBENCH-015 #12210: observational-view identity](https://github.com/Luminous-Dynamics/symthaea/issues/12210)
- [AOBENCH-017 #12242: anytime-valid inference](https://github.com/Luminous-Dynamics/symthaea/issues/12242)
- [AOBENCH-018 #12251: delayed outcomes and staggered entry](https://github.com/Luminous-Dynamics/symthaea/issues/12251)
- [AOBENCH-019 #12278: causal transport sensitivity benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/12278)
- Existing domain-specific transport: [#7041](https://github.com/Luminous-Dynamics/symthaea/issues/7041), [#8705](https://github.com/Luminous-Dynamics/symthaea/issues/8705), [#10752](https://github.com/Luminous-Dynamics/symthaea/issues/10752), [#10792](https://github.com/Luminous-Dynamics/symthaea/issues/10792), [#7623](https://github.com/Luminous-Dynamics/symthaea/issues/7623).

## Claim boundary

A sensitivity interval is only as credible as the registered Lambda model, source support, target stratum masses and outcome measurement contract. Lambda is not learned or verified merely because the computed interval contains the target truth in a synthetic oracle. The finite v0 is a deterministic sharp-bound primitive, not evidence that arbitrary real-world transport assumptions hold. Exact-head Rust qualification and broader coverage/robustness studies are required before statistical performance claims.
