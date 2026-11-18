# Capability-Gated Superconductivity Experiment Planning

**Contract:** SC-CAPABILITY-PLAN-0.1  
**Status:** exact two-parameter synthetic planning oracle and myopic baseline; no real experiment is authorized  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12819

## Why this is distinct

Cost-aware Bayesian optimization already selects measurements using predictive value and cost. The missing challenge is that some actions create capabilities rather than directly returning scientific information.

Examples in a real evidence programme could include instrument qualification, reference calibration, sample preparation, or setting up a data-reduction pipeline. Such an action may have zero immediate information gain, yet unlock a valuable follow-up measurement. A myopic information-gain-per-cost policy can therefore stop at a locally attractive plan and leave the target prediction underidentified.

This benchmark connects the specimen-linked inverse model (#12811) to experiment selection and the synthesis/process work (#12671). It is distinct from compute allocation across DFT/ML funnel stages (#12644).

Relevant research baselines:
- Alvi et al. (2026), *Deep Gaussian process-based cost-aware batch Bayesian optimization for complex materials design campaigns*, npj Computational Materials 12, 105: https://www.nature.com/articles/s41524-026-01981-7
- Hassoon & Dredze (2026 preprint), *Capability-Gated Planning: Cost-to-Goal Discovery and the Limits of Myopic Experiment Selection*: https://arxiv.org/abs/2608.05085
- Jakeman et al. (2026 preprint), *Risk-Aware Goal-Oriented Bayesian Optimal Experimental Design*: https://arxiv.org/abs/2609.08023

These motivate the testbed; they are not evidence that a superconductivity-specific decision policy has already been validated.

## Initial mathematical testbed

The first synthetic version uses two model parameters, a Gaussian prior represented by a precision matrix \(P_0\), and a target prediction direction \(g\). Each measurement action contributes a positive-semidefinite 2×2 Fisher-information matrix \(I_a\). For a selected plan \(S\):

\[
P(S)=P_0+\sum_{a\in S} I_a,
\qquad
V_g(S)=g^T P(S)^{-1}g.
\]

The objective is target predictive variance reduction:

\[
\Delta V_g(S)=V_g(\varnothing)-V_g(S).
\]

This is deliberately a small conjugate-Gaussian benchmark. It is not a general posterior solver and does not claim that Fisher information is the correct likelihood for every physical measurement.

Each action has a cost, required capabilities, capabilities provided, an information contribution, and a hard eligibility flag. A feasible plan must fit within budget, contain only eligible actions, and admit an ordering in which every action's prerequisites are available before that action runs.

Capability actions are constrained to contribute zero direct measurement information. They earn value only by unlocking downstream actions.

## Synthetic adversarial example

Budget: 5 resource units. Prior precision: identity. Goal direction: equal weight on amplitude and phase-stiffness parameters.

Available actions:
- amplitude-sensitive map: cost 2, immediately measures amplitude;
- calorimetry workflow qualification: cost 2, unlocks heat-capacity measurement;
- heat-capacity scan: cost 1, measures the second parameter;
- low-cost transport proxy: cost 1, partially informs both parameters;
- vortex-imaging qualification: cost 3, unlocks vortex mapping;
- vortex map: cost 2;
- unreviewed high-field test: deliberately ineligible due to missing independent safety/facility qualification.

The exact feasible-plan oracle should choose amplitude mapping + calorimetry qualification + heat-capacity scan (total cost 5). A myopic marginal variance-reduction-per-cost policy ignores the zero-immediate-information qualification action and instead selects cheap transport plus amplitude mapping. The benchmark measures the lost target variance reduction.

This is an adversarial synthetic example, not a recommendation to perform or prioritize any actual experiment.

## Implemented planner

`sc_capability_plan_audit`:
- validates a JSON Schema 2020-12 instance;
- checks the prior precision matrix is symmetric positive definite;
- checks each action information matrix is symmetric positive semidefinite;
- rejects duplicate action IDs and invalid capability declarations;
- rejects any capability-only action that claims direct information;
- excludes ineligible actions and records their reason;
- enumerates all action subsets for up to 20 actions;
- checks cost budget and capability closure;
- constructs a feasible ordered sequence for every selected subset;
- computes the exact small-instance oracle maximizing target variance reduction;
- computes a myopic marginal target-variance-reduction-per-cost baseline;
- reports oracle regret / efficiency and exact input hashes.

The exact oracle is exponential in action count and is intended for small benchmark instances. Larger campaigns need a dynamic-programming, branch-and-bound or stochastic-planning implementation and must be compared with this oracle on tractable subsets.

## Hard constraints and limitations

- An ineligible action is never selected, regardless of its information matrix. The synthetic high-field action demonstrates this non-compensable gate.
- Capability requirements must be available initially or provided by an earlier selected action.
- The first objective assumes additive independent Fisher information. It does not yet model shared calibration errors, common-mode noise, correlated observations, stochastic capability failures, destructive sample consumption or instrument queueing.
- Cost is currently a single dimensionless integer resource. Time, instrument-hours, specimen mass/volume and budgeted failure probability are later constraints.
- No actual measurement action is authorized by this benchmark.

## Benchmark ladder

1. Exact enumeration on small action graphs and comparison to greedy baselines.
2. Correlated measurement noise / shared calibration nuisance parameters.
3. Stochastic action outcomes and capability failure probabilities.
4. Multi-resource constraints: cost, instrument time, specimen consumption and transfer logistics.
5. Goal-oriented risk objectives for rare/high-consequence failure modes.
6. Receding-horizon plans with stop / abstain and model-discrepancy detection.
7. Replay on held-out historical measurement campaigns, then prospective evaluation only after institutional review.

## Proposed targets (research targets, not achievements)

- Exact oracle recovered on every small enumerated instance.
- Capability-gated cases demonstrate measurable regret for immediate-gain-per-cost baselines.
- Lower oracle regret than greedy and cost-aware baselines on a frozen benchmark suite.
- Robustness to prior misspecification, correlated errors, capability failures and cost uncertainty.
- Correct exclusion of all ineligible actions, even when their nominal information is arbitrarily large.
- Calibrated target-prediction uncertainty on held-out synthetic ground truth.
- Report plan quality, resource use, runtime, action count, unused budget and failure rates.

## Claim ceiling

This is a synthetic decision-planning benchmark. It does not prescribe physical experiments, replace institutional safety review, authorize synthesis or establish any superconducting material or property.
