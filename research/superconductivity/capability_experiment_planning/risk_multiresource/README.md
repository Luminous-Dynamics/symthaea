# SC-RISK-MULTIRESOURCE-0.1

**Status:** synthetic exact-oracle implementation; no real experiment is authorized.  
**Parent:** [#12819](https://github.com/Luminous-Dynamics/symthaea/issues/12819)  
**Tracking:** [#12946](https://github.com/Luminous-Dynamics/symthaea/issues/12946)

## Objective

Extend the risk-neutral scalar-budget EVSI planner with independently constrained resource vectors and an ex-ante precommitment tail-risk value-of-information objective. Expected-loss EVSI and joint-world-state mutual information remain separate comparators.

## Mathematical contract

For world state M, joint observation outcome Y_S, decision policy pi:Y_S->D, and declared loss L:

rho_alpha(S) = min_pi CVaR_alpha(L(pi(Y_S), M))

EVSI_alpha_pre(S) = rho_alpha(empty) - rho_alpha(S)

The policy is selected before observing Y_S, but maps each possible observation to a downstream decision. The CVaR distribution is the joint predictive distribution of (M,Y_S). This is intentionally **not** the expectation of separately optimized posterior CVaRs.

The finite-support Rockafellar-Uryasev formula is:

CVaR_alpha(Z) = min_eta [ eta + E[(Z-eta)_+] / (1-alpha) ]

For fixed eta, the best downstream decision can be chosen independently for each observation outcome. The implementation enumerates the finite set of loss thresholds and evaluates that exact objective. Unit tests cover alpha=0 (mathematical-helper boundary), fractional tail mass, and alpha approaching one.

The fixture declares alpha=0.90. The schema restricts the operational range to [0.5,1).

## Resource semantics

Every resource has an explicit ID, unit, capacity and semantic description. Action consumption is an integer vector. Each dimension is checked independently; there is no exchange rate between money, instrument time, specimen mass, furnace time, synthesis runs or compute allocation. Unknown resource IDs, unit mismatches, duplicate entries, non-positive quantities, arithmetic overflow, over-capacity plans and ineligible actions fail closed.

The synthetic fixture uses:
- financial budget in USD cents;
- cryostat and magnetometer booking minutes;
- specimen mass in mg;
- furnace minutes and synthesis run slots;
- precursor mass and compute core-minutes.

The financial-budget resource is the declared reference denominator for greedy-per-cost comparisons. It does not replace vector feasibility.

Queue waiting, interval overlap, time windows, stochastic yield and random execution success are not implemented. The resource model uses deterministic conservative consumption bounds and does not model shared facility calendars.

## Evidence and world-state semantics

The fixture uses explicit joint world states. Each state identifies model family (including none-of-the-above), phase-realization status, evidence-qualification status and engineering-use status. Unknown remains distinct from a condition-qualified negative.

All world priors, likelihoods, shared nuisance probabilities and decision losses are invented fixture values. They are not calibrated to a sample, instrument, lab, material, application or economic decision.

## Exact oracle and comparators

The Rust binary sc_risk_multiresource_audit:
- validates the versioned schema and synthetic-only claim ceiling;
- integrates one shared nuisance state once over all selected observations;
- exact-enumerates feasible action subsets (at most 12 actions and 10 binary observations);
- validates capability prerequisites and eligibility as hard constraints;
- enforces all resource capacities independently;
- emits expected-loss EVSI, precommitment CVaR-EVSI, joint-world-state mutual information and resource-use receipts;
- computes exact oracles and marginal-per-financial-resource greedy baselines;
- emits an outcome-contingent decision policy and exact schema/fixture SHA-256 digests.

The exact oracle is exponential and is intended only for small contract fixtures. It is not a production-scale campaign scheduler.

## Negative controls

Tests cover:
- CVaR equals expectation at alpha=0 and exact fractional tail-boundary mass;
- convergence to worst-case loss as alpha approaches one;
- capability prerequisites and hard exclusion of the unreviewed high-field action;
- independent resource feasibility and per-unit accounting;
- resource/action JSON order invariance;
- invariance of the selected plan under additive loss shifts and positive loss scaling;
- zero decision value when every decision has identical statewise losses;
- wrong units, unknown resource IDs and duplicate consumption entries;
- a synthetic rare-catastrophe case where expected EVSI and precommitment CVaR-EVSI prefer different experiments.

## Claim ceiling

This is a synthetic software-contract result only. It does not prescribe physical experiments, authorize synthesis, establish a pairing mechanism, or prove a superconducting material or application property. Any transfer to historical or prospective campaigns requires rights-cleared, specimen-linked evidence, independent scientific review, and all required facility and safety approvals.
