# SPECTRA Decision-Directed Allocation — Research Specification v0.1

**Status:** Research specification; no allocator implementation or performance claim.
**Parent implementation:** [PR #10599 — anytime robust simulation allocation](https://github.com/Luminous-Dynamics/symthaea/pull/10599)
**Research tracking:** [#10615 — decision sensitivity and value of computation](https://github.com/Luminous-Dynamics/symthaea/issues/10615), [#10601 — locked allocator comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10601)

## 1. Objective

Move from spending simulation budget to reduce uncertainty generally toward spending it where uncertainty reduction can change the current scientifically defensible decision.

SPECTRA should optimize **cost to a valid decision**, not information gain, interval width, or mean score in isolation. The current anytime confidence sequences and robust epsilon certificate remain the inferential authority. Allocation policies only choose the next predictable batch.

Three outcomes are legitimate:

1. issue a supported epsilon-optimality certificate;
2. continue sampling because a feasible batch can materially reduce decision uncertainty;
3. abstain because the budget is exhausted, no supported batch is available, or remaining uncertainty is not resolvable under the declared policy.

A no-certificate outcome is not a system failure. It is preferable to an unjustified winner.

## 2. Existing certificate and decision gap

For each frozen action `a`, scenario `s`, and replication history, retain a simultaneous time-uniform utility interval:

`[l_{a,s}, u_{a,s}]`.

For the frozen monotone robust aggregation function `rho` (currently Maximin or lower-tail CVaR), define:

`L_a = rho(l_{a,*})`

`U_a = rho(u_{a,*})`.

Choose the incumbent deterministically:

`a_star = argmax_a L_a`

using the existing stable action-ID tie-break. The certificate gap is:

`Gamma = max_a U_a - L_{a_star}`.

The existing certificate condition remains exactly:

`Gamma <= epsilon`.

No allocation policy may weaken this inequality, alter the frozen objective, change epsilon after observing outcomes, or treat a point estimate as a certificate.

## 3. Why interval width is not yet decision value

The #10599 implementation's `compute_cell_priorities` multiplies a utility interval width by scenario-tail contributions. Its current tail helper combines active lower-bound and upper-bound tails for each competitive action.

That heuristic is transparent and useful as a baseline. However, the two sides of the certificate play different roles:

- The incumbent's robust **lower** endpoint is subtracted from the maximum upper endpoint.
- The current threatening action's robust **upper** endpoint contributes to that maximum.
- A non-incumbent lower endpoint ordinarily does not tighten the current certificate gap.
- The incumbent upper endpoint ordinarily does not tighten the gap unless the incumbent also attains the maximum upper endpoint.
- For Maximin and lower-tail CVaR, the active scenario set for a lower endpoint can differ from the active set for an upper endpoint.

Therefore, width × combined tail mass can allocate to an endpoint that is not currently certificate-limiting. This is a limitation to quantify, not grounds to discard the existing implementation before a locked comparison.

### 3.1 Worked counterexample for a regression fixture

Consider two scenarios under Maximin:

| Action | Scenario 1 utility interval | Scenario 2 utility interval | Robust interval |
|---|---:|---:|---:|
| A (incumbent) | [0.80, 0.85] | [0.82, 0.92] | [0.80, 0.85] |
| B (competitor) | [0.20, 0.90] | [0.35, 0.84] | [0.20, 0.84] |

Action A is selected because its robust lower bound is 0.80. The maximum robust upper bound is 0.85, so the current certificate gap is 0.05. With epsilon 0.25, B passes the current competitive-action filter because 0.84 >= 0.80 - 0.25.

However, shrinking B's intervals alone cannot change the current gap under radius-only contraction:
- B's robust upper bound (0.84) is below A's upper bound (0.85), so reducing it does not change the maximum upper endpoint.
- B's robust lower bound (0.20) is below A's incumbent lower bound (0.80), so increasing it without overtaking A does not change the incumbent lower endpoint.

The current combined-tail helper assigns B positive contribution to its lower-tail scenario 1 and upper-tail scenario 2. Their widths are 0.70 and 0.49, respectively. A's active lower and upper endpoint both come from scenario 1, whose interval width is 0.05. Under the current width × tail-mass score, B's scenario-1 and scenario-2 cells can therefore outrank A's scenario-1 cell despite having zero radius-only certificate-gap contraction.

Use this exact case as a deterministic negative-control fixture:
- the current heuristic must reproduce its declared priorities;
- Policy A must report zero certificate-gap contraction for both B cells;
- Policy A must report positive contraction for A's active scenario-1 cell when the hypothetical radius is reduced;
- certification semantics and the actual confidence bounds must remain unchanged.


## 4. Policy A — exact radius-only certificate contraction

Implement a deterministic, interpretable next-batch policy using the production interval and robust-aggregation functions.

For each feasible candidate batch (b):

1. Freeze the current raw samples, point estimates, ranges, robust objective, action/scenario snapshots, epsilon, and confidence-sequence method.
2. Compute the current gap (Gamma_t).
3. Hypothetically advance the selected cell's sample count by the candidate batch size and recompute the time-uniform radius using the registered method. Hold the current point estimate fixed solely for this radius-only comparison.
4. For an independent-per-cell batch, shrink only the candidate action × scenario cell's interval.
5. For an atomic CRN batch, shrink all action cells in the selected scenario together at aligned replication indices. Charge the entire batch, not one action's cost.
6. Recompute every affected (L_a), (U_a), incumbent, and the robust gap using the exact production aggregation and tie-breaking logic.
7. Rank candidates by:

`I(b) = max(0, Gamma_before - Gamma_after_radius_only) / cost(b)`

where `cost(b)` is a frozen, declared batch-cost estimate in an explicit resource unit.

This evaluates the exact certificate's response to interval-radius contraction for the hypothetical batch. It is not a prediction of the new sample values, not expected value of information, and not a theorem of allocation optimality.

### 4.1 Resource accounting

Do not assume all action/scenario replications cost the same. The cost contract should support a predeclared cost per cell or batch and a versioned calibration source. Record predicted cost and observed cost separately.

Keep simulation compute, wall-clock, memory, energy, and participant-time as distinct resource units. A policy may expose a Pareto frontier; it must not silently convert participant burden into CPU time or collapse units with undocumented weights.

### 4.2 Zero-gain behavior

If all feasible batches have zero radius-only gap contraction, return an explicit no-current-certificate-pressure disposition. A separately versioned exploration/audit policy may still be invoked to preserve required coverage, but its budget and purpose must be visible.

Do not relabel ordinary interval-width reduction as decision value. Do not falsely claim that a zero-gain result proves no future observation can change the decision: the radius-only calculation holds sample means fixed and does not model future outcomes.

## 5. Policy B — predictive value of computation

Research a one-step and short-horizon alternative that estimates the expected value of a possible batch.

For candidate batch `b`, use a versioned predictive distribution `P_hat_t`, fitted only from information available at time `t`, to generate possible batch outcomes. Define a declared decision loss `J`, for example robust regret or compute-to-valid-certificate, and estimate:

`EVC_t(b) = [J_t - E_{X_tilde ~ P_hat_t}(J_{t+1}(b, X_tilde))] / cost(b)`.

The exact loss, prediction horizon, Monte Carlo error of this estimate, and cost unit must be frozen before comparison.

The predictive model affects allocation efficiency only. It cannot replace the time-uniform confidence sequence, independent receipt verifier, model-adequacy veto, or epsilon certificate. A Bayesian or predictive allocation score does not become a frequentist coverage guarantee. Any betting parameter must remain predictable with respect to the inferential filtration required by its method.

Keep Policy A and uniform anytime allocation as fallbacks. Disable Policy B in regimes where predictive calibration fails its preregistered gate.

## 6. Receipt and independent replay contract

An allocation receipt must bind:

- exact score receipt and its method, ranges, samples, and frozen snapshots;
- robust objective, scenario weights, epsilon, and certificate method;
- current incumbent, maximum-upper-bound challenger set, and certificate gap;
- candidate batch definitions, per-cell counts, resource-cost model, and cost units;
- hypothetical post-batch intervals and recomputed certificate gaps for Policy A;
- selected marginal gap reduction per cost and deterministic tie-break;
- predictive-model ID, version, training provenance, and prediction diagnostics for Policy B;
- exploration-floor use, fallback, no-progress, budget exhaustion, or abstention disposition;
- per-replication identities and seed commitments when integrated with #10597.

An independent verifier must reconstruct candidate priorities from the frozen receipts and method versions. Caller-supplied digests remain opaque commitments unless a separately reviewed canonical hashing and signature system is introduced.

## 7. Benchmark protocol

Add the following to the locked #10601 arena:

1. fixed equal allocation;
2. uniform anytime allocation;
3. current #10599 interval-width × tail-mass policy;
4. exact radius-only certificate-contraction policy;
5. predictive value-of-computation policy, only after calibration;
6. robust OCBA / additive robust OCBA where the published assumptions apply;
7. Track-and-Stop and Top Two on separate scalar-arm sub-benchmarks where their assumptions apply;
8. oracle allocation as an explicitly unattainable reference.

Include unique winners, exact ties, epsilon-indifference zones, incumbent reversals, multiple active worst scenarios, CVaR boundary ties, unequal scenario weights, positive/zero/negative CRN correlation, unequal costs, variable batch sizes, budget exhaustion, and non-evaluable cases.

Use at least 10,000 independent outer campaigns per primary regime and untouched evaluation seeds. Report Monte Carlo uncertainty for PCS and false-selection probabilities. Keep all failures and non-evaluable campaigns in denominators.

### Primary outcomes

- familywise PCS_epsilon, false-winner probability, and no-certificate probability;
- expected robust opportunity cost and worst-decile regret;
- total simulations and per-cell allocation;
- compute, wall-clock, and other resource cost to a valid certificate;
- allocation spent on certificate-inactive cells;
- certificate-gap trajectory and selection stability;
- time-uniform coverage over all looks;
- paired variance reduction where valid;
- exact receipt and allocation replay.

## 8. Proposed qualification targets — not results

- No degradation of the declared time-uniform coverage or certificate semantics.
- Zero Policy A allocations to cells with exactly zero radius-only certificate sensitivity, except separately declared audit/exploration obligations.
- At least 10% lower median compute-to-certificate than the current #10599 heuristic in preregistered certificate-bottleneck regimes.
- At least 20% lower median compute-to-certificate than uniform anytime allocation at equal certified (PCS_epsilon) in supported regimes.
- No increase in false selection or worst-decile regret to obtain those savings.
- 100% deterministic allocation replay; zero accepted duplicate, missing, or mismatched paired identities.
- Policy B remains disabled wherever its predictive calibration gate fails.

These are targets to preregister and test, not present achievements.

## 9. Research comparators and claim boundaries

- Khosravi & Huo (2026), [Rank Confidence Sequences: Anytime-valid Leaderboards](https://arxiv.org/abs/2609.32211): pairwise betting e-processes and closed testing for dependent model scores on shared items. Its scalar rank result is not automatically a robust-CVaR theorem.
- Waudby-Smith & Ramdas, [Estimating Means of Bounded Random Variables by Betting](https://doi.org/10.1093/jrsssb/qkad009): variance-adaptive bounded-mean confidence sequences.
- Wan, Li & Hong (2025), [New Additive OCBA Procedures for Robust Ranking and Selection](https://doi.org/10.1142/S0217595925400032): ambiguity-aware fixed-budget robust ranking and selection.
- Cao et al. (European Journal of Operational Research 2025), [A budget-adaptive allocation rule for optimal computing budget allocation](https://doi.org/10.1016/j.ejor.2025.04.015): finite-budget OCBA explicitly accounts for budget scale rather than relying only on asymptotic allocation proportions; add as a fixed-budget comparator.
- Wang & Zhou (INFORMS Journal on Optimization 2025), [Optimal Computing Budget Allocation for Data-Driven Ranking and Selection](https://doi.org/10.1287/ijoo.2024.0035): data-driven ranking-and-selection with Bayesian estimation / large-deviation allocation; compare where its input-uncertainty assumptions fit.
- Wen et al. (ICML/PMLR 2026), [Robust Sequential Experimental Design for A/B Testing](https://proceedings.mlr.press/v306/wen26i.html): worst-case MSE guarantees and technology-company evaluation under a different treatment-effect estimand.

Do not claim SOTA or industry superiority until exact-head implementation, independent statistical review, locked campaigns, and appropriate shared-estimand comparisons succeed.

## 10. Implementation sequence

1. Keep #10599 as the interval-width baseline and obtain exact-head Rust verification.
2. Add a pure, independently testable certificate-gap recomputation helper for candidate hypothetical intervals.
3. Implement Policy A without changing score-bound or certificate semantics.
4. Add receipt replay and mutation/negative controls.
5. Integrate explicit paired replication identities from #10597 and test atomic CRN batch cost accounting.
6. Add nonuniform cost snapshots and deterministic tie handling.
7. Add the Policy A comparison to #10601 before attempting Policy B.
8. Research and calibrate Policy B separately; do not enable it by default.
9. Publish failure-inclusive locked results before making efficiency claims.

## Non-goals

This specification does not implement a human-subject adaptive experiment, a sample-optimal OCBA theorem, an anytime-valid model-adequacy test, or an operational authorization. The frozen scenario bank remains a separate scientific assumption; see [#10618 — scenario-bank assurance](https://github.com/Luminous-Dynamics/symthaea/issues/10618).
