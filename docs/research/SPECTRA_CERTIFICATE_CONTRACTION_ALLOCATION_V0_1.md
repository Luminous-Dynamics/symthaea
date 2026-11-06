# SPECTRA Certificate-Contraction Allocation v0.1

**Status:** research implementation; draft PR; not a statistical claim of superiority.  
**Base:** anytime score bounds and robust indifference-zone selection.  
**Policy ID:** `robust-certificate-contraction-allocation-v1`.

## Motivation: allocate against the decision, not a proxy

The current v1 allocator ranks action × scenario cells by

`interval_width × robust_tail_contribution`.

This is a useful uncertainty heuristic, but it is not the objective used by the stopping rule. The robust selection certificate is controlled by the global regret upper bound

`Γ = max_a U(a) − L(a_selected)`,

where `L(a_selected)` is the largest robust lower bound and `U(a)` is the robust upper bound for each action. A cell can have a wide interval yet be irrelevant to the current maximum upper endpoint or selected lower endpoint. Spending there may not move `Γ` at all.

Policy A directly scores a candidate batch by the change in that exact certificate quantity:

`priority(batch) = max(0, Γ_before − Γ_after_radius_only(batch)) / fully_charged_batch_cost`.

## Deterministic projection

For each competitive action × scenario candidate:

1. Freeze all observed utility point estimates and every untouched cell.
2. Compute the projected per-cell count after the full next batch.
3. Recompute the registered anytime Hoeffding radius with the same per-claim alpha and declared utility range.
4. Replace only the target cell's lower and upper utility endpoints. In common-random-number mode, replace all actions in the scenario, because the requested batch is atomic across those actions.
5. Recompute every action's Maximin or lower-tail CVaR lower/upper robust interval with the same scenario weights, sorting, and incumbent tie-breaking used by the production certificate.
6. Compute the projected global regret upper bound. Divide positive contraction by the fully charged projected batch cost.

No synthetic future score values are invented. The projection is intentionally radius-only: it estimates *where uncertainty contraction would matter if the empirical means stayed fixed*. It is not an expected value of computation calculation and does not predict what the next samples will be.

The projected cost is the number of action × replication score records required by the candidate batch. For an independent cell batch of size b the cost is b. For a CRN batch with A actions the cost is A × b. The CRN batch remains indivisible.

If no candidate yields a strictly positive projected reduction in `Γ`, Policy A returns `NoCertificateContraction` and requests no batch. This is an explicit abstention, not a certificate. The caller may choose a different batch policy or investigate a model/scenario limitation; the decision-directed allocator does not manufacture progress.

## Implemented surface

- `allocate_next_score_batch`: original width × tail policy, with its original method ID retained.
- `allocate_next_score_batch_decision_directed`: new Policy A.
- `counterfactual_regret_upper_bound`: crate-private ranking projection helper; not a public certificate.
- `verify_adaptive_allocation_receipt`: method-ID dispatch and deterministic replay for both registered policies.
- Receipt priorities record `certificate_gap_contraction`, `projected_batch_cost`, and `projected_regret_upper_bound`.
- Regression fixtures cover the wide-but-irrelevant challenger endpoint, CRN full action cost under lower-tail CVaR, and zero-gain abstention.

## Claim boundary and known limitations

The policy changes only the next-batch allocation. It does **not** modify:
- the underlying anytime score confidence sequence or its alpha schedule;
- the familywise claim count;
- the robust objective or scenario weights;
- the preregistered epsilon;
- the stop/certification rule;
- operational mapping, model adequacy, scenario coverage, or human-study authorization.

The policy is a deterministic sensitivity heuristic, not an optimal allocation theorem, EVI estimate, independent verifier, cryptographic receipt authentication, or evidence that the simulator is adequate. Its projected post-batch interval uses current means and a hypothetical radius; actual future means are unknown. The certificate remains conditional on the original bounded independent replication contract and frozen scenario bank.

A lack of projected gap contraction may mean that a different cell or a larger candidate batch is needed. It may also reveal that the current decision is limited by an unchanged endpoint, a scenario/tail tie, or a model/scenario adequacy gate. Do not reinterpret `NoCertificateContraction` as evidence that further data cannot help.

## Locked evaluation plan

Compare on identical preregistered DGPs, frozen scenario banks, root seeds, costs, and stopping/certificate thresholds:

| Policy | Role | Important caveat |
|---|---|---|
| Equal allocation | transparent baseline | wastes effort on clearly separated actions/cells |
| SPECTRA width × tail v1 | exact in-repo baseline | uncertainty proxy, not certificate sensitivity |
| OCBA and budget-adaptive OCBA (Cao et al., 2025) | fixed-budget PCS comparator | classical mean/variance best-design setting; adapt robustly before comparing |
| Robust OCBA (Wan, Li & Hong, 2025) | ambiguity-set robust R&S comparator | fixed-budget PICS objective; confirm objective and assumptions match |
| COpt (Faella et al., ECAI 2025; online 2026) | fixed-budget nonlinear convex BAI comparator | sub-Gaussian best-arm setting; compare compute overhead as well as sampling efficiency |
| DBCARE (Harding & Kandasamy, NeurIPS 2025) | cost + decision-loss comparator | different risk-functional formulation; include cost and regret together |
| Policy A | certificate-directed comparator | heuristic until locked empirical evidence supports any efficiency claim |
| Calibrated predictive EVC / R-IDeA-inspired policy | research track after Policy A | predictor can affect allocation only; never certify or narrow confidence bounds |

### Required benchmark regimes

- Large-gap and near-tie best-action instances.
- Unequal variances, skew, heavy tails, bounded but highly asymmetric distributions.
- Unequal simulation cost and per-batch startup cost.
- Maximin and lower-tail CVaR with fractional tail-mass boundaries and tied scenarios.
- CRN benefits, weak pairing, and partial-batch failure injection.
- Many actions × scenarios, including noncompetitive actions and very large action sets.
- Misspecified generators, hidden scenario shifts, omitted failure modes, and updated input data streams.
- Small budgets where large-batch lookahead is especially sensitive to the budget horizon.
- Model-adequacy and operational-mapping failures, which must remain blocked even when ranking appears confident.

### Metrics and proposed gates

Report all outcomes, including failures and abstentions:

- Familywise false-certificate rate (target ≤ 5% under the declared valid sampling contract).
- False winner / false epsilon-optimal recommendation rate (target ≤ 5% under matched objective and assumptions).
- Probability of certificate by total budget and the full cost-to-certificate distribution.
- Simple regret / robust regret at fixed budgets; median and high-quantile simulation cost to a valid certificate.
- Budget utilization, wasted/non-decision-relevant replications, and atomic CRN cost.
- Wall-clock allocator overhead and memory; report by action/scenario scale.
- Hidden-world false robustness, model-adequacy false pass, and unsupported-decision rate (all target zero in mandatory-gate negative controls).
- Receipt replay acceptance for unmodified receipts and 100% rejection of deterministic field mutations.

**Pre-registered efficiency aspiration (not a result):** at least 10% median cost-to-certificate improvement versus the best applicable baseline, with no increase in false certificates, false winners, hidden-world false robustness, or unsupported decisions. Report confidence intervals over independent DGP instances; do not promote the aspiration to a claim if only one regime improves.

For uncertainty comparisons, include a failure-inclusive paired bootstrap or exact randomization interval over DGP instances. Preserve the same statistical validity gate for every allocation policy. Compare cost and decision quality on a Pareto frontier; do not use a single weighted score that can hide rare catastrophic failures.

## Research context

- Cao et al. propose a budget-adaptive OCBA rule because asymptotic OCBA can misallocate under small budgets; this is directly relevant to our small-budget stress tests.
- Faella, Magliocca & Sauro's COpt formulates fixed-budget best-arm allocation as nonlinear convex optimization over empirical estimates and unknown variances. It is a serious algorithmic comparator, but does not by itself supply SPECTRA's robust scenario and time-uniform certificate contract.
- Harding & Kandasamy's DBCARE explicitly balances observation cost with the penalty for a suboptimal recommendation. This motivates a decision-risk/cost frontier alongside cost-to-certificate.
- Tang, Sloman & Kaski's R-IDeA argues that acquisition under misspecification should combine representativeness, informativeness, and de-amplification. This is complementary to certificate contraction: future policies may seek informative, representative, non-error-amplifying scenarios, but must never let a predictor rewrite inferential bounds.
- Wang & Zhou's simultaneous budget allocation handles streaming input data and simulation budgets under heterogeneous input distributions, including non-IID simulation outputs. SPECTRA should only ingest such evolving outputs through a separately qualified estimator/receipt, never append them naively to the current per-cell IID confidence sequence.

References:
1. Cao et al. (2025), *A budget-adaptive allocation rule for optimal computing budget allocation*, European Journal of Operational Research 325(2), 247–260. DOI: 10.1016/j.ejor.2025.04.015.
2. Faella, Magliocca & Sauro (ECAI 2025; first published online 2026), *Convex Optimization Yields Empirically Superior Best-Arm Identification*. DOI: 10.3233/FAIA251030.
3. Harding & Kandasamy (NeurIPS 2025), *Balancing Performance and Costs in Best Arm Identification*. DOI: 10.52202/085713-4080.
4. Tang, Sloman & Kaski (AISTATS 2026), *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*. PMLR 300:3016–3024.
5. Wang & Zhou (2025), *Data-Driven Ranking and Selection With Simultaneous Input Data Collection and Simulation*. Naval Research Logistics. DOI: 10.1002/nav.22275.
6. Wan, Li & Hong (2025), *New Additive OCBA Procedures for Robust Ranking and Selection*. Asia-Pacific Journal of Operational Research 42(6). DOI: 10.1142/S0217595925400032.
