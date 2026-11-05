# SPECTRA Finite-Horizon Sequential Inference — Research Specification v0.1

**Status:** Research specification; no implementation or empirical performance claim.
**Research issue:** [#10633 — finite-horizon alpha spending and terminal-test induction](https://github.com/Luminous-Dynamics/symthaea/issues/10633)
**Baseline implementation:** [#10599 — anytime robust simulation allocation](https://github.com/Luminous-Dynamics/symthaea/pull/10599)
**Benchmark:** [#10601 — locked allocator comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10601)

## 1. Objective

SPECTRA currently uses an infinite-horizon alpha-spending Hoeffding confidence sequence. The allocator also accepts a finite, predeclared maximum total simulation budget. Investigate whether this known terminal horizon can be used to reduce conservatism while preserving validity at every inspected look.

Keep the current infinite-horizon method as the safe reference. Compare finite-horizon spending and sequential tests induced from terminal procedures; do not replace the baseline until proof, exact-head verification and locked evaluation are complete.

## 2. Baseline

For each action × scenario cell, the current six one-sided score claims receive base allowance:

`delta = alpha / (6 * number_of_actions * number_of_scenarios)`.

The current method spends:

`delta_n = delta / [n * (n + 1)]`, for all `n >= 1`,

giving the Hoeffding radius:

`range_width * sqrt(log(n * (n + 1) / delta) / (2 * n))`.

This spends no more than delta over an unbounded number of looks because the reciprocal-product series sums to one.

## 3. Track A — finite campaign horizon

Let `B` be the **frozen maximum total score-replication count for the entire campaign**. The allocator currently counts the per-cell sample counts toward this total. Therefore every individual action × scenario cell has a local sample count no greater than B.

For each one-sided claim, allocate:

`delta_n = delta / B`, for `n = 1, ..., B`.

Then the total spending over all reachable local looks is:

`sum(delta_n, n=1..B) = delta`.

The corresponding radius is:

`range_width * sqrt(log(B / delta) / (2 * n))`.

A union bound over the frozen cells, six one-sided claims per cell, and every local look through B gives the stated familywise coverage under the same bounded independent-within-cell stream assumptions. Dependence between cells or metrics does not invalidate this union bound. A practical implementation must also preserve the current radius cap at the full declared score range; when the uncapped factor is at least one, the bound is the full range rather than a larger radius.

### 3.1 Numerical example

For 24 one-sided claims, familywise alpha 0.05, B=1000, and n=1000:

- Current infinite-horizon radius: approximately 0.1000 times the declared score range.
- Uniform finite-horizon radius: approximately 0.0809 times the declared score range.
- Terminal radius reduction: approximately 19.1%.

This is a calculation from the formulas, not a measured Rust benchmark. The production implementation caps radii at the full declared score range, so the comparison must apply the same cap. With that cap, B=1000 gives identical full-range radii at n=1 and n=2; at n=5 the finite-horizon radius is about 2.2% wider, at n=10 about 9.7% wider, and at n=20 about 3.5% wider. The curves cross near n=31; at n=100 the finite-horizon radius is about 7.8% smaller, and at n=1000 it is about 19.1% smaller. Plot and report the complete interval-width trajectory, stopping distribution and certificate probability; do not present only the terminal improvement.

### 3.2 Horizon and budget semantics

- B is frozen before the first simulation and is part of the method receipt.
- B counts total score replications across all action × scenario cells; a cell cannot exceed B.
- If the campaign exhausts B without certification, return no certificate.
- A caller may not change B mid-campaign and recompute old prefixes under the new schedule.
- A budget increase creates a new claim scope. Investigate explicit optional-continuation epochs with conditional alpha only if a valid transition is derived and independently reviewed.
- If B is not fixed or a potentially unbounded continuation is required, use the existing infinite-horizon method or another method with a proved horizon-independent guarantee.

### 3.3 Alternative finite-horizon schedules

Compare a preregistered family of spending weights `w_n >= 0`, with:

`sum(w_n, n=1..B) <= 1`.

Spend `delta_n = delta * w_n` and derive each radius from `delta_n`. Candidate schedules may prioritize early, middle or terminal looks. The schedule must be frozen before confirmatory outcomes are observed. Do not optimize weights on the confirmatory seed bank.

## 4. Track B — induce sequential tests from terminal tests

Study Koning & van Meer (JRSSB 2026), [Anytime validity is free: inducing sequential tests](https://doi.org/10.1093/jrsssb/qkag050).

The paper shows that, for a valid terminal test under a null distribution P, a sequential test can be induced using the conditional expectation of the terminal test given the observed filtration. The induced process is a Doob martingale and agrees with the terminal procedure at the predeclared horizon.

This is a promising comparator, not a ready-made distribution-free confidence sequence for SPECTRA.

### 4.1 Composite-null obstacle

SPECTRA's bounded score mean is unknown and generally has a composite null. The conditional expectation of a terminal test can depend on the entire null distribution, not just its mean. A plug-in estimate of that distribution is not automatically valid.

Before implementation:
- define the null family and filtration;
- provide a statistic computable from observed history without oracle parameters;
- for composite nulls, show a uniform validity proof, such as a valid least-favorable conditional envelope, or restrict the method to a justified known-null model;
- derive one-sided test inversion and simultaneous coverage across all score claims;
- account for adaptive cell selection and CRN dependence;
- report computational cost of the induced process at each look.

If no useful data-only construction exists under the intended assumptions, record that negative result and limit the method to domains where its assumptions genuinely hold.

## 5. Statistical assumptions

Every method must state whether it requires:
- IID bounded draws within each action × scenario × metric stream;
- a fixed conditional mean / martingale difference condition;
- known null distribution or nuisance parameters;
- predictable allocation / betting choices;
- independent or paired replication identities;
- a fixed finite horizon.

Cross-cell dependence from CRN may be compatible with a union bound over marginal confidence sequences. It does not authorize treating paired outputs as independent. Paired inference still requires the identity and seed-integrity contract from #10597.

A valid Monte Carlo confidence sequence quantifies uncertainty about a supplied simulator's score mean. It does not validate the simulator, the score contract, model adequacy, or the scenario bank.

## 6. Receipt contract

The method receipt must bind:
- frozen B and the definition of total replication budget;
- method ID, alpha and exact spending weights / induced-test definition;
- number of claims and their allocation;
- frozen ranges, action/scenario snapshots and opaque commitments;
- generator and PRNG versions, root seed, and replication ordering/identity;
- raw samples and every bound needed for independent recomputation;
- null family, nuisance restrictions and least-favorable construction for Track B;
- exact stop reason and certificate result.

Changing B, method, null family, ranges, action/scenario set or score definition requires a new receipt and a new claim scope. Do not alter or relabel prior receipts.

## 7. Locked benchmark

Add to #10601:

1. Infinite-horizon alpha-spending Hoeffding (#10599).
2. Uniform finite-horizon spending (Track A).
3. Preregistered nonuniform finite-horizon schedules.
4. Empirical-Bernstein and betting confidence sequences (#10598).
5. Induced sequential tests from terminal procedures, only where a valid observable construction exists.
6. Fixed-horizon terminal intervals as terminal-only baselines.

Use bounded Bernoulli, rare-event, two-point endpoint, skewed, near-constant, mixture and actual SPECTRA DGP score streams. Include exact ties, epsilon-indifference zones, maximin and CVaR objectives, unequal cell allocation, CRN and independent modes, and both small and large frozen budgets.

Report:
- familywise coverage at every look and across all cells/claims;
- width trajectory from early through terminal looks;
- terminal power / certificate rate versus fixed-horizon procedures;
- median, upper-tail and worst-case replications to certificate;
- probability of no certificate at each budget;
- false-winner rate, robust regret and worst-decile regret;
- CPU time per update and total campaign runtime;
- replay success, mutation rejection and failure-inclusive dispositions.

Use untouched evaluation seeds and at least 10,000 outer campaigns per primary regime where computationally feasible. Report Monte Carlo uncertainty for coverage and PCS estimates.

## 8. Proposed qualification targets — not achieved

- At least 95% familywise coverage across all predeclared looks and claims, with Monte Carlo uncertainty.
- At B=1000, at least 15% lower terminal radius than the infinite-horizon baseline in low/moderate-variance regimes, without undercoverage.
- No more than 5% false-winner probability in exact ties and declared indifference zones.
- If Track B is validly implemented, exact agreement with its terminal test at B within declared randomization/numerical tolerance.
- No silent horizon changes or budget extension.
- 100% exact receipt replay and zero acceptance of mismatched method/horizon metadata.

The early-look trade-off, compute overhead and no-certificate probability remain primary outcomes. Terminal width alone is insufficient.

## 9. Research comparators

- Koning & van Meer (JRSSB 2026), [Anytime validity is free: inducing sequential tests](https://doi.org/10.1093/jrsssb/qkag050).
- Waudby-Smith & Ramdas (JRSSB 2024), [Estimating means of bounded random variables by betting](https://doi.org/10.1093/jrsssb/qkad009).
- Howard et al. (Annals of Statistics 2021), [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://doi.org/10.1214/20-AOS1991).
- Khosravi & Huo (2026), [Rank Confidence Sequences: Anytime-valid Leaderboards](https://arxiv.org/abs/2609.32211), on its applicable shared-item dependent-ranking subproblem.

## 10. Execution order

1. Preserve #10599 unchanged as the infinite-horizon reference.
2. Implement a formula-only finite-horizon reference and boundary checks for B=1, n=1, n=B, invalid B and alpha.
3. Independently verify the finite-horizon union-bound derivation and budget semantics.
4. Add finite-horizon score receipt and verifier as a separately versioned method.
5. Compare interval trajectories and certificate rates against #10599 and #10598.
6. Study terminal-test induction for composite bounded-mean nulls; prove or reject a data-only construction before production code.
7. Integrate all methods into #10601's locked campaign.
8. Promote no method until exact-head tests, independent statistical review and confirmatory results pass.

## Non-goals

This specification does not change #10599, weaken its certificate, authorize human-subject adaptive experimentation, or claim that terminal-test induction automatically solves composite bounded-mean inference.
