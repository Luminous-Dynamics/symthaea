# SPECTRA Few-for-Many Portfolio Bounds v0.2

**Status:** deterministic Rust primitive and exact small-K family certificate declarations; not yet compiled or executed.
**Research parent:** #10730
**Implementation PR:** #10768
**Portfolio-bound method:** spectra-few-for-many-portfolio-degradation-interval-v1
**Family method:** spectra-few-for-many-exact-small-k-family-identification-v1

## 1. Research problem and paper semantics

Xue et al., *Few-for-Many Best-Arm Identification in Multi-Objective Bandits*, arXiv:2610.16840 (submitted 12 October 2026), define the few-for-many objective as choosing representatives that jointly cover objective-wise ideals while limiting worst weighted degradation.

Paper: https://arxiv.org/abs/2610.16840

For an action set A and objective m, let:
- μ*_m = max over a in A of μ[a,m]
- coverage_m(S) = max over a in S of μ[a,m]
- Φ(S) = max over objectives m of λ_m(μ*_m - coverage_m(S))

The paper distinguishes:
- **GFM-Elim:** known objective groups; exploits the group structure and shared observations.
- **AFM-Elim:** latent objective grouping induced by the representative set; the candidate family contains exact-size-L subsets and is combinatorial.
- **BiAFM-Greedy:** a polynomial-time bicriteria relaxation that can return O(L log M) representatives; it trades a larger representative set for tractability and is not the same exact-size-L identification guarantee.

These algorithms and their assumptions remain future matched baselines. This PR implements deterministic interval algebra and an exact small-K oracle, not GFM-Elim, AFM-Elim or BiAFM-Greedy.

## 2. At-most-L versus exact-size-L

The SPECTRA API uses an at-most-L deployment cap: 1 ≤ |S| ≤ L. The paper's latent-group formulation uses exact-size-L representative sets.

With monotone max-coverage and a fixed feasible action universe, if K≥L, any portfolio smaller than L can be padded with additional feasible actions without reducing coverage or increasing degradation. Therefore the minimum objective value over at-most-L and exact-size-L families agrees under that simple objective. However:
- the set of identified portfolios and tie semantics can differ;
- representative-specific costs, incompatibilities, diversity requirements or feasibility constraints can invalidate simple padding;
- direct paper reproduction must use and report exact-size-L portfolios.

Keep exact-size-L and at-most-L protocols distinct in results.

## 3. Interval-bound derivation

A portfolio S is a nonempty subset of a frozen feasible action universe A. Each objective is direction-normalized so larger is better. The objective coefficient c_m is positive, pre-registered and already includes any unit normalization.

Assume a joint event E on which every action/objective mean lies in its supplied interval [L[a,m], U[a,m]]. For any nonempty portfolio S:

- ideal_m = max_a μ[a,m]
- covered_m(S) = max_(a in S) μ[a,m]
- degradation_m(S) = ideal_m - covered_m(S)

Then, on E:

- lower_m(S) = max(0, max_a L[a,m] - max_(a in S) U[a,m])
- upper_m(S) = max(0, max_a U[a,m] - max_(a in S) L[a,m])

and lower_m ≤ degradation_m ≤ upper_m. For frozen positive coefficients c_m:

D_lower(S) = max_m c_m lower_m(S)
D_upper(S) = max_m c_m upper_m(S)

so D_lower(S) ≤ D(S) ≤ D_upper(S).

### Why there is no additional union bound over portfolios

E covers the complete frozen action × objective interval grid. Every portfolio bound is deterministic post-processing of this same event. Therefore, even if S is selected adaptively from an exponentially large family, another union bound over portfolios is unnecessary. The base intervals must still carry valid simultaneous coverage over every frozen action/objective claim.

This does not repair invalid source/target support, false feasibility, estimated coefficients, inferred group membership, nonstationarity or provenance authenticity. Those remain separate qualification/evidence requirements.

## 4. Rust implementation

File: tools/spectra-uvsd-fit/src/few_for_many.rs

### Frozen contract and input validation

- Frozen feasible action IDs, objective IDs, positive coefficients, maximum portfolio size and epsilon threshold.
- Requires exactly one finite ordered interval for every frozen action × objective pair.
- Rejects unknown, duplicate and missing records; invalid weights; duplicate or unknown portfolio actions; empty portfolios; over-cap portfolios; malformed contracts.

### Per-portfolio decision

Computes per-objective lower/upper degradation and weighted worst-objective bounds.

Disposition:
- CertifiedWithinThreshold only when D_upper(S) ≤ epsilon.
- CertifiedAboveThreshold only when D_lower(S) > epsilon.
- Unresolved otherwise.

This says whether a candidate satisfies a registered degradation threshold. It does not by itself say that it is the best portfolio in the candidate family.

### Exact small-K portfolio family certificate

New function: certify_exact_bounded_portfolio_family.

1. Enumerates every nonempty portfolio up to the cap, using the declared max-candidate budget.
2. Computes bounds for every portfolio.
3. Selects the portfolio with the smallest degradation upper bound, breaking ties lexicographically by action ID.
4. Computes min_S D_lower(S), a lower bound on the best achievable degradation over the complete enumerated family.
5. Certifies family optimality only when D_upper(S_selected) ≤ min_S D_lower(S). Otherwise it returns Unresolved.

The reported optimality gap upper bound is max(0, D_upper(S_selected) - min_S D_lower(S)). This is conditional on the upstream simultaneous intervals and the complete enumerated family.

Exact enumeration is only an oracle for small K. It is guarded by checked combinatorial counting and a candidate budget; it fails closed rather than attempting an unbounded family. No scalable branch-and-bound or elimination guarantee is implemented.

## 5. Tests declared

- Exact objective-wise coverage yields zero degradation.
- Positive coefficient weighting and strict above-threshold disposition.
- Overlapping intervals remain unresolved.
- Input-order invariance.
- Lexicographic small-K enumeration.
- Exact enumeration succeeds exactly at the budget and rejects a budget one below the candidate count.
- Malformed grids, empty/duplicate/unknown portfolios, invalid coefficients and over-cap members fail closed.
- Deterministic generated property corpus over 128 campaigns, 2–6 actions, 1–4 objectives and all nonempty portfolios up to size 3; checks every true weighted degradation is inside its computed interval.
- Exact family certificate succeeds for a uniquely separated point-interval optimum.
- Overlapping intervals do not produce a false optimality certificate.
- Exact-family enumeration respects the explicit candidate budget.

Rust tests are declarations only. No cargo/rustc/rustfmt are available in the current environment, so no compile, test, formatting, clippy or CI pass is claimed.

## 6. What is deliberately not included

- No GFM-Elim or AFM-Elim implementation.
- No latent objective-group estimator or group uncertainty model.
- No BiAFM-Greedy bicriteria approximation.
- No scalarized trade-off of hard constraints; feasibility gates remain upstream and mandatory.
- No large-K branch-and-bound or elimination theorem.
- No simultaneous interval estimator or confidence-level claim.
- No receipt signing, source verification or external provenance authentication.
- No performance or SOTA claim.

## 7. Locked benchmark

### Reproduce the paper first

Match the Few-for-Many paper's reward family, known/latent group structures, exact-size-L family and representative-budget protocol. Compare GFM-Elim, AFM-Elim, its Gaussian oracle Track-and-Stop where feasible, and the BiAFM-Greedy bicriteria relaxation as a separately labelled runtime/portfolio-size trade-off.

For small K, use SPECTRA's exact family certificate as a correctness oracle; do not treat exact enumeration as an efficient baseline for large K.

### Extend one assumption at a time

- bounded non-Gaussian and correlated objective rewards;
- hard feasibility constraints;
- uncertain/ambiguous latent groups;
- heteroscedasticity and heavy tails only with valid inference;
- fixed conditional means and separately labelled drift estimands;
- target shift and support gaps;
- multi-fidelity discrepancy;
- missing-not-at-random outcomes;
- action-specific cost, incompatibility and deployment constraints.

Each extension needs a corresponding confidence/identification argument; the paper's guarantees must not be carried over by analogy.

### Sweep and outcomes

Sweep actions 4–128, objectives 4–64 and portfolio cap 1–8. Record:
- familywise coverage of base intervals;
- probability of certified D(S)≤epsilon;
- false portfolio-threshold and false optimality certificates;
- unresolved rate;
- true degradation and optimality gap;
- portfolio size and exact-size-L versus at-most-L semantics;
- fully charged cost-to-certificate;
- p50/p95/p99 search and verifier time;
- allocations, memory, group recovery and group-uncertainty calibration.

At least 10,000 outer campaigns per primary regime, with Monte Carlo uncertainty on false-certification rates. Preserve all negative-control results.

Proposed goal in high-redundancy, correctly grouped regimes: ≥20% lower fully charged cost to a valid bounded-portfolio certificate than objective-wise MO-BAI, with at least 95% portfolio success and zero mandatory-gate false passes. This is an aspiration, not an observed result.
