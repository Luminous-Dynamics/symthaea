# SPECTRA Few-for-Many Portfolio Bounds v0.1

**Status:** deterministic Rust primitive and small-K oracle declarations; not yet compiled or executed.
**Research parent:** #10730
**Method ID:** spectra-few-for-many-portfolio-degradation-interval-v1

## Scope

A portfolio S is a nonempty subset of a frozen feasible action universe A, with |S|≤L. For each objective m, larger is better and the objective-wise ideal is max over a in A of μ[a,m]. Portfolio coverage is max over a in S of μ[a,m]. The degradation is ideal minus covered utility; the registered scalar decision objective is the maximum across objective coefficients times degradation.

This module consumes action × objective intervals. It does not estimate score means, prove that intervals are simultaneous, qualify hard constraints, infer objective groups, or authenticate provenance. The contract's feasible action list must already be qualified upstream. Objective directions must be normalized and the coefficients must be fixed before outcomes are observed.

## Bound derivation

Assume a joint event E on which every action/objective mean lies in its supplied interval [L[a,m], U[a,m]]. For any nonempty portfolio S:

- ideal_m = max_a μ[a,m]
- covered_m(S) = max_{a in S} μ[a,m]
- degradation_m(S) = ideal_m - covered_m(S)

Then, on E:

- lower_m(S) = max(0, max_a L[a,m] - max_{a in S} U[a,m])
- upper_m(S) = max(0, max_a U[a,m] - max_{a in S} L[a,m])

and lower_m ≤ degradation_m ≤ upper_m. For frozen positive coefficients c_m:

D_lower(S) = max_m c_m lower_m(S)
D_upper(S) = max_m c_m upper_m(S)

so D_lower ≤ D(S) ≤ D_upper.

The important multiplicity point is that E covers the entire frozen action × objective grid. Every portfolio bound is deterministic post-processing of this same event, so even if S is selected adaptively from an exponentially large candidate family, another union bound over portfolios is not required. The base intervals must still carry valid simultaneous coverage over all action/objective claims. This result does not repair invalid source/target support, false feasibility, estimated weights, or inferred group membership.

## Rust implementation

File: tools/spectra-uvsd-fit/src/few_for_many.rs

- Frozen contract identifies feasible action IDs, objective IDs and positive coefficients, portfolio size cap and degradation threshold.
- Requires exactly one finite ordered interval for every frozen action × objective pair; rejects unknown, duplicate and missing records.
- Validates that every portfolio is nonempty, within cap, duplicate-free and entirely inside the frozen feasible universe.
- Computes objective-wise lower/upper degradation and weighted worst-objective bounds.
- Returns CertifiedWithinThreshold only when the upper bound is at or below epsilon; CertifiedAboveThreshold only when the lower bound is strictly above epsilon; otherwise Unresolved.
- Provides a lexicographic exact enumerator for small K, with checked combinatorial counting and an explicit max-candidate budget. This is an oracle, not a scalable optimizer.

## Tests declared

- exact objective-wise coverage yields zero degradation;
- positive coefficient weighting and strict above-threshold disposition;
- overlapping intervals remain unresolved;
- input-order invariance;
- lexicographic small-K enumeration and candidate-budget rejection;
- malformed grids and portfolio members fail closed;
- deterministic generated property corpus over 128 campaigns, 2–6 actions, 1–4 objectives and all nonempty portfolios up to size 3; checks that every true weighted degradation lies inside its computed interval.

Rust tests are declarations only. No cargo/rustc/rustfmt are available in the current environment, so no compile, test, formatting, clippy or CI pass is claimed.

## What is deliberately not included

- no known-group or latent-group estimator (GFM-Elim / AFM-Elim must be reproduced as separate matched algorithms);
- no scalarized trade-off of hard constraints (hard gates remain upstream and mandatory);
- no large-K branch-and-bound or elimination theorem;
- no simultaneous interval estimator or confidence-level claim;
- no receipt signing, source verification or external provenance authentication;
- no performance or SOTA claim.

## Locked benchmark extension

Match the Few-for-Many paper's assumptions first: Gaussian vector rewards, known objective groups and latent objective groups with a representative budget. Compare GFM-Elim, AFM-Elim, its Gaussian oracle Track-and-Stop where feasible, objective-wise MO-BAI, constrained Pareto PSI, scalarization and an exact enumerator on small K.

Then extend one assumption at a time: bounded non-Gaussian rewards, correlated objectives, hard feasibility constraints, uncertain groups, heteroscedasticity, fixed conditional means, target shift, multi-fidelity and missing-not-at-random outcomes. Each extension needs a corresponding confidence/identification argument; the paper's guarantees must not be carried over by analogy.

Sweep actions 4–128, objectives 4–64 and portfolio cap 1–8. Record familywise coverage, probability of certified D(S)≤epsilon, false portfolio certification, unresolved rate, true degradation, portfolio size, fully charged cost-to-certificate, p50/p95/p99 search/verifier time, allocation and memory. At least 10,000 outer campaigns per primary regime with Monte Carlo uncertainty on false-certification rates.

Proposed goal in high-redundancy, correctly grouped regimes: ≥20% lower fully charged cost to a valid bounded-portfolio certificate than objective-wise MO-BAI, with at least 95% portfolio success and zero mandatory-gate false passes. This is an aspiration, not an observed result.
