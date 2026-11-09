# Exact rational regret-profile quotient and Pareto frontier

This benchmark covers the exact profile layer introduced in
`crates/bridges/symthaea-complexity-lab/src/finite_family_portfolio.rs`.
It consumes the complete model-major policy-value table produced by the
finite-family exhaustive solver.

## Mathematical object

For each deterministic time-dependent Markov policy \(\pi\), define its
exact regret profile across the ordered finite family:

```
r(pi) = (V*_0 - V_0(pi), ..., V*_(m-1) - V_(m-1)(pi))
```

For a portfolio, the objective is:

```
max over models i: min over policies pi in portfolio: r_i(pi)
```

This is a min over portfolio members *within each model*, followed by a
max over models. It is not the maximum of independent singleton regrets.

## Exact profile equivalence

Policies with identical full regret vectors are equivalent for every
portfolio objective. They are grouped into one profile class, retaining all
original canonical policy indices. The smallest policy index is the
representative.

## Strict Pareto dominance

Profile a strictly dominates b iff:
- a_i <= b_i for every model i; and
- a_i < b_i for at least one model.

A dominated policy can never improve the per-model minimum regret of a
portfolio that already contains its dominator. Replacing it by the
dominating policy cannot worsen any model's portfolio regret. Thus at least
one minimax-optimal portfolio exists using only undominated profile classes.

The result retains:
- every exact unique profile;
- all original policy indices in each equivalence class;
- canonical dominator witnesses for every dominated class;
- the ordered Pareto frontier group indices;
- source family, policy manifest and policy-value-table digests;
- a versioned SHA-256 over the full quotient/witness/frontier output.

This is an optimum-preserving search reduction. It does **not** enumerate all
raw tied portfolios that include redundant or dominated policies. The
exhaustive reference remains the authority for complete raw tie enumeration.

## Input/source validation

The function checks family identity, dimensions, deterministic Markov policy
manifest, complete model-policy evaluation count, row sizes and the
policy-value SHA-256. It recomputes all modelwise regret components with
checked rational subtraction, rejects negative regrets, and enforces the
configured rational-width bound.

The profile routine consumes a policy-value table returned by the exact
finite-family solver. For externally persisted or modified inputs, run
`verify_finite_family_policy_portfolio_result` first with the source
solver's budget; a matching digest is an integrity binding, not a substitute
for validating an untrusted table's provenance.

## Budgets

`PolicyRegretProfileBudget` caps:
- profile entries (policy_count x model_count);
- worst-case component comparisons for equality grouping and directed
  dominance checks;
- rational numerator/denominator bit width;
- conservative structural output and scratch memory.

All count/byte arithmetic is checked. Profile rows, equivalence indices,
frontier indices and dominance witnesses use fallible reservations. Exhaustion
returns an explicit error, never a partial frontier marked complete.

## Exact fixtures

### Two specialist models

M0 has rewards [1,0]; M1 has rewards [0,1]. Their only policies have profiles
(0,1) and (1,0). Neither dominates the other, so both remain on the frontier.
The full-profile and frontier portfolio optima agree exactly:
- k=1: regret 1;
- k=2: regret 0.

### Duplicate and dominated profiles

One model has three actions with rewards [3,3,1]. The three policy profiles
are [0], [0], [2].
- Two policies are grouped in one exact equivalence class with indices [0,1].
- The profile [2] is strictly dominated by [0].
- The Pareto frontier retains only the [0] profile class.
- The full policy portfolio optimum and the frontier-restricted optimum
  both remain exactly zero regret for k=1.

Known answers for this fixture:
- Model family SHA-256:
  `4b004bccbec0e45c3fbaf42b29e5303811aa0085ce61014df274ceb6dab067d2`.
- Policy-value table SHA-256:
  `f78dfbd81c56209d3f6c444c5b6b59c7eb473432ea5f60d8f9ef5f6afae5768e`.
- Profile frontier certificate SHA-256:
  `05fd42a60adf45e9b6496be260a1dd8ecb524033f7fe60a4549e030c225dbbac`.

These hashes are independently reconstructed from the documented canonical
little-endian encodings. They are regression targets, not evidence that Rust
tests have executed.

## Paradigm connection

Adaptive Policy Portfolios (arXiv:2608.17929, August 2026) uses regret
profiles and approximate k-means clustering to choose a small policy set.
This implementation provides a deterministic exact quotient and a
componentwise Pareto filter over finite rational model families. It creates
the foundation for, but does not yet implement:
1. threshold feasibility: at regret threshold r, policy pi covers models
   whose profile component is <= r;
2. set-cover / branch-and-price portfolio construction with policy columns
   generated on demand;
3. a one-run Pareto frontier over all policy budgets k;
4. online evidence-conditioned selection among pre-certified policy classes.

## Current limitation

The implementation is currently a profile analysis layer over the exact
finite-family value table. It is not yet wired into the outer partition B&B,
does not support continuous parameter polytopes, and does not cover randomized,
history-dependent or partially observable policies.
