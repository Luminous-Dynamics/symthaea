# AR-5B19: Exact Finite Policy-Frontier LP for Robust Randomized Continuation v0.1

**Status:** exact bounded reference draft; seven tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → #9854 → #9856 → #9860 → #9895 → #9897 → this branch.  
**Tracking:** #9898.

## 1. Research shift: from pairwise mixing to the policy frontier

AR-5B17 enumerates deterministic stop/continue policies and selects the best robust-utility policy. AR-5B18 convexifies a pair of complete policies. Neither alone solves the finite policy-frontier randomization problem when the best robust strategy needs support on three or more deterministic policies.

This reference accepts a bounded set of policy receipts and solves the exact finite maximin LP over all distributions on that supplied frontier.

The important distinction: it optimizes over the policies provided to it. It does not invent new stopping policies or claim to solve a general robust MDP.

## 2. LP formulation

For deterministic policies i = 1,...,M, define:
- q_i: ex-ante probability of selecting policy i;
- z: worst-case expected utility across the declared alternatives;
- s_ij: type-I size of policy i under null model j;
- u_ik: decision utility under alternative k;
- c_ik: expected evaluator cost under alternative k.

The LP is:

```
maximize z

subject to:
    sum_i q_i = 1
    q_i >= 0

    sum_i q_i * s_ij <= alpha
        for every declared null model j

    sum_i q_i * c_ik <= cost_cap
        for every alternative k, when a cap is supplied

    z <= sum_i q_i * u_ik
        for every declared alternative k
```

The optimum of this finite LP occurs at a basic feasible vertex. The implementation:
1. Builds the non-negativity, per-null size, per-alternative cost, and utility hypograph inequalities.
2. Enumerates every choice of M active inequalities, combined with the probability-sum equality.
3. Solves each square system using deterministic partial-pivot Gaussian elimination.
4. Reconstructs the mixture and independently checks every null and cost constraint.
5. Selects the feasible vertex with maximum worst-case utility, then lower maximum cost, lower worst-case null size, and lexicographically smaller canonical probability vector.

Bounds: at most six policy receipts, four null models, and four alternatives. With all limits at their maximum and a cost cap, the basis count is C(18, 6) = 18,564, within the 100,000-basis ceiling. This is a small exact oracle, not a scalable LP backend.

## 3. Three-policy fixture: why two-policy support is not enough

There are three alternatives and three deterministic policies:

| Policy | alt-a utility/power | alt-b utility/power | alt-c utility/power | evaluator cost |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 1 |
| B | 0 | 1 | 0 | 2 |
| C | 0 | 0 | 1 | 3 |

Every policy has null-a size 0.02 and null-b size 0.03. Set alpha=0.05.

No deterministic policy has worst-case utility above 0. A mixture over only two policies also has worst-case utility 0, because one of the three alternatives receives zero utility.

The exact three-policy optimum is:
- q_A = 1/3;
- q_B = 1/3;
- q_C = 1/3;
- worst-case utility = **1/3**;
- worst-case power = **1/3**;
- null sizes = 0.02 and 0.03;
- maximum expected cost = 2.0;
- mixture entropy = ln(3) = **1.098612 nats**.

With a hard evaluator cost cap of 1.6, the optimum shifts:
- q_A = 0.6;
- q_B = 0.2;
- q_C = 0.2;
- worst-case utility/power = **0.2**;
- expected cost = **1.6**.

The constrained solution exposes the actual price of evaluator budget; it does not hide it inside a soft cost coefficient.

These are exact constructed finite-frontier arithmetic fixtures, not empirical performance measurements.

## 4. Algorithm and numerical boundaries

The implementation enumerates active constraint bases rather than invoking an external solver. It uses f64, deterministic pivot selection, tolerance-checked feasibility, and independent metric reconstruction after solving each basis.

This is suitable for:
- tiny reference problems;
- verifier-side checks of an independently generated finite frontier;
- regression fixtures for a future scalable LP/column-generation implementation.

It is not suitable for hundreds or thousands of policies. A production solver should use a maintained LP backend and must be cross-checked against this exact bounded oracle on shared small fixtures.

## 5. Fail-closed controls

The seven authored tests cover:
- a true three-way optimum requiring all three policies;
- a hard-cost-cap solution that changes the mixture;
- per-null composite validity;
- mismatched model sets and utility contracts;
- infeasible cost caps and invalid non-finite receipts;
- policy frontier size limits;
- canonical input reordering and report tampering.

A future independent LP cross-check remains required. No test execution or solver-equivalence result is claimed.

## 6. Comparison to industry and SOTA

The reference hierarchy is now:
1. best deterministic policy (#9895);
2. best mixture between two supplied policies (#9897);
3. best distribution over the supplied finite policy frontier (this tranche);
4. future scalable LP / column generation over a larger generated frontier;
5. future robust MDP / online policy generation with independently verified conditional-error constraints.

Benchmarks should report:
- robust utility gain over the best deterministic policy;
- gain over the best pairwise mixture;
- optimal support size and mixture entropy;
- null size for every null distribution;
- per-alternative power, utility and expected cost;
- solver wall time, basis count, memory and frontier-size scaling;
- optimality gap versus an independent LP implementation;
- held-out and omitted-model stress separately.

A useful industry anchor is Spotify's July 2026 sample-size report: the company reports 8–20% simulation savings (14% under common WSKR settings) and median 9.5% saving across 713 metrics from 283 production experiments. Those are vendor-reported findings, not an independent replication. The target for Symthaea should be more ambitious than raw sample savings: reduce worst-case decision loss and evaluator cost at matched power and error guarantees.

Reference: https://confidence.spotify.com/blog/accurate-sample-size-always-valid

## 7. Proposed acceptance gates (targets, not results)

- Exact basis enumeration finishes within the declared 100,000 ceiling.
- The three-way fixture returns equal policy weights and utility 1/3.
- The cost-capped fixture returns weights (0.6, 0.2, 0.2) and cost 1.6.
- Every declared null size and alternative cost is reconstructed independently and remains feasible.
- Randomized bounded fixtures agree with an independent LP solver within declared numerical tolerances.
- Exact deterministic reconstruction is invariant to policy and model input order.
- No SOTA statement until matched real or independently seeded benchmark runs exist.

## 8. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering policy_frontier_lp -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

Seven tests are authored. No compile, formatting, clippy, unit-test, CI, benchmark, independent solver cross-check or SOTA pass is claimed.
