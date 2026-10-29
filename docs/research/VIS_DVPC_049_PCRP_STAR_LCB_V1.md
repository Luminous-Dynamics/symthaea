# VIS-DVPC-049 — pCRP-star lower confidence bound for OPE selection

**Status:** standalone numerical comparator draft. Exact-head tests and the focused workflow must pass before treating outputs as Rust-qualified. This is not the complete COLT 2025 experimental reproduction and makes no SOTA or production claim.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8958

## Why this tranche

Ryu, Kwon, Koppe and Jun, *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing*, COLT 2025, Section 3 develops a variance-adaptive lower confidence bound for nonnegative random variables using universal portfolios. Their off-policy selection method applies that bound separately to each candidate policy's importance-weighted reward and selects the largest lower confidence bound.

Reference: https://proceedings.mlr.press/v291/ryu25a.html  
Full text: https://arxiv.org/abs/2502.10826

The implemented primitive is the penalized best constant-rebalanced-portfolio bound (pCRP-star-LCB), paper equations (3.3)–(3.5). For nonnegative scores Y_1,...,Y_n, candidate mean nu>0, and a constant bet b in [0,1]:

log W_CRP(b; nu) = sum_t log(1 - b + b Y_t / nu)

log W_pCRP-star(nu) = sup_b log W_CRP(b; nu) - 0.5 log(pi(n+1))

The lower bound is the smallest nu whose penalized wealth is at most 1/delta. The implementation solves the one-dimensional bet optimum and root numerically. The confidence-bound argument relies on the paper's nonnegative-process assumptions; in the frozen synthetic evaluation, logger, policies and reward mechanism are stationary and actions/rewards are IID.

## Numerical-safety design

A confidence bound should not become optimistic because the optimizer's numerical error changes the direction of its guarantee. The implementation evaluates the log wealth at a feasible bet; any feasible bet's wealth is less than or equal to the supremum over bets. The root inversion uses that feasible wealth as a lower witness for the supremum. Underestimating the supremum can only move the lower confidence bound down, making it more conservative; it must not be replaced by an upper envelope, which could raise the LCB and invalidate coverage. A small downward floating-point margin is applied to the evaluated log wealth. This is a numerical engineering precaution, not a machine-checked interval-arithmetic proof.

The code includes the endpoint cases, zero rewards, non-finite / negative inputs, probability validation, importance-weight support checks, and a distinct NOT_IDENTIFIED status for structural zero support. It never creates a numeric estimate or bound for unsupported target-policy mass.

## Frozen comparison

- Three candidate policies with P(A=1) in {0.10, 0.50, 0.90}.
- Reward means: E[R|A=0]=0.25, E[R|A=1]=0.75; analytical values are 0.30, 0.50 and 0.70.
- Healthy fixed logger: P(A=1)=0.50.
- Weak-overlap fixed logger: P(A=1)=0.99.
- Structural zero-support logger: P(A=1)=1; target policies put positive mass on action zero, so results must be suppressed.
- Same logged stream and seed schedule for all candidates and both comparison methods.
- Horizons 64, 256 and 512; 32 replicates in the initial deterministic engineering sweep. This is a smoke-scale engineering panel, not enough to establish tight statistical performance comparisons.

Methods:
1. pCRP-star-LCB with delta_policy = 0.05 / M (variance-adaptive; confidence bound is time-uniform under its assumptions).
2. A bounded-reward Hoeffding LCB with the same per-policy alpha allocation and a known importance-score range (fixed-horizon comparator, not advertised as anytime-valid).

Metrics include familywise lower-bound coverage, top-1 policy selection accuracy, simple regret, selected lower bound, lower-bound gap to the selected policy's known truth, Monte Carlo standard errors, and elapsed calculation time. Timing rows are excluded from deterministic replay comparisons.

## Acceptance and limits

- Analytic constant-stream root and zero-stream tests pass.
- The tangent envelope dominates a dense grid of log wealth in a deterministic fixture.
- The known expectation identity for importance-weighted reward matches the analytical policy value.
- Nonnegative finite score validation and structural-zero-support fail-closed behavior pass.
- Repeated smoke runs produce identical non-timing output; exact source/output digests are published.
- The full workflow must compile and test the exact PR head. Queue status, skipped general benchmark jobs, or a passing syntax check alone do not qualify the Rust result.

This is a component-level reproduction of the pCRP-star bound, not the full Cover universal-portfolio integral, not the full COLT 2025 selection experiment, and not the separate freezing-based off-policy learning method. Do not claim superiority until a faithful comparison on matched workloads and an external confirmation dataset have been completed.

## Reference

Ryu, J. J., Kwon, J., Koppe, B., and Jun, K.-S. (2025). *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing.* Proceedings of Machine Learning Research 291, 5015–5053. https://proceedings.mlr.press/v291/ryu25a.html
