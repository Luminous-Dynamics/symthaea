# VIS-DVPC-039 — Independently verified MOPE constrained-bet optimizer

**Status:** numerical kernel draft; no full confidence-sequence or coverage claim.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8837

## Purpose

The full MOPE method in Karampatziakis, Mineiro & Ramdas (ICML 2021) chooses a predictable two-component bet by maximizing a concave quadratic lower bound on log wealth over a small feasible region. Because every later confidence-sequence bound depends on these bets, the optimizer is a critical implementation boundary. This PR isolates the kernel before it is integrated into a stream processor.

## Frozen mathematical problem

Given a symmetric positive-semidefinite 2x2 matrix A, vector b, maximum importance weight W >= 1, and

psi = 2 - 4 ln(2) < 0,

maximize

f(lambda) = psi * lambda^T A lambda + b^T lambda

subject to

- lambda_2 >= 0
- lambda_1 + lambda_2 <= 1/2
- (1-W) lambda_1 + lambda_2 <= 1/2.

For W > 1, the feasible set is a triangle with vertices

- (1 / (2(1-W)), 0)
- (0, 1/2)
- (1/2, 0).

The optimizer checks the unconstrained stationary point when A is invertible and feasible. It also optimizes the quadratic along each edge exactly as a one-dimensional concave quadratic. This handles singular or flat boundary directions without relying on a matrix inverse. Vertices are always included.

For W = 1, valid importance ratios are exactly one, so lambda_1's contribution and its sufficient statistics vanish. The kernel verifies that degeneracy invariant and optimizes only lambda_2 over [0, 1/2], avoiding division by zero in the vertex formula.

The result includes the objective value, active constraints, constraint slacks, and whether a feasible stationary point was used. Non-finite values, W below one, and non-PSD A fail closed.

## Verification strategy

Eight deterministic numeric fixtures exercise:
- a known interior optimum;
- mixed and singular positive-semidefinite matrices;
- the lambda_2 = 0 edge;
- a generic high-weight boundary case;
- the W = 1 degeneracy;
- interior optima on both slanted edges.

Each fixture is compared to a dense feasible triangular grid. The exact candidate must be at least as good as the grid maximum within a stated numeric tolerance. The grid is only an independent diagnostic; it is not the optimizer.

The focused workflow compiles and tests the Rust source, emits a machine-readable result table for every fixture, checks feasibility and known optima, and saves source/output hashes and the compiler version.

## Evidence boundary

This is only the **constrained-bet optimizer kernel** from the paper's "Avoiding Grid Search" section. It does not yet implement the sufficient-statistic updates, lower-bound polynomial, hedged upper/lower streams, root finding, running intersection, or any complete OPE confidence sequence. Passing these geometry tests would establish numerical kernel behavior only—not statistical coverage, production readiness or SOTA performance.

## Research source

Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021: https://proceedings.mlr.press/v139/karampatziakis21a.html

The exact equations and active-set algorithm are also available in the authors' paper source, section "Avoiding Grid Search", Algorithm argmax: https://github.com/n17s/off-policy-confidence-sequences/blob/7a25831a9ba5763e6a166a0d74ed61e54a7a7e4a/opecs.tex
