# VIS-DVPC-040 — Streaming MOPE sufficient-statistic identity audit

**Status:** algebraic streaming kernel; no confidence-sequence claim.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8848

## Why this layer exists

Full MOPE chooses a two-component predictable bet, then updates a quadratic lower bound on log wealth as a polynomial in the candidate value v. The lower bound is only useful if the per-sample sufficient-statistic expansions have exactly the right signs and coefficients. This PR implements the fixed-memory accumulator and compares it with direct O(n) recomputation at every prefix before confidence-bound roots are used in later work.

The numerical constrained optimizer is reused from parent PR #8838 rather than copied. This branch depends on that PR's branch until the parent kernel is merged.

## Algebraic identities

For each observation (w_i,r_i), let z_i=w_i-1 and d_i=w_i r_i. Define b_i(v)=[z_i,d_i-v] and A_i(v)=b_i(v)b_i(v)^T. Maintain:

A0 = Σ [[z_i²,z_i d_i],[z_i d_i,d_i²]]
A1 = Σ [[0,-z_i],[-z_i,-2d_i]]
A2 = Σ [[0,0],[0,1]]
b0 = Σ [z_i,d_i]
b1 = Σ [0,-1].

For every v,

Σ A_i(v) = A0 + v A1 + v² A2,
Σ b_i(v) = b0 + v b1.

For each sample, the predictable bet λ_i is selected from prior observations; its lower-bound contribution is

λ_iᵀ b_i(v) + ψ λ_iᵀ A_i(v) λ_i,

where ψ = 2 - 4 ln(2). The stream stores the five scalar coefficients

C = Σ λ_iᵀ[z_i,d_i],   S = Σ λ_{2,i},
Q = Σ ψ λ_iᵀ A_i(0) λ_i,
T = Σ ψ λ_iᵀ A_i^(1) λ_i,
U = Σ ψ λ_{2,i}².

Therefore P(v)=C-Sv+Q+Tv+Uv² must equal the sum of the direct per-observation lower-bound contributions at every candidate value. The source maintains these coefficients incrementally, without storing the sample history.

## Predict-then-observe order

- A stream starts with λ=(0,0).
- The next observed sample updates the polynomial statistics using the already-selected, predictable λ.
- The same sample updates the v-polynomial sufficient statistics.
- Only after the observation may a caller request the optimizer's next bet for a candidate value.
- Both sample and bet are checked against the frozen input bounds / paper feasible region.

The stream holds only matrices, vectors, five polynomial coefficients, the sample count, and the current bet. The number of retained numeric values does not grow with the stream length.

## Exact regression design

Frozen sequences cover:
- unit importance weights W=1;
- zero weights and fractional weights;
- a high-importance-weight observation at W=100;
- all-zero rewards.

At every prefix, the implementation compares the streamed expansions against independent direct O(n) recomputation for v=0.00,0.05,...,1.00. It also invokes the parent optimizer on updated sufficient statistics and checks returned bets are feasible. Separate tests exercise quadratic, linear and no-real-root polynomial cases.

The run reports algebraic identity success, not statistical coverage. No interval endpoints are emitted or interpreted as a confidence sequence here.

## References and boundaries

Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021, eqs. (7)-(14), (18)-(22), and the LCS algorithm:
https://proceedings.mlr.press/v139/karampatziakis21a.html
Paper source:
https://github.com/n17s/off-policy-confidence-sequences/blob/7a25831a9ba5763e6a166a0d74ed61e54a7a7e4a/opecs.tex

This is a sufficient-statistic and polynomial identity audit only—not the complete MOPE confidence sequence, not evidence of beating SOTA, and not a production result. The next layer must verify the one-sided endpoint/root-selection semantics and the hedged intersection against direct wealth on exact finite paths before publishing any coverage claim.
