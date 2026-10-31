# Primal-dual receipts for utility-aware closed-eBH recertification

**Status:** research implementation on a branch from PR #9308. Not production-ready; remote CI has not been run for this branch.

## Motivation

The capacity-aware optimizer in PR #9308 already has a primal feasibility story: its separation oracle checks the most violated closed-eBH subset at each cardinality. Feasibility alone does not establish that the optimizer found the best utility value.

This change adds a second, independently checkable claim: a primal-dual optimality receipt.

Keep the claims separate:

1. **Statistical validity:** supplied e-values meet their theorem contract. This verifier does not prove that.
2. **Primal decision validity:** the rejection-count law is a valid probability distribution and satisfies every closed-eBH intersection constraint.
3. **Optimization optimality:** a dual-feasible upper bound matches the primal utility within a declared tolerance.

## LP and dual certificate

The solver uses cumulative charge variables

\[
S_r=\sum_{j=r}^{K}c_j,\qquad S_{K+1}=0,
\]

with per-rank charges \(c_r=S_r-S_{r+1}\) and rejection tails

\[
q_r=rS_r-(r-1)S_{r+1}.
\]

For sorted e-values \(e_{[1]}\ge\dots\ge e_{[K]}\) and nonnegative predeclared rank utility \(u_r\), the objective is

\[
\max_S \sum_{r=1}^{K}u_rq_r = \max_S a^\top S.
\]

The complete closed-eBH certificate requires, for every nonempty subset \(T\),

\[
\sum_{r\in T}(S_r-S_{r+1})
\leq \frac{\alpha}{|T|}\sum_{r\in T}e_{[r]}.
\]

Monotonicity, convexity, nonnegativity and the probability bound are also imposed. The cutting-plane solver starts with suffix rows and adds exact violated subset rows using the top-\(t\) separation oracle.

Write the generated LP as

\[
\max a^\top S\quad\text{s.t. }AS\leq b,\ S\geq0.
\]

A dual witness \(y\) proves a global upper bound when

\[
y\geq0,\qquad A^\top y\geq a.
\]

Then \(a^\top S\leq b^\top y\) for every feasible point. Constraints omitted by cutting-plane generation receive dual multiplier zero; this is valid for the dual of the full finite LP. The verifier reconstructs the semantics of every row carrying a nonzero (or explicit zero) multiplier rather than trusting an opaque solver matrix.

The receipt reports the primal objective, dual objective and gap. It is an optimality certificate only when both independent primal checks and dual feasibility pass and the gap is within tolerance.

## Independent verifier

The module validation/utility_recertification_certificate.py:

- binds the original e-value vector, utility vector and alpha to a canonical SHA-256 digest;
- reconstructs monotonicity, convexity, suffix and intersection rows from typed descriptors;
- checks the primal variables and the derived rejection-count distribution;
- independently checks the charge/probability identity;
- checks all closed-eBH subsets without enumerating them: for each cardinality \(t\), the most violated subset consists of the \(t\) largest values of \(c_r-\alpha e_{[r]}/t\);
- checks dual multiplier signs, \(A^\top y-a\), the dual upper bound and the primal/dual gap;
- returns an explicit failure list and a worst-subset witness.

The subset-separation check costs \(O(K^2\log K)\) with a straightforward sort per cardinality. This is a reference verifier, not the intended final high-throughput implementation.

## Reproduction

From the repository root:

    PYTHONPATH=validation python -m unittest -v test_utility_recertification.py
    PYTHONPATH=validation python -m unittest -v test_utility_recertification_certificate.py

The new tests cover a valid solver-generated receipt, input-digest mutation, a negative dual multiplier, primal tampering and noncanonical subset descriptors.

A separate development sanity screen reconstructed the same LP row semantics and solved 280 random instances with \(K=2,\ldots,15\). The largest absolute primal/dual gap was approximately \(3.6\times10^{-15}\), the minimum computed dual slack was approximately \(-7.8\times10^{-14}\), and the largest subset violation was approximately \(1.4\times10^{-16}\). This is numerical development evidence only: it is not a run of the committed test suite or remote CI.

## Scope boundaries

- Fixed-family, offline closed-eBH only.
- No online FDR, FDR-sup, or SupFDR claim.
- No theorem-level verification that input e-values are valid; that requires a separate assumption/evidence contract.
- No claim that the LP objective is downstream truth or societal utility. Utilities must be predeclared and external to unknown null labels.
- Floating-point primal/dual residuals are tolerance-bounded, not exact rational proofs. A future exact or interval-arithmetic verifier could strengthen this layer.
- A valid dual receipt proves optimality for the encoded finite LP. It does not repair an incorrect LP formulation or an invalid statistical theorem.

## Next gates

1. Run committed tests locally against the exact branch checkout and capture the exact commit SHA.
2. Add randomized differential tests against full all-subsets primal and dual LPs for \(K\leq10\).
3. Add adversarial degeneracy tests: tied e-values, zero utilities, zero e-values, extreme scales, duplicate cuts and near-zero slacks.
4. Add interval-arithmetic residual checking or exact-rational small-instance verification.
5. Benchmark certificate generation and verification separately at \(K=100,1{,}000,10{,}000\).
6. Only then propose a Rust port into the active statistics architecture.

## Related work

- [PR #9308 — utility-aware recertification](https://github.com/Luminous-Dynamics/symthaea/pull/9308)
- [Issue #9357 — proof-carrying risk decisions](https://github.com/Luminous-Dynamics/symthaea/issues/9357)
- [Issue #9488 — predictable-capacity online e-closure](https://github.com/Luminous-Dynamics/symthaea/issues/9488)
- [Bai & Ramdas (2026), arXiv:2610.35821](https://arxiv.org/abs/2610.35821)
