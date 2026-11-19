# Verified finite-cover symmetry-defect reference v0.1

First executable slice for [SYMM-DEFECT-COVER-001 #12936](https://github.com/Luminous-Dynamics/symthaea/issues/12936), composed with [SYMM-ERROR-ENVELOPE-001 #12830](https://github.com/Luminous-Dynamics/symthaea/issues/12830).

## Paradigm shift: turn samples into a domain-wide bound, not a sample maximum

For the one-step defect function

d(x) = ||F(Tx) - T(F(x))||,

a finite set of small measured residuals does not prove that the defect is small between samples. V0 requires:
- an externally qualified upper bound on the defect Lipschitz constant K;
- a certified upper bound d_i at each sample centre x_i;
- closed interval cells [x_i-r_i, x_i+r_i] whose union is checked exactly against the complete declared one-dimensional domain.

For every point in cell i, Lipschitz continuity gives

d(x) <= d_i + ceil(K r_i).

Since the code verifies that the cells cover the full domain, it derives

sup_D d(x) <= max_i (d_i + ceil(K r_i)).

The upper bound uses the largest cell-specific value, not the largest observed sample residual. Boundary-touching cells are valid; any gap causes a fail-closed error.

## Exact verification and provenance

The interval union is checked by an exact integer sweep after clipping cells to the declared domain. Input, sample, Lipschitz and domain certificate identifiers are bound into canonical versioned bytes. The receipt includes every cell interval, residual upper bound, coverage inflation, witness sample and global maximum. Receipt verification recomputes the cover and every bound.

The method verifies the geometry of the one-dimensional cover; it does not verify that the external sample residual and Lipschitz certificates are true. Their validity remains a separate premise. It also does not prove that the declared interval is the correct physical domain.

## Connection to the finite-horizon error envelope

The function to_uniform_domain_step_premise turns a verified uniform whole-domain bound into a StepPremiseV1 with zero additional state-dependent defect term. That conversion replays the original receipt. It is sound only if both the reference and transformed trajectories are independently certified to remain in the same covered domain at every step. The error-envelope engine still requires its own domain-invariance and boundary-margin evidence.

## Current scope and limits

One-dimensional closed intervals only. This is a deterministic exact cover, not random/probabilistic coverage. It avoids a high-dimensional geometric claim by keeping the geometry verifier narrow and exact.

For higher dimensions, require a separately verified polytope/ball cover or a formal geometry certificate. Random points or nearest-neighbour distances alone are not a cover.

## Research comparisons

- CARe (CDC 2025) bounds Hamilton–Jacobi residual error over the domain using SMT and closes the loop with counterexample-guided refinement. It is the formal residual-verification comparator when the defect expression is SMT-encodable: https://doi.org/10.1109/cdc57313.2025.11312709
- On the Limits of Sampling-Based Reachability (CoRL 2026) shows that upgrading sampling-based recovery to Hausdorff guarantees requires geometric and Lipschitz regularity; its sample complexity remains exponential in dimension and horizon in the general setting. This motivates reporting cover cardinality rather than claiming random samples solve the curse of dimensionality: https://arxiv.org/abs/2607.18606

This crate is a one-dimensional exact-cover baseline, not a replacement for SMT/CEGIS or general reachability analysis.

## Tests

- exact touching and nonuniform cover cells;
- gap detection, boundary clipping and a single-point domain;
- unit mismatch and missing sample evidence;
- overflow in cell geometry and Lipschitz inflation;
- duplicate IDs, invalid domains and forged/mutated receipts;
- conversion into the error-envelope step premise.

## Reproduction

    cargo fmt --manifest-path research/symm-defect-cover-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symm-defect-cover-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symm-defect-cover-001/Cargo.toml --all-targets -- -D warnings

## Benchmark targets (prospective)

- 100% exact cover acceptance/rejection on exhaustive one-dimensional cover cases;
- 0 accepted incomplete covers;
- 100% receipt replay agreement;
- 0 arithmetic-overflow acceptance;
- report the ratio uniform_bound / exact_domain_supremum, sample count, largest radius, total covered length, wall time, memory and receipt size;
- sweep domain span, cell count, Lipschitz constant and defect variation; report both bound tightness and verification cost.

## Claim boundary

This reference proves only that the supplied intervals cover the supplied one-dimensional integer domain and computes the corresponding conditional Lipschitz upper bound. It does not prove the sample residual or Lipschitz certificates are correct, establish a high-dimensional cover, validate physical invariance, prove approximate group structure, certify safety, or authorize quotient pruning.
