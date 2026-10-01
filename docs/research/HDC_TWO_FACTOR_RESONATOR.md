# HDC Two-Factor Resonator Harness

This harness is the next research layer after single-unknown associative cross-validation:
it evaluates genuinely coupled two-factor factorization.

## Problem definition

A clean query is constructed as:

`X ⊛ Y`

where `X` and `Y` are independently selected from the same deterministic bipolar
codebook. Query corruption is then applied to the bound product, so the resonator and
the exhaustive reference receive the same observed vector.

The resonator solves the coupled system:

- `Y ⊛ X ≈ query`
- `X ⊛ Y ≈ query`

with deterministic initialization seeds. Because both unknowns participate in each
other's update, this is materially different from the one-unknown associative case.

## Reference baseline

For each query, the harness also performs exhaustive pair retrieval over all
`n²` ordered pairs and reports:

- whether the target factor set was recovered, allowing X/Y swap because bipolar binding is commutative;
- exhaustive top-1 accuracy;
- top-1/top-2 similarity margin;
- explicit search-space size formula (`n²`), while avoiding a false ordered-pair claim;

The exhaustive path is a reference measurement, not a score against which a composite
ranking is constructed.

## Resonator outcomes

Every resonator trial is partitioned into:

- **correct convergence** — both factor estimates converged and both decoded factors
  match the target pair;
- **spurious convergence** — both estimates converged but at least one decoded factor
  is wrong;
- **non-convergence** — at least one factor did not satisfy the convergence condition
  within the configured budget.

Factor-level correctness, joint correctness, iterations, and factor margins remain
separate measurements.

## Why this layer matters

Recent resonator work treats factorization as a coupled dynamical problem whose
difficulty grows with the combinatorial search space, and explicitly separates
correct, spurious, and non-converged terminal states. This harness adopts that
measurement discipline while retaining a transparent exhaustive reference. See Yeung, Poduval, and Imani (2026), *A comparative study of nonlinear cleanup rules in resonator networks*, for the corresponding failure-mode analysis.

This is deliberately **not** a claim that resonators outperform exhaustive search
at these small sizes, nor a claim that any dimension is universally optimal. It is a
mechanism-validation surface for the implementation in Symthaea.

## Default CI qualification

The default qualification matrix is:

- dimensions: 1K, 2K, 4K, 8K;
- codebook sizes: 4, 8;
- query corruption: 0.00, 0.20, 0.35;
- three deterministic solver seeds;
- 16 maximum solver iterations.

That is 24 evidence cells. Larger factor counts, codebooks, and dimensions should be
run as extended research surfaces rather than making the qualification workflow
unbounded.

## Reproducibility

The evidence identity commits the dimension ladder, codebook-size ladder, query-noise
ladder, solver-seed ladder, fixture seed, query count, and iteration budget. The
fixture revision and final artifact digest are also emitted.

Focused validation:

`cargo test -p symthaea-core --lib two_factor_resonator_harness`

Evidence example:

`cargo run -p symthaea-core --example hdc_two_factor_resonator`
