# Plasma physics reference oracles

This standalone Rust package contains two deliberately narrow verification
primitives. It is not a production plasma solver.

## 1. Periodic 1D variational particle oracle

For particles with effective charges q_i, positions x_i, periodic length L,
permittivity epsilon_0, and fixed positive Fourier modes k_m = 2*pi*m/L,
the package defines the zero-mode-removed, self-interaction-excluded pair
energy

$$
U(X)=\frac{2}{L\epsilon_0}
\sum_{m=1}^{M}\sum_{i<j}
\frac{q_iq_j}{k_m^2}\cos(k_m(x_i-x_j)).
$$

The analytic negative gradient is

$$
F_i=-\frac{\partial U}{\partial x_i}
=\frac{2q_i}{L\epsilon_0}
\sum_{m=1}^{M}\sum_{j\ne i}
\frac{q_j}{k_m}\sin(k_m(x_i-x_j)).
$$

Positions are reduced modulo L. The zero Fourier mode is omitted, which
corresponds to a periodic neutralising-background convention. Pair self-energy
terms are excluded. This convention must be kept fixed when comparing energies
or training a learned scalar functional.

The oracle costs O(M N^2) and is intended for small-N known answers,
gradient checks, symmetry tests, and falsification of variational models. It
does not claim an untruncated Coulomb kernel, a time integrator, exact
time-discrete energy conservation, or a scalable PIC field solve.

## 2. Periodic 1D staggered-grid Gauss verifier

The verifier accepts right-face electric fields E_(i+1/2) and cell-centred
charge densities rho_i, on a uniform periodic mesh. Its discrete divergence is

$$
(D_hE)_i=\frac{E_{i+1/2}-E_{i-1/2}}{\Delta x},
\qquad
r_i=(D_hE)_i-\frac{\rho_i}{\epsilon_0}.
$$

It reports each cell residual, the maximum absolute residual, the RMS
residual, and integrated net charge. A periodic 1D divergence sums to zero,
so periodic Poisson data must have zero integrated net charge after the
declared background convention is applied.

This is a verifier for a specified staggered finite-volume operator. It does
not create or correct the electric field and does not reproduce the local
Gauss-preserving GP-PIC method from the October 2026 paper. Two- and
three-dimensional methods, alternative staggering, nonuniform meshes,
boundary terms, and source-code adapters require their own pinned discrete
operators and known-answer fixtures.

## Reproducible checks

Run from this directory:

~~~sh
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
~~~

No third-party Rust dependencies are used.

## 3. Coordinate normalization and periodic-topology audit

The `coordinate_audit` module provides four small mathematical helpers:

- sample min–max normalization, where changing an extremal coordinate can move every other normalized coordinate;
- fixed-domain periodic normalization, where one particle's coordinate does not depend on other particles;
- the periodic embedding `(sin(2*pi*x/L), cos(2*pi*x/L))`;
- a maximum unaffected-particle displacement metric for paired before/after coordinates.

Tests quantify extremal-particle perturbation at multiple particle counts and verify seam continuity/integer-period invariance of the periodic embedding. These are properties of the coordinate maps only. They do not prove a defect in NEOPIC or measure the trained model's field/trajectory response. The implementation-level tests remain tracked in [issue #12980](https://github.com/Luminous-Dynamics/symthaea/issues/12980).

This fixed-domain map is appropriate only for a correctly declared periodic domain. Open-boundary coordinate systems require their own physical normalization and must not be forced into a periodic embedding.

## 4. Residual-certified periodic Poisson warm starts

The `poisson_certificate` module certifies a potential estimate for one
specific operator: the uniform periodic 1D second-order finite-difference
Laplacian, restricted to the mean-zero subspace. Given the projected residual
norm and smallest nonzero eigenvalue, it reports conservative Euclidean
(`L2` vector) bounds for potential error and the forward-difference electric
field error.

The routine rejects charge-density and potential means outside caller-declared
tolerances. Small zero-mode projections are returned explicitly, including the
net charge before projection. The certificate does not bound pointwise errors,
particle-interpolated force errors, other stencils, nonuniform grids, open
boundaries, or higher-dimensional solvers.

Tests use manufactured Fourier eigenmodes, a deliberately perturbed potential,
actual-versus-certified potential and field errors, zero-mode reporting, and
invalid-input negative controls. This is a mathematical certificate for the
specified discrete operator, not a claim that a neural solver is already using
it. See [issue #12992](https://github.com/Luminous-Dynamics/symthaea/issues/12992).

## 5. Fail-closed correction routing

The `correction_policy` module routes a field proposal to one of four states:

- accept only when the field L2 error certificate meets the declared tolerance;
- spend a trusted correction step when a certificate is missing or its bound is too large and budget remains;
- use the trusted full-solver fallback when the certificate is invalid or correction budget is exhausted;
- remain unresolved when fallback is unavailable.

The initial implementation only accepts the pinned periodic 1D finite-difference Poisson operator identity. It verifies the certificate's operator identity, spectral gap, finite values and internally consistent residual-derived bounds before allowing acceptance. This is routing logic only; it does not implement the correction solver or provide a tamper-proof signed receipt. The integrating solver must persist immutable source/config/certificate/execution digests and re-evaluate the certificate after every correction.

## 6. Weighted particle-measure field known answers

The base variational oracle stores effective macro-particle charge in
`Particle1d::charge`; any macro-particle weight must already be included.
The new `periodic_fourier_field_at_probes` evaluates the truncated Fourier
electric field at arbitrary periodic probe positions from that weighted
particle measure.

Because the field is linear in source charge, the oracle should be invariant
(up to floating-point summation order) when a weighted source is split into
co-located fragments whose charges sum to the original, or when the particle
records are permuted. A negative control changes the total represented charge
and must change the field. Another test verifies that multiplying the probe
field by the probe charge matches the existing variational force.

This is a small-N Fourier reference with an explicitly omitted zero mode. It
does not prove that production solvers or the current NEOPIC implementation
support arbitrary particle weights, and it does not settle how a split/merge
operation must treat self-energy in the particle-particle energy functional.
See [issue #12991](https://github.com/Luminous-Dynamics/symthaea/issues/12991).
