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
