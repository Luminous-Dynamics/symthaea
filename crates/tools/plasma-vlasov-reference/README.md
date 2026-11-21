# Periodic 1D1V Vlasov–Poisson reference prototype

This isolated, dependency-free Rust crate is the first distribution-space trajectory engine for the plasma programme. It is a numerical reference prototype, not a production PIC solver, not a continuum certificate, and not evidence of an SOTA result.

## Equations and convention

The normalized collisionless electrostatic cohort is

\[
\partial_t f + v\,\partial_x f + (q/m)E\,\partial_v f=0,
\qquad \partial_x E=\rho/\epsilon_0,
\]

with

\[
\rho(x,t)=n_{\mathrm{background}}-\int f(x,v,t)\,dv.
\]

The electron charge is normalized to -1, so the mass is `m = -1/(q/m)` and the configured charge-to-mass ratio must be finite and negative. The positive background charge density is `n_background`; neutrality at unit electron number density uses `n_background=1`. Kinetic moments include this particle mass, and field energy includes the configured permittivity.

The spatial coordinate is periodic on a circle. Physical velocity is unbounded, but the numerical solver uses a finite periodic interval `[velocity_min, velocity_max)`. The velocity interval and its periodic boundary condition are a numerical truncation. A valid reference study must increase the velocity-domain extent and resolution and demonstrate insensitivity of the reported observables.

## Numerical method

A time step uses Strang splitting:

1. exact Fourier translation for (v\,\partial_x f) over (\Delta t/2);
2. recompute the self-consistent field from the intermediate density and apply Fourier translation for ((q/m)E\,\partial_v f) over (\Delta t);
3. exact Fourier translation for (v\,\partial_x f) over (\Delta t/2).

The periodic Poisson equation is solved spectrally with a zero-mean gauge. The zero and collocation Nyquist modes are set to zero. Any non-neutral or unresolved Nyquist charge remains visible in the reported Poisson residual; it is not silently projected into a pass.

The in-house radix-2 FFT has no external dependencies. The time integrator is second-order Strang splitting. It is not an energy-preserving discrete-gradient or variational integrator. Small time-step drift must be measured; it must not be described as exact conservation.

## Fourier interpolation / zero-padding oracle

`zero_pad_periodic_line` refines a uniformly sampled real periodic line by zero-padding its Fourier coefficients. It preserves the trigonometric interpolant at old grid nodes and at new nodes, with the source Nyquist coefficient split evenly across the positive and negative target modes. It rejects non-power-of-two grids, coarsening, too-small inputs and non-finite samples.

This is a standalone transfer oracle. It does not mutate the Vlasov state, apply a spectral-tail trigger, or assert that a resolution transition is physically safe. Adaptive expansion requires a separate receipt for mass, momentum, energy, phase-space discrepancy, field consistency and full transfer cost.

## Diagnostics

The solver reports mass, momentum, kinetic energy, field energy, total energy, integrated net charge, maximum collocation spectral Poisson residual, minimum distribution value and velocity-endpoint occupancy ratio. For a requested spatial mode it also exposes the complex Fourier coefficient (real part, imaginary part, magnitude and principal phase), so oscillatory field modes are not reduced to a magnitude-only time trace. The spectral-tail diagnostic separately reports high-frequency and Nyquist energy fractions along x and v for a frozen mode-band threshold.

These are raw observations. In particular:

- the endpoint ratio is a truncation warning, not a rigorous tail bound;
- spectral interpolation does not guarantee positivity;
- small Poisson residual only qualifies the exact spectral operator used here;
- energy drift is not automatically covered by the Poisson residual;
- phase-space MMD and W1 are not built into this solver and must remain separate diagnostics with their own contracts.

## Verification tests

The unit suite contains 19 tests:
- radix-2 FFT forward/inverse round-trip;
- homogeneous Maxwellian equilibrium;
- manufactured free-streaming characteristic solution;
- analytic single-mode spectral Poisson solution;
- non-neutral periodic negative control;
- coupled-step mass, momentum, total-energy and residual checks;
- observed second-order timestep-convergence ratio on a coupled trajectory;
- phase-space grid-size overflow rejection;
- invalid grid, non-finite input and invalid timestep rejection;
- complex electric-mode coefficient, phase and amplitude consistency;
- spectral-tail axis separation for high x-only and v-only modes;
- invalid spectral-tail threshold rejection;
- exact Fourier zero-padding of band-limited signals and old nodes;
- symmetric Nyquist splitting under refinement;
- constant-line and identity-grid interpolation;
- invalid grid/non-finite interpolation rejection;
- truncated-velocity-domain negative control despite low v spectral tail.

Run with Rust 1.96:

```sh
cargo fmt --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml -- --check
cargo test --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml
cargo clippy --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --all-targets -- -D warnings
```

## Reproducible probes

Matched Landau amplitude sweep at fixed physical/numerical configuration (same Nx, Nv, domain, velocity interval, timestep and horizon), with amplitudes 1e-3 and 1e-2:

```sh
cargo run --release --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --example landau_amplitude_sweep
```

The 1e-2 lane matches the VlasovTT paper's stated Landau perturbation amplitude, but it does not reproduce the paper's 1024-by-1024 TT/dense numerical configuration.

Longer phase-resolved Landau-damping and spectral-tail trace to t=10 (complex Fourier coefficient, magnitude, phase, x/v tail fractions and Nyquist fractions):

```sh
cargo run --release --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --example landau_reference
```

Coupled space/velocity and independent timestep refinement probe:

```sh
cargo run --release --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --example refinement_probe
```

The refinement example independently varies spatial grid size, velocity grid size, velocity-domain extent, and timestep at a common final time. It reports electric fundamental-mode amplitude, mass, total energy, Poisson residual, velocity-endpoint ratio, minimum distribution, and independent x/v high-frequency tail fractions. It is an initial convergence probe, not a complete L3 admission study. A full reference admission still requires separate spatial, velocity-grid, velocity-domain and timestep studies; comparison with an independent second formulation; long-horizon drift; and explicit reference uncertainty.

The focused GitHub workflow executes both examples on its exact head after formatting, unit tests and strict Clippy pass, then uploads the generated CSV traces as a workflow artifact. These hosted-runner traces are reproducible measurements for that build, not a substitute for independent physical validation or a matched external baseline. Spectral-tail fractions are warning indicators only; they are not residual-to-error certificates and do not by themselves authorise grid growth.

## Relationship to the rest of the plasma stack

This crate does not import or reuse the L0 particle Fourier oracle, the staggered Gauss verifier, the finite-difference Poisson certificate, the MMD implementation, W1 transport, or the trajectory ledger. This intentional isolation avoids circular validation. Future adapters should consume its raw states and bind their receipts to the exact solver and diagnostic revisions.

The L0 Fourier particle operator, the L0 staggered finite-volume Gauss operator, and this distribution-space spectral Poisson solver are three separate discrete operators. A certificate for one must never be transferred to another.


## Offline representation-transition qualification

`src/transition.rs` exposes a dependency-free, fail-closed evaluator for
proposed representation handoffs:

- `qualify_transition(&TransitionEvidence, TransitionBudget)` returns
  `Accepted`, `Rejected`, or `Inconclusive`.
- An `Accepted` decision requires every mandatory measurement, valid
  source/destination SHA-256 identities, and all measured errors and resource
  costs within the frozen budget.
- A measured fidelity or resource-budget violation is `Rejected`.
- Missing or malformed observations, identities, or budgets are
  `Inconclusive`; they cannot be interpreted as a pass.
- Positivity and negative-mass checks run only when the caller explicitly
  declares `DistributionSemantics::Nonnegative`. Signed delta-f states do not
  receive an invalid positivity gate.

The evidence contract expects weighted L1/L2 distribution discrepancies on a
common declared diagnostic domain and quadrature, invariant errors computed
with a frozen denominator convention, a destination field residual for the
destination's own discrete operator, and end-to-end transition memory/time
measurements. The negative-mass fraction is
`integral(max(-f, 0)) / integral(abs(f))`.

This is an **offline qualification gate**, not a transfer operator, adaptive
controller, rigorous continuum error bound, or permission to change a live
solver state. Thresholds must be calibrated and preregistered for the relevant
equation cohort. Persist the complete evidence and budget next to each receipt;
the returned Rust value is not itself a JSON serializer. Persist missing measurements as explicit JSON null values when recording an inconclusive attempt; the schema allows these nulls, while an accepted record requires complete numeric measurements and valid SHA-256 identities.

The portable evidence shape is described in
`benchmarks/plasma/schemas/representation-transition-v0.1.schema.json`.
The evaluator tests acceptance, measured failure, missing evidence, malformed
hashes/measurements, invalid budgets, resource exhaustion, and the signed vs
nonnegative distribution distinction.


## Affine velocity-frame moment oracle

The optional research module `velocity_frame.rs` records exact raw-moment
identities for an affine velocity coordinate (v=u+\alpha\xi), with
(alpha>0) and computational density
(g(\xi)=\alpha f(u+\alpha\xi)). The convention makes the zeroth moment
invariant:

[
M_0=\mu_0,\qquad
M_1=u\mu_0+\alpha\mu_1,\qquad
M_2=u^2\mu_0+2u\alpha\mu_1+\alpha^2\mu_2.
]

The inverse transform is also provided. This is an algebraic moment oracle,
not a remap of sampled values, adaptive Hermite projection, positivity
preserver, or live representation switch. It applies equally to signed
perturbation moments; positivity is not assumed. The second raw moment is
reported before multiplying by particle mass or one half, which belong to the
physical model's kinetic-energy convention.

The tests cover known-answer identities, inverse and inter-frame round trips,
zeroth-moment invariance, signed moments, invalid shifts/scales, non-finite
moments and overflow. The module is research infrastructure for
[#13130 — adaptive Hermite coordinates and kinetic residual compression](https://github.com/Luminous-Dynamics/symthaea/issues/13130);
it does not claim an adaptive Hermite implementation or invariant preservation
for a discretized, remapped distribution.


### Analytic affine-moment sensitivities

`VelocityFrame::moment_jacobian` returns the exact first derivatives of
((M_0,M_1,M_2)) with respect to the computational moments, frame shift
(u), and positive frame scale (alpha). The shift and scale derivatives
are checked against central finite differences on a manufactured moment
state. Non-finite Jacobians are rejected rather than propagated.

This is a narrowly scoped analytic sensitivity primitive. It does not
differentiate a distribution remap, Hermite coefficient projection, kinetic
time integrator, or adaptive representation-selection policy. Those paths
need their own discrete derivative and conditioning qualification.


## Source-derived mixed Hermite-Legendre basis oracle

`src/mixed_basis.rs` implements the normalized asymmetrically weighted
Hermite recurrence and finite-window normalized Legendre recurrence from the
pinned ADEPT source
(`ergodicio/adept@0c85ce0d3d66abab51212191c2644f58a31a49cd`,
`adept/_hermite_legendre_1d/vector_field.py`, blob
`f65b1700f2c98ed942c0d8ed9baefca1b30bb822`).

The Hermite evaluator returns row-major values using the source shift and
positive scale. The Legendre evaluator returns row-major values on the
configured closed velocity interval and exact zeros outside it. Both reject
invalid inputs and shape overflow, and reserve the output allocation before
writing values.

The tests cover:
- Hermite known answers at transformed coordinates zero and one;
- the normalized Hermite recurrence through 256 modes;
- Legendre values at the interval midpoint and endpoints;
- exact zero support outside the Legendre interval;
- invariance under affine rescaling of the Legendre interval;
- invalid shift, scale, window and velocity rejection;
- empty shapes and multiplication overflow.

This is a basis-evaluation oracle, not a full mixed-basis solver. It does not
reconstruct a total distribution, evolve coefficients, apply ADEPT's
Hermite-to-Legendre closure, solve Poisson, or claim positivity/conservation.
It is intended to make the source-faithful basis contract independently
testable before implementing a coefficient reconstruction or trajectory
adapter.
