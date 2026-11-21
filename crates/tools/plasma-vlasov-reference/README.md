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

The common electron convention is (q/m=-1) and (n_{\mathrm{background}}=1). All parameters are explicit in `VlasovPoissonConfig`.

The spatial coordinate is periodic on a circle. Physical velocity is unbounded, but the numerical solver uses a finite periodic interval `[velocity_min, velocity_max)`. The velocity interval and its periodic boundary condition are a numerical truncation. A valid reference study must increase the velocity-domain extent and resolution and demonstrate insensitivity of the reported observables.

## Numerical method

A time step uses Strang splitting:

1. exact Fourier translation for (v\,\partial_x f) over (\Delta t/2);
2. recompute the self-consistent field from the intermediate density and apply Fourier translation for ((q/m)E\,\partial_v f) over (\Delta t);
3. exact Fourier translation for (v\,\partial_x f) over (\Delta t/2).

The periodic Poisson equation is solved spectrally with a zero-mean gauge. The zero and collocation Nyquist modes are set to zero. Any non-neutral or unresolved Nyquist charge remains visible in the reported Poisson residual; it is not silently projected into a pass.

The in-house radix-2 FFT has no external dependencies. The time integrator is second-order Strang splitting. It is not an energy-preserving discrete-gradient or variational integrator. Small time-step drift must be measured; it must not be described as exact conservation.

## Diagnostics

The solver reports mass, momentum, kinetic energy, field energy, total energy, integrated net charge, maximum collocation spectral Poisson residual, minimum distribution value and velocity-endpoint occupancy ratio.

These are raw observations. In particular:

- the endpoint ratio is a truncation warning, not a rigorous tail bound;
- spectral interpolation does not guarantee positivity;
- small Poisson residual only qualifies the exact spectral operator used here;
- energy drift is not automatically covered by the Poisson residual;
- phase-space MMD and W1 are not built into this solver and must remain separate diagnostics with their own contracts.

## Verification tests

The unit suite contains:
- radix-2 FFT forward/inverse round-trip;
- homogeneous Maxwellian equilibrium;
- manufactured free-streaming characteristic solution;
- analytic single-mode spectral Poisson solution;
- non-neutral periodic negative control;
- coupled-step mass and residual check;
- invalid grid, non-finite input and invalid timestep rejection;
- electric-mode diagnostic validation.

Run with Rust 1.96:

```sh
cargo fmt --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml -- --check
cargo test --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml
cargo clippy --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --all-targets -- -D warnings
```

## Reproducible probes

Landau-damping diagnostic trace:

```sh
cargo run --release --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --example landau_reference
```

Coupled space/velocity and independent timestep refinement probe:

```sh
cargo run --release --locked --manifest-path crates/tools/plasma-vlasov-reference/Cargo.toml --example refinement_probe
```

The refinement example reports electric fundamental-mode amplitude, mass and Poisson residual at a common final time. It is an initial convergence probe, not a complete L3 admission study. A full reference admission still requires separate spatial, velocity-grid, velocity-domain and timestep studies; comparison with an independent second formulation; long-horizon drift; and explicit reference uncertainty.

## Relationship to the rest of the plasma stack

This crate does not import or reuse the L0 particle Fourier oracle, the staggered Gauss verifier, the finite-difference Poisson certificate, the MMD implementation, W1 transport, or the trajectory ledger. This intentional isolation avoids circular validation. Future adapters should consume its raw states and bind their receipts to the exact solver and diagnostic revisions.

The L0 Fourier particle operator, the L0 staggered finite-volume Gauss operator, and this distribution-space spectral Poisson solver are three separate discrete operators. A certificate for one must never be transferred to another.
