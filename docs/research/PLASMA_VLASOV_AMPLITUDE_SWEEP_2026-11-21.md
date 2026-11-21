# Landau amplitude sweep: matched weak and paper-amplitude traces

**Date:** 2026-11-21  
**Source branch:** `research/plasma-vp-spectral-reference-v0-1`  
**Source commit:** `dff72ad9a5e42d5e26acb580f9b624f3a1f47676`  
**Workflow:** [44905069931](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44905069931)  
**Artifact:** [14707492942](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44905069931/artifacts/14707492942)  
**Artifact SHA-256:** `561a800261722e60bf8d589df65e0dc0106e4cc11a7f2959331007ffa9b933d9`  
**Machine-readable receipt:** [run-manifest-v0.1.json](../../benchmarks/plasma/runs/vlasov-reference-rust1.96/2026-11-21-amplitude-sweep/run-manifest-v0.1.json)

## Exact-head engineering qualification

The Rust 1.96 workflow passed on the PR merge candidate containing source commit `dff72ad9a5e42d5e26acb580f9b624f3a1f47676`:

- strict `cargo fmt --check`;
- 19 unit tests, 0 failures;
- strict Clippy with `-D warnings`;
- weak-Landau reference trace to (t=10);
- independent space / velocity-resolution / velocity-domain / timestep probe;
- matched-amplitude sweep to (t=10);
- CSV artifact upload.

The tested merge candidate was `a7077b7360c3313ffcec518ba26a3321bf706ca9`, merging the PR source head into base candidate `36eb3adc1dc2d5ad1ae643863fee828d594472d4`. The run's source branch head and merge checkout are both recorded; this is not a test of an unrecorded local working tree.

The artifact contains three CSV files. SHA-256 digests and row counts are recorded in the manifest. The artifact expires on 2026-12-21.

## Matched physical and numerical setup

Both cases use exactly the same solver, grid, timestep, final time and all physical parameters apart from the initial density perturbation amplitude:

| Parameter | Value |
|---|---:|
| Equation cohort | Collisionless electrostatic Vlasov–Poisson, 1D1V |
| Spatial period | (4\pi) |
| (N_x\times N_v) | (64\times128) |
| Velocity interval | ([-6,6]), periodic numerically |
| Thermal speed | 1 |
| Fundamental spatial mode | 1, hence (k=0.5) |
| Charge-to-mass ratio | -1 |
| Permittivity / background density | 1 / 1 |
| Timestep | 0.01 |
| Final time | 10 |
| Output interval | 0.1 |
| Amplitudes | (10^{-3}) and (10^{-2}) |

Each case contains 101 samples from (t=0) through (t=10). The CSV records the real and imaginary parts, magnitude and phase of the positive fundamental electric-field Fourier coefficient, along with mass, momentum, total energy, Poisson residual, velocity-endpoint occupancy and separate spatial/velocity spectral-tail fractions.

The Fourier coefficient magnitude is not the real-space cosine peak amplitude. The damping fits below use the magnitude of the coefficient's imaginary component at sampled post-initial extrema.

## Observations

Four post-initial extrema appear at the same sampled times for both amplitudes:

| Time | (alpha=10^{-3}) imaginary coefficient | (alpha=10^{-2}) imaginary coefficient |
|---:|---:|---:|
| 2.5 | (-5.056582033394\times10^{-4}) | (-5.056323559188\times10^{-3}) |
| 4.7 | (+3.532724335582\times10^{-4}) | (+3.532202947605\times10^{-3}) |
| 7.0 | (-2.510287827752\times10^{-4}) | (-2.509348424696\times10^{-3}) |
| 9.2 | (+1.788269546570\times10^{-4}) | (+1.787021454843\times10^{-3}) |

A descriptive least-squares regression of (ln(|E_{k,mathrm{imag}}|)) against the four extremum times gives:

| Initial amplitude | Descriptive rate (gamma) | (R^2) |
|---:|---:|---:|
| (10^{-3}) | -0.15444615 | 0.999670 |
| (10^{-2}) | -0.15454296 | 0.999674 |

A window-sensitivity fit excluding the first post-initial extremum gives:

| Initial amplitude | Last-three-extrema rate (gamma) | (R^2) |
|---:|---:|---:|
| (10^{-3}) | -0.15127274 | 0.999886 |
| (10^{-2}) | -0.15139496 | 0.999884 |

The amplitude-pair ratio

[
\frac{|E_k(\alpha=10^{-2})|}{10|E_k(\alpha=10^{-3})|}
]

ranges from 0.994594 to 1.000377 over the 101 aligned samples. Its maximum absolute deviation from unity is about 0.541%; at (t=10), the ratio is 0.999363.

This is a strong approximate 10x scaling observation for the paired trajectories on this fixed numerical configuration. It suggests that increasing the initial perturbation by a factor of ten did not materially change the measured mode envelope over this horizon and discretization.

## Interpretation boundary

The paired response is useful, but it is not yet a validated linear Landau damping reproduction:

1. Only one (64\times128) phase-space grid and one timestep are used for the (t=10) amplitude pair.
2. The existing independent refinement probe ends at (t=1); it does not establish long-horizon convergence.
3. The envelope extraction and fitting interval were selected after observing the original trace.
4. Only four post-initial extrema are available, sampled every 0.1 time units; sub-sample extrema localization is not implemented.
5. No independent distribution-space reference has been matched yet.
6. The finite periodic velocity interval is a numerical truncation; velocity-domain adequacy needs independent convergence.
7. No runtime, memory, or energy-to-solution advantage has been measured.

The VlasovTT paper's stated amplitude of 0.01 is now represented as a distinct input lane, but this run does not reproduce its 1024-by-1024 numerical setup or its dense/TT algorithms.

## Analytic calibration still required

For the admitted unit-thermal-speed Maxwellian and (k=0.5), the standard linear dielectric equation is

[
\epsilon(k,\omega)=1+\frac{1}{k^2}\left[1+\zeta Z(\zeta)\right]=0,
\qquad \zeta=\frac{\omega}{\sqrt{2}k}.
]

An independent numerical Faddeeva-function solve gives a root near

[
\omega\approx1.41566189-0.15335947i.
]

This corresponds to a field-amplitude damping rate near (-0.15335947). The standard weak-Landau reference is approximately (-0.1533), while the VlasovTT paper reports approximately (-0.151). These values must remain separately identified until normalization and estimation conventions are reconciled. A fit to field energy would have twice the field-amplitude decay exponent in the linear regime.

The reproducible analytic root solver, cross-check and uncertainty decomposition are tracked in [Issue #13119](https://github.com/Luminous-Dynamics/symthaea/issues/13119).

## Next qualification steps

1. Preregister the envelope estimator and fitting window before evaluating holdout trajectories; add sub-sample peak localization.
2. Add independently checked analytic dispersion roots and verify frequency and damping separately.
3. Repeat (t=10) while independently refining (N_x), (N_v), velocity extent and timestep; retain coupled refinements.
4. Reproduce the same initial condition with an independent distribution-space solver and VlasovTT dense reference path.
5. Establish a noise/floor rule for late-time field-mode amplitudes and retain failed or unscorable fits.
6. Only after those gates, compare TT, Fourier–Hermite, adaptive spectral and particle representations at matched fidelity and full cost.

**Claim ceiling:** successful execution of the paired-amplitude experiment and a strong approximate scaling observation. No validated damping-rate reproduction, t=10 convergence, adaptive-grid safety, SOTA superiority or production performance claim.
