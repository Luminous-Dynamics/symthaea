# Vlasov–Poisson reference: first t=10 phase-resolved Landau trajectory

**GitHub Actions run:** [44895652946](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44895652946)  
**Source commit:** `7ba2dd5124f8ba36ea5c4fa9a7096a4cef90071b`  
**Exact-head workflow:** strict format, 19 tests, strict Clippy, both examples and artifact upload all passed.  
**Artifact:** `14703563532`, SHA-256 `edf9a35c5a9e50b5d0151c0c45d810563ba9fa98a770b019e512401315cfdfeb`, 9,961 bytes.

The artifact contains the full 101-row phase-resolved Landau trace and the independent refinement CSV. Its retained GitHub Actions artifact expires 21 December 2026; the run manifest is committed beside this report.

## Configuration

- Periodic spatial length (L=4\pi), (N_x=64).
- (N_v=128), numerical velocity interval ([-6,6]).
- Normalized electron charge-to-mass ratio -1, permittivity 1, background density 1.
- Maxwellian thermal speed 1, spatial mode 1, initial density perturbation (\alpha=10^{-3}).
- (Delta t=0.01), 1,000 steps, (t_{final}=10).
- Samples every 0.1 time units.
- Spectral-tail occupancy threshold: 75% of Nyquist, separately for x and v.

**Comparability caveat:** the VlasovTT paper reports a Landau perturbation amplitude (A=0.01); this Symthaea run uses (0.001), ten times smaller. The spatial domain, velocity interval, thermal speed and fundamental mode match the paper's stated physical setup, but this is not an exact paper initial-condition reproduction or a reproduction of the paper's numerical configuration.

## Exploratory Landau envelope observation

The positive spatial Fourier coefficient is almost purely imaginary, consistent with the initial cosine-density / sine-field parity. The imaginary coefficient oscillates through zero and alternates sign; the magnitude alone cannot be interpreted as a monotone envelope.

The sampled absolute-mode trace has these post-initial extrema:

| Time | Signed imaginary coefficient | Type |
|---:|---:|---|
| 2.5 | -5.056582033394e-4 | Negative trough |
| 4.7 | +3.532724335582e-4 | Positive peak |
| 7.0 | -2.510287827752e-4 | Negative trough |
| 9.2 | +1.788269546570e-4 | Positive peak |

A descriptive least-squares regression of (ln(|E_k|_{peak})) against these four post-initial extremum times gives:

[
\gamma_{fit}=-0.154446\quad (R^2=0.99967).
]

Dropping the first post-initial trough and fitting the last three extrema gives (-0.151273) with (R^2=0.999886). Including the initial state at (t=0), however, changes the fitted slope to (-0.181471), demonstrating sensitivity to transient/window choice.

The VlasovTT paper reports a linear Landau amplitude damping rate near (-0.151\,\omega_{pe}). The exploratory post-transient slopes are numerically close to that value. This is a promising consistency check, **not a validated reproduction**: the fit window was selected after observing the trajectory, only four post-initial extrema are available, amplitude differs by 10x, and no independent reference, grid/time refinement at t=10, or independent-seed study has been performed. The conventional regression standard error is not a valid uncertainty interval for a deterministic, correlated numerical trajectory.

## Invariant and numerical observations

- Total mass: (12.56637061436) at both start and end to the printed precision.
- Momentum: from (-4.2947885\times10^{-8}) to (-4.2963027\times10^{-8}).
- Total energy: from (6.283197404413) to (6.283197404582).
- Maximum observed collocation spectral Poisson residual: (4.73\times10^{-16}).
- Velocity endpoint ratio remains approximately (2.66\times10^{-8}) to (2.68\times10^{-8}).
- Maximum x high-frequency tail fraction is approximately (4.32\times10^{-29}); v high-frequency tail fraction stays around (2.66\times10^{-19}).

These figures apply to this small-grid, short-to-intermediate horizon prototype. They do not certify continuum fidelity, positivity or long-horizon stability.

## What is now executable

The same PR also adds `zero_pad_periodic_line`, a standalone transfer oracle that:
- interpolates a real periodic line to a finer power-of-two grid by Fourier zero-padding;
- splits the coarse-grid Nyquist coefficient evenly between positive and negative target harmonics;
- preserves the trigonometric interpolant at old nodes and tests the refined analytic signal;
- rejects invalid grids and non-finite samples.

Its 19-test suite includes band-limited reconstruction, old-node preservation, Nyquist splitting, constant and identity cases, malformed input, and a counterexample where low v spectral-tail occupancy does not reveal a too-narrow physical velocity interval.

This function does not change the simulation's live grid. No adaptive resolution controller or physical transition certificate is claimed.

## Next experiment

The next meaningful validation is a matched amplitude sweep at (\alpha=10^{-3}) and (10^{-2}), followed by independent (N_x/N_v) and timestep refinements at (t=10), using a preregistered peak-envelope extraction rule. Then compare against an independent 1D1V distribution-space solver and the VlasovTT dense reference path at genuinely matched initial conditions.
