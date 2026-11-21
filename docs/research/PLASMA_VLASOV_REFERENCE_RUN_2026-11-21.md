# Vlasov–Poisson reference prototype: first exact-head trajectory run

**Run:** GitHub Actions run [44894225928](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44894225928)  
**Source:** `98215ac7b1931416b900feaa437a282b8c7ded23`  
**Runner/toolchain:** `ubuntu-latest`, `x86_64-unknown-linux-gnu`, Rust 1.96.0  
**Artifact:** ID `14702877947`, SHA-256 `794602594082b5f13e32919fe3c456efdbdb1692aeee4479de79f84ea7f8bf86`

This is the first exact-head execution of the prototype's focused workflow. It passed strict formatting, all 11 unit tests, strict Clippy, both executable examples and CSV artifact upload. The raw CSVs are committed beside this report.

## What was executed

### Landau diagnostic trace

Configuration: (N_x=64), (N_v=128), (L=4\pi), (v\in[-6,6]), density perturbation (alpha=10^{-3}), spatial mode 1 ((k=0.5)), thermal speed 1, (Delta t=0.01), final time (t=2). The CSV samples every 10 steps.

Observed:
- Mass stayed at approximately (12.56637061436).
- Momentum changed from (-4.2947885\times10^{-8}) to (-4.2953911\times10^{-8}), an absolute drift of about (6.03\times10^{-12}).
- Total energy changed from (6.283197404413) to (6.283197404612), an absolute drift of about (1.99\times10^{-10}).
- Maximum observed spectral Poisson residual over the sampled trace was approximately (2.58\times10^{-16}).
- The velocity-endpoint occupancy ratio remained about (2.68\times10^{-8}).

The electric-mode *magnitude* is not monotone: it falls to (7.31\times10^{-6}) at (t=1.5) then rises to (3.71\times10^{-4}) at (t=2). This is a warning against fitting a damping rate to a straight line in (log|E_k|) using this magnitude-only trace. The next diagnostic increment records the complex Fourier coefficient and phase, enabling a phase-aware mode analysis. **No Landau damping rate is claimed from this run.**

### Resolution and timestep probe at (t=1)

| Sweep | Result |
|---|---|
| Spatial grid (N_x=32,64,128), (N_v=128) | Electric-mode magnitude agrees to about (2\times10^{-16}) across this smooth, low-mode test |
| Velocity resolution (N_v=64,128,256), (N_x=64) | Electric-mode magnitude changes by about (3.85\times10^{-12}) from coarsest to finest |
| Velocity interval ([-4,4]) | Endpoint ratio (4.33\times10^{-4}), minimum (f\approx1.34\times10^{-4}), total energy (6.27644195); insufficient tail margin relative to wider intervals |
| Velocity interval ([-6,6]) | Endpoint ratio (2.68\times10^{-8}), minimum (f\approx6.07\times10^{-9}), total energy (6.28319740) |
| Velocity interval ([-8,8]) | Endpoint ratio (2.37\times10^{-14}), total energy (6.28319787), minimum (f\approx-9.46\times10^{-15}), consistent with roundoff-scale negativity in this spectral prototype |
| Timestep (Delta t=0.04,0.02,0.01,0.005) | Successive amplitude-difference ratios (4.00085) and (4.00021), consistent with the expected second-order temporal trend for this test |

The ±4 velocity interval has visibly inadequate tail margin. The ±6 and ±8 runs are much closer for this initial condition and final time, but this is not a rigorous velocity truncation error bound. The tiny negative minimum at ±8 also confirms that this spectral method does not enforce positivity by construction.

## What this establishes—and what it does not

Established on the recorded exact head:
- the small reference solver compiles and runs under Rust 1.96;
- 11 focused unit tests, strict formatting and strict Clippy pass;
- a manufactured free-streaming case, single-mode Poisson answer and non-neutral negative control pass;
- a coupled small-amplitude trajectory and initial space/velocity/time refinement probes execute and emit reproducible CSV artifacts.

Not established:
- continuum convergence or a qualified L3 reference;
- a fitted Landau damping rate or agreement with the theoretical (-0.151\,\omega_{pe}) target;
- positivity preservation;
- long-horizon stability or the absence of recurrence;
- external solver reproduction or matched-budget SOTA comparison;
- runtime, memory or energy-to-solution advantage.

The next code increment is phase-resolved Fourier diagnostics and a careful fit window / transient analysis. The next physics increment is adaptive spectral-tail monitoring and wavenumber-domain expansion, motivated by the 2026 adaptive Fourier spectral work; see [issue #13106](https://github.com/Luminous-Dynamics/symthaea/issues/13106).

