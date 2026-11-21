# Vlasov–Poisson reference: phase-resolved modes and independent x/v spectral tails

**Run:** GitHub Actions [44894949853](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44894949853)  
**Source:** `2c2b91ce43890b3d9175b4949b1354fd09fdc2c4`  
**Toolchain:** Rust 1.96.0, x86_64 Linux on `ubuntu-latest`  
**Artifact:** ID `14703438168`, SHA-256 `02a620b3123416a6c245f76ffdf29fd2a9524c414a9a6199c9b12120a4fedc66`

The exact-head workflow passed strict format, **14 unit tests**, strict Clippy, both examples and artifact upload.

## Phase-resolved electric mode

The run uses the same short weak-Landau configuration as the first trace: (N_x=64), (N_v=128), (L=4\pi), (v\in[-6,6]), (alpha=10^{-3}), (k=0.5), thermal speed 1, (Delta t=0.01), and (t_{final}=2).

The complex Fourier coefficient remains almost purely imaginary (real component around (10^{-13}) or smaller), as expected from the even-in-x density perturbation. Its imaginary part is positive through (t=1.4), crosses zero between (t=1.4) and (t=1.5), and is negative by (t=1.5). The phase correspondingly changes from approximately (+pi/2) to (-pi/2).

This resolves the apparent magnitude “rebound” from the prior magnitude-only trace: it is an oscillatory sign reversal of the spatial mode. It does **not** by itself determine the Landau damping rate. A meaningful rate estimate still needs a longer trajectory and a declared phase-/transient-aware fitting procedure.

Mass, momentum, total-energy and Poisson-residual observations remain consistent with the earlier short-run trace. No physical error bound or continuum convergence result follows from this small-grid test.

## Spectral-tail observations

At the fixed 75%-of-Nyquist threshold, the initial and evolved low-mode Landau state has:
- x high-frequency tail fractions around (10^{-30});
- v high-frequency tail fractions around (2.66\times10^{-19});
- Nyquist fractions several orders smaller still.

This is expected for this deliberately smooth, short-horizon case. It establishes that the x and v indicators are measured independently and the monitor reports low occupancy in this particular fixture; it does **not** validate trigger thresholds or prove the absence of unresolved fine structure.

The velocity-domain sweep is a useful counterexample to relying on spectral-tail energy alone. At (v\in[-4,4]), the endpoint occupancy is about (4.33\times10^{-4}), yet the v high-frequency fraction is only about (1.78\times10^{-11}). At (v\in[-6,6]), endpoint occupancy falls to about (2.68\times10^{-8}). Thus velocity-domain adequacy requires its own tail/endpoint and expansion study; a small Fourier tail is not a velocity truncation certificate.

## Refinement results remain consistent

The space and velocity resolution sweeps at (t=1) remain essentially unchanged from the initial run, and timestep differences retain ratios close to four. Those observations apply only to this smooth low-mode case and the fixed numerical operator.

## Next step

Extend the Landau trajectory to a longer horizon, preserve signed mode phase, and define a preregistered damped-mode fit window with transient sensitivity analysis. Separately, build a zero-padding interpolation oracle and transfer receipt before enabling any adaptive grid expansion. These are tracked in [#13106](https://github.com/Luminous-Dynamics/symthaea/issues/13106).

