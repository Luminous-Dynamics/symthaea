# ADEPT mixed Hermite–Legendre Vlasov–Poisson baseline admission

**Admission date:** 2026-11-21  
**Upstream repository:** https://github.com/ergodicio/adept  
**Pinned source:** 0c85ce0d3d66abab51212191c2644f58a31a49cd  
**License:** MIT  
**Upstream test run:** [41496559182](https://github.com/ergodicio/adept/actions/runs/41496559182)  
**Dedicated test job:** [125452928400](https://github.com/ergodicio/adept/actions/runs/41496559182/job/125452928400)  
**Machine-readable admission:** [adept-mixed-hermite-legendre-v0.1.json](../../benchmarks/plasma/references/adept-mixed-hermite-legendre-v0.1.json)

## Why this comparator matters

The plasma programme now has a much stronger comparator than a generic “Hermite versus Fourier” proposal.

ADEPT already implements a mixed Hermite–Legendre 1D1V Vlasov–Poisson solver:
\[
f(x,v,t)=f_0(x,v,t)+\delta f(x,v,t).
\]

The near-Maxwellian bulk \(f_0\) is carried by an asymmetrically weighted Hermite basis on the unbounded velocity line. The non-Maxwellian residual is carried by Legendre modes on a finite velocity window \([v_a,v_b]\). A one-way coupling sends the highest Hermite coefficient into the Legendre correction, while both representations contribute to the self-consistent Poisson solve.

The implementation includes:
- Fourier spectral treatment in periodic space;
- prediagonalized exact streaming exponentials;
- structured local Cayley updates for the force terms;
- Lawson-RK4 and IMEX alternatives;
- optional low-mode conservation correction;
- explicit Legendre-window boundary diagnostics;
- separate artificial Hermite/Legendre high-mode damping;
- optional BGK and Dougherty collision models.

This design attacks a real weakness of a single global velocity basis: near-Maxwellian bulk and localized beams, plateaus, or trapped structures need not be represented by the same basis family.

## Exact upstream admission

The pinned public repository is MIT-licensed. Its source, lockfile, test files, configs and immutable Git blob hashes are in the JSON manifest. The main commit has a GitHub-verified signature.

The upstream CPU workflow completed successfully on the pinned commit. The dedicated **test-hermite-legendre-1d** job ran:

\[
31\ \text{tests passed}, \quad 0\ \text{failed}
\]

in 205.57 seconds. This is evidence that the upstream implementation passes its own test suite. It is not a Symthaea reproduction, a cross-method comparison, or independent confirmation of the authors' larger claims.

## Important distinction: driven Landau is not our current initial-value Landau case

ADEPT's dedicated Landau test is a driven-resonance experiment. It starts from a uniform Maxwellian, applies an external longitudinal field tuned to the kinetic dispersion root, ramps the driver off, and measures the free ring-down frequency and damping.

Its tested cases use \(k\lambda_D=0.30\) and \(0.35\), \(N_x=64\), \(N_h=256\), \(N_l=16\), \(dt=0.05\), \(t_{max}=80\), and a fit interval from 45 to 70. Its test tolerances are 2% for frequency and 15% for damping.

That is a useful and mature resonance-validation lane, but it must not be conflated with Symthaea's initial-value, cosine-density perturbation at \(k=0.5\) and \(\alpha=10^{-3}\) or \(10^{-2}\). Both should be retained as distinct benchmark profiles.

## Reproduction cases

### 1. Driven Landau resonance

Reproduce the exact source test before modification, including the driver envelope, dispersion root, fit interval and mode extraction. Then add a separate initial-value case matched to Symthaea's density-perturbed Maxwellian.

### 2. Two-stream instability

The source **two-stream.yaml** profile uses \(L_x=4\pi\), \(N_x=64\), \(N_h=85\), \(N_l=171\), \(v\in[-2.5,2.5]\), \(dt=0.002\), and \(t_{max}=35\), with initial amplitude 0.01 and Lawson integration.

The companion **two-stream-imex.yaml** uses the same physical case but \(dt=0.02\) with IMEX. The upstream code includes a negative control where explicit Lawson becomes non-finite at \(dt=0.01\), while IMEX remains finite and mass-conserving at that step. This is a stability/fidelity comparison, not an automatic tenfold speedup: implicit dissipation, phase error and instability growth must be measured.

### 3. Bump-on-tail

The configured source profile uses \(L_x=20\pi\), \(N_x=128\), \(N_h=32\), \(N_l=128\), \(v\in[2,18]\), \(dt=0.1\), \(t_{max}=120\), a beam fraction of 0.01, drift 10 and thermal speed 1.

The YAML comments also describe a separate \(t=700\) experiment, a shorter-window failure, and a comparison of \(N_h=32\) versus \(128\). These are **source-reported observations**, not the configured \(t=120\) profile and not yet independently reproduced by Symthaea. They are useful falsification targets, not accepted benchmark results.

### 4. Manufactured free streaming

ADEPT includes an analytic characteristic-solution test:
\[
f(x,v,t)=f(x-vt,v,0).
\]
It reconstructs the total distribution from both coefficient families and compares relative \(L^2\) error against the known solution. This is especially useful for testing transfer maps without relying on a self-consistent nonlinear field.

### 5. Collision and conservation behavior

The upstream test suite separately checks local moments through BGK and Dougherty collision paths. These model collisions are distinct from artificial high-mode damping and from the collisionless Vlasov–Poisson cohort. Comparisons must keep these equation/physics lanes separate.

## What this changes in our research strategy

The new representation tournament should now distinguish at least these families:

1. Fixed dense Fourier Vlasov (Symthaea reference).
2. Adaptive Fourier zero-padding.
3. Fixed Hermite or Hermite–Fourier.
4. Mixed asymmetrically weighted Hermite bulk + bounded Legendre residual (ADEPT).
5. Adaptive shifted/scaled Hermite.
6. Moment-constrained low-rank / LoMaC.
7. TT/MPS and quantics compression.
8. Particle-measure and PIC methods on matched particle cohorts.

The mixed Hermite–Legendre solver also motivates a deeper extension: preserve the macroscopic moment backbone, select a bulk coordinate frame, and allocate a residual basis according to the geometry of the non-Maxwellian structures. But the basis controller must be allowed to choose **multiple residual components** or abstain when one bounded Legendre window is inadequate for several separated beams or long tails.

Do not simply combine all named methods into a larger architecture. The hybrid only counts as an improvement if the complete physical trajectory is more accurate or more economical at matched fidelity.

## Matched comparison protocol

All comparable methods must use the same:
- normalized equation and charge/background convention;
- initial distribution, physical domain and final time;
- mode/field observable definitions;
- common diagnostic grid and velocity quadrature;
- precision, hardware and resource budget for paired comparisons.

Measure:
- complex fundamental field mode, phase, frequency and damping/growth;
- field energy and spectrum;
- reconstructed total distribution discrepancy (weighted L1/L2 and a geometry-aware metric);
- mass, momentum, kinetic/field/total energy;
- Hermite coefficient spectrum and Legendre spectrum;
- Legendre endpoint occupancy and high-mode energy;
- total-\(f\) minimum/negative mass when physically nonnegative;
- correction norms, split/IMEX residuals and collision/damping settings;
- full wall time, peak memory, allocation, bytes moved and energy-to-solution;
- failure, instability, fallback and unqualified-run rates.

For the first tournament, run paired CPU comparisons on the same machine and precision. Then separately compare each method's best documented hardware path. Do not claim an algorithmic speedup from comparing JAX/GPU against Rust/CPU.

## New paradigm hypothesis: moment backbone + adaptive multi-component residual atlas

The current methods suggest a stronger architecture to investigate, not assume:

- carry density, momentum and energy as an explicit invariant backbone;
- track a shifted/scaled bulk frame using physical moments;
- represent non-Maxwellian structure as one or more residual components selected for shape (beam, plateau, trapped vortex, tail);
- compress each residual with the best qualified local representation;
- hand off only through the fail-closed transition evaluator in [PR #13120](https://github.com/Luminous-Dynamics/symthaea/pull/13120).

The residual can be signed. Positivity belongs to the reconstructed physical \(f\), only for cohorts whose \(f\) is truly nonnegative. It must not be forced onto signed delta-f variables.

The first cheap step is exactly the affine velocity-moment oracle in [PR #13130](https://github.com/Luminous-Dynamics/symthaea/pull/13130). It verifies the raw-moment coordinate identities and round trips, but it does not yet remap distribution samples or adapt Hermite coefficients.

## Immediate execution sequence

1. Execute the upstream source-faithful ADEPT test/config lanes under a pinned environment.
2. Add an adapter that emits the Symthaea canonical trajectory and diagnostics schema.
3. Match the driven Landau resonance profile separately from the initial-value density perturbation profile.
4. Reproduce two-stream and bump-on-tail, including Legendre boundary and Hermite/Legendre spectrum diagnostics.
5. Compare split, Lawson and IMEX at matched physical fidelity and count all correction/solve cost.
6. Compare dense Fourier versus mixed Hermite–Legendre on free streaming, weak Landau, two-stream and bump-on-tail.
7. Only then investigate automatic frame updates, residual splitting and compression.

## Claim ceiling

ADEPT's exact source commit and upstream 31-test CI result are admitted as an external baseline. Symthaea has not independently reproduced its numerical results. No head-to-head win, speedup, stronger invariant guarantee, positivity guarantee or production superiority is claimed.
