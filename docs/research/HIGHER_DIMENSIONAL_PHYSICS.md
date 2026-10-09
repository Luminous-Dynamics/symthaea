# Higher-Dimensional Physics: Spatial-Wave Benchmark

**Status:** research branch; implementation and checks have been added but must be run in CI before being treated as passing evidence.  
**Scope:** a falsifiable software/physics benchmark, not a claim that extra spatial dimensions exist.

## Why this comes next

The physics bridge already has two related but distinct results:

- **Stage A** uses the existing autonomous-invariant search to rediscover global energy for small one-dimensional discretized wave chains.
- **Stage B** checks local continuity/flux relations. Its own current notes document an honest negative for the tested genetic-programming flux-recovery configurations, which is valuable evidence that local-current discovery is a harder problem than discovering a global conserved scalar.

The next useful benchmark should vary the number of **physical spatial axes**, not only the ODE state-vector length. Symtropy already has dimension-generic rigid-body geometry, but that is not a substitute for a field solver. This bridge-level benchmark gives the discovery machinery an independently known field model before investing in continuum PDE or relativistic geometry.

## Model

For each spatial dimension *d = 1..6*, use two free lattice sites along every axis, so the lattice has *N = 2ᵈ* sites. Each site coordinate is encoded by the bits of its index. Boundary values are fixed to zero (Dirichlet conditions), and the normalized wave speed and spacing are *c = h = 1*.

The first-order state is:

- *u[0..N]*: field displacement at the free sites;
- *v[0..N]*: time derivative of the field.

For site *i*, the semi-discrete scalar wave equation is:

    du_i/dt = v_i
    dv_i/dt = Σ_(a=0..d-1) (u_(i xor 2^a) - 2 u_i)

Here, toggling bit *a* selects the adjacent interior site along axis *a*; the other adjacent site on that axis is a fixed-zero boundary. This is a minimal lattice intended to isolate dimensional dependence, not to establish continuum convergence.

The known discrete Hamiltonian is:

    H = 1/2 Σ_i v_i²
        + 1/2 Σ_a [Σ_(i: bit_a(i)=0) (u_i - u_(i xor 2^a))² + Σ_i u_i²]

This expression counts each interior bond once and includes one fixed-zero boundary face per site and axis. The equations imply *dH/dt = 0*. At *d = 1*, the RHS and Hamiltonian reduce to the existing two-free-site Stage A benchmark.

## What is implemented on this branch

- Dimension validation from state length (*2 × 2ᵈ* values), with malformed sizes rejected by the validator.
- Generic RHS, discrete Hamiltonian, and analytic *dH/dt* evaluation.
- Fixed-step RK4 trajectories for conservation-drift checks.
- Tests comparing the one-dimensional RHS and Hamiltonian with Stage A, evaluating analytic and finite-difference *dH/dt* through six spatial dimensions, and measuring RK4 energy drift through six dimensions.
- A deterministic discovery runner that does **not** seed the known Hamiltonian into the search. It uses five paired search seeds (`42, 1337, 2718, 7919, 31415`) for each dimension while holding train and holdout trajectories fixed, and reports candidate counts, screened candidates, `symbolically_proven` metadata, and elapsed time.

## Resolution refinement (separate from dimensionality)

The new pde_grid_refinement module adds a one-dimensional, arbitrary-resolution Dirichlet wave operator. This is deliberately separate from the small \(2^d\)-site dimensional sweep above, so increasing the number of grid points does not get confused with adding physical dimensions.

For \(n\) interior sites on \(0 < x < 1\), the spacing is \(h=1/(n+1)\). The semi-discrete operator uses
\[
\ddot u_i = (u_{i-1}-2u_i+u_{i+1})/h^2,
\]
with fixed-zero boundary values. A matching discrete energy is
\[
H_h = \frac{h}{2}\sum_i v_i^2 + \frac{1}{2h}\sum_{i=0}^{n}(u_{i+1}-u_i)^2,
\]
where \(u_0=u_{n+1}=0\).

The tests check (a) the fundamental sine mode is an eigenvector of the discrete operator, (b) the discrete fundamental frequency converges to the continuum frequency \\(\\pi\\) at approximately second order, (c) RK4 exhibits fourth-order temporal convergence against the exact semi-discrete mode at fixed resolution, and (d) the integrated \\(u(x,t)=\\sin(\\pi x)\\cos(\\pi t)\\) mode converges to the continuum solution with \\(\\Delta t \\leq h^2\\). The space-time test uses a non-special final phase \\(t=3/4\\), reports combined and component-wise discrete spatial \\(L^2\\) errors for both displacement and velocity (the velocity error is normalized by the continuum frequency \\(\\pi\\) before forming the combined norm), and checks second-order behavior on every adjacent pair of the 8/16/32/64-point grids. This makes the continuum comparison less dependent on a single component or a single refinement interval. The time step is chosen small enough that fourth-order temporal error should not dominate second-order spatial error. A second space-time test superposes the first two sine modes with phase offsets and nonzero initial velocity, then checks displacement, velocity, and combined error orders on every adjacent pair of the 12/24/48/96-point grids. These are still smooth manufactured solutions; they do not prove convergence for arbitrary data or higher spatial dimensions.

Research basis: summation-by-parts methods provide a systematic framework for energy estimates in wave-equation discretizations, while convergence analysis shows that stability or energy estimates alone do not guarantee the expected convergence order near boundaries. See [Svärd & Nordström's review](https://doi.org/10.1016/j.jcp.2014.02.031), [Wang & Kreiss on convergence for the wave equation](https://doi.org/10.1007/s10915-016-0297-3), and [Wang, Appelö & Kreiss on energy-based wave discretizations](https://doi.org/10.1007/s10915-022-01829-4). The present central-difference operator is a simpler baseline, not an implementation of their SBP-SAT schemes.

## Run and interpret

    cargo test -p symthaea-physics-bridge pde_hypercubic_wave
    cargo test -p symthaea-physics-bridge pde_grid_refinement
    cargo run -p symthaea-physics-bridge --example higher_dimensional_wave_discovery -- 2

The optional example argument is the maximum spatial dimension (1–4). The runner uses five deterministic paired search seeds and prints elapsed time per seed and dimension. The discovery run is deliberately a *measurement harness*, not a CI assertion that the search must succeed. It may report zero accepted candidates. That is an honest result and should not be converted to a pass by loosening the thresholds after observing results.

The example's screening rule is predeclared:

1. Lie-derivative variance must be finite and below *1e-6* on training and on a differently initialized holdout trajectory.
2. The candidate must pass *is_informatively_conserved* on the training trajectory.
3. At least *50%* of holdout samples must meet the existing relative gradient-informativeness floor (the same non-degeneracy criterion used by *is_informatively_conserved*). This closes an asymmetry where holdout variance could pass despite a mostly near-flat candidate.
4. Absolute Pearson correlation with the hand-derived Hamiltonian must be at least *0.995* on both trajectories.

These are screening gates only. Correlation is not symbolic equivalence; a passing candidate is not a proof of being the Hamiltonian or a novel physical law. Formula identity, invariance across more initial conditions, solver convergence, and independent derivation would be stronger follow-up checks.

## Acceptance and failure handling

The benchmark should keep separate records for:

- **Oracle correctness:** analytic *dH/dt* residual and finite/nonnegative Hamiltonian.
- **Integrator behavior:** maximum relative Hamiltonian drift under the stated RK4 step size.
- **Discovery recall:** number of candidates passing the predeclared screening gates, per spatial dimension and seed, plus the number of seeds with at least one screened candidate.
- **Generalization:** fixed independent holdout initial conditions and, later, larger grid resolutions not used to tune candidate search.
- **Search cost:** elapsed time per seed/dimension, candidate count, seed, configuration, and candidate expressions.

A correct oracle does not imply the search found it. A candidate with low training variance does not imply it is conserved. A conserved quantity does not imply it is the Hamiltonian. A result from this finite-dimensional semidiscrete model does not establish a continuum theorem.

## Next scientific steps

1. Run and retain CI evidence for the new tests and example; do not mark them as passed beforehand.
2. Add a seed-paired dimension sweep with fixed search budgets, then report recall and cost across dimensions rather than presenting one favorable run.
3. Generalize lattice resolution independently of spatial dimension, using more than two free sites per axis. Check convergence as grid spacing decreases and separate discretization error from discovery error.
4. Extend to local energy density/current once the global-invariant baseline is stable. Reuse the negative Stage B flux-recovery findings as constraints on search design; compare structural priors, factorized candidate generation, and direct vector vs. lossy HDC dedup under a predeclared factorial.
5. Only after the flat-space field benchmark is solid, build or integrate a separate metric/curvature/field-equation solver for a known higher-dimensional general-relativity solution. Symtropy's N-dimensional rigid-body collision engine can supply geometric utilities and visualization, but must not be described as solving Einstein's equations.

## Research references

- Makke & Chawla (2024), [Interpretable scientific discovery with symbolic regression: a review](https://link.springer.com/article/10.1007/s10462-023-10622-0).
- Royal Society (2025), [Introduction to the special issue on symbolic regression in the physical sciences](https://doi.org/10.1098/rsta.2024.0600).
- Emparan & Reall (2008), [Black Holes in Higher Dimensions](https://link.springer.com/article/10.12942/lrr-2008-6).

These sources motivate the methodology and the future relativistic target; they are not evidence that this benchmark's implementation is correct.
