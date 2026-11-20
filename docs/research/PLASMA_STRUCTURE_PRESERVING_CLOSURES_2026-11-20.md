# Structure-preserving plasma closure benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-STRUCTURE-PRESERVING-CLOSURE-001 #12947](https://github.com/Luminous-Dynamics/symthaea/issues/12947)  
**Companion algebra audit:** Symtropy [#6306](https://github.com/Luminous-Dynamics/symtropy/issues/6306) / draft PR pending  
**Status:** research protocol; no plasma GENERIC-FNO model or Levermore reproduction performed

## 1. Research thesis

The next step beyond a more accurate closure is to make the closure's physical structure part of its parameterization.

Compare two families:
- **realizable distribution parameterizations**: predict a positive velocity distribution or positive sufficient-statistic representation, then derive pressure, heat flux and higher moments from it;
- **structure-preserving field operators**: parameterize reversible and dissipative dynamics with the correct skew/symmetric-positive operators and degeneracy conditions where the governing equations support them.

This could move realizability, energy consistency and entropy behaviour from soft loss penalties into hard construction. It can also reduce model flexibility and add quadrature/solver cost. The benchmark must measure both benefits and costs.

## 2. Primary methodological references

### Levermore maximum-entropy moment hierarchy

C. D. Levermore, “Moment closure hierarchies for kinetic theories,” *Journal of Statistical Physics* 83 (1996), 1021–1065, DOI [10.1007/BF02179552](https://doi.org/10.1007/BF02179552).

The hierarchy derives moment systems from positive exponential-family velocity distributions and is constructed to provide hyperbolicity and a well-defined entropy on the realizable domain. The Maxwellian 5-moment and anisotropic-Gaussian 10-moment closures are mandatory baselines.

**Limitations to test:** realizable-domain boundaries, conditioning, computational cost, tails and the ability to represent strongly non-Maxwellian kinetic structure. Maximum entropy is a baseline, not a universal winner.

### GENERIC-FNO (non-plasma prior art)

J. Sulskis & S. Ravi, “GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators,” arXiv:2606.08343 (6 June 2026).

The method represents dynamics as
[
\partial_t u=L\,\frac{\delta E}{\delta u}
+M\,\frac{\delta S}{\delta u},
]
where (L) is skew-adjoint, (M) is symmetric positive semidefinite, and
[
L\,\frac{\delta S}{\delta u}=0,
\qquad
M\,\frac{\delta E}{\delta u}=0.
]
The degeneracies couple the reversible and dissipative generators so that energy is conserved and entropy is non-decreasing in continuous time.

The preprint reports degeneracy residuals around (10^{-13}), zero-shot transfer over a 4× super-resolution range (64 to 256), and tests on four **non-plasma** PDEs. It also reports finite-step energy drift under explicit time stepping (per-step residual around (10^{-6})) and explains that the ((E,S,L,M)) decomposition has gauge freedom. These are source-reported non-plasma results, not plasma closure performance.

The paper's gauge-invariant dissipation diagnostic is important: do not interpret learned reversible/irreversible channel contributions as unique physical quantities without an independently defined invariant diagnostic.

### Structure-preserving kinetic reference

D. S. Finn, J. V. Pusztay, M. G. Knepley & M. F. Adams, “Structure preservation using discrete gradients in the Vlasov-Poisson-Landau system,” *Journal of Computational Physics* 554 (2026), 114749, DOI [10.1016/j.jcp.2026.114749](https://doi.org/10.1016/j.jcp.2026.114749), published 1 June 2026.

The paper combines PIC discretization and discrete-gradient time integration for the Vlasov-Poisson-Landau system, reporting conservation of mass, momentum and energy and monotonic entropy production in both time-continuous and discrete systems. The article says its tools/tests are available in open-source PETSc; pin the exact PETSc version/build, solver inputs and commit before adopting it as a reproduction baseline.

This is a **collisional** kinetic reference. It is not interchangeable with the collisionless Landau-damping data, magnetosheath PIC reconnection data, or ICF heat-transport data.

## 3. Non-negotiable physical distinction

Do not impose one entropy rule on every plasma task.

| Task | Correct structural emphasis | What not to assume |
|---|---|---|
| Collisionless Vlasov-Poisson / Landau damping | Reversible Hamiltonian structure, charge/field/particle energy invariants, phase-mixing and electric-field energy decay | Fine-grained entropy must monotonically increase. Landau damping is phase mixing, not collisional entropy production. |
| Collisional Vlasov-Poisson-Landau / VFP | Collision invariants, total energy conservation, collision-operator entropy production and its discrete-time counterpart | A generic learned entropy functional is automatically the physical kinetic entropy. |
| Collisionless magnetosheath / reconnection | Frame/symmetry assumptions, pressure tensor realizability, pressure-force and particle/field energy exchange | A generic pointwise positive local entropy-production density is valid for collisionless PIC. |
| Collisional ICF nonlocal heat transport | Total electron-energy balance and the correctly derived nonlocal transport dissipation/entropy functional | Pointwise entropy production must be nonnegative for every nonlocal flux closure. |

Every structure gate must cite its governing equation, physical regime, boundary conditions, exact discrete operator and time integrator.

## 4. Candidate architectures

### A. Positive distribution and realizability

1. Unconstrained six-component pressure/heat-flux prediction.
2. Symmetry-tied pressure head with PSD factorization.
3. Levermore 5-moment Maxwellian and 10-moment anisotropic-Gaussian closures.
4. Positive mixture-of-Gaussians velocity distribution, with positive weights and positive-definite component covariances; derive moments analytically.
5. Positive exponential-family distribution with learned sufficient-statistic potentials; derive moments by documented quadrature.
6. Multiplicative positive residual around a Maxwellian / anisotropic-Gaussian base, with explicit normalization, mean and energy constraints.

The positive-mixture and exponential-family candidates are hypotheses. Positivity of (f(v)) only at sampled quadrature points is not a global proof; test tail behaviour, quadrature convergence and moment reconstruction residual.

### B. Structure-preserving field and closure operators

1. Unconstrained FNO / FCNN.
2. Hamiltonian/skew-adjoint neural operator for reversible tasks.
3. GENERIC/metriplectic operator with (L) skew, (M\succeq0), (L\delta S=0), (M\delta E=0).
4. Onsager / positive-semidefinite dissipative operator where a collisional constitutive law is derived.
5. Learned constitutive closure combined with a trusted discrete-gradient / invariant-preserving solver.

For GENERIC, report both algebraic degeneracy residuals and the independent gauge-invariant dissipation diagnostic. A finite-dimensional skew/PSD/degeneracy test is not a proof of a Poisson bracket: the Jacobi identity, differential-operator boundary terms and function-space consistency still need qualification.

## 5. Benchmark lanes

### Lane 1 — collisionless electrostatic Vlasov-Poisson / Landau damping

Compare analytic Hammett–Perkins, Markovian and source online-FNO closures, Levermore 5-/10-moment baselines, positive-distribution candidates and Hamiltonian/memory models.

Measure field/particle energy transfer, charge consistency, phase mixing, electric-field energy and full a-posteriori trajectories. Do not use a blanket monotone entropy gate. Reuse #12907 and #12914.

### Lane 2 — collisional Vlasov-Poisson-Landau

First pin and reproduce the Finn et al. discrete-gradient PIC solver / PETSc implementation, or a separately verified equivalent. Compare:
- exact kinetic reference;
- Levermore maximum-entropy closures;
- unconstrained neural closure;
- distribution-parameterized closure;
- GENERIC-style learned operator;
- hybrid model with exact discrete-gradient correction.

Measure mass, momentum, energy and discrete entropy production using the same collision operator, time integrator and boundary convention. Vary collision rate, perturbation amplitude, phase-space/grid resolution and timestep. Collisionless Landau-damping data do not substitute for this lane.

### Lane 3 — magnetosheath pressure tensor

Compare source FCNN/MLP/CGL/symbolic baselines against PSD distribution-derived, symmetry-tied and equivariant outputs. Preserve componentwise pressure scores, off-diagonal channels, exact solver-matched div(P), pressure-strain and energy accounting. No generic entropy monotonicity rule is activated for collisionless PIC.

### Lane 4 — ICF nonlocal heat transport

Compare the source FNO, LINN, TCNN, SNB and appropriate local baselines. Distribution-based closure or GENERIC-style transport is eligible only after the constitutive entropy/energy functional is derived for the same collisional regime and nonlocal operator. Compare in the same implicit electron-energy solver and on matched PIC data/resolutions.

## 6. Metrics and hard gates

**Distribution:** positivity on a converged velocity quadrature, normalization residual, moment reconstruction error, exact mean/pressure/energy consistency, tail error, PSD pressure and realizable-domain coverage.

**GENERIC algebra:** (\|L+L^T\|), (\|M-M^T\|), minimum eigenvalue of (M), (\|L\nabla S\|), (\|M\nabla E\|), energy rate and entropy rate.

**Discrete solver:** drift in mass/momentum/energy/charge invariants, discrete entropy production where applicable, nonlinear residual history, timestep sensitivity and long-horizon stability.

**Closure quality:** componentwise pressure/heat-flux error, a-priori operator error, a-posteriori trajectory error, OOD selective risk and false-safe rate.

**Economics:** velocity-space quadrature, model inference, training, data-generation and solver costs; cost per stable qualified plasma time.

Hard fail if the distribution is materially negative, required pressure realizability fails, the exact task-specific conservation law exceeds its registered tolerance, a collisional entropy law is violated, or the coupled solver becomes unstable. Do not hide hard failures inside an aggregate accuracy score.

## 7. Prospective experiments / target gates (not results)

1. Reproduce the classical Levermore baselines before claiming a better learned realizable closure.
2. For GENERIC candidates, verify both degeneracies, skewness and PSD independently; report time-discretization drift separately from continuous-time algebra.
3. For distribution-derived closures, demonstrate quadrature convergence and exact moment reconstruction under independent velocity-grid refinement.
4. Compare physical accuracy at matched parameter count and matched compute; report the extra velocity-space integration cost.
5. Demonstrate zero-shot spatial resolution transfer with both structural identities and a-posteriori solver stability.
6. The collisionless lane must pass reversible/invariant tests without an unjustified entropy-production condition; the collisional lane must pass its pinned discrete entropy law.
7. New methods must improve the solver-in-the-loop Pareto frontier, not merely the a-priori closure loss.

## 8. Source / reproduction state

| Source | Status |
|---|---|
| Levermore 1996 hierarchy | Mathematical baseline identified; implementation/reproduction pending |
| GENERIC-FNO arXiv:2606.08343 | Non-plasma preprint identified; not a plasma result |
| Finn et al. 2026 VPL discrete gradients | Kinetic paper identified; exact PETSc revision/build/input admission pending |
| Plasma GENERIC-FNO implementation | Not implemented |
| Positive mixture / exponential-family closure | Not implemented |
| Plasma structure-preserving tournament | Not run |

## 9. Companion Symtropy algebra audit

Symtropy [#6306](https://github.com/Luminous-Dynamics/symtropy/issues/6306) and draft [PR pending] provide a finite-dimensional local audit of skewness, PSD, degeneracy, energy rate and entropy rate. It deliberately does **not** check the Poisson Jacobi identity, spatial adjointness/boundary terms, the physical validity of E/S, or discrete-time preservation. It is a test harness, not a plasma solver or proof of function-space GENERIC.

## 10. Claim ceiling

This protocol does not claim a plasma GENERIC-FNO, positive distribution closure, Levermore reproduction, Vlasov-Poisson-Landau solver reproduction or SOTA superiority. All source-reported results remain distinct from our own measurements.
