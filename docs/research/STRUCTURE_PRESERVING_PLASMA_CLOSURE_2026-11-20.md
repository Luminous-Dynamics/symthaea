# Structure-preserving plasma closure benchmark v0.1

Date: 2026-11-20
Tracking: https://github.com/Luminous-Dynamics/symthaea/issues/12958
Status: mathematical protocol only; no plasma GENERIC-FNO implementation or reproduction.

## Research question

Can a learned plasma closure be parameterised inside the reversible/dissipative geometry of the equations, instead of learning an unconstrained map and hoping soft penalty terms preserve physical structure?

Keep two physical lanes separate.

## Lane A: collisionless Hamiltonian Vlasov-Poisson

The model is Hamiltonian. The correct Poisson bracket, Hamiltonian and Casimirs govern the structure. Report total particle-plus-field energy, mass, momentum, charge/Poisson consistency, applicable Casimirs, phase mixing, damping rate, recurrence, bracket/skew-adjointness residual and long-horizon behaviour.

Do not impose the collisional Landau entropy-production monotonicity test as a universal collisionless pass/fail rule.

## Lane B: collisional Vlasov-Poisson-Landau

The dynamics combine reversible Vlasov-Poisson evolution with dissipative Landau collisions. The benchmark must preserve mass, momentum and total energy while checking entropy monotonicity under the exact discrete convention. Include two-species thermalisation, collision-frequency and time-step sweeps, equilibrium approach, positivity/realizability, and long-horizon stability.

## Source references

### Plasma-native discrete-gradient solver

Finn, Pusztay, Knepley & Adams, "Structure preservation using discrete gradients in the Vlasov-Poisson-Landau system," Journal of Computational Physics 554 (2026), article 114749, DOI 10.1016/j.jcp.2026.114749, published 1 June 2026.

The paper describes a PIC discretisation with discrete-gradient integrators for Hamiltonian Vlasov-Poisson and dissipative Landau operators. It reports mass, momentum and energy conservation with monotonic entropy production in the continuous and discrete systems. PETSc implementation is used; the paper says tools/tests are available in PETSc and specific data are available on request. Exact PETSc revision and run configuration remain to be pinned.

### Transferable function-space GENERIC prior art

Sulskis & Ravi, "GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators," arXiv:2606.08343, submitted 6 June 2026.

For a GENERIC system:

    du/dt = L(u) δE/δu + M(u) δS/δu

with L skew-adjoint and M positive semidefinite, the degeneracy conditions are:

    L δS/δu = 0
    M δE/δu = 0

These express that reversible dynamics do not produce entropy and irreversible dynamics conserve energy. The preprint reports machine-precision degeneracy residuals in continuous time, but explicit time stepping introduces drift. In one 2D 128x128 experiment it reports approximately 4.4x FNO forward latency, 10x training-step latency and 2.3x peak memory. These are general-PDE results, not demonstrated plasma results.

## Candidate methods

1. Plasma-native PETSc discrete-gradient Vlasov-Poisson-Landau reference.
2. Unconstrained FNO or learned closure.
3. Conservation-penalty FNO.
4. Post-step invariant projection.
5. Hamiltonian/Poisson-structure-preserving operator for collisionless Vlasov-Poisson.
6. Metriplectic/GENERIC-parameterised operator for collisional Vlasov-Poisson-Landau.
7. Positive-semidefinite learned dissipative operator.
8. Continuous-time latent closure only after it is mapped to the correct bracket and invariants.

## Required mathematical admission

Before implementation, pin state variables and spaces, energy E, entropy S, Poisson/skew operator L, dissipative operator M, mass/momentum/charge/Casimirs, boundary terms, discrete gradient, quadrature and particle-shape conventions, and the exact entropy definition. Fine-grained kinetic entropy, discrete particle entropy and fluid closure entropy are not interchangeable without derivation.

## Benchmark metrics

Structure: energy, mass and momentum residuals; charge/Poisson/Gauss residual where applicable; ||L δS|| and ||M δE||; minimum eigenvalue/negative spectrum of M; collisional entropy increments and violation count; boundary-flux/bracket residual.

Accuracy: field/distribution error; damping and field-energy histories; phase mixing; two-species relaxation and equilibrium error; pressure/heat-flux operator error where relevant.

Stability/portability: long-horizon drift and failure time; timestep convergence; nonlinear solver iterations; grid/particle-count transfer; OOD selective risk and false-safe rate.

Economics: operator latency; functional-gradient and Jacobian-vector-product cost; implicit iteration count; training/inference memory; complete end-to-end wall time.

## Prospective gates (not results)

1. Reproduce the plasma-native discrete-gradient reference first.
2. Derive the proper structure for each equation lane before implementing a learned operator.
3. Verify continuous-time degeneracy identities before time stepping.
4. Verify discrete rollout invariants under the deployed integrator.
5. Compare unconstrained, penalty, projection and structure-parameterised candidates at matched accuracy and cost.
6. Do not transfer GENERIC-FNO general-PDE claims to plasma without a plasma-specific derivation and reproduction.
7. Treat projection or residual-correction wrappers as separate methods and count their cost.

## Admission ledger

- Vlasov-Poisson-Landau paper / DOI: identified.
- PETSc tools/tests availability: stated by paper.
- Exact PETSc revision and run options: not pinned.
- Vlasov-Poisson-Landau reproduction: not performed.
- GENERIC-FNO paper: identified.
- GENERIC-FNO source code: anonymous repository stated by paper.
- Plasma GENERIC-FNO implementation: not implemented.
- Matched learned-closure tournament: not performed.

## Claim ceiling

This is a mathematical research specification. It does not claim a plasma GENERIC-FNO implementation, universal entropy guarantee, SOTA advantage or production/fusion-device validation.
