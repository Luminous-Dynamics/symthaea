# Amplitude–Phase / Vortex-Core Benchmark

**Contract:** SC-AMPLITUDE-PHASE-0.1  
**Status:** model-input contract and synthetic gauge-invariance controls; no Monte Carlo result is claimed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12793

## Why a separate model is required

The classical XY benchmark (#12755) and quantum rotor benchmark (#12779) deliberately hold the local order-parameter amplitude fixed. They isolate phase fluctuations but cannot represent suppression of the order parameter in vortex cores or the feedback between amplitude fluctuations, vortex-core energy and phase ordering.

Relevant source literature:
- Furutani, Midei, Perali & Salasnich, *Amplitude, phase, and topological fluctuations shaping the complex phase diagram of two-dimensional superconductors*, Phys. Rev. B 110, 134501 (2024), https://doi.org/10.1103/PhysRevB.110.134501.
- Alvarez & Fort, *Phase transitions driven by vortices in two-dimensional superfluids and superconductors: From Kosterlitz-Thouless to first order*, Phys. Rev. B 63, 132504 (2001), https://doi.org/10.1103/PhysRevB.63.132504. Their lattice GL study identifies a regime where strong amplitude fluctuations change a BKT-like transition to first-order behaviour.
- Enss et al., *Interplay of spin waves and vortices in the two-dimensional XY model at small vortex-core energy*, Phys. Rev. B 96, 174505 (2017), https://doi.org/10.1103/PhysRevB.96.174505.
- *Universal scaling behavior of resistivity under two-dimensional superconducting phase fluctuations*, npj Computational Materials (2025), https://www.nature.com/articles/s41535-025-00822-y, a useful fixed-amplitude XY transport comparator.

These papers motivate a model ladder. They do not establish one universal GL model for every unconventional superconductor.

## Starting functional

The first model is a static, classical, gauge-covariant lattice GL functional:

\[
F[\psi,A]=\sum_i\left[a_i(T)|\psi_i|^2+\frac{b_i}{2}|\psi_i|^4\right]
+\sum_{\langle ij\rangle}K_{ij}\left|\psi_i-e^{iA_{ij}}\psi_j\right|^2.
\]

- \(\psi_i=\rho_i e^{i\phi_i}\) is a complex, coarse-grained order parameter. \(\rho_i\) fluctuates and can approach zero in a vortex core.
- \(a_i(T)\) may be negative in the ordered regime; \(b_i>0\) stabilizes the local potential.
- \(K_{ij}>0\) is the gradient / inter-site coupling.
- \(A_{ij}\) is an oriented link gauge phase, dimensionless in the declared convention.
- The initial contract uses a fixed external link field. Self-consistent Maxwell screening is a later model extension and must not be inferred from a fixed-field calculation.

The link convention is \(\psi_i-e^{iA_{ij}}\psi_j\). Under

\[
\psi_i\mapsto e^{i\chi_i}\psi_i,\qquad
A_{ij}\mapsto A_{ij}+\chi_i-\chi_j,
\]

the free energy is invariant. The auditor includes a synthetic numerical invariance test. This is a software/model-contract property, not evidence that a physical material follows the GL functional.

## What this adds over phase-only models

1. Amplitude can be suppressed at vortex centres; vortex-core energy is no longer an arbitrary fixed input.
2. Amplitude healing length, phase stiffness, vortex density and core free energy can respond to the same parameter changes.
3. Disorder can act separately on the local quadratic coefficient, quartic coefficient and bond coupling.
4. Gauge fields and vortex winding can be represented without assigning a phase to a site whose amplitude has vanished.
5. A phase-only XY model can be treated as a controlled large-amplitude-rigidity limit and checked against the full complex field.

## Model ladder

### A. Static, uniform, zero-field GL

Start with 2D square and 3D cubic lattices, uniform coefficients and zero gauge links. Validate the local potential, gradient term, boundary conditions and gauge invariance before introducing disorder.

### B. Amplitude-rigid / XY limit

Increase amplitude rigidity while holding the matched phase coupling and geometry fixed. Demonstrate convergence of stiffness and phase-correlation observables toward the XY baseline. The limit must be established numerically; do not equate a large quartic coefficient by itself with a universal amplitude-free limit without checking the parameter rescaling.

### C. Disorder and interfaces

Allow spatially varying \(a_i\), \(b_i\), \(K_{ij}\), and correlated disorder fields. Preserve the exact realization, seed and spatial covariance. Keep local pair-amplitude inhomogeneity separate from bond-network topology.

### D. Vortex-core energetics

Measure:
- amplitude profile around a constrained vortex;
- phase winding on plaquettes whose corner amplitudes exceed a declared threshold;
- core free energy using constrained-vortex free-energy differences or thermodynamic integration;
- finite-size dependence and core-energy uncertainty.

A plaquette with a near-zero corner amplitude has an ambiguous phase and must not be assigned an ordinary XY winding without an explicit core convention.

### E. Electromagnetic response

The first contract represents a fixed external link field. A future self-consistent gauge-field extension must add a declared Maxwell term, boundary conditions, penetration-depth / screening convention and magnetic energy. Do not silently interpret fixed external flux as self-consistent screening.

### F. Dynamics and quantum amplitude fluctuations

Only after the static model is qualified should the project add time-dependent GL or a quantum complex-field action. Specify the kinetic term, damping, noise, fluctuation–dissipation relation and conservation laws. A static free-energy functional does not determine real-time conductivity.

## Main research hypotheses

- H0: in a declared amplitude-rigid regime, full GL observables converge to matched phase-only XY observables.
- H1: amplitude suppression changes vortex-core free energy and shifts non-universal BKT parameters.
- H2: strong amplitude fluctuations can change transition character in some regions of GL parameter space.
- H3: correlated local amplitude disorder may dominate simple graph-topology descriptors in selected regimes.
- H4: amplitude-aware models improve held-out prediction of thermodynamic and same-specimen measurements over fixed-amplitude baselines.

Null and adverse results are required.

## Observable separation

Do not collapse:
- \(P(|\psi|)\), mean-square amplitude and amplitude correlation length;
- phase-correlation function and coherence length;
- helicity modulus / stiffness from a declared boundary twist;
- gauge-invariant link current;
- vortex density with a stated core/threshold convention;
- vortex-core free energy and its finite-size dependence;
- specific heat / susceptibility;
- transport, which requires a separately validated dynamic model.

\(|\psi|\) is a coarse-grained field and is not automatically a measured spectroscopic gap. A nonzero local amplitude is not bulk superconductivity.

## Benchmark gates

1. Reproduce the qualitative amplitude-fluctuation regime reported by Alvarez–Fort and the Furutani et al. fluctuation corrections using their exact declared assumptions before asserting a new effect.
2. Verify local gauge invariance under arbitrary node-wise \(\chi_i\) transformations.
3. Recover the matched XY limit in a preregistered amplitude-rigid sequence.
4. Measure vortex-core profile and free energy rather than fitting only a BKT curve.
5. Distinguish BKT, crossover and first-order signatures using stiffness, vortex density, energy histograms, Binder cumulants and finite-size scaling.
6. Use at least three spatial sizes and multiple seeds; report equilibration, autocorrelation, uncertainty and finite-size drift.
7. Hold out complete parameter and disorder-generation families.
8. Compare full GL, amplitude-rigid GL and phase-only XY under matched geometry and compute budget.

## Claim ceiling

This is a coarse-grained GL research model. It does not calculate microscopic electronic pairing, prove a cuprate mechanism, predict a material's measured Tc without calibration, establish room-temperature superconductivity or authorize physical synthesis.

## Interfaces

- Same-specimen pairing / coherence evidence: #12677.
- Classical phase-network / XY baseline: #12755.
- Quantum phase-only rotor baseline: #12779.
- Synthesis protocol → realized phase / microstructure: #12671.
- Application-specific translation: #12651.
