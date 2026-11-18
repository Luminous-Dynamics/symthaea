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


## Executable input auditor and static free-energy evaluator

The initial Rust binary is `sc_amplitude_phase_audit` under `tools/superconductivity-amplitude-phase-harness`. It is deliberately an evaluator for a supplied synthetic complex field, not a thermal sampler or minimizer.

For the declared link convention, it evaluates:

\[
F_{\mathrm{local}}=\sum_i\left[a_i|\psi_i|^2+\frac{b_i}{2}|\psi_i|^4\right],
\qquad
F_{\mathrm{gradient}}=\sum_{\langle ij\rangle}K_{ij}|\psi_i-e^{iA_{ij}}\psi_j|^2.
\]

The receipt includes both terms, their total, the mean amplitude, graph/boundary metadata, and SHA-256 hashes of the exact schema and fixture. The parser rejects non-finite fields, non-positive quartic or gradient coefficients, duplicate IDs/bonds, unresolved endpoints, inconsistent periodic-image geometry, and hidden nonzero gauge links under a zero-field declaration.

### Gauge invariance

The implementation tests the transformation

\[
\psi_i' = e^{i\chi_i}\psi_i,
\qquad
A_{ij}'=A_{ij}+\chi_i-\chi_j.
\]

It requires the free-energy residual to be below \(10^{-9}\) dimensionless lattice-energy units. The unit tests exercise both the uniform fixture and nonuniform complex fields with nonzero background links. A wrong-sign transformation is a negative control and must change the energy.

### Amplitude-aware plaquette winding

For complete 2D rectangular integer-coordinate grids, it evaluates a gauge-invariant integer plaquette winding. With a stored oriented link \(A_{ij}\), the oriented covariant phase difference is

\[
\delta_{ij}^{\mathrm{cov}}=
\operatorname{wrap}(\arg\psi_j-\arg\psi_i+A_{ij}).
\]

The integer winding is recovered by subtracting the oriented plaquette link flux:

\[
q_p=\operatorname{round}\left[
\frac{\sum_{(ij)\in\partial p}\delta_{ij}^{\mathrm{cov}}
-\sum_{(ij)\in\partial p}A_{ij}}{2\pi}
\right].
\]

This convention is consistent with the link term \(|\psi_i-e^{iA_{ij}}\psi_j|^2\) and the gauge transformation above. It is not a field-theoretic continuum magnetic flux density. Plaquettes touching a site with \(|\psi_i|\le\max(\epsilon_{\mathrm{core}},\epsilon_{\mathrm{machine}})\) are marked skipped: the phase at a zero-amplitude core is undefined. The receipt reports positive and negative integer charge, absolute charge sum, valid plaquettes and skipped low-amplitude plaquettes separately.

The synthetic 4×4 periodic phase-winding fixture control generates compensating periodic images: two positive and two negative unit plaquette charges. The same winding counts must survive a local gauge transformation. Suppressing one site to zero amplitude must skip the four adjacent plaquettes rather than invent a phase.

### Exact synthetic checks

For the committed 4×4 uniform fixture, \(a=-1\), \(b=1\), \(|\psi|=1\), and \(K=1\) with zero link field:
- local potential energy is \(-8\);
- gradient energy is \(0\);
- total static free energy is \(-8\).

If one site is set to \(\psi=0\), its local contribution changes by \(+0.5\), while its four incident unit-coupling links contribute gradient energy \(+4\), giving local \(-7.5\), gradient \(+4\), total \(-3.5\).

These are algebraic/software controls, not measured material energies.

### Current limitations

Not implemented or qualified:
- thermal Metropolis / hybrid Monte Carlo sampling of the full complex field;
- equilibration, autocorrelation, independent-chain error estimates and finite-size scaling;
- thermodynamic integration of constrained vortex free energy;
- BKT versus first-order phase-transition classification;
- self-consistent Maxwell screening or a magnetic-field minimizer;
- dynamics, noise, fluctuation-dissipation and conductivity.

The current vortex output is a phase-winding count with an explicit amplitude threshold. It does not yet estimate vortex-core free energy, distinguish bound from unbound pairs, or classify the transition.


## Complex-field Metropolis pilot

The same executable now has an optional `--monte-carlo` path. It uses a single-site symmetric proposal in the complex plane:

\[
\psi_i' = \psi_i + \delta_x + i\delta_y,
\qquad
\delta_x,\delta_y\sim\mathrm{Uniform}[-s,s],
\]

and accepts with \(\min(1,e^{-\Delta F/T})\), where \(F\) is the exact lattice functional above and the declared dimensionless temperature is positive.

The update computes \(\Delta F\) from the changed site's local quartic potential and incident gauge-covariant links. A unit test compares this local delta to a full-lattice recomputation at every site for nonuniform complex fields and nonzero link phases. The full energy is recomputed at each measurement point to bound accumulated floating-point drift.

Sampling protocol is part of the fixture: RNG seed, warmup sweeps, measurement sweeps, thinning, proposal scale and independent-chain count. CLI overrides are included in the receipt. Each chain reports:
- initial/final energy and proposal acceptance / invalid counts;
- per-node total, local and gradient energy;
- mean \(|\psi|\) and \(|\psi|^2\);
- amplitude-aware vortex charge density and skipped core plaquettes when the graph supports the 2D grid diagnostic;
- sample standard deviations, initial-positive-sequence integrated autocorrelation estimates, effective sample sizes and autocorrelation-corrected standard errors when the series is long enough.

The multi-chain receipt preserves every chain/seed and reports between-chain dispersion. This does not replace within-chain autocorrelation checks or prove equilibration. The current CI pilot uses two short chains only to test determinism, update plumbing and receipt structure. It is not a thermodynamic measurement.

The sampler uses a fixed proposal scale and is intentionally simple. It is a baseline, not a claim of state-of-the-art mixing. The next algorithmic comparison should include adaptive-but-frozen-after-warmup Metropolis, overrelaxation or hybrid Monte Carlo where mathematically appropriate, and replica exchange near suspected first-order regimes. Compare effective independent samples per CPU-second at matched acceptance, observable error and system size. Do not judge samplers by acceptance rate alone.


### Gauge-invariant twist response and helicity modulus

The GL model now differentiates its link functional with respect to a global boundary twist. For each oriented bond, define

\[
A_{ij}(\boldsymbol\theta)=A_{ij}
+\sum_\mu \theta_\mu\frac{\Delta x_{ij,\mu}}{L_\mu}.
\]

The configuration-level derivatives are recorded as

\[
J_\mu=\frac{\partial F}{\partial\theta_\mu}
=2\sum_{\langle ij\rangle}K_{ij}\,
\operatorname{Im}(\psi_i^*e^{iA_{ij}}\psi_j)
\frac{\Delta x_{ij,\mu}}{L_\mu},
\]

\[
D_{\mu\nu}=\frac{\partial^2 F}{\partial\theta_\mu\partial\theta_\nu}
=2\sum_{\langle ij\rangle}K_{ij}\,
\operatorname{Re}(\psi_i^*e^{iA_{ij}}\psi_j)
\frac{\Delta x_{ij,\mu}\Delta x_{ij,\nu}}{L_\mu L_\nu}.
\]

For a thermal ensemble, the twist free-energy Hessian is estimated as

\[
\Upsilon_{\mu\nu}
=\langle D_{\mu\nu}\rangle
-\frac{\langle J_\mu J_\nu\rangle-
\langle J_\mu\rangle\langle J_\nu\rangle}{T}.
\]

The code preserves the raw twist-response tensor in the per-chain receipt and aggregates it across independent seeds. Its normalization is the second derivative with respect to the declared total twist vector; it is not silently converted into a three-dimensional superfluid density or material penetration depth. For the uniform 4×4, zero-field, \(|\psi|=1,K=1\) fixture, the bare diamagnetic twist curvature is \(2\) along each axis and zero off-diagonal. A gauge-transformation test verifies that both current and diamagnetic tensors are invariant.

The current Metropolis pilot reports a finite-chain helicity-modulus estimate from sample means/covariances. It does not yet have blocked jackknife uncertainties or finite-size scaling. For BKT classification, compare the size-dependent helicity-modulus crossing/jump and vortex behaviour; do not enforce a universal jump where strong amplitude fluctuations may drive a first-order transition.
