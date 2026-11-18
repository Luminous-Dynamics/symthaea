# Quantum / Thermal / Dissipative Phase-Coherence Benchmark

**Contract:** SC-QUANTUM-PHASE-0.1  
**Status:** input specification and benchmark plan; no QMC result is claimed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12779

## Why this exists

The classical XY workstream (#12755) holds pair amplitude fixed and simulates thermal phase fluctuations. It has no charge-number operator and cannot represent quantum phase fluctuations, charging-energy competition, a quantum critical point, or quantum current-current response.

Ponticelli et al. (Phys. Rev. B 114, 134503, 3 September 2026) study a two-dimensional quantum XY model with on-site interactions using path-integral quantum Monte Carlo. Their abstract reports a temperature–interaction critical line ending at a zero-temperature quantum critical point, resistance scaling above the line, an insulating crossover at stronger interaction, and frequency-dependent conductivity extracted from current-current correlations.

Source: https://doi.org/10.1103/jbz7-l9h4

A second, foundational environmental comparator is Chakravarty, Ingold, Kivelson & Zimanyi (Phys. Rev. B 37, 3283, 1988), who model resistively shunted Josephson arrays with a bath reproducing Ohmic resistance in the classical limit:
https://doi.org/10.1103/PhysRevB.37.3283

The 2026 paper's full Hamiltonian, parameter definitions, units, numerical values, finite-size protocol and observables must be reconstructed from the full article and supplemental material before claiming reproduction. The abstract alone is insufficient to recreate its exact Hamiltonian or numerical phase boundary.

## Paradigm shift

Treat phase coherence as a coupled **material + interaction + environment + frequency** problem, rather than a single equilibrium scalar:

- Josephson coupling energy \(E_{J,ij}\) favours phase locking.
- Charging matrix \(U_{ij}\) penalizes charge fluctuations and generates quantum phase dynamics.
- Offset charges \(n_i^g\) can alter quantum interference.
- Gauge links \(A_{ij}\) encode applied flux / frustration.
- Temperature introduces thermal fluctuations and vortices.
- A specified bath spectral density can add dissipation and memory.
- Current-current correlations connect the phase model to finite-frequency electrodynamics.

A generic compact quantum-rotor model is

\[
H = \frac{1}{2}\sum_{ij}(n_i-n_i^g)U_{ij}(n_j-n_j^g)
-\sum_{\langle ij\rangle}E_{J,ij}\cos(\phi_i-\phi_j-A_{ij}),
\]

with \([\phi_i,n_j]=i\delta_{ij}\) in the \(\hbar=1\) convention. This is a declared generic model, not an assertion that it is exactly the Hamiltonian used by Ponticelli et al. The source-specific reproduction must transcribe the paper's own model exactly.

## Model ladder

### 1. Classical XY reference

Use #12755 as the thermal, fixed-amplitude baseline. It cannot represent charge quantization, charging energy, zero-temperature quantum criticality or quantum current-current correlations.

### 2. Quantum rotor / quantum phase model

The input contract declares:
- node identities and integer Cooper-pair number convention;
- edge identities, Josephson energies and oriented gauge-link phases;
- full symmetric charging matrix \(U_{ij}\), not an unexplained scalar;
- offset charges;
- temperature, \(\hbar\), \(k_B\), energy and charge conventions;
- spatial boundary conditions and external-flux declaration;
- imaginary-time discretization and charge-basis truncation;
- solver family, seed and exact input hashes.

The first generic synthetic fixture uses a small periodic ring with a diagonal positive charging matrix. It is a software fixture, not a material or device model.

### 3. Dissipative extension

Only after the no-bath quantum rotor baseline is qualified, add a bath with an explicit spectral density \(J_{bath}(\omega)\), cutoff, units, coupling operator and influence-kernel convention. The Ohmic / sub-Ohmic / super-Ohmic classification must be derived from the declared function and low-frequency exponent.

A scalar labelled only \(\alpha\) is insufficient without the spectral function and normalisation. A substrate or shunt mapping requires measured impedance / dielectric context.

### 4. Dynamic response

Keep these distinct:
- static helicity modulus / stiffness;
- imaginary-time phase correlations;
- winding and vortex observables;
- current-current correlation tensor;
- Matsubara conductivity;
- real-frequency conductivity after analytic continuation;
- dc resistance after declared zero-frequency and thermodynamic limits.

Analytic continuation is ill-posed. Preserve the imaginary-time correlator, continuation method, priors/regularisation, resolution tests and uncertainty. A smooth spectrum is not a unique spectrum.

## Research hypotheses

- H0: in a declared large-\(E_J/U\) regime, the classical XY model approximates specified static observables within quantified error.
- H1: charging energy and quantum fluctuations shift coherence boundaries at fixed classical coupling graph.
- H2: a specified dissipative bath changes phase coherence and transport in a model-dependent manner.
- H3: thermal vortex unbinding dominates the finite-temperature BKT regime while quantum fluctuations become central near a zero-temperature critical region.
- H4: classical graph descriptors alone are insufficient once capacitance matrix, offset charges and bath memory enter.
- H5: finite-frequency fluctuation conductivity can be mistaken for a dc superconducting response if the limits and spectral resolution are not handled carefully.

Every hypothesis can be rejected. Null and adverse outcomes are part of the benchmark.

## Benchmark plan

1. Reconstruct the exact Ponticelli et al. model from the full source and supplement; record equations, parameters, units, estimator definitions and figure/data locators.
2. Reproduce at least one source parameter slice and its qualitative phase-boundary structure before adding new physics.
3. Independently validate the classical limit against #12755.
4. Test spatial-size, imaginary-time-slice, charge-cutoff and temperature-grid convergence separately.
5. Estimate quantum-critical behaviour using finite-size / finite-temperature scaling with correction terms and uncertainty.
6. Add dissipation only with an explicit spectral density and influence kernel.
7. Validate current conservation, gauge invariance and conductivity sum rules before attempting real-frequency continuation.
8. Compare matched topology families from #12755 while independently varying \(U_{ij}\), offset charges and bath parameters.
9. Translate to a physical device only when its geometry, capacitance, impedance, flux and measurement temperature are known.

## Acceptance metrics

- Source reproduction error for the phase-boundary points and observables, with the source's own uncertainty and digitization uncertainty.
- Finite-size drift and scaling-collapse residuals.
- Imaginary-time / charge-cutoff convergence.
- Integrated autocorrelation time and effective independent samples.
- Gauge-transformation and graph-relabeling invariance.
- Current continuity / Ward-identity residuals where applicable.
- Conductivity sum-rule residual.
- Analytic-continuation resolution and uncertainty diagnostics.
- Compute, memory and sample cost.
- Leave-parameter-family-out prediction and calibration, never random-only splits.

Do not claim superiority to the published QMC based on a smoke test. Any speed/accuracy claim must match the physical model, lattice sizes, observables, error bars and compute budgets.

## Industry and device relevance

This track is principally relevant to granular superconductors, Josephson-junction arrays, thin films, interfaces and devices where capacitance, impedance or dissipation are material variables. It does not replace bulk cuprate / hydride microscopic calculations or the conductor-engineering benchmark.

For industrial translation, keep the following metrics:
- measured device / material resistance and its temperature-frequency limits;
- stiffness / superfluid density with geometry and calibration;
- critical current and current density at declared \(B,T,\theta\);
- conductor-form engineering current density and cost per kA·m only when a real conductor geometry exists;
- quench / thermal stability, reliability and fabrication yield.

## Claim levels

1. Schema-valid model input: no physics established.
2. Reproduction-ready solver input: still no result.
3. Reference-reproduced QMC: one model/parameter set checked against source evidence.
4. Converged quantum phase diagram: finite-size and algorithmic systematics reported.
5. Dissipative extension: bath assumptions and kernel validated.
6. Device translation: measured parameters and device observables support the mapping.

A quantum rotor phase diagram is not a microscopic pairing calculation, not a measured material \(T_c\), and not room-temperature superconductivity.

## Related programme

- Pair formation / bulk coherence evidence: https://github.com/Luminous-Dynamics/symthaea/issues/12677
- Classical topology / thermal XY network: https://github.com/Luminous-Dynamics/symthaea/issues/12755
- Quantum / thermal / dissipative dynamics: https://github.com/Luminous-Dynamics/symthaea/issues/12779
- Synthesis protocol to phase to property: https://github.com/Luminous-Dynamics/symthaea/issues/12671


## Exact source reconstruction: Ponticelli et al. (2026)

The full preprint is openly accessible as arXiv:2512.20476 (23 December 2025); it was subsequently published as Phys. Rev. B 114, 134503 (3 September 2026):
- Published DOI: https://doi.org/10.1103/jbz7-l9h4
- Preprint: https://arxiv.org/abs/2512.20476

The exact phase-representation Hamiltonian in Eq. (1) is:

\[
H=-t\sum_{\langle ij\rangle}\cos(\phi_i-\phi_j)
+\frac{U}{2}\sum_i\left(-i\frac{\partial}{\partial\phi_i}\right)^2.
\]

The model is a nearest-neighbour \(L\times L\) square lattice. The authors set \(\hbar=k_B=2e=1\) and use \(t\) as the energy unit. The source model is therefore a special case of the generic compact quantum rotor, with a diagonal on-site charging matrix \(U_{ij}=U\delta_{ij}\), uniform nearest-neighbour Josephson energy \(E_J=t\), zero link gauge field and no dissipative bath.

The Suzuki–Trotter / Villain mapping gives:

\[
S=-\sum_{\tau,\langle ij\rangle}K_s\cos(\phi_i(\tau)-\phi_j(\tau))
-\sum_{\tau,i}K_\tau\cos(\phi_i(\tau)-\phi_i(\tau+\Delta\tau)),
\]

with \(K_s=t\Delta\tau\), \(K_\tau=1/(U\Delta\tau)\), \(N_\tau=\beta/\Delta\tau\), and periodic imaginary-time boundary conditions. **The spatial and imaginary-time couplings are anisotropic in the source's finite-\(\Delta\tau\) scheme.** Replacing them with an isotropic 3D XY model changes non-universal critical values; the paper explicitly discusses this source of disagreement with earlier values.

### Source-reported numerical targets

- Finite-temperature transition: BKT for \(U<U_c\), with finite-size criterion \(\Upsilon_L(U,T_c(L))=2T_c(L)/(t\pi)\) and \(T_c(L)=T_c(\infty)+a_T/[\ln(b_TL)]^2\).
- Zero-temperature critical point: \(U_c(0)/t=5.72(1)\), obtained by extrapolating finite-\(\Delta\tau\) estimates linearly to \(\Delta\tau\to0\).
- At finite \(\Delta\tau=1/150\), the reported finite-step estimate is \(U_c/t=5.63(2)\); do not substitute this for the extrapolated result.
- Quantum critical scaling: \(\Upsilon=L^{-z}f((U-U_c)L^{1/\nu})\), with \(z=1\) and \(\nu=0.671\); at criticality \(\Upsilon L\) should be size-independent.
- The finite-temperature BKT line persists for \(U<U_c\); for \(U>U_c\) the source reports finite resistance and a metallic-to-insulating crossover rather than a finite-temperature superconducting transition.
- Quantum fluctuations increase total vortex density at low temperatures. The source's total vortex count does not separate bound pairs from unbound vortices.

### Numerical burden and convergence targets

The source demonstrates why an arbitrary small \(N_\tau\) is unsafe:
- At \(L=48\), \(U/t=4\), stiffness convergence near the transition required \(N_\tau=1000\) in the displayed comparison.
- A lower-temperature scaling case uses \(T/t=0.05\) and \(N_\tau=5000\).
- The paper shows an example with \(L=48\), \(N_\tau=1000\), \(U/t=4\), and temperature values around \(T/t=0.3\) and \(0.72\).
- For the zero-temperature scaling path it uses \(L=\beta\), \(z=1\), and extrapolates multiple \(\Delta\tau\) values.

Those values are source-specific reference points, not a universal minimum for every implementation. Record the chosen \(L\), \(N_\tau\), \(\Delta\tau\), \(\beta\), \(U/t\), random seed, cluster/Metropolis update schedule, equilibration and autocorrelation evidence for every run. Enforce \(N_\tau\Delta\tau=\beta=1/T\) in the input auditor.

### Stiffness and transport

The paper's superfluid stiffness is obtained from a boundary twist and a current-fluctuation estimator. The source's normalization is tied to its conventions \(\hbar=2e=k_B=1\); compare formulas only after matching these conventions.

For transport, the source computes the imaginary-time current-current correlator \(F(\tau)=\langle I(\tau)I(0)\rangle\), then reconstructs the real-frequency regular conductivity through:

\[
F(\tau)=\frac{1}{\pi}\int_0^\infty d\omega\,
\sigma_{\mathrm{reg}}(\omega)\,
\frac{\omega\cosh[\omega(\beta-2\tau)/2]}{\sinh(\omega\beta/2)}.
\]

The paper uses a constrained Lorentzian-plus-symmetrized-Gaussian ansatz above the transition and a symmetrized-Gaussian ansatz below it. It explicitly warns that finite-size effects prevent physically reliable analytic continuation near the critical region. Our benchmark must reproduce the source method for a baseline comparison but also test alternative continuation methods, uncertainty and resolution; it must not elevate one fitted spectrum to a unique result.

## Contract update motivated by the source

The input schema now records:
- \(\hbar\), \(k_B\), and Cooper-pair charge \(2e\) conventions separately;
- imaginary-time step \(\Delta\tau\) and slice count \(N_\tau\);
- whether the solver uses a phase-path-integral representation or an integer-charge basis;
- an explicit null/not-applicable charge-basis cutoff for a phase-path-integral solver.

The auditor enforces \(N_\tau\Delta\tau=1/T\) in the declared \(k_B=1\) units. A phase-path-integral input cannot silently claim an integer charge-state truncation. This prevents a generic rotor contract from misrepresenting the source paper's numerical discretisation.


## QXY path-integral solver prototype

A first restricted Rust execution path has been added in `tools/superconductivity-quantum-phase-harness/src/bin/qxy_phase_pimc.rs`. It is specific to the source Hamiltonian above; it is not a generic QMC engine and does not yet reproduce the paper's published results.

### Implemented model mapping

For a declared uniform square lattice with zero offset charge, no bath and zero gauge field, the executable constructs the anisotropic spacetime action using

\[
K_s=t\Delta\tau,\qquad K_\tau=\frac{1}{U\Delta\tau},
\qquad N_\tau\Delta\tau=\beta=\frac{1}{T}.
\]

Each spatial time-slice has both positive-direction nearest-neighbour bonds; each spatial site has a periodic imaginary-time bond. The Wolff embedded-cluster update uses the corresponding bond coupling \(K_s\) or \(K_\tau\) and is followed by a random-site Metropolis sweep. The solver refuses nonuniform Josephson energies, nonzero offset charges, gauge links / flux, baths, non-square lattices and malformed/duplicate lattice bonds rather than silently running a different model.

### Source observables implemented

- Euclidean action per spacetime site.
- Source-normalized superfluid stiffness estimator
  \[
  \Upsilon=\frac{1}{t}\frac{\langle S_s\rangle-\beta\langle I^2\rangle}{2L^2},
  \qquad
  S_s=\frac{1}{\beta}\sum_{\tau,\langle ij\rangle}K_s\cos(\Delta\phi),
  \qquad
  I=\frac{t\Delta\tau}{\beta}\sum_{\tau,\langle ij\rangle}\sin(\Delta\phi).
  \]
- Total spatial-plaquette vortex density using the source's wrapped phase-circulation definition.

A deterministic aligned-phase unit test checks the normalization \(\Upsilon=1\), zero vortex density and the expected action per spacetime site. Input tests separately check the source anisotropic couplings and the \(N_\tau\Delta\tau=\beta\) constraint.

### Exact smoke inputs and boundaries

- `superconductivity_quantum_phase_QXY_smoke_fixture_v0_1.json`: L=4, U/t=4, T/t=0.3, Nτ=16. This is deliberately under-resolved and is only suitable for testing execution, determinism and finite-valued output.
- `superconductivity_quantum_phase_ponticelli_2026_U4_T03_input_fixture_v0_1.json`: L=4, U/t=4, T/t=0.3, Nτ=1000. This checks the source's discretisation/parameter mapping; it is not the paper's L=48 simulation.

The smoke receipt includes exact schema and fixture SHA-256 values, instance identity, model parameters, couplings, actual seed, update counts, Metropolis acceptance and the observable outputs. The workflow's short run is not a statistical measurement.

**Not implemented / not qualified yet:** production-size L=48 and larger runs, error bars, integrated autocorrelation / effective-sample-size analysis, warmup and convergence criteria, Δτ extrapolation, BKT finite-size scaling, zero-temperature 3D-XY scaling, quantum-critical finite-size collapse, current-current imaginary-time correlators, conductivity sum rules, analytic continuation and the dissipative bath extension.

The reported \(U_c/t=5.72(1)\), \(z=1\), and \(\nu=0.671\) remain paper-reported targets. We have not reproduced them. The source's \(N_\tau=1000\) at L=48 and \(N_\tau=5000\) at T/t=0.05 illustrate the computational burden; do not promote the L=4 smoke to source parity.


### Sampling diagnostics and evidence receipts

The QXY smoke receipt is bound to the exact schema and fixture SHA-256 hashes, instance ID, source parameter set and actual random seed. For action, stiffness and vortex-density series it reports the sample mean, sample standard deviation, an explicitly labelled **iid-only** standard error, sample count and lag-one autocorrelation.

These statistics are not a replacement for integrated autocorrelation time, effective sample size, thermalization analysis, multiple independent chains or systematic finite-\(N_\tau\) extrapolation. In particular, the iid standard error must never be used as a publication-quality uncertainty when lagged dependence is present or unquantified.
