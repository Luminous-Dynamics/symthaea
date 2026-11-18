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
