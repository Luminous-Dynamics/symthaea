# Microscopic-to-Effective Superconductivity Evidence Bridge

**Contract:** SC-MICROSCOPIC-GL-TRANSLATION-0.1  
**Status:** source metadata and identifiability auditor; no microscopic calculation has been reproduced  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12840

## Purpose

This workstream connects material-specific calculations and measurements to the effective GL models in #12793 and #12826 without treating phenomenological coefficients as free, source-less numbers.

It connects:
- conventional SCDFT / Eliashberg / DFPT-EPC calculations;
- correlated and unconventional many-body calculations;
- experiment-calibrated specimen-linked observables (#12677 and #12811);
- scalar amplitude-phase GL (#12793);
- multicomponent/interband/PDW GL (#12826);
- synthesis / phase realization (#12671), prospective discovery (#12634), and capability-aware follow-up (#12819).

The model family remains explicit. Conventional SCDFT results are not automatically applicable to cuprates, nickelates, or PDW states.

## Source benchmark motivating this bridge

Kawamura et al., *First-Principles Calculation of Coherence Length and Penetration Depth Based on Density Functional Theory for Superconductors*, Phys. Rev. Lett. 137, 196001 (published 3 November 2026), DOI https://doi.org/10.1103/lrhl-j9bt.

The paper reports a SCDFT method including finite-momentum Cooper-pair response and calculations of Tc, microscopic coherence length xi0 and London penetration depth lambdaL. Its reported application set includes elemental superconductors, V3Si and high-pressure H3S. The first fixture records only this paper-level method summary and material coverage; exact material tables, supplementary details and numerical inputs/outputs have not yet been extracted or reproduced.

The PNAS perspective *The path to room-temperature superconductivity: A programmatic approach* (2026), DOI https://doi.org/10.1073/pnas.2520324123, motivates a closed loop between prediction, synthesis thermodynamics, engineering controls and experiment. A model that predicts a high Tc but cannot identify a realizable phase or useful coherence/response is not a complete discovery result.

## Non-equivalence rules

Keep these quantities distinct:
- microscopic/BCS/Pippard coherence length xi0;
- near-Tc GL coherence length xi_GL(T);
- London penetration depth lambdaL(T), scalar only under an explicit isotropy assumption or otherwise a tensor;
- superfluid stiffness/helicity modulus under a declared normalization;
- GL gradient coefficient K;
- quartic GL coefficient b;
- microscopic pairing gap and coarse-grained |psi|.

The GL expansion is represented as

\[
F-F_n=\int d^dr\left[
a(T)|\psi|^2+\frac{b}{2}|\psi|^4+
K_{\mu\nu}(D_\mu\psi)^*(D_\nu\psi)+\cdots
\right].
\]

The normalization of psi changes the numerical coefficients. Therefore every coefficient requires a basis/field normalization, units, temperature and field regime, method, evidence status, source linkage and uncertainty/covariance.

Neither Tc, xi0 nor lambdaL alone determines the full set a(T), b, K_mu_nu, multiband coupling matrices and current response. The auditor rejects unsupported coefficient promotion.

## Evidence status vocabulary

- **derived**: extracted from an explicit, versioned microscopic free-energy, pair-susceptibility or response calculation with declared normalization and fit procedure.
- **constrained**: bounded by evidence but not uniquely identified.
- **phenomenological_fit**: fitted to experimental observations through a declared observation model/prior and state linkage.
- **not_identifiable**: current evidence does not identify the coefficient; numeric value must be null.
- **not_applicable**: coefficient is outside the chosen model family; numeric value must be null.

Paper-reported values are not independently reproduced values. The source metadata fixture preserves this distinction and intentionally carries zero numeric derived-observable rows.

## Implemented input contract and auditor

Files:
- schema: `data/schemas/superconductivity_microscopic_gl_translation_v0_1.schema.json`
- fixture: `data/research/superconductivity_microscopic_gl_translation_fixture_v0_1.json`
- binary: `sc_microscopic_gl_audit` in `tools/superconductivity-microscopic-gl-translation-harness`

The Rust auditor validates:
- explicit material state identity and state linkage for every observable and GL coefficient;
- calculation family and mechanism scope;
- finite scalar/vector/matrix observable values and non-negative finite uncertainty;
- scalar lambda_L only with an explicit isotropy assumption;
- lambda_L tensor only as a matrix with at least two declared axes;
- xi0_microscopic and xi_GL as distinct typed quantities with distinct length-scale definitions;
- derived a(T) slope only with quadratic pair susceptibility / free-energy expansion evidence;
- derived b only with quartic free-energy response;
- derived K only with finite-q pair response, a q-gradient expansion and basis normalization;
- phenomenological-fit coefficients only with experimental evidence and an experimental-effective model family;
- no promotion of conventional SCDFT outputs as correlated-unconventional GL coefficients;
- exact schema and fixture SHA-256 values in the audit receipt.

The source-method starter record contains:
- source method: SCDFT with finite-q pair response;
- paper-reported material coverage descriptions for elemental cases, V3Si and high-pressure H3S;
- no invented numerical values, meshes, cutoffs, pressure points or raw output digests;
- a(T) slope, b and K marked not_identifiable until the full quantitative response and coefficient normalization are extracted.

The K status is deliberately not_identifiable, rather than constrained, because the metadata-level source record does not include a quantitative finite-q response curve or fitted gradient coefficient.

## Next executable gates

1. Retrieve and freeze the full PRL article, methods, supplementary material and any author-provided inputs/outputs.
2. Extract exact pseudopotential/basis, XC, SOC, k/q mesh, energy cutoff, phonon treatment, Coulomb kernel and convergence.
3. Reconstruct at least one elemental case, V3Si and high-pressure H3S with exact material states and pressure.
4. Compare Tc, xi0, lambdaL and type-I/type-II classification with the paper and experiment under matched conditions.
5. Where the microscopic source actually provides the required response, fit the near-Tc GL expansion and derive only identifiable a, b and K parameters.
6. Verify GL xi_GL(T) only in the GL regime; never relabel xi0.
7. Pass identified/constrained coefficients into #12793/#12826 with covariance and provenance. Preserve non-identifiable values as null.
8. Evaluate held-out cross-modal predictions via #12811 and choose high-value next measurements/computations via #12819.

## Research targets (not results)

- Reproduce at least one published SCDFT material case with method-aligned inputs and reported uncertainty.
- 100% of exported GL coefficients have declared normalization, units, regime and evidence status.
- Zero silent xi0 ↔ xi_GL conversions.
- Zero cross-state joins across pressure, specimen, phase, doping or strain.
- Correct abstention for non-identifiable coefficients.
- Synthetic parameter recovery with nominal 90% intervals achieving 85–95% coverage for identified parameters; non-identifiable parameters must not be forced into narrow intervals.
- A held-out cross-modal prediction that beats a simpler baseline and is calibrated on grouped holdouts.

## Claim ceiling

This is a provenance and identifiability auditor. It does not reproduce the PRL calculations, derive coefficients from missing data, establish a room-temperature superconductor, prove any unconventional mechanism or authorize experiments.
