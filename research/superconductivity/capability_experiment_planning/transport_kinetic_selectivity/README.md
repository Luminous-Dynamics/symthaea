# SC-TRANSPORT-KINETIC-SELECTIVITY-0.1

**Status:** reduced synthetic two-route transport / reaction competing-risk model; no physical phase yield or experimental protocol is predicted.  
**Research issue:** [#13000](https://github.com/Luminous-Dynamics/symthaea/issues/13000)  
**Thermochemical network:** [#12997 / PR #12999](https://github.com/Luminous-Dynamics/symthaea/pull/12999)  
**Reaction selectivity:** [PR #12996](https://github.com/Luminous-Dynamics/symthaea/pull/12996)

## Motivation

Hull distance, target-vs-competitor \(\Delta G\), and activation barriers are not enough when formation is limited by transport through a product layer. A 2026 *Nature Materials* study, “Ion correlations explain kinetic selectivity in diffusion-limited solid-state synthesis reactions,” uses ML-derived transport properties within a thermodynamic cellular reaction model and reports that cross-ion transport coefficients matter for phase selection in Ba–Ti–O synthesis. This is a methodological comparator, not superconductivity-specific evidence: https://www.nature.com/articles/s41563-026-02596-5

This v0.1 contract isolates the mechanism in a small synthetic two-mobile-species model. It is not a reproduction of that paper.

## Transport model

For mobile species A and B, compute diagonal diffusivities using Arrhenius temperature dependence:

\[
D_{AA}(T)=D^0_A e^{-E^D_A/(k_BT)},\qquad
D_{BB}(T)=D^0_B e^{-E^D_B/(k_BT)}.
\]

The symmetric cross coefficient is declared through a dimensionless correlation:

\[
D_{AB}(T)=\rho\sqrt{D_{AA}(T)D_{BB}(T)}.
\]

v0.1 requires \(|\rho|<1\), so the 2×2 matrix is positive definite. The target reaction requires common-mode co-transport of A and B:

\[
D_{\mathrm{co}}=\frac{D_{AA}+D_{BB}+2D_{AB}}{2}.
\]

The competing reaction uses A only. The declared product-layer thickness \(L\) yields characteristic transport rates \(d_j=2D_{\mathrm{eff},j}/L^2\).

The reaction waiting rate is computed separately:

\[
k_j(T)=k^0_j e^{-E_{a,j}/(k_BT)} a_j,
\]

where \(a_j\) is the declared route-availability fraction. Thermodynamic \(\Delta G\), activation barrier, transport and process duration remain separate receipt fields.

## Two-stage competing-risk model

Each route's completion time is approximated as the sum of two independent exponential waiting times: transport at rate \(d_j\), then reaction at rate \(k_j\). For unequal rates:

\[
S_j(t)=\frac{k_j e^{-d_jt}-d_j e^{-k_jt}}{k_j-d_j},\qquad
f_j(t)=\frac{d_jk_j(e^{-d_jt}-e^{-k_jt})}{k_j-d_j}.
\]

The equal-rate Erlang-2 limit is implemented explicitly.

For independently competing route-completion clocks, the cumulative incidence fractions are:

\[
F_T(t)=\int_0^t f_T(u)S_C(u)\,du,\qquad
F_C(t)=\int_0^t f_C(u)S_T(u)\,du,
\]

\[
F_{\mathrm{unreacted}}(t)=S_T(t)S_C(t).
\]

Composite Simpson quadrature evaluates the two integrals. The audit requires all three fractions to sum to one within \(10^{-6}\). Since the route completion distributions differ, their relative cumulative incidence can change over time; this permits a short-hold / long-hold phase-selection reversal.

This is a phenomenological reduced model. It assumes independent route clocks and does not model shared precursor depletion, spatial concentration fields, moving product layers, nucleation/growth, post-formation phase transformations, correlated waiting times or a real Onsager transport field. It is not a spatial reaction-diffusion solver.

## Frozen fixture

The main fixture holds thermodynamic intervals, barriers, prefactors, layer thickness and temperature fixed while changing only the cross-ion correlation or hold time.

For the negative-correlation cases:
- at 5 s, target cumulative incidence exceeds competitor cumulative incidence;
- at 30 s, the competitor overtakes the target.

For the 30 s positive-correlation case, the larger target co-transport projection raises target cumulative incidence. The diagonal-diffusion baseline sets the cross coefficient to zero; a reaction-only baseline ignores transport entirely.

These are outputs of invented parameters. The fixture is designed to expose the mechanism, not to represent a real material or laboratory schedule.

## Adversarial controls and tests

Registry: \`tools/superconductivity-discovery-harness/fixtures/transport_kinetic_selectivity_suite_v0_1.json\`.

- Same thermodynamic preference but opposite cross-ion transport correlations.
- Short-hold and long-hold negative-correlation cases with a phase-selection reversal.
- Positive-correlation target co-transport case.
- Diagonal-diffusion and reaction-only matched baselines.
- A non-positive-definite transport matrix with \(|\rho|>1\), which must be rejected.
- A same-rate test exercises the Erlang-2 analytic limit.
- Every phase fraction must be non-negative and sum to one within the frozen quadrature tolerance.

## Comparators and next benchmarks

The output provides three transparent baselines: coupled transport, zero-cross-term diagonal diffusion, and reaction-only kinetics. The next benchmark should add a 1D finite-volume reaction-diffusion reference, concentration-dependent transport matrices, moving product-layer thickness, precursor-ratio effects and uncertainty in transport/activation parameters. Compare the reduced model against the spatial reference on identical initial conditions and process schedules before comparing to published ML-transport / thermodynamic-cellular approaches.

Relevant 2026 comparators:
- Ion correlations and diffusion-limited solid-state phase selectivity (*Nature Materials*, 2026): https://www.nature.com/articles/s41563-026-02596-5
- Closed-loop LLM solid-state synthesis planning (Advanced Materials, 2026; 4,407-paper recipe corpus and iterative oxy-selenide demonstrations): https://doi.org/10.1002/adma.74502
- Chemically aware protocol validation and SAC benchmark (Journal of Cheminformatics, 2026; 425 protocols from 239 papers): https://doi.org/10.1186/s13321-026-01222-5

The latter two are protocol generation/validation comparators, not phase-yield or superconductivity prediction benchmarks. Protocol validity, actual execution, phase realization and superconducting characterization remain distinct endpoints.

## Claim ceiling

This is a synthetic transport-coupled kinetic-selectivity contract. It does not establish a measured diffusion coefficient, real reaction rate, realized phase fraction, superconducting material, process recipe or laboratory authorization.
