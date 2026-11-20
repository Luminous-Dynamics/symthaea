# SC-SPATIAL-REACTION-DIFFUSION-0.1

**Status:** bounded synthetic 1D finite-volume reference; no measured phase yield or real process prediction.  
**Research issue:** [#13003](https://github.com/Luminous-Dynamics/symthaea/issues/13003)  
**Reduced transport model:** [#13000](https://github.com/Luminous-Dynamics/symthaea/issues/13000)  
**Thermochemical network:** [PR #12999](https://github.com/Luminous-Dynamics/symthaea/pull/12999)

## Why a spatial reference is required

The reduced two-stage model in #13000 compresses each competing route into independent transport and reaction waiting times. It is computationally cheap and can expose time-dependent ranking changes, but it cannot express spatial gradients, shared local reactant depletion or moving reaction fronts.

This v0.1 finite-volume model is a small reference problem against which that reduction can be measured. It is not a reproduction of the 2026 *Nature Materials* cross-ion cellular model and not yet a moving-boundary product-layer solver.

## Governing equations

For one-dimensional mobile species A and B:

\[
\begin{bmatrix}J_A\\J_B\end{bmatrix}
=-\begin{bmatrix}D_{AA}&D_{AB}\\D_{AB}&D_{BB}\end{bmatrix}
\begin{bmatrix}\partial_x c_A\\\partial_x c_B\end{bmatrix}
\]

\[
\partial_t c_A=-\partial_xJ_A-r_T-r_C,\qquad
\partial_t c_B=-\partial_xJ_B-r_T
\]

\[
r_T=k_Tc_Ac_B,\qquad r_C=k_Cc_A.
\]

The target consumes one A and one B; the competing phase consumes one A. Concentrations are dimensionless relative to the declared reservoir reference. Diffusivities are m²/s, length m, time s and rate constants s⁻¹ under this normalized concentration convention.

Boundary conditions:
- fixed reservoir concentration at the exposed surface \(x=0\);
- zero-flux boundary at \(x=L\);
- uniform initial mobile-species profiles from the fixture.

## Conservative finite-volume discretization

Each internal face flux is computed once and used with opposite signs in neighboring cells. The exposed reservoir flux is calculated across the half-cell distance; the inner boundary flux is zero. The update is explicit in time and uses:
- diffusion stability guard from the largest transport-matrix eigenvalue;
- reaction stability guard from the maximum declared reaction rates and concentration ceiling;
- a hard error if the requested timestep exceeds either bound;
- a hard error for materially negative concentration, with only sub-\(10^{-12}\) floating-point dust set to zero.

The auditor integrates target/competitor production and reservoir influx over the full run. It closes two separate inventory ledgers:

\[
A_0+I_A=A_f+P_T+P_C,\qquad
B_0+I_B=B_f+P_T.
\]

Both relative residuals must remain below \(10^{-8}\). The receipt preserves the spatial final concentration profiles and the boundary influx, rather than reporting only a scalar phase score.

## Frozen fixture cases

The five main cases compare:
1. negative cross coefficient / short hold;
2. the same negative cross coefficient / long hold;
3. positive cross coefficient / long hold;
4. diagonal diffusion (cross coefficient zero) / long hold;
5. a thicker layer under otherwise matched conditions.

The suite also has two fail-closed cases:
- non-positive-definite transport matrix;
- explicit timestep above the diffusion/reaction stability bound.

The negative/positive cross terms change the A+B co-transport flux; the competitor consumes only A. Thickness and hold time alter penetration and reaction opportunity. All transport values, rates, lengths and time windows are synthetic.

## Comparison against #13000

The #13000 reduced model gives a fast independent-route completion approximation. This solver resolves a 1D concentration field with local reaction consumption and a reservoir boundary. The benchmark should compare both on matched transport matrices, kinetics, thickness and process duration.

Report:
- target and competitor integrated product amounts;
- target/competitor ratios;
- A/B concentration profiles and front position;
- cumulative reservoir influx;
- A and B relative conservation residuals;
- sensitivity to grid and timestep refinement;
- rank agreement and phase-fraction error of the reduced #13000 model against the finite-volume reference.

The finite-volume solver is itself only a reference for this simplified 1D model. It does not include lateral microstructure, moving phase boundaries, concentration-dependent mobility, nucleation, nonideal chemical potentials, elastic stress, or product-layer evolution.

## Next numerical benchmarks

- Grid refinement at 24/48/96/192 cells with timestep scaled to the stability bound.
- Timestep refinement at fixed grid.
- Positivity and conservation under high cross-correlation and high reaction rate.
- Compare explicit finite volume with an implicit solver for stiffer regimes.
- Extend to a moving product-layer interface only after the fixed-domain reference is verified.
- Add uncertainty ensembles for \(D_{AA},D_{BB},D_{AB},k_T,k_C\), not just point estimates.

## Literature and SOTA boundary

The 2026 *Nature Materials* study “Ion correlations explain kinetic selectivity in diffusion-limited solid-state synthesis reactions” identifies cross-ion transport correlations as important in Ba–Ti–O phase selection and uses ML-derived transport inputs in a thermodynamic cellular reaction framework: https://www.nature.com/articles/s41563-026-02596-5

This work is a methodological inspiration. v0.1 does not reproduce its equations, parameters, Ba–Ti–O outcomes or model performance. The relevant next SOTA comparison is a faithful published-model reproduction on the same initial states and conditions, plus a converged spatial reference. Do not claim superiority based on a synthetic 1D fixture.

## Claim ceiling

This is a synthetic finite-volume calculation. It does not establish measured diffusion, reaction kinetics, phase yield, material synthesizability, superconductivity or a safe/authorized laboratory protocol.
