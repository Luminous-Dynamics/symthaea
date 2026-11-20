# Local Gauss-law-preserving electrostatic PIC benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-LOCAL-GAUSS-PIC-001 #12962](https://github.com/Luminous-Dynamics/symthaea/issues/12962)  
**Status:** source review / reproduction protocol only; no implementation or simulation performed

## 1. Primary source

Zhonghua Qiao, Zhenli Xu, Qian Yin & Shenggao Zhou, “Intrinsic local Gauss’s law preserving PIC method: A self-consistent field-particle update scheme for plasma simulations,” *Computer Physics Communications* 327 (October 2026), article 110296, DOI [10.1016/j.cpc.2026.110296].

The source describes an electrostatic Vlasov–Ampere PIC method that updates the electric field locally and self-consistently with particle motion. Particle trajectories are split into directional substeps; local field updates follow the charge flux induced by each particle-motion substep; a local curl-free relaxation minimises electric-field free energy subject to Gauss's law. The authors report exact discrete Gauss-law preservation to machine precision and tests on Landau damping, two-stream and Diocotron instability.

The paper describes the method as avoiding an explicit global Poisson or Ampere solve during its field update. The local relaxation is reported to scale linearly with the number of local mesh operations per sweep and is presented as suitable for nearest-neighbour communication. These claims apply to the paper's exact algorithm and assumptions; they are not a general guarantee for arbitrary geometry, time integrator or electromagnetic model.

No exact author code repository, raw run bundle, tag, or artifact licence has been admitted in this review. Do not infer these from the article page.

## 2. Physical scope

This is an **electrostatic Vlasov–Ampere** benchmark. It is not automatically:
- a full electromagnetic Vlasov–Maxwell method;
- an unstructured-grid solver;
- an energy- or momentum-conserving method merely because Gauss's law is preserved;
- a neural operator.

Keep this lane separate from electromagnetic Maxwell, magnetosheath 2D–3V, ICF heat transport and collisional Vlasov–Poisson–Landau scorecards.

## 3. Reproduction contract

Pin the source revision, implementation release, exact equations/normalisation, periodic-domain and neutralising-background convention, particle shape/deposition/interpolation, directional substep ordering, local charge-flux field update, curl-free relaxation equation and stopping threshold, time integration, mesh, particle count, timestep, precision and random seed.

If the exact implementation or numerical configuration remains unavailable, the lane stays source-described and unscored.

## 4. Baseline tournament

Compare:
1. source local Gauss-preserving GP-PIC;
2. standard CIC / grid PIC with its discrete Poisson/Ampere solve;
3. PIF / Fourier PIC;
4. energy-conserving Fourier Vlasov–Ampere PIC;
5. NEOPIC direct particle-measure field operator;
6. variational particle-energy functional with gradient-derived force (prospective #12960);
7. neural initial guess plus unchanged local GP-PIC correction.

A neural proposal cannot replace the reference constraints. In the hybrid, GP-PIC remains the final local Gauss-law authority; failed proposals incur full correction/fallback cost.

## 5. Frozen task cohorts

- 2D-2V Landau damping;
- 2D-2V two-stream instability;
- Diocotron instability using the source's exact geometry, boundary and equilibrium assumptions;
- particle-count, grid-resolution and timestep sweeps;
- domain decomposition / nearest-neighbour communication scaling.

Do not pool these with 3D electromagnetic instabilities.

## 6. Qualification metrics

**Discrete structure:** per-cell Gauss residual and maximum normalised residual; charge continuity; electrostatic curl-free residual; global charge/neutrality; field-energy positivity where applicable; total energy drift and momentum residual measured separately; local relaxation sweeps and residual contraction.

**Kinetic behaviour:** particlewise electric-field, position and velocity error; Landau damping rate and field-energy spectrum; two-stream growth rate, saturation time and peak; Diocotron mode frequency, growth and confinement; long-horizon distribution and trajectory error.

**Performance and scaling:** local field-update time per cell/particle; relaxation sweeps and convergence time; bytes communicated, halo exchanges, global collectives and synchronization count; complete PIC step including particle push; weak/strong scaling; memory.

## 7. Prospective gates (not results)

1. Reproduce source exact-Gauss tests before claiming the baseline.
2. Evaluate Gauss and charge continuity at every time step and per cell, not only in a global norm.
3. Neural competitors must pass the same discrete Gauss and trajectory tests.
4. Claim communication advantages only from measured message/byte/collective counts and matched scaling experiments.
5. Report energy, momentum and Gauss separately; one does not imply the others.
6. Keep electrostatic and electromagnetic solver results in different lanes.

## 8. Admission state

| Item | State |
|---|---|
| Published paper / DOI | Identified |
| Electrostatic Vlasov–Ampere scope | Identified |
| Source-reported Gauss-law property | Recorded, not reproduced |
| Public author code revision | Not admitted |
| Exact run data / seeds / configs | Not admitted |
| Code/data licence | Unknown pending artifact admission |
| Reproduction | Not performed |
| Matched NEOPIC/PIF/CIC comparison | Not performed |
| Communication scaling | Not measured |

## 9. Claim ceiling

This is a benchmark specification. It does not claim GP-PIC has been reproduced, code is public, the method is universally communication-free, or that a neural competitor meets Gauss's law.
