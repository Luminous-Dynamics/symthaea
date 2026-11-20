# ICF resolution-robust heat-flux closure benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-ICF-RESOLUTION-CLOSURE-001 #12938](https://github.com/Luminous-Dynamics/symthaea/issues/12938)  
**Evidence state:** source review / benchmark specification; no data admitted, model run or article reproduction performed

## 1. Primary reference

M. Luo, A. R. Bell, F. Miniati, S. M. Vinko & G. Gregori, “Resolution-Robust Machine Learning Heat Flux Closure for Inertial Confinement Fusion Plasmas,” *PRX Intelligence* 1, 013017 (2026), DOI [10.1103/9l4n-mnz6](https://doi.org/10.1103/9l4n-mnz6), published 3 September 2026.

The paper learns the nonlocal constitutive map from electron temperature profile to heat-flux divergence using a Fourier neural operator trained on fully kinetic OSIRIS PIC simulations. It embeds the learned operator in the electron-energy equation and solves implicitly/iteratively. The published abstract and accessible article text describe:
- hot-spot relaxation and Epperlein–Short temperature perturbation cases;
- hot-spot width interpolation and temporal extrapolation beyond the training horizon;
- coarse-data training and zero-shot deployment in substantially finer-resolution implicit solvers;
- comparison against the Schurtz–Nicolaï–Busquet nonlocal transport model;
- nonlocal regimes including a reported Knudsen number around 0.17.

The article is CC BY 4.0. This does not automatically grant rights to separate PIC datasets, model weights or third-party figures. No public author code or source dataset artifact/record licence has been admitted by this project yet.

## 2.1 Required adjacent neural-transport baselines

The SOTA comparison must also include two directly related neural heat-transport methods:

- **LINN / nonlocal theory-informed neural kernel:** Luo et al., *Physical Review Research* 7, L042017 (2025), DOI [10.1103/m215-3f6c](https://doi.org/10.1103/m215-3f6c). This approach learns spatiotemporal heat-flux kernels from kinetic PIC data across local and nonlocal regimes, using LMV transport theory as a prior.
- **TCNN / time-embedded CNN:** Luo et al., *Physical Review E* 113, 035303 (2026), DOI [10.1103/xx8k-xw91](https://doi.org/10.1103/xx8k-xw91). It models the coupled evolution of normalised heat flux and a characteristic nonlocality parameter, addressing a limitation of quasi-stationary LMV-informed kernels under strongly time-dependent transport.

These papers are required prior-art families, not matched score rows by default. Their kinetic datasets, collisionality ranges, input/target conventions, solver and geometry may differ from the PRX FNO paper. First reproduce each on its own source task; then adapt them to a common hot-spot / Epperlein–Short benchmark only after source and data admission.

The matched common-task tournament must therefore distinguish:
1. source FNO operator;
2. LINN theory-informed time-varying kernel;
3. TCNN time-embedded coupled heat-flux/nonlocality model;
4. SNB nonlocal transport;
5. applicable local transport baseline;
6. mesh-/coordinate-aware or graph/finite-volume neural operator;
7. neural initial guess / trusted implicit residual-correction hybrid.

## 2. Important source-faithful details

The article identifies the learned operator as Te(x) → d_x q_e(x), and the coupled electron energy equation in the source model as a relation of the form:

    (3/2) ∂Te/∂t + F_FNO(Te) = S_ext

The studied cases use an implicit iterative scheme. The paper reports training samples over t in (0,20] and evaluating selected predictions through t=30. For the hot-spot case, selected Gaussian widths are used for training and intermediate widths for testing. The Epperlein–Short case varies perturbation wavenumber.

The source uses its own notation for spatial and temporal downsampling factors. Do not infer or silently rename those factors. Obtain the exact supplement/configuration before a numerical reproduction.

## 3. Research paradigm

The strongest conceptual contribution is not simply an FNO architecture. It is a **constitutive operator that can be inserted into an implicit solver and evaluated on a different mesh**.

A model that predicts flux divergence accurately in isolation may still fail when placed inside a nonlinear iterative solve. Our core question is therefore:

> Does the learned operator remain accurate, stable, physically admissible and computationally advantageous after the actual discrete solver consumes it at a new resolution and over a longer physical time?

## 4. Matched reproduction tracks

### Track A — source-faithful reproduction

- Obtain source training arrays, PIC outputs, weights, exact configuration, preprocessing/normalisation, supplementary material and solver scripts from authors or an explicit artifact record.
- Pin OSIRIS version/build, kinetic parameters, boundary conditions, particle statistics and exact temporal/spatial sampling.
- Reconstruct hot-spot and Epperlein–Short training/test splits.
- Reproduce the FNO, SNB and any source local-transport comparator.
- Reproduce the electron-energy equation, implicit iteration algorithm, convergence criteria and temperature histories.
- Record all source unknowns instead of guessing from figures.

### Track B — resolution transfer tournament

Compare:
- source FNO;
- grid-matched CNN;
- coordinate-/mesh-aware neural operator;
- graph or finite-volume message-passing operator;
- SNB;
- applicable classical local transport baseline;
- kinetic PIC reference where admitted.

Interventions:
- train coarse → test fine (zero-shot);
- train fine → test coarse;
- multi-resolution training;
- independent spatial and temporal resolution shifts;
- unchanged physical profile under mesh refinement;
- domain-size and boundary transfer when supported by the source equations.

Separate field prediction error from the exact discrete heat-flux-divergence error used in the solver.

### Track C — implicit solver-in-the-loop

Freeze one electron-energy equation, source term, boundary contract, time integrator, nonlinear iteration method, solver tolerance and stopping rule.

Compare source FNO, SNB, local closure, Markovian learned closure, and a hybrid where the learned operator serves only as a preconditioner/initial guess for a trusted residual solve.

Record per iteration:
- residual norm and relative residual;
- update norm;
- nonlinear/linear iteration count;
- damping or line-search state;
- convergence/failure and fallback;
- time spent in FNO, solver, transfer and synchronisation.

Do not count one FNO forward pass as an accelerated solve.

### Track D — physical / thermodynamic qualification

- integrated electron-energy balance;
- boundary heat flux and total energy transport;
- positive finite temperature and applicable state constraints;
- flux-limit comparisons only within the applicable physical assumptions;
- timestep and mesh refinement;
- long-horizon stability;
- thermodynamic/passivity diagnostics derived for the nonlocal operator.

A nonlocal flux operator need not have pointwise nonnegative entropy production. Do not impose local sign constraints without deriving them from the declared constitutive law. Qualify the correct global/operator-level dissipation property.

### Track E — OOD and uncertainty

Hold out hot-spot widths beyond both training boundaries, Epperlein–Short wavenumbers, amplitude/gradient/Knudsen combinations, time horizons, and spatial/temporal resolutions. Expand to collision or other plasma-regime changes only when matched kinetic references exist.

Evaluate calibrated intervals, coverage/width, selective risk, false-safe rate, abstention and failure-detection lead time by regime. Unknown regime must be represented as unsupported/unscoreable or abstained, not silently extrapolated.

## 5. Scorecard

**A-priori operator:** d_x q_e MAE/RMSE/relative-L2, spatial spectrum, error by gradient scale/amplitude/wavenumber, integrated flux balance.

**A-posteriori solver:** Te(x,t) trajectory error, peak temperature, heat-front position/speed, energy-integrated error, temporal extrapolation, iterations, convergence failure rate.

**Physics:** total electron-energy residual, boundary flux consistency, positive temperature, valid flux limits, derived nonlocal thermodynamic diagnostics.

**Resolution:** zero-shot fine-grid error, refinement trend, space/time transfer, cost per physical domain-time.

**Economics:** PIC data generation, preprocessing, training, inference, nonlinear iterations, memory, host/device transfer and full end-to-end wall time.

**Uncertainty:** prediction interval coverage/width, proper interval score / CRPS when probabilistic output exists, selective risk/coverage, false-safe rate and abstention by OOD stratum.

## 6. Prospective gates (not achieved)

1. Reproduce source FNO and SNB temperature histories before superiority claims.
2. Zero-shot fine-grid use must preserve registered solver convergence and energy-residual bounds.
3. Beat SNB only on the same kinetic reference, solver and physical regime.
4. Show full solve-time savings, not just inference speed.
5. Preserve accuracy and physical gates under independent spatial and temporal refinement.
6. Unsupported physical regimes trigger declared abstention or a qualified baseline path.
7. No single aggregate score can hide solver failure or energy residual failure.

## 7. Source admission ledger

| Item | State |
|---|---|
| Published article / DOI | Identified |
| Article CC BY 4.0 | Observed |
| OSIRIS as PIC generator | Identified by article |
| Exact source data arrays | Not admitted |
| Author code repository | Not located / not admitted |
| Model weights/configuration | Not admitted |
| Dataset/software licences | Unknown pending artifact inventory |
| FNO/SNB reproduction | Not performed |
| Implicit solver reproduction | Not performed |
| Cross-resolution replication | Not performed |
| Better-than-SOTA result | Not claimed |

## 8. Claim ceiling

This is a source review and benchmark specification. It is not an ICF solver, data reproduction, production closure, validated resolution-invariant model, experimental qualification or facility-control authority.
